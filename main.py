import argparse
import logging

from torch.backends import cudnn
from fastai.vision import *
from fastai.callbacks.general_sched import GeneralScheduler, TrainingPhase

from semimtr.callbacks.callbacks import IterationCallback, TextAccuracy, TopKTextAccuracy, EMA
from semimtr.dataset.dataset import ImageDataset, TextDataset, collate_fn_filter_none
from semimtr.dataset.dataset_selfsupervised import ImageDatasetSelfSupervised
from semimtr.dataset.dataset_consistency_regularization import ImageDatasetConsistencyRegularization
from semimtr.dataset.weighted_sampler import WeightedDatasetRandomSampler
from semimtr.losses.losses import MultiCELosses
from semimtr.losses.seqclr_loss import SeqCLRLoss
from semimtr.losses.consistency_regularization_loss import ConsistencyRegularizationLoss
from semimtr.utils.utils import Config, Logger, MyDataParallel, \
    MyConcatDataset, if_none
from semimtr.utils.test import test_on_each_ds


def _set_random_seed(seed):
    cudnn.deterministic = True
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        logging.warning('You have chosen to seed training. '
                        'This will slow down your training!')


def _get_training_phases(config, n):
    lr = np.array(config.optimizer_lr)
    periods = config.optimizer_scheduler_periods
    sigma = [config.optimizer_scheduler_gamma ** i for i in range(len(periods))]
    phases = [TrainingPhase(n * periods[i]).schedule_hp('lr', lr * sigma[i])
              for i in range(len(periods))]
    return phases


def _get_dataset(ds_type, paths, is_training, config, **kwargs):
    kwargs.update({
        'img_h': config.dataset_image_height,
        'img_w': config.dataset_image_width,
        'max_length': config.dataset_max_length,
        'case_sensitive': config.dataset_case_sensitive,
        'charset_path': config.dataset_charset_path,
        'data_aug': config.dataset_data_aug,
        'deteriorate_ratio': config.dataset_deteriorate_ratio,
        'multiscales': config.dataset_multiscales,
        'data_portion': config.dataset_portion,
        'filter_single_punctuation': config.dataset_filter_single_punctuation,
    })
    datasets = []
    for p in paths:
        subfolders = [f.path for f in os.scandir(p) if f.is_dir()]
        if subfolders:  # Concat all subfolders
            datasets.append(_get_dataset(ds_type, subfolders, is_training, config, **kwargs))
        else:
            datasets.append(ds_type(path=p, is_training=is_training, **kwargs))
    if len(datasets) > 1:
        return MyConcatDataset(datasets)
    else:
        return datasets[0]


def _get_language_databaunch(config):
    kwargs = {
        'max_length': config.dataset_max_length,
        'case_sensitive': config.dataset_case_sensitive,
        'charset_path': config.dataset_charset_path,
        'smooth_label': config.dataset_smooth_label,
        'smooth_factor': config.dataset_smooth_factor,
        'use_sm': config.dataset_use_sm,
    }
    train_ds = TextDataset(config.dataset_train_roots[0], is_training=True, **kwargs)
    valid_ds = TextDataset(config.dataset_valid_roots[0], is_training=False, **kwargs)
    data = DataBunch.create(
        path=train_ds.path,
        train_ds=train_ds,
        valid_ds=valid_ds,
        bs=config.dataset_train_batch_size,
        val_bs=config.dataset_test_batch_size,
        num_workers=config.dataset_num_workers,
        pin_memory=config.dataset_pin_memory)
    logging.info(f'{len(data.train_ds)} training items found.')
    if not data.empty_val:
        logging.info(f'{len(data.valid_ds)} valid items found.')
    return data


def _get_databaunch(config):
    bunch_kwargs = {}
    ds_kwargs = {}
    bunch_kwargs['collate_fn'] = collate_fn_filter_none
    if config.dataset_scheme == 'supervised':
        dataset_class = ImageDataset
    elif config.dataset_scheme == 'selfsupervised':
        dataset_class = ImageDatasetSelfSupervised
        if config.dataset_augmentation_severity is not None:
            ds_kwargs['augmentation_severity'] = config.dataset_augmentation_severity
        ds_kwargs['supervised_flag'] = if_none(config.model_contrastive_supervised_flag, False)
    elif config.dataset_scheme == 'consistency_regularization':
        dataset_class = ImageDatasetConsistencyRegularization
        if config.dataset_augmentation_severity is not None:
            ds_kwargs['augmentation_severity'] = config.dataset_augmentation_severity
        ds_kwargs['supervised_flag'] = if_none(config.model_consistency_regularization_supervised_flag, True)
    else:
        raise NotImplementedError(f'dataset_scheme={config.dataset_scheme} is not supported')
    train_ds = _get_dataset(dataset_class, config.dataset_train_roots, True, config, **ds_kwargs)
    valid_ds = _get_dataset(dataset_class, config.dataset_valid_roots, False, config, **ds_kwargs)
    if config.dataset_test_roots is not None:
        test_ds = _get_dataset(dataset_class, config.dataset_test_roots, False, config, **ds_kwargs)
        bunch_kwargs['test_ds'] = test_ds
    data = ImageDataBunch.create(
        train_ds=train_ds,
        valid_ds=valid_ds,
        bs=config.dataset_train_batch_size,
        val_bs=config.dataset_test_batch_size,
        num_workers=config.dataset_num_workers,
        pin_memory=config.dataset_pin_memory,
        **bunch_kwargs,
    ).normalize(imagenet_stats)
    ar_tfm = lambda x: ((x[0], x[1]), x[1])  # auto-regression only for dtd
    data.add_tfm(ar_tfm)
    if config.dataset_train_weights is not None:
        weighted_sampler = WeightedDatasetRandomSampler(dataset_weights=config.dataset_train_weights,
                                                        dataset_sizes=[len(ds) for ds in train_ds.datasets])
        data.train_dl = data.train_dl.new(shuffle=False, sampler=weighted_sampler)

    logging.info(f'{len(data.train_ds)} training items found.')
    if not data.empty_val:
        logging.info(f'{len(data.valid_ds)} valid items found.')
    if data.test_dl:
        logging.info(f'{len(data.test_ds)} test items found.')

    return data


def _get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    # logging.info(model)
    return model


def _get_learner(config, data, model):
    if config.global_stage == 'pretrain-language':
        metrics = [TopKTextAccuracy(
            k=if_none(config.model_k, 5),
            charset_path=config.dataset_charset_path,
            max_length=config.dataset_max_length + 1,
            case_sensitive=config.dataset_eval_case_sensitive,
            model_eval=config.model_eval)]
    elif config.dataset_scheme == 'selfsupervised' and not config.model_contrastive_supervised_flag:
        metrics = None
    else:
        metrics = [TextAccuracy(
            charset_path=config.dataset_charset_path,
            max_length=config.dataset_max_length + 1,
            case_sensitive=config.dataset_eval_case_sensitive,
            model_eval=config.model_eval)]
    opt_type = getattr(torch.optim, config.optimizer_type)
    if config.dataset_scheme == 'supervised':
        loss_func = MultiCELosses()
    elif config.dataset_scheme == 'selfsupervised':
        loss_func = SeqCLRLoss(supervised_flag=config.model_contrastive_supervised_flag)
    elif config.dataset_scheme == 'consistency_regularization':
        loss_func = ConsistencyRegularizationLoss(
            supervised_flag=config.model_consistency_supervised_flag,
            all_teacher_layers_to_all_student_layers=config.model_consistency_all_to_all,
            teacher_layer=config.model_consistency_teacher_layer,
            student_layer=config.model_consistency_student_layer,
            teacher_one_hot_labels=config.model_consistency_teacher_one_hot,
            consistency_kl_div=config.model_consistency_kl_div,
            teacher_stop_gradients=config.model_consistency_teacher_stop_gradients,
            use_threshold=config.model_consistency_use_threshold,
        )
    else:
        raise NotImplementedError(f'dataset_scheme={config.dataset_scheme} is not supported')
    learner = Learner(data, model, silent=True, model_dir='.',
                      true_wd=config.optimizer_true_wd,
                      wd=config.optimizer_wd,
                      bn_wd=config.optimizer_bn_wd,
                      path=config.global_workdir,
                      metrics=metrics,
                      opt_func=partial(opt_type, **config.optimizer_args or dict()),
                      loss_func=loss_func)

    phases = _get_training_phases(config, len(learner.data.train_dl))
    learner.callback_fns += [
        partial(GeneralScheduler, phases=phases),
        partial(GradientClipping, clip=config.optimizer_clip_grad),
        partial(IterationCallback, name=config.global_name,
                show_iters=config.training_show_iters,
                eval_iters=config.training_eval_iters,
                save_iters=config.training_save_iters,
                start_iters=config.training_start_iters,
                stats_iters=config.training_stats_iters,
                hist_iters=config.training_hist_iters,
                debug=config.global_debug)]

    if config.model_consistency_ema:
        learner.callback_fns += [partial(EMA)]

    if torch.cuda.device_count() > 1:
        logging.info(f'Use {torch.cuda.device_count()} GPUs.')
        learner.model = MyDataParallel(learner.model)

    if config.model_checkpoint:
        with open(config.model_checkpoint, 'rb') as f:
            buffer = io.BytesIO(f.read())
        learner.load(buffer, strict=config.model_strict)
        logging.info(f'Read model from {config.model_checkpoint}')
    elif config.global_phase == 'test':
        learner.load(f'best-{config.global_name}', strict=config.model_strict)
        logging.info(f'Read model from best-{config.global_name}')

    return learner


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help='batch size')
    parser.add_argument('--run_only_test', action='store_true', default=None,
                        help='flag to run only test and skip training')
    parser.add_argument('--test_root', type=str, default=None,
                        help='path to test datasets')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint')
    parser.add_argument('--vision_checkpoint', type=str, default=None,
                        help='path to vision model pretrained')
    parser.add_argument('--debug', action='store_true', default=None,
                        help='flag for running on debug without saving model checkpoints')
    parser.add_argument('--model_eval', type=str, default=None,
                        choices=['alignment', 'vision', 'language'],
                        help='model decoder that outputs predictions')
    parser.add_argument('--workdir', type=str, default=None,
                        help='path to workdir folder')
    parser.add_argument('--subworkdir', type=str, default=None,
                        help='optional prefix to workdir path')
    parser.add_argument('--epochs', type=int, default=None,
                        help='number of training epochs')
    parser.add_argument('--eval_iters', type=int, default=None,
                        help='evaluate performance on validation set every this number iterations')
    args = parser.parse_args()
    config = Config(args.config)
    if args.batch_size is not None:
        config.dataset_train_batch_size = args.batch_size
        config.dataset_valid_batch_size = args.batch_size
        config.dataset_test_batch_size = args.batch_size
    if args.run_only_test is not None:
        config.global_phase = 'Test' if args.run_only_test else 'Train'
    if args.test_root is not None:
        config.dataset_test_roots = [args.test_root]
    args_to_config_dict = {
        'checkpoint': 'model_checkpoint',
        'vision_checkpoint': 'model_vision_checkpoint',
        'debug': 'global_debug',
        'model_eval': 'model_eval',
        'workdir': 'global_workdir',
        'subworkdir': 'global_subworkdir',
        'epochs': 'training_epochs',
        'eval_iters': 'training_eval_iters',
    }
    for args_attr, config_attr in args_to_config_dict.items():
        if getattr(args, args_attr) is not None:
            setattr(config, config_attr, getattr(args, args_attr))
    return config


def main():
    config = _parse_arguments()
    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    _set_random_seed(config.global_seed)
    logging.info(config)

    logging.info('Construct dataset.')
    if config.global_stage == 'pretrain-language':
        data = _get_language_databaunch(config)
    else:
        data = _get_databaunch(config)

    logging.info('Construct model.')
    model = _get_model(config)

    logging.info('Construct learner.')
    learner = _get_learner(config, data, model)

    if config.global_phase == 'train':
        logging.info('Start training.')
        learner.fit(epochs=config.training_epochs,
                    lr=config.optimizer_lr)
        logging.info('Finish training.')

    logging.info('Start testing')
    test_on_each_ds(learner)


if __name__ == '__main__':
    main()
