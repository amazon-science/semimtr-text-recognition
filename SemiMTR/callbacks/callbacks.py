import logging

import editdistance as ed
import torchvision.utils as vutils
from SemiMTR.utils.utils import CharsetMapper, Timer, blend_mask
from fastai.callbacks.tensorboard import (LearnerTensorboardWriter)
from fastai.vision import *
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms


class IterationCallback(LearnerTensorboardWriter):
    "A `TrackerCallback` that monitor in each iteration."

    def __init__(self, learn: Learner, name: str = 'model', checpoint_keep_num=5,
                 show_iters: int = 50, eval_iters: int = 1000, save_iters: int = 20000,
                 start_iters: int = 0, stats_iters=20000, hist_iters=20000, debug=False):
        super().__init__(learn, base_dir='.', name=learn.path, loss_iters=show_iters,
                         stats_iters=stats_iters, hist_iters=hist_iters)
        self.name, self.bestname = Path(name).name, f'best-{Path(name).name}'
        self.show_iters = show_iters
        self.eval_iters = eval_iters
        self.save_iters = save_iters
        self.start_iters = start_iters
        self.checpoint_keep_num = checpoint_keep_num
        self.metrics_root = 'metrics/'  # rewrite
        self.timer = Timer()
        self.host = True
        self.debug = debug

    def _write_metrics(self, iteration: int, names: List[str], last_metrics: MetricsList) -> None:
        "Writes training metrics to Tensorboard."
        for i, name in enumerate(names):
            if last_metrics is None or len(last_metrics) < i + 1: return
            scalar_value = last_metrics[i]
            self._write_scalar(name=name, scalar_value=scalar_value, iteration=iteration)

    def _write_sub_loss(self, iteration: int, last_losses: dict) -> None:
        "Writes sub loss to Tensorboard."
        for name, loss in last_losses.items():
            scalar_value = to_np(loss)
            tag = self.metrics_root + name
            self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def _save(self, name):
        if self.debug: return
        if isinstance(self.learn.model, DistributedDataParallel):
            tmp = self.learn.model
            self.learn.model = self.learn.model.module
            self.learn.save(name)
            self.learn.model = tmp
        else:
            self.learn.save(name)

    def _validate(self, dl=None, callbacks=None, metrics=None, keeped_items=False):
        "Validate on `dl` with potential `callbacks` and `metrics`."
        dl = ifnone(dl, self.learn.data.valid_dl)
        metrics = ifnone(metrics, self.learn.metrics)
        cb_handler = CallbackHandler(ifnone(callbacks, []), metrics)
        cb_handler.on_train_begin(1, None, metrics);
        cb_handler.on_epoch_begin()
        if keeped_items: cb_handler.state_dict.update(dict(keeped_items=[]))
        val_metrics = validate(self.learn.model, dl, self.loss_func, cb_handler)
        cb_handler.on_epoch_end(val_metrics)
        if keeped_items:
            return cb_handler.state_dict['keeped_items']
        else:
            return cb_handler.state_dict['last_metrics']

    def jump_to_epoch_iter(self, epoch: int, iteration: int) -> None:
        try:
            self.learn.load(f'{self.name}_{epoch}_{iteration}', purge=False)
            logging.info(f'Loaded {self.name}_{epoch}_{iteration}')
        except:
            logging.info(f'Model {self.name}_{epoch}_{iteration} not found.')

    def on_train_begin(self, n_epochs, **kwargs):
        # TODO: can not write graph here
        # super().on_train_begin(**kwargs)
        self.best = -float('inf')
        self.timer.tic()
        if self.host:
            checkpoint_path = self.learn.path / 'checkpoint.yaml'
            if checkpoint_path.exists():
                os.remove(checkpoint_path)
            open(checkpoint_path, 'w').close()
        return {'skip_validate': True, 'iteration': self.start_iters}  # disable default validate

    def on_batch_begin(self, **kwargs: Any) -> None:
        self.timer.toc_data()
        super().on_batch_begin(**kwargs)

    def on_batch_end(self, iteration, epoch, last_loss, smooth_loss, train, **kwargs):
        super().on_batch_end(last_loss, iteration, train, **kwargs)
        if iteration == 0: return

        if iteration % self.loss_iters == 0:
            last_losses = self.learn.loss_func.last_losses
            self._write_sub_loss(iteration=iteration, last_losses=last_losses)
            self.tbwriter.add_scalar(tag=self.metrics_root + 'lr',
                                     scalar_value=self.opt.lr, global_step=iteration)

        if iteration % self.show_iters == 0:
            log_str = f'epoch {epoch} iter {iteration}: loss = {last_loss:6.4f},  ' \
                      f'smooth loss = {smooth_loss:6.4f} '
            logging.info(log_str)
            # log_str = f'data time = {self.timer.data_diff:.4f}s, runing time = {self.timer.running_diff:.4f}s'
            # logging.info(log_str)

        if iteration % self.eval_iters == 0:
            self._eval_model(iteration, epoch)

        if iteration % self.save_iters == 0 and self.host:
            logging.info(f'Save model {self.name}_{epoch}_{iteration}')
            filename = f'{self.name}_{epoch}_{iteration}'
            self._save(filename)

            checkpoint_path = self.learn.path / 'checkpoint.yaml'
            if not checkpoint_path.exists():
                open(checkpoint_path, 'w').close()
            with open(checkpoint_path, 'r') as file:
                checkpoints = yaml.load(file, Loader=yaml.FullLoader) or dict()
            checkpoints['all_checkpoints'] = (
                    checkpoints.get('all_checkpoints') or list())
            checkpoints['all_checkpoints'].insert(0, filename)
            if len(checkpoints['all_checkpoints']) > self.checpoint_keep_num:
                removed_checkpoint = checkpoints['all_checkpoints'].pop()
                removed_checkpoint = self.learn.path / self.learn.model_dir / f'{removed_checkpoint}.pth'
                os.remove(removed_checkpoint)
            checkpoints['current_checkpoint'] = filename
            with open(checkpoint_path, 'w') as file:
                yaml.dump(checkpoints, file)

        self.timer.toc_running()

    def _eval_model(self, iteration=None, epoch=None):
        if iteration is None or epoch is None:
            msg_start = f'last iteration'
        else:
            msg_start = f'epoch {epoch} iter {iteration}'
        # 1. Record time
        log_str = f'average data time = {self.timer.average_data_time():.4f}s, ' \
                  f'average running time = {self.timer.average_running_time():.4f}s'
        logging.info(log_str)

        # 2. Call validate
        last_metrics = self._validate()
        self.learn.model.train()
        names = self._metrics_to_logging(last_metrics, msg_start)
        if len(last_metrics) > 1:
            current_eval_loss = last_metrics[2]
        else:  # only eval loss
            current_eval_loss = last_metrics[0]

        if iteration is not None:
            self._write_metrics(iteration, names, last_metrics)

        # 3. Save best model
        if current_eval_loss is not None and current_eval_loss > self.best:
            logging.info(f'Better model found at {msg_start} with accuracy value: {current_eval_loss:6.4f}.')
            self.best = current_eval_loss
            self._save(f'{self.bestname}')

    @staticmethod
    def _metrics_to_logging(metrics, msg_start, dl_len=None):
        log_str = f'{msg_start}: '
        if dl_len is not None:
            log_str += f'dataset size = {dl_len} '
        log_str += f'eval loss = {metrics[0]:6.3f},  '
        names = ['eval_loss']
        if len(metrics) > 1:
            log_str += f'ccr = {metrics[1]:6.3f},  cwr = {metrics[2]:6.3f},  ' \
                       f'ted = {metrics[3]:6.3f},  ned = {metrics[4]:6.0f},  ' \
                       f'ted/w = {metrics[5]:6.3f}, '
            names += ['ccr', 'cwr', 'ted', 'ned', 'ted/w']
        logging.info(log_str)
        return names

    def on_train_end(self, **kwargs):
        logging.info('Train ended')
        self._eval_model()
        self.learn.load(f'{self.bestname}', purge=False)
        logging.info(f'Loading best model from {self.learn.path}/{self.learn.model_dir}/{self.bestname}.pth')

    def on_epoch_end(self, last_metrics: MetricsList, iteration: int, **kwargs) -> None:
        self._write_embedding(iteration=iteration)


class EMA(LearnerCallback):
    def on_step_end(self, **kwargs):
        if isinstance(self.learn.model, nn.DataParallel):
            self.learn.model.module.update_teacher()
        else:
            self.learn.model.update_teacher()


class TextAccuracy(Callback):
    _names = ['ccr', 'cwr', 'ted', 'ned', 'ted/w']

    def __init__(self, charset_path, max_length, case_sensitive, model_eval):
        self.charset_path = charset_path
        self.max_length = max_length
        self.case_sensitive = case_sensitive
        self.charset = CharsetMapper(charset_path, self.max_length)
        self.names = self._names

        self.model_eval = model_eval or 'alignment'
        assert self.model_eval in ['vision', 'language', 'alignment']

    def on_epoch_begin(self, **kwargs):
        self.total_num_char = 0.
        self.total_num_word = 0.
        self.correct_num_char = 0.
        self.correct_num_word = 0.
        self.total_ed = 0.
        self.total_ned = 0.

    @staticmethod
    def _extract_output_list(last_output):
        if isinstance(last_output, (tuple, list)):
            return last_output
        elif isinstance(last_output, dict) and 'supervised_outputs_view0' in last_output:
            return last_output['supervised_outputs_view0']
        elif isinstance(last_output, dict) and 'teacher_outputs' in last_output:
            return last_output['teacher_outputs']
        else:
            return

    def _get_output(self, last_output):
        output_list = self._extract_output_list(last_output)
        if output_list is not None:
            if isinstance(output_list, (tuple, list)):
                for res in output_list:
                    if res['name'] == self.model_eval: output = res
            else:
                output = output_list
        else:
            output = last_output
        return output

    def _update_output(self, last_output, items):
        output_list = self._extract_output_list(last_output)
        if output_list is not None:
            if isinstance(output_list, (tuple, list)):
                for res in output_list:
                    if res['name'] == self.model_eval: res.update(items)
            else:
                output_list.update(items)
        else:
            last_output.update(items)
        return last_output

    def on_batch_end(self, last_output, last_target, **kwargs):
        output = self._get_output(last_output)
        logits, pt_lengths = output['logits'], output['pt_lengths']
        pt_text, pt_scores, pt_lengths_ = self.decode(logits)
        if not (pt_lengths == pt_lengths_).all():
            for pt_lengths_i, pt_lengths_i_, pt_text_i in zip(pt_lengths, pt_lengths_, pt_text):
                if pt_lengths_i != pt_lengths_i_:
                    logging.warning(f'{pt_lengths_i} != {pt_lengths_i_} for {pt_text_i}')
        last_output = self._update_output(last_output, {'pt_text': pt_text, 'pt_scores': pt_scores})

        pt_text = [self.charset.trim(t) for t in pt_text]
        label = last_target[0]
        if label.dim() == 3: label = label.argmax(dim=-1)  # one-hot label
        gt_text = [self.charset.get_text(l, trim=True) for l in label]

        for i in range(len(gt_text)):
            if not self.case_sensitive:
                gt_text[i], pt_text[i] = gt_text[i].lower(), pt_text[i].lower()
            distance = ed.eval(gt_text[i], pt_text[i])
            self.total_ed += distance
            self.total_ned += float(distance) / max(len(gt_text[i]), 1)

            if gt_text[i] == pt_text[i]:
                self.correct_num_word += 1
            self.total_num_word += 1

            for j in range(min(len(gt_text[i]), len(pt_text[i]))):
                if gt_text[i][j] == pt_text[i][j]:
                    self.correct_num_char += 1
            self.total_num_char += len(gt_text[i])

        return {'last_output': last_output}

    def on_epoch_end(self, last_metrics, **kwargs):
        mets = [self.correct_num_char / self.total_num_char,
                self.correct_num_word / self.total_num_word,
                self.total_ed,
                self.total_ned,
                self.total_ed / self.total_num_word]
        return add_metrics(last_metrics, mets)

    def decode(self, logit):
        """ Greed decode """
        # TODO: test running time and decode on GPU
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = self.charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(self.charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, self.max_length))  # one for end-token
        pt_scores = torch.stack(pt_scores)
        pt_lengths = pt_scores.new_tensor(pt_lengths, dtype=torch.long)
        return pt_text, pt_scores, pt_lengths


class TopKTextAccuracy(TextAccuracy):
    _names = ['ccr', 'cwr']

    def __init__(self, k, charset_path, max_length, case_sensitive, model_eval):
        self.k = k
        self.charset_path = charset_path
        self.max_length = max_length
        self.case_sensitive = case_sensitive
        self.charset = CharsetMapper(charset_path, self.max_length)
        self.names = self._names

    def on_epoch_begin(self, **kwargs):
        self.total_num_char = 0.
        self.total_num_word = 0.
        self.correct_num_char = 0.
        self.correct_num_word = 0.

    def on_batch_end(self, last_output, last_target, **kwargs):
        logits, pt_lengths = last_output['logits'], last_output['pt_lengths']
        gt_labels, gt_lengths = last_target[:]

        for logit, pt_length, label, length in zip(logits, pt_lengths, gt_labels, gt_lengths):
            word_flag = True
            for i in range(length):
                char_logit = logit[i].topk(self.k)[1]
                char_label = label[i].argmax(-1)
                if char_label in char_logit:
                    self.correct_num_char += 1
                else:
                    word_flag = False
                self.total_num_char += 1
            if pt_length == length and word_flag:
                self.correct_num_word += 1
            self.total_num_word += 1

    def on_epoch_end(self, last_metrics, **kwargs):
        mets = [self.correct_num_char / self.total_num_char,
                self.correct_num_word / self.total_num_word,
                0., 0., 0.]
        return add_metrics(last_metrics, mets)
