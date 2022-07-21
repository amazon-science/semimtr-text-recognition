from torch.utils.data import ConcatDataset
from fastai.vision import *
from semimtr.callbacks.callbacks import IterationCallback


def test_on_each_ds(learner):
    test_dls = dataset_to_dataloader_list(
        learner.data.test_ds,
        batch_size=learner.data.test_dl.batch_size,
        device=learner.data.device,
        transforms=learner.data.test_dl.tfms,
        collate_fn=learner.data.test_dl.collate_fn
    )
    last_metrics_list = []
    ds_sizes = []
    loss_dict = {}
    for dl in test_dls:
        dl_name = dl.dataset.name
        last_metrics = learner.validate(dl=dl)
        last_metrics_list.append(last_metrics)
        IterationCallback._metrics_to_logging(last_metrics, f'{dl_name} test', dl_len=len(dl.dataset))
        ds_sizes.append(len(dl.dataset))
        loss_dict[dl_name] = [ds_sizes[-1]] + last_metrics

    last_metrics_average = np.average(last_metrics_list, axis=0, weights=ds_sizes)
    names = IterationCallback._metrics_to_logging(last_metrics_average, f'average test')
    loss_dict['Average'] = [sum(ds_sizes)] + list(last_metrics_average)
    df = pd.DataFrame.from_dict(loss_dict, orient='index', columns=['size'] + names)
    df.T.to_csv(learner.path / learner.model_dir / f'test_results.csv')


def dataset_to_dataloader_list(dataset, batch_size, device, transforms, collate_fn):
    if isinstance(dataset, ConcatDataset):
        test_dls = []
        for ds in dataset.datasets:
            test_dls.extend(dataset_to_dataloader_list(ds, batch_size, device, transforms, collate_fn))
        return test_dls
    else:
        return [DeviceDataLoader(DataLoader(dataset, batch_size), device, transforms, collate_fn)]
