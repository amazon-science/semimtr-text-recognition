import numpy as np
import torch
from torchvision import transforms

from semimtr.utils.transforms import ImageToPIL, ImageToArray
from semimtr.dataset.dataset import ImageDataset
from semimtr.dataset.augmentation_pipelines import get_augmentation_pipeline


class ImageDatasetSelfSupervised(ImageDataset):
    """
    Image Dataset for Self Supervised training that outputs pairs of images
    """

    def __init__(self, augmentation_severity: int = 1, supervised_flag=False, **kwargs):
        super().__init__(**kwargs)
        self.supervised_flag = supervised_flag
        if self.data_aug:
            if augmentation_severity == 0 or (not self.is_training and supervised_flag):
                regular_aug = self.augment_tfs.transforms if hasattr(self, 'augment_tfs') else []
                self.augment_tfs = transforms.Compose([ImageToPIL()] + regular_aug + [ImageToArray()])
            else:
                self.augment_tfs = get_augmentation_pipeline(augmentation_severity).augment_image

    def _process_training(self, image):
        image = np.array(image)
        image_views = []
        for _ in range(2):
            if self.data_aug:
                image_view = self.augment_tfs(image)
            else:
                image_view = image
            image_views.append(self.totensor(self.resize(image_view)))
        return np.stack(image_views, axis=0)

    def _process_test(self, image):
        return self._process_training(image)

    def _label_postprocessing(self, text):
        y = super()._label_postprocessing(text)
        if text.lower() == 'unlabeleddata':
            y['length'] = torch.tensor(0).to(dtype=torch.long)  # don't calculate cross entropy on this image
        return y
