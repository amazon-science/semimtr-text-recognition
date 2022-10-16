import torch
import torch.nn as nn

from semimtr.modules.model_vision import BaseVision
from semimtr.modules.model_language import BCNLanguage
from semimtr.modules.model_alignment import BaseAlignment
from semimtr.utils.utils import if_none


class ABINetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_alignment = if_none(config.model_use_alignment, True)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        # self.vision_no_grad = if_none(config.model_vision_no_grad, False)
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        if self.use_alignment: self.alignment = BaseAlignment(config)

    def forward(self, images, *args, **kwargs):
        v_res = self.vision(images, *args, **kwargs)
        v_tokens = torch.softmax(v_res['logits'], dim=-1)
        v_lengths = v_res['pt_lengths'].clamp_(2, self.max_length)  # TODO:move to langauge model

        samples = {'label': v_tokens, 'length': v_lengths}
        l_res = self.language(samples, *args, **kwargs)
        if not self.use_alignment:
            return l_res, v_res
        l_feature, v_feature = l_res['feature'], v_res['feature']

        a_res = self.alignment(l_feature, v_feature, *args, **kwargs)
        return a_res, l_res, v_res
