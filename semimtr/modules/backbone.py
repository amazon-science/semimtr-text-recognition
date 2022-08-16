import torch.nn as nn

from semimtr.modules.model import _default_tfmer_cfg
from semimtr.modules.resnet import resnet45
from semimtr.modules.transformer import (PositionalEncoding,
                                         TransformerEncoder,
                                         TransformerEncoderLayer)
from semimtr.utils.utils import if_none


class ResTranformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = resnet45()

        self.d_model = if_none(config.model_vision_d_model, _default_tfmer_cfg['d_model'])
        nhead = if_none(config.model_vision_nhead, _default_tfmer_cfg['nhead'])
        d_inner = if_none(config.model_vision_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = if_none(config.model_vision_dropout, _default_tfmer_cfg['dropout'])
        activation = if_none(config.model_vision_activation, _default_tfmer_cfg['activation'])
        num_layers = if_none(config.model_vision_backbone_ln, 2)

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=8 * 32)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead,
                                                dim_feedforward=d_inner, dropout=dropout, activation=activation)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, images):
        feature = self.resnet(images)
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).permute(2, 0, 1)
        feature = self.pos_encoder(feature)
        feature = self.transformer(feature)
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        return feature
