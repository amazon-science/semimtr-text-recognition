import logging
import torch.nn as nn

from semimtr.modules.model import Model, _default_tfmer_cfg
from semimtr.modules.projections import BidirectionalLSTM, AttnLinear
from semimtr.utils.utils import if_none


class SeqCLRProj(Model):
    def __init__(self, config):
        super().__init__(config)
        vision_d_model = if_none(config.model_vision_d_model, _default_tfmer_cfg['d_model'])
        self.working_layer = if_none(config.model_proj_layer, 'feature')
        if self.working_layer in ['feature', 'backbone_feature', 'alignment_feature']:
            projection_input_size = vision_d_model
        else:
            raise NotImplementedError(f'SeqCLR projection head does not support working layer of {self.working_layer}.')

        if config.model_proj_scheme is None:
            self.projection = nn.Identity()
            projection_output_size = projection_input_size
        elif config.model_proj_scheme == 'bilstm':
            projection_hidden_size = if_none(config.model_proj_hidden, projection_input_size)
            projection_output_size = if_none(config.model_proj_output, projection_input_size)
            self.projection = BidirectionalLSTM(projection_input_size,
                                                projection_hidden_size,
                                                projection_output_size)
        elif config.model_proj_scheme == 'linear_per_column':
            projection_output_size = if_none(config.model_proj_output, projection_input_size)
            self.projection = nn.Linear(projection_input_size, projection_output_size)
        elif config.model_proj_scheme == 'attn_linear_per_column':
            projection_hidden_size = if_none(config.model_proj_hidden, projection_input_size // 2)
            projection_output_size = if_none(config.model_proj_output, self.charset.num_classes)
            self.projection = AttnLinear(projection_input_size,
                                         projection_hidden_size,
                                         projection_output_size)
        else:
            raise NotImplementedError(f'The projection scheme of {config.model_proj_scheme} is not supported.')

        if config.model_instance_mapping_frame_to_instance:
            self.instance_mapping_func = nn.Identity()
        else:
            instance_mapping_fixed = if_none(config.model_instance_mapping_fixed, 'instances')
            w = if_none(config.model_instance_mapping_w, 5)
            if instance_mapping_fixed == 'instances':
                self.instance_mapping_func = nn.AdaptiveAvgPool2d((w, projection_output_size))
            elif instance_mapping_fixed == 'frames':
                self.instance_mapping_func = AvgPool(kernel_size=w, stride=w)
            else:
                raise NotImplementedError(f'instance_mapping_fixed of {instance_mapping_fixed} is not supported')

        if config.model_proj_checkpoint is not None:
            logging.info(f'Read projection head model from {config.model_proj_checkpoint}.')
            self.load(config.model_proj_checkpoint)

    def _single_forward(self, output):
        features = output[self.working_layer]
        if self.working_layer == 'backbone_feature':
            features = features.permute(0, 2, 3, 1).flatten(1, 2)  # (N, E, H, W) -> (N, H*W, E)
        projected_features = self.projection(features)
        projected_instances = self.instance_mapping_func(projected_features)
        return {'instances': projected_instances, 'name': 'projection_head'}

    def forward(self, output, *args):
        if isinstance(output, (tuple, list)):
            return [self._single_forward(o) for o in output]
        else:
            return [self._single_forward(output)]


class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
