from semimtr.modules.model_vision import BaseVision
from semimtr.modules.model import Model
from semimtr.modules.seqclr_proj import SeqCLRProj
from semimtr.utils.utils import if_none


class SeqCLRModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.vision = BaseVision(config)
        self.seqclr_proj = SeqCLRProj(config)
        self.loss_weight = if_none(config.model_contrastive_loss_weight, 1.0)

    def forward(self, images, *args, **kwargs):
        v_res_view0 = self.vision(images[:, 0], *args, **kwargs)
        v_res_view1 = self.vision(images[:, 1], *args, **kwargs)

        projected_features_view0 = self.seqclr_proj(v_res_view0)[0]
        projected_features_view1 = self.seqclr_proj(v_res_view1)[0]

        return {'supervised_outputs_view0': v_res_view0,
                'supervised_outputs_view1': v_res_view1,
                'instances_view0': projected_features_view0['instances'],
                'instances_view1': projected_features_view1['instances'],
                'loss_weight': self.loss_weight,
                'name': 'seqclr_vision'}
