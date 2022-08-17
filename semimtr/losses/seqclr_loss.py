import torch
import torch.nn as nn
import torch.nn.functional as F

from semimtr.losses.losses import MultiCELosses


class SeqCLRLoss(nn.Module):
    def __init__(self, temp=0.1, reduction="batchmean", record=True, supervised_flag=False):
        super().__init__()
        self.reduction = reduction
        self.temp = temp
        self.record = record
        self.supervised_flag = supervised_flag
        self.supervised_loss = MultiCELosses()

    @property
    def last_losses(self):
        return self.losses

    def _seqclr_loss(self, features0, features1, n_instances_per_view, n_instances_per_image):
        instances = torch.cat((features0, features1), dim=0)
        normalized_instances = F.normalize(instances, dim=1)
        similarity_matrix = normalized_instances @ normalized_instances.T
        similarity_matrix_exp = (similarity_matrix / self.temp).exp_()
        cross_entropy_denominator = similarity_matrix_exp.sum(dim=1) - similarity_matrix_exp.diag()
        cross_entropy_nominator = torch.cat((
            similarity_matrix_exp.diagonal(offset=n_instances_per_view)[:n_instances_per_view],
            similarity_matrix_exp.diagonal(offset=-n_instances_per_view)
        ), dim=0)
        cross_entropy_similarity = cross_entropy_nominator / cross_entropy_denominator
        loss = - cross_entropy_similarity.log()

        if self.reduction == "batchmean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean_instances_per_image":
            loss = loss.sum() / n_instances_per_image
        return loss

    def forward(self, outputs, gt_dict, *args, **kwargs):
        if isinstance(outputs, (tuple, list)):
            raise NotImplementedError
        self.losses = {}
        ce_loss = 0
        if self.supervised_flag:
            ce_loss += self.supervised_loss(outputs['supervised_outputs_view0'], gt_dict, record=True)
            ce_view0_last_losses = self.supervised_loss.last_losses
            ce_loss += self.supervised_loss(outputs['supervised_outputs_view1'], gt_dict, record=True)
            ce_view1_last_losses = self.supervised_loss.last_losses
            self.losses.update({k: (v + ce_view1_last_losses[k]) / 2 for k, v in ce_view0_last_losses.items()})

        loss_name = outputs.get('name')
        gt_lengths = gt_dict['length']
        seqclr_loss = 0
        if loss_name == 'seqclr_fusion':
            pt_length = outputs['pt_lengths']
            pt_length[gt_lengths != 0] = gt_lengths[gt_lengths != 0]  # Use ground truth length if available
            # TODO: spread on gpus
            for features0, features1 in zip(outputs['instances_view0'], outputs['instances_view1']):
                features0 = MultiCELosses._flatten(sources=features0, lengths=pt_length)
                features1 = MultiCELosses._flatten(sources=features1, lengths=pt_length)
                n_instances_per_image = pt_length.float().mean()
                n_instances_per_view = features0.shape[0]
                seqclr_loss += self._seqclr_loss(features0, features1, n_instances_per_view, n_instances_per_image)
            seqclr_loss /= len(outputs['instances_view0'])  # Average seqclr losses
        else:
            features0 = torch.flatten(outputs['instances_view0'], start_dim=0, end_dim=1)
            features1 = torch.flatten(outputs['instances_view1'], start_dim=0, end_dim=1)
            n_instances_per_image = outputs['instances_view0'].shape[1]
            n_instances_per_view = outputs['instances_view0'].shape[0] * n_instances_per_image
            seqclr_loss += self._seqclr_loss(features0, features1, n_instances_per_view, n_instances_per_image)
        seqclr_loss *= outputs['loss_weight']

        if self.record and loss_name is not None:
            self.losses[f'{loss_name}_loss'] = seqclr_loss
        return seqclr_loss + ce_loss
