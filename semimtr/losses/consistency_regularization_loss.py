import torch
import torch.nn as nn
import torch.nn.functional as F

from semimtr.losses.losses import MultiCELosses

layer_name_to_output_ind = {'alignment': 0, 'language': 1, 'vision': 2}


class ConsistencyRegularizationLoss(nn.Module):
    def __init__(
            self,
            record=True,
            supervised_flag=False,
            all_teacher_layers_to_all_student_layers=False,
            teacher_layer='vision',
            student_layer='all',
            teacher_one_hot_labels=False,
            consistency_kl_div=False,
            teacher_stop_gradients=True,
            use_threshold=False,
            threshold_value=0.9
    ):
        super().__init__()
        if not all_teacher_layers_to_all_student_layers:
            if teacher_layer not in layer_name_to_output_ind.keys():
                raise NotImplementedError(f'Teacher layer can be one of {list(layer_name_to_output_ind.keys())}')
            if student_layer != 'all' and student_layer not in layer_name_to_output_ind.keys():
                raise NotImplementedError(f'Student layer can be \'all\' or '
                                          f'one of {list(layer_name_to_output_ind.keys())}')
            self.teacher_layer_ind = layer_name_to_output_ind[teacher_layer]
            self.student_layer_ind = None if student_layer == 'all' else layer_name_to_output_ind[student_layer]
        self.record = record
        self.supervised_flag = supervised_flag
        self.supervised_ce_loss = MultiCELosses()
        self.consistency_ce_loss = MultiCELosses(kl_div=consistency_kl_div)
        self.all_teacher_layers_to_all_student_layers = all_teacher_layers_to_all_student_layers
        self.teacher_one_hot_labels = teacher_one_hot_labels
        self.teacher_stop_gradients = teacher_stop_gradients
        self.use_threshold = use_threshold
        self.threshold_value = threshold_value

    @property
    def last_losses(self):
        return self.losses

    def forward(self, outputs, *args):
        if isinstance(outputs, (tuple, list)):
            raise NotImplementedError
        self.losses = {}
        ce_loss = 0
        if self.supervised_flag:
            ce_loss_teacher = self.supervised_ce_loss(outputs['teacher_outputs'], *args)
            self.losses.update({f'{k}_teacher': v for k, v in self.supervised_ce_loss.last_losses.items()})
            ce_loss_student = self.supervised_ce_loss(outputs['student_outputs'], *args)
            self.losses.update({f'{k}_student': v for k, v in self.supervised_ce_loss.last_losses.items()})
            ce_loss += ce_loss_teacher + ce_loss_student

        if not self.all_teacher_layers_to_all_student_layers:
            teacher_predictions = outputs['teacher_outputs'][self.teacher_layer_ind]
            pt_labels_teacher, pt_lengths_teacher, threshold_mask = self.create_teacher_labels(teacher_predictions)
            if self.student_layer_ind is not None:
                student_predictions = outputs['student_outputs'][self.student_layer_ind]
            else:
                student_predictions = outputs['student_outputs']
            pt_teacher = {'label': pt_labels_teacher, 'length': pt_lengths_teacher}
            ce_loss_student_teacher = self.consistency_ce_loss(student_predictions, pt_teacher, *args[1:],
                                                               mask=threshold_mask)
        else:
            ce_loss_student_teacher = 0
            for teacher_predictions, student_predictions in zip(outputs['teacher_outputs'], outputs['student_outputs']):
                pt_labels_teacher, pt_lengths_teacher, threshold_mask = self.create_teacher_labels(teacher_predictions)
                pt_teacher = {'label': pt_labels_teacher, 'length': pt_lengths_teacher}
                ce_loss_student_teacher += self.consistency_ce_loss(student_predictions, pt_teacher, *args[1:],
                                                                    mask=threshold_mask)
        self.losses.update({f'{k}_teacher_student': v for k, v in self.consistency_ce_loss.last_losses.items()})
        ce_loss += outputs['loss_weight'] * ce_loss_student_teacher
        return ce_loss

    def create_teacher_labels(self, teacher_predictions):
        if isinstance(teacher_predictions, list):
            teacher_predictions = teacher_predictions[-1]
        pt_lengths_teacher = teacher_predictions['pt_lengths']
        pt_logits_teacher = teacher_predictions['logits']
        if self.teacher_stop_gradients:
            pt_logits_teacher = pt_logits_teacher.detach()
        pt_labels_teacher = F.softmax(pt_logits_teacher, dim=-1)
        max_values, max_indices = torch.max(pt_labels_teacher, dim=-1, keepdim=True)
        if self.teacher_one_hot_labels:
            pt_labels_teacher = torch.zeros_like(pt_logits_teacher).scatter_(-1, max_indices, 1)
        threshold_mask = (max_values.squeeze() > self.threshold_value).float() if self.use_threshold else None
        return pt_labels_teacher, pt_lengths_teacher, threshold_mask
