from fastai.vision import *
import torch.nn as nn

from SemiMTR.modules.model_abinet_iter import ABINetIterModel


class TeacherStudentFusionEMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.student = ABINetIterModel(config)
        self.teacher = ABINetIterModel(config)
        self.teacher.load_state_dict(self.student.state_dict())
        self.loss_weight = ifnone(config.model_teacher_student_loss_weight, 1.0)
        self.decay = ifnone(config.model_teacher_student_ema_decay, 0.9999)

    def update_teacher(self):
        with torch.no_grad():
            for param_student, param_teacher in zip(self.student.parameters(), self.teacher.parameters()):
                param_teacher.data.mul_(self.decay).add_((1 - self.decay) * param_student.detach().data)

    def forward(self, images, *args):
        with torch.no_grad():
            a_res_teacher, l_res_teacher, v_res_teacher = self.teacher(images[:, 0])
        a_res_student, l_res_student, v_res_student = self.student(images[:, 1])

        return {'teacher_outputs': [a_res_teacher, l_res_teacher, v_res_teacher],
                'student_outputs': [a_res_student, l_res_student, v_res_student],
                'loss_weight': self.loss_weight,
                'name': 'teacher_student_fusion'}
