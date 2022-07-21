from fastai.vision import *

from SemiMTR.modules.model_abinet_iter import ABINetIterModel


class ConsistencyRegularizationFusionModel(ABINetIterModel):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_teacher_student_loss_weight, 1.0)

    def forward(self, images, *args):
        a_res_teacher, l_res_teacher, v_res_teacher = super().forward(images[:, 0])
        a_res_student, l_res_student, v_res_student = super().forward(images[:, 1])

        return {'teacher_outputs': [a_res_teacher, l_res_teacher, v_res_teacher],
                'student_outputs': [a_res_student, l_res_student, v_res_student],
                'loss_weight': self.loss_weight,
                'name': 'teacher_student_fusion'}
