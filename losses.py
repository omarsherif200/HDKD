
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
from typing import List
import torch.nn as nn


class LogitDistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            #But we also experiments output_kd.size(0)
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        return loss


class FeatureDistillationLoss(torch.nn.Module):
    """
    This module wraps the Feature distillation criterion which is added to the Logit distillation loss
    and acts as regulaization term and enhances the model generalization
    """
    def __init__(self, teacher_model: torch.nn.Module, student_model: torch.nn.Module, alpha: float,
                 teacher_layers: List[int] = [1, 2, 3], student_layers: List[int] =[1, 2, 3]):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.alpha=alpha
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers

    def forward(self,inputs):
        fkd_loss=0
        for i, layers in enumerate(zip(self.teacher_layers,self.student_layers)):
            teacher_layer,student_layer=layers[0],layers[1]
            teacher_layers=nn.Sequential(*list(self.teacher_model.children())[:teacher_layer])
            student_layers=nn.Sequential(*list(self.student_model.children())[:student_layer])
            with torch.no_grad():
                teacher_feature=teacher_layers(inputs)
            student_feature=student_layers(inputs)
            if teacher_feature.shape != student_feature.shape:
                raise ValueError("""This feature distillation loss requires the selected features
                of the teacher and student models to have the same dimensions while
                the given teacher feature dimension is {} and the student feature dimension is {}
                """.format(teacher_feature.shape,student_feature.shape))
            fkd_loss+= nn.MSELoss()(teacher_feature,student_feature)* pow(i+1,self.alpha)

        return fkd_loss


class TotalDistillationLoss(torch.nn.Module):
    """
    This module wraps the Total distillation loss which is a weighted combination of logit and 
    feature distillation losses. 
    """

    def __init__(self, LogitDistillationLoss: torch.nn.Module, FeatureDistillationLoss: torch.nn.Module, _lambda=10):
        super().__init__()
        self.LogitDistillationLoss=LogitDistillationLoss
        self.FeatureDistillationLoss=FeatureDistillationLoss
        self._lambda=_lambda

    def forward(self,inputs,outputs,labels):
        lkd_loss = self.LogitDistillationLoss(inputs,outputs,labels)
        fkd_loss = self.FeatureDistillationLoss(inputs)
        return lkd_loss + self._lambda * fkd_loss

