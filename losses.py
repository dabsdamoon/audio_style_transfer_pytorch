import torch
import torch.nn as nn
import torch.nn.functional as F

##### Define loss module
class CustomLoss(nn.Module):
    def __init__(self, target, loss_type="custom"):
        super(CustomLoss, self).__init__()
        self.target = target.detach()
        self.loss_type = loss_type

    def forward(self, input):
        if self.loss_type == "custom":
            self.loss = self.custom_tf_nn_l2(input, self.target)
        elif self.loss_type == "MSE":
            self.loss = F.mse_loss(input, self.target, reduction="sum")
        else:
            raise ValueError("Loss type mismatch")

        return input

    def custom_tf_nn_l2(self, a,b):
        '''
        Notice that tensorflow.nn.l2 loss is a bit different from regular MSE loss.
        For more information, visit: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/l2_loss
        '''
        t = a-b
        output = torch.sum(t**2) / 2
        return output