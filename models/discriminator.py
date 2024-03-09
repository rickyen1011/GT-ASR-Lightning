import os
from typing import Optional, Dict, Tuple, Any

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from utils.loss import FocalLoss


class Discriminator(nn.Module):
    def __init__(self, base_args):
        super(Discriminator, self).__init__()
        for key, value in base_args.items():
            setattr(self, key, value)
        
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=self.feature_dim, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=self.num_domain)
        )
        # weights = torch.FloatTensor(self.class_weights).cuda()
        # self.criterion = FocalLoss(gamma=0.7, alpha=weights)

    def _compute_loss(self, outputs_d, targets_d):
        xe_loss = F.cross_entropy(outputs_d, targets_d)
        # xe_loss = self.criterion(outputs_d, targets_d)
        return xe_loss, {'CrossEntropyLoss': xe_loss.detach()}
    
    def _forward(self, x):
        outputs_d = self.discriminator(x)
        outputs_d = torch.mean(outputs_d, dim=1)
        outputs_d = F.softmax(outputs_d, dim=-1)
        
        return outputs_d
    
    def forward(self, x, targets_d) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        The forward pass for training.
        Args:
            x (torch.Tensor): The input feature
            target_d (torch.Tensor): The target domain label
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: The loss and log statistics
        """
        outputs_d = self._forward(x)
        return self._compute_loss(outputs_d, targets_d)

    def inference(self, x):
        outputs_d = self._forward(x)
        return outputs_d

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None