import os, sys
from typing import Optional, Dict, Tuple, Any

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import torchaudio

from models.ssl import SSLASR

# Adverserial Reprogramming layer
class ARTLayer(nn.Module):
    def __init__(self, W_regularizer=0.05, **kwargs):
        super(ARTLayer, self).__init__(**kwargs)

        self.W = nn.Parameter(torch.randn(16000, 1), requires_grad=True)
        init.xavier_uniform_(self.W)

    def forward(self, x, dropout=0.4):
        # prog = nn.Dropout(dropout)(self.W) # remove K.tanh
        x = x.unsqueeze(-1)
        out = x + self.W
        return out.squeeze(-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], input_shape[2])

class SSLASRNMR(SSLASR):
    def __init__(self, base_args, SSL_backbone_args):
        super(SSLASRNMR, self).__init__(base_args, SSL_backbone_args)
        if self.nmr:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder_phn.parameters():
                param.requires_grad = False

    
    def forward(self, x, target, input_lengths, label_lengths) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        The forward pass for training.
        Args:
            x (torch.Tensor): The input waveform
            target (torch.Tensor): The target label
            input_lengths (torch.Tensor)
            label_lengths (torch.Tensor)
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: The loss and log statistics
        """
        x = self.nmr_layer(x)
        outputs = self._forward(x)

        return self._compute_loss(outputs, target, input_lengths, label_lengths)
    
    def set_nmr(self):
        self.nmr_layer = ARTLayer()
    
