import os, sys
import copy
import math
from typing import Optional, Dict, Tuple, Any

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import torchaudio

from models.ssl import SSLASR

class SSLASRLP(SSLASR):
    def __init__(self, base_args, SSL_backbone_args):
        super(SSLASRLP, self).__init__(base_args, SSL_backbone_args)
        if self.linear_prob:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _compute_loss(self, outputs, target, input_lengths, label_lengths, kld_target=None):
        ctc_loss = self.ctc_criterion(
            F.log_softmax(outputs, dim=-1).transpose(0, 1), target, input_lengths, label_lengths
        )
        if self.map:
            if self.weight_target.device != self.decoder_phn.weight.device:
                self.weight_target = self.weight_target.to(self.decoder_phn.weight.device)
                self.bias_target = self.bias_target.to(self.decoder_phn.bias.device)
            assert self.weight_target.requires_grad == False
            assert self.bias_target.requires_grad == False
            regularization_loss = nn.MSELoss()(self.decoder_phn.weight, self.weight_target) + nn.MSELoss()(self.decoder_phn.bias, self.bias_target)
            loss = ctc_loss + self.regularization_weight * regularization_loss
            return loss, {'CTCLoss': ctc_loss.detach(), 'RegularizationLoss': regularization_loss.detach()}
        elif self.kld:
            assert kld_target != None
            if kld_target.device != outputs.device:
                kld_target = kld_target.to(outputs.device)
            assert kld_target.requires_grad == False
            log_outputs = F.log_softmax(outputs, dim=-1)
            log_kld_target = F.log_softmax(kld_target, dim=-1)
            kld_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)(log_outputs, log_kld_target)
            loss = ctc_loss + self.kld_weight * kld_loss
            return loss, {'CTCLoss': ctc_loss.detach(), 'KLDLoss': kld_loss.detach()}
        else:
            return ctc_loss, {'CTCLoss': ctc_loss.detach()}

    def _forward_li_model(self, x):
        rep, _ = self.encoder.extract_features(x)
        outputs_phn = self.pretrained_decoder(rep[-1])
        
        return outputs_phn.detach()
    
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
        outputs = self._forward(x)
        if self.kld:
            if not hasattr(self, "pretrained_decoder"):
                self.set_pretrained_model()
            outputs_target = self._forward_li_model(x)
            return self._compute_loss(outputs, target, input_lengths, label_lengths, outputs_target)
        else:
            return self._compute_loss(outputs, target, input_lengths, label_lengths)

    def set_target_weight(self):
        self.weight_target = self.decoder_phn.weight.detach()
        self.bias_target = self.decoder_phn.bias.detach()

    def set_pretrained_model(self):
        self.pretrained_decoder = nn.Linear(self.d_encoder, self.num_class+1)
        self.pretrained_decoder.weight = nn.Parameter(self.decoder_phn.weight.clone())
        self.pretrained_decoder.bias = nn.Parameter(self.decoder_phn.bias.clone())
        self.pretrained_decoder.weight.requires_grad = False
        self.pretrained_decoder.bias.requires_grad = False
        self.pretrained_decoder.weight = self.pretrained_decoder.weight.to(self.decoder_phn.weight.device)
        self.pretrained_decoder.bias = self.pretrained_decoder.bias.to(self.decoder_phn.bias.device)
    
    def expand_linear(self):
        new_weights = torch.empty((self.num_class_target+1, self.d_encoder))
        new_bias = torch.empty((self.num_class_target+1))

        init.kaiming_uniform_(new_weights, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(new_weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(new_bias, -bound, bound)
        
        new_weights[:self.num_class+1][:] = self.decoder_phn.weight.clone()
        new_bias[:self.num_class+1] = self.decoder_phn.bias.clone()

        self.decoder_phn = nn.Linear(self.d_encoder, self.num_class_target+1)
        self.decoder_phn.weight = nn.Parameter(new_weights)
        self.decoder_phn.bias = nn.Parameter(new_bias)
