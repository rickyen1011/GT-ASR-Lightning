import os, sys
from typing import Optional, Dict, Tuple, Any

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio


class SSLKIASR(nn.Module):
    def __init__(self, base_args, SSL_backbone_args):
        super(SSLKIASR, self).__init__()
        for key, value in base_args.items():
            setattr(self, key, value)
        for key, value in SSL_backbone_args.items():
            setattr(self, key, value)
        
        if self.pretrained_model == 'hubert':
            bundle =  torchaudio.pipelines.HUBERT_LARGE
        elif self.pretrained_model == 'wav2vec2_base':
            bundle =  torchaudio.pipelines.WAV2VEC2_BASE
        else:
            raise NotImplementedError()
        
        self.encoder = bundle.get_model()
        self.decoder = nn.ModuleDict({{}})
        
        for i, num_class in enumerate(self.num_classes):
            attr = self.attributes[i]
            self.decoder[attr] = nn.Linear(self.d_encoder, num_class+1)
            self.ctc_criterion[attr] = \
                nn.CTCLoss(blank=num_class, zero_infinity=True)

    def _compute_loss(self, outputs, target, input_lengths, label_lengths):
        ctc_loss_dict = {}
        attr_comb = self.attributes[-1]
        ctc_loss = self.ctc_criterion[attr_comb](
            torch.log(F.normalize(outputs[attr_comb], p=1, dim=-1).transpose(0, 1)+1e-8), 
            target[attr_comb], input_lengths, label_lengths
        )
        ctc_loss_dict['CTCLoss-'+attr_comb] = ctc_loss.detach()
        if self.mt:
            for attr in self.attributes[:-1]:
                attr_ctc_loss = self.ctc_criterion[attr](
                    F.log_softmax(outputs[attr], dim=-1).transpose(0, 1), 
                    target[attr], input_lengths, label_lengths
                )
                ctc_loss += attr_ctc_loss
                ctc_loss_dict['CTCLoss-'+attr] = attr_ctc_loss.detach()
        return ctc_loss, ctc_loss_dict
    
    def _forward(self, x):
        rep, _ = self.encoder.extract_features(x)
        outputs = {}
        for attr in self.attributes[:-1]:
            attr_score = self.decoder[attr](rep[-1])
            outputs[attr] = attr_score
            attr_score = F.softmax(attr_score, dim=-1)
            outputs2comb[attr] = attr_score * attr2comb[attr]
        
        comb_score = torch.ones_like(attr_score)
        for score in outputs2comb.values()
            comb_score *= score
        outputs[self.attributes[-1]] = comb_score
        
        return outputs
    
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
        return self._compute_loss(outputs, target, input_lengths, label_lengths)

    def inference(self, x):
        outputs = self._forward(x)

        return outputs