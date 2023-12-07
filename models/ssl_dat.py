import os
from typing import Optional, Dict, Tuple, Any

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio


class SSLDATASR(nn.Module):
    def __init__(self, base_args, SSL_backbone_args):
        super(SSLDATASR, self).__init__()
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
        self.decoder_phn = nn.Linear(self.d_encoder, self.num_class+1)
            
        self.ctc_criterion = nn.CTCLoss(blank=self.num_class, zero_infinity=True)

    def _compute_loss(self, outputs, target, input_lengths, label_lengths):
        ctc_loss = self.ctc_criterion(
            F.log_softmax(outputs, dim=-1).transpose(0, 1), target, input_lengths, label_lengths
        )
        return ctc_loss, {'CTCLoss': ctc_loss.detach()}
    
    def _forward(self, x):
        rep, _ = self.encoder.extract_features(x)
        outputs_phn = self.decoder_phn(rep[-1])
        
        return outputs_phn, rep[-1]
    
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
        outputs, rep = self._forward(x)
        return self._compute_loss(outputs, target, input_lengths, label_lengths), rep

    def inference(self, x):
        outputs, rep = self._forward(x)

        return outputs, rep