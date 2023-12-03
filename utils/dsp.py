"""
This module is borrowed from Facebook denoiser, Demcus.
"""
import math
from typing import Dict

import torch
import numpy as np
import torch.nn.functional as F

def get_phase_direc(clean_phase, noisy_phase):
    phase_diff = clean_phase - noisy_phase
    phase_diff = torch.where(phase_diff > torch.pi, phase_diff - 2*torch.pi, phase_diff)
    phase_diff = torch.where(phase_diff < -1*torch.pi, phase_diff + 2*torch.pi, phase_diff)
    return (phase_diff > 0).float()

def compute_weighted_cosine_similarity(
    weight: torch.Tensor, 
    clean_phase: torch.Tensor, 
    estimated_clean_phase: torch.Tensor
) -> torch.Tensor:
    """
    Compute the weighted cosine similarity between two phases.

    Args:
        weight (torch.Tensor): Weight for each element in the computation.
        clean_phase (torch.Tensor): Ground truth clean phase.
        estimated_clean_phase (torch.Tensor): Estimated clean phase.

    Returns:
        torch.Tensor: Weighted cosine similarity between clean and estimated clean phases.
    """
    cos_similarity = torch.sum(weight * torch.cos(clean_phase - estimated_clean_phase)) / torch.sum(weight)
    return cos_similarity


def convert_to_different_features(complex_spec: torch.ComplexType) -> Dict:
    """
    complex_spec: [B * F * T]
    """
    features = {}
    features["real"] = torch.real(complex_spec)
    features["imag"] = torch.imag(complex_spec)
    features["mag"] = torch.abs(complex_spec)
    features["phase"] = torch.angle(complex_spec)
    
    features["phase"][:, 0, :] = 0.0 # The DC values are always real numbers 

    return features

def compute_gd(phase):
    """
    phase: B * F * T
    """
    _, freq_dim, _ = phase.shape
    gd = phase.clone()
    gd[:, 1:, :] -= phase[:, :freq_dim-1, :]
    return compute_principle_angle(gd)

def compute_iaf(phase):
    """
    phase: B * F * T
    """
    _, _, time_length = phase.shape
    iaf = phase.clone()
    iaf[:, :, 1:] -= phase[:, :, :time_length-1]
    return compute_principle_angle(iaf)

def compute_principle_angle(angles: torch.tensor):
    rect = torch.complex(real = torch.cos(angles), imag=torch.sin(angles))
    return torch.angle(rect)

def inverse_compute_gd(gd: torch.Tensor):
    """
    gd: B * F * T, the output from compute_gd
    initial_phase: B * T, the initial phase values for each batch and time frame
    """
    phase = torch.zeros_like(gd, device=gd.device)
    phase[:, 0, :] = gd[:, 0, :]

    for f in range(1, gd.shape[1]):
        phase[:, f, :] = phase[:, f - 1, :] + gd[:, f, :]

    return phase

def inverse_compute_gd_with_phase_diff_regularization(gd: torch.Tensor, phase_diff: torch.Tensor, noisy_phase: torch.Tensor):
    """
    gd: B * F * T, the output from compute_gd
    initial_phase: B * T, the initial phase values for each batch and time frame
    """
    phase = torch.zeros_like(gd, device=gd.device)

    for f in range(1, gd.shape[1]):
        phase[:, f, :] = compute_principle_angle(phase[:, f - 1, :] + gd[:, f, :])
        phase[:, f, :] = noisy_phase[:, f, :] + torch.where(phase[:, f, :] > noisy_phase[:, f, :], phase_diff[:, f], -1 * phase_diff[:, f])
        
    return phase


def inverse_compute_iaf(iaf: torch.Tensor):
    """
    Inverse of compute_iaf function

    iaf: B * F * T
    """
    _, _, time_length = iaf.shape
    phase = torch.zeros_like(iaf, device=iaf.device)
    for t in range(1, time_length):
        phase[:, :, t] = phase[:, :, t-1] + iaf[:, :, t]
    return phase

def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    return 700 * (10**(m / 2595) - 1)


def mel_frequencies(n_mels, fmin, fmax):
    low = hz_to_mel(fmin)
    high = hz_to_mel(fmax)
    mels = np.linspace(low, high, n_mels)
    return mel_to_hz(mels)


class LowPassFilters(torch.nn.Module):
    """
    Bank of low pass filters.
    Args:
        cutoffs (list[float]): list of cutoff frequencies, in [0, 1] expressed as `f/f_s` where
            f_s is the samplerate.
        width (int): width of the filters (i.e. kernel_size=2 * width + 1).
            Default to `2 / min(cutoffs)`. Longer filters will have better attenuation
            but more side effects.
    Shape:
        - Input: `(*, T)`
        - Output: `(F, *, T` with `F` the len of `cutoffs`.
    """

    def __init__(self, cutoffs: list, width: int = None):
        super().__init__()
        self.cutoffs = cutoffs
        if width is None:
            width = int(2 / min(cutoffs))
        self.width = width
        window = torch.hamming_window(2 * width + 1, periodic=False)
        t = np.arange(-width, width + 1, dtype=np.float32)
        filters = []
        for cutoff in cutoffs:
            sinc = torch.from_numpy(np.sinc(2 * cutoff * t))
            filters.append(2 * cutoff * sinc * window)
        self.register_buffer("filters", torch.stack(filters).unsqueeze(1))

    def forward(self, input):
        *others, t = input.shape
        input = input.view(-1, 1, t)
        out = F.conv1d(input, self.filters, padding=self.width)
        return out.permute(1, 0, 2).reshape(-1, *others, t)

    def __repr__(self):
        return "LossPassFilters(width={},cutoffs={})".format(self.width, self.cutoffs)