#!/usr/bin/env/python3
import numpy as np
import librosa
import scipy

PCS = np.ones(257)      # Perceptual Contrast Stretching
PCS[0:3] = 1
PCS[3:6] = 1.070175439
PCS[6:9] = 1.182456140
PCS[9:12] = 1.287719298
PCS[12:138] = 1.4       # Pre Set
PCS[138:166] = 1.322807018
PCS[166:200] = 1.238596491
PCS[200:241] = 1.161403509
PCS[241:256] = 1.077192982

maxv = np.iinfo(np.int16).max


def get_PCS(n_fft=257):
    assert n_fft == 257
    return PCS

def apply_PCS(signal):
    signal_length = signal.shape[0]
    n_fft = 512
    y_pad = librosa.util.fix_length(signal, size=signal_length + n_fft // 2)

    F = librosa.stft(y_pad, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)

    Lp = PCS * np.transpose(np.log1p(np.abs(F)), (1, 0))
    phase = np.angle(F)

    PCS_mag = np.expm1(np.transpose(Lp, (1, 0)))
    Rec = np.multiply(PCS_mag, np.exp(1j*phase))
    PCS_signal = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming, length=signal_length)


    return PCS_signal