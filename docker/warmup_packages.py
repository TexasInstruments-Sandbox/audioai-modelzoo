#!/usr/bin/env python3
"""Warmup PyTorch/Torchaudio to trigger lazy initialization and validate installation"""
import warnings
warnings.filterwarnings('ignore')

print("Warming up packages...")

sr = 16000

# NumPy - fast import validation
try:
    print("  - NumPy...", end=" ", flush=True)
    import numpy as np
    audio = np.random.randn(sr).astype(np.float32)
    print("✓")
except ImportError:
    print("skip")

# PyTorch - may have lazy module loading
try:
    print("  - PyTorch...", end=" ", flush=True)
    import torch
    x = torch.randn(1, sr)
    torch.stft(x.squeeze(), n_fft=512, hop_length=256, win_length=512, 
               window=torch.hann_window(512), return_complex=True)
    print("✓")
except (ImportError, Exception):
    print("skip")

# Torchaudio - may have lazy module loading
try:
    print("  - Torchaudio...", end=" ", flush=True)
    import torchaudio.transforms as ta_trans
    ta_trans.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=512, n_mels=64)(x)
    ta_trans.Resample(48000, sr)(torch.randn(1, 48000))
    print("✓")
except (ImportError, Exception):
    print("skip")

# Matplotlib - validate installation
try:
    print("  - Matplotlib...", end=" ", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plt.close('all')
    print("✓")
except (ImportError, Exception):
    print("skip")

# SoundFile - validate installation
try:
    print("  - SoundFile...", end=" ", flush=True)
    import soundfile as sf
    print("✓")
except (ImportError, Exception):
    print("skip")

print("Warmup complete!")

