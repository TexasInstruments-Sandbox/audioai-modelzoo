#!/usr/bin/env python3
"""Pre-compile critical packages for audio inference (Numba, PyTorch, etc.)"""
import os
import warnings
warnings.filterwarnings('ignore')

print("Warming up packages...")

# Dummy data
sr = 16000

# NumPy
try:
    print("  - NumPy...", end=" ", flush=True)
    import numpy as np
    audio = np.random.randn(sr).astype(np.float32)
    print("✓")
except ImportError:
    print("skip")

# Librosa (Numba JIT compilation)
try:
    print("  - Librosa (Numba JIT)...", end=" ", flush=True)
    import librosa
    stft = librosa.stft(audio, n_fft=512, hop_length=256, win_length=512)
    librosa.istft(stft, hop_length=256, win_length=512)
    librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
    print("✓")
except (ImportError, Exception):
    print("skip")

# PyTorch
try:
    print("  - PyTorch...", end=" ", flush=True)
    import torch
    x = torch.randn(1, sr)
    torch.stft(x.squeeze(), n_fft=512, hop_length=256, win_length=512, 
               window=torch.hann_window(512), return_complex=True)
    print("✓")
except (ImportError, Exception):
    print("skip")

# Torchaudio
try:
    print("  - Torchaudio...", end=" ", flush=True)
    import torchaudio.transforms as ta_trans
    ta_trans.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=512, n_mels=64)(x)
    ta_trans.Resample(48000, sr)(torch.randn(1, 48000))
    print("✓")
except (ImportError, Exception):
    print("skip")

# Matplotlib + Librosa display
try:
    print("  - Matplotlib...", end=" ", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    librosa.display.specshow(np.random.randn(257, 100), sr=sr, hop_length=256, ax=ax)
    plt.close('all')
    print("✓")
except (ImportError, Exception):
    print("skip")

# SoundFile
try:
    print("  - SoundFile...", end=" ", flush=True)
    import soundfile as sf
    print("✓")
except (ImportError, Exception):
    print("skip")

print("Warmup complete!")
