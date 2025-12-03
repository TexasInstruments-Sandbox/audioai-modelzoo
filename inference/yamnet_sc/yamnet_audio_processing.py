"""
YAMNet Audio Preprocessing Module

This module provides audio preprocessing functions for YAMNet inference,
adapted from torch_audioset package to be standalone.
"""

import numpy as np
import torch
import torchaudio.transforms as ta_trans
import soundfile as sf


# Audio processing parameters (from CommonParams and YAMNetParams)
TARGET_SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.01
MEL_MIN_HZ = 125.0
MEL_MAX_HZ = 7500.0
NUM_MEL_BANDS = 64
LOG_OFFSET = 0.001
PATCH_WINDOW_IN_SECONDS = 0.96
PATCH_HOP_SECONDS = 0.96  # Non-overlapping patches


class VGGishLogMelSpectrogram(ta_trans.MelSpectrogram):
    """
    Log mel-spectrogram transform that adheres to the transform
    used by Google's VGGish/YAMNet model input processing pipeline.
    """

    def forward(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (..., time)

        Returns:
            torch.Tensor: Log mel frequency spectrogram of size (..., n_mels, time)
        """
        specgram = self.spectrogram(waveform)
        # Google's implementation uses np.abs on fft output (power 1)
        # torchaudio.MelSpectrogram default power is 2.0, so we sqrt it
        specgram = specgram ** 0.5

        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + LOG_OFFSET)
        return mel_specgram


class WaveformToInput(torch.nn.Module):
    """
    Convert audio waveform to YAMNet input format (log mel spectrogram patches).
    """

    def __init__(self):
        super().__init__()
        window_length_samples = int(round(
            TARGET_SAMPLE_RATE * STFT_WINDOW_LENGTH_SECONDS
        ))
        hop_length_samples = int(round(
            TARGET_SAMPLE_RATE * STFT_HOP_LENGTH_SECONDS
        ))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        
        # Validate expected values
        assert window_length_samples == 400
        assert hop_length_samples == 160
        assert fft_length == 512
        
        self.mel_trans_ope = VGGishLogMelSpectrogram(
            TARGET_SAMPLE_RATE, 
            n_fft=fft_length,
            win_length=window_length_samples, 
            hop_length=hop_length_samples,
            f_min=MEL_MIN_HZ,
            f_max=MEL_MAX_HZ,
            n_mels=NUM_MEL_BANDS
        )

    def wavform_to_log_mel(self, waveform, sample_rate):
        """
        Convert waveform to log mel spectrogram patches.
        
        Args:
            waveform: torch tensor [num_audio_channels, num_time_steps]
            sample_rate: audio sample rate per second
            
        Returns:
            patches: torch tensor of shape [num_patches, 1, 96, 64]
            spectrogram: numpy array of full spectrogram for visualization
        """
        # Average over channels to get mono
        x = waveform.mean(axis=0, keepdims=True)
        
        # Resample to target sample rate if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = ta_trans.Resample(sample_rate, TARGET_SAMPLE_RATE)
            x = resampler(x)
        
        # Extract log mel spectrogram
        x = self.mel_trans_ope(x)
        x = x.squeeze(dim=0).T  # [1, C, T] -> [T, C]
        spectrogram = x.cpu().numpy().copy()

        # Calculate window size in frames
        window_size_in_frames = int(round(
            PATCH_WINDOW_IN_SECONDS / STFT_HOP_LENGTH_SECONDS
        ))

        if PATCH_HOP_SECONDS == PATCH_WINDOW_IN_SECONDS:
            # Non-overlapping patches
            num_chunks = x.shape[0] // window_size_in_frames

            # Reshape into chunks of non-overlapping sliding window
            num_frames_to_use = num_chunks * window_size_in_frames
            x = x[:num_frames_to_use]
            # [num_chunks, 1, window_size, num_freq]
            x = x.reshape(num_chunks, 1, window_size_in_frames, x.shape[-1])
        else:
            # Overlapping patches with custom hop
            patch_hop_in_frames = int(round(
                PATCH_HOP_SECONDS / STFT_HOP_LENGTH_SECONDS
            ))
            patch_hop_num_chunks = (x.shape[0] - window_size_in_frames) // patch_hop_in_frames + 1
            num_frames_to_use = window_size_in_frames + (patch_hop_num_chunks - 1) * patch_hop_in_frames
            x = x[:num_frames_to_use]
            x_in_frames = x.reshape(-1, x.shape[-1])
            x_output = np.empty((patch_hop_num_chunks, window_size_in_frames, x.shape[-1]))
            for i in range(patch_hop_num_chunks):
                start_frame = i * patch_hop_in_frames
                x_output[i] = x_in_frames[start_frame: start_frame + window_size_in_frames]
            x = x_output.reshape(patch_hop_num_chunks, 1, window_size_in_frames, x.shape[-1])
            x = torch.tensor(x, dtype=torch.float32)
            
        return x, spectrogram


def load_audio(audio_path):
    """
    Load audio file and convert to torch tensor.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        waveform: torch.Tensor of shape [channels, samples]
        sample_rate: int
    """
    # Load audio (soundfile returns float32 in range [-1, 1])
    waveform, sr = sf.read(audio_path, dtype='float32', always_2d=True)
    # Convert to [channels, samples] format
    waveform = waveform.T
    waveform_tensor = torch.from_numpy(waveform)
    return waveform_tensor, sr


def preprocess_audio_to_patches(audio_path):
    """
    Load and preprocess audio file to YAMNet input patches.
    
    Args:
        audio_path: Path to audio file (WAV format)
        
    Returns:
        patches: numpy array of shape [num_patches, 1, 96, 64]
        spectrogram: numpy array of full spectrogram
        sample_rate: original sample rate
        duration: audio duration in seconds
    """
    # Load audio
    waveform, sr = load_audio(audio_path)
    duration = waveform.shape[1] / sr
    
    # Create preprocessor and extract patches
    preprocessor = WaveformToInput()
    patches, spectrogram = preprocessor.wavform_to_log_mel(waveform, sr)
    
    return patches.cpu().numpy(), spectrogram, sr, duration
