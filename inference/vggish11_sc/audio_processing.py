import torch
import torchaudio
import soundfile as sf

def preprocess_audio_waveform(file_path, sample_rate=16000, duration=4.0, num_samples=None):
    """
    Load and preprocess an audio file into a normalized waveform tensor.

    Args:
        file_path: Path to the audio file
        sample_rate: Target sample rate in Hz
        duration: Target duration in seconds

    Returns:
        waveform: Preprocessed audio waveform tensor of shape [channels, time_samples]
                 where channels=1 (mono) and time_samples = sample_rate * duration
    """
    # waveform: [channels, samples]
    # Use soundfile directly to avoid torchcodec dependency issues
    waveform_np, sr = sf.read(file_path, dtype='float32')
    # Convert to torch tensor and ensure shape is [channels, samples]
    waveform = torch.from_numpy(waveform_np.T).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension for mono audio

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Ensure fixed length
    if num_samples is None:
        num_samples = int(sample_rate * duration)

    if waveform.shape[1] < num_samples:
        # pad if too short
        padding = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif waveform.shape[1] > num_samples:
        # truncate if too long
        waveform = waveform[:, :num_samples]

    return waveform

def log_mel_spectrogram(waveform, config):
    """
    Extract log mel spectrogram features from waveform
    waveform: [channels, time_samples]
    log_mel_spec: [channels, mel_bins, time_frames]
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['dataset']['sample_rate'],
        n_fft=config['dataset']['n_fft'],
        hop_length=config['dataset']['hop_length'],
        n_mels=config['dataset']['n_mels'],
        # center=False, #==> True is default, True increased the STFT frame number (126 vs 124)
        center=True,
    )

    mel_spec = mel_transform(waveform)
    log_mel_spec = torch.log(mel_spec + 1e-9)

    return log_mel_spec

