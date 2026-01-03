import numpy as np
import mirdata
from torch.utils.data import Dataset
import torch
import torchaudio.functional as F
from scipy import signal

class MIRDataset(Dataset):
    def __init__(self, dataset_name: str, mix : float = 0.0):
        # Initialize the loader, download if required, and validate
        self.loader = mirdata.initialize(dataset_name)
        # self.loader.download()
        # self.loader.validate()
        
        # batch size must be 1 because here we do not pad the items
        self.mix = mix

    def __len__(self) -> int:
        return len(self.loader.track_ids)

    def generate_rir(self, sr, duration=1.0):
        # Generate simple synthetic RIR (exponential decay noise)
        t = torch.linspace(0, duration, int(duration * sr))
        envelope = torch.exp(-5.0 * t)
        noise = torch.randn_like(t)
        rir = noise * envelope
        return (rir / torch.norm(rir)).numpy()

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Unpack the current track
        track_id = self.loader.track_ids[item]
        track = self.loader.track(track_id)

        # Get the audio and annotations
        audio_signal, sample_rate = track.audio
        audio_signal = audio_signal.mean(axis=-1) if audio_signal.ndim > 1 else audio_signal

        if self.mix > 0:
            rir = self.generate_rir(sample_rate)
            # Convolve
            rev = signal.fftconvolve(audio_signal, rir, mode="full")
            rev = rev[:audio_signal.shape[0]]  # Keep original length
            
            # Normalize energy
            audio_signal_norm = np.linalg.norm(audio_signal)
            rev_norm = np.linalg.norm(rev)
            if rev_norm > 0:
                rev = rev * (audio_signal_norm / rev_norm)
            
            audio_signal = (1 - self.mix) * audio_signal + self.mix * rev

        times = track.f0.times
        frequencies = track.f0.frequencies

        return (
            audio_signal.astype(np.float32),
            times.astype(np.float32),
            frequencies.astype(np.float32),
        )