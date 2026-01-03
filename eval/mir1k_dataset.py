import torch
import os
import torchaudio
import torchaudio.functional as F
import glob
import numpy as np

class MIR1KDataset(torch.utils.data.Dataset):
    def __init__(self, root, mix=0.0, target_sr=16000):
        self.wav_files = sorted(glob.glob(os.path.join(root, "Wavfile", "*.wav")))
        self.pitch_dir = os.path.join(root, "PitchLabel")
        self.mix = mix
        self.target_sr = target_sr
        self.resampler_cache = {}

    def __len__(self):
        return len(self.wav_files)

    def _get_resampler(self, sr):
        if sr not in self.resampler_cache:
            self.resampler_cache[sr] = T.Resample(sr, self.target_sr)
        return self.resampler_cache[sr]

    def generate_rir(self, sr, duration=1.0):
        # Generate simple synthetic RIR (exponential decay noise)
        t = torch.linspace(0, duration, int(duration * sr))
        envelope = torch.exp(-5.0 * t)
        noise = torch.randn_like(t)
        rir = noise * envelope
        return rir / torch.norm(rir)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        basename = os.path.basename(wav_path)
        pv_name = basename.replace(".wav", ".pv")
        pv_path = os.path.join(self.pitch_dir, pv_name)

        x, sr = torchaudio.load(wav_path)
        x = x[1:2, :] # take vocals only
        if sr != self.target_sr:
            x = self._get_resampler(sr)(x)

        # Apply Reverb
        if self.mix > 0:
            rir = self.generate_rir(self.target_sr).to(x.device).unsqueeze(0) # [1, T_rir]
            # Convolve
            rev = F.fftconvolve(x, rir, mode="full")
            rev = rev[:, :x.shape[1]]  # Keep original length
            
            # Normalize energy
            x_norm = torch.norm(x)
            rev_norm = torch.norm(rev)
            if rev_norm > 0:
                rev = rev * (x_norm / rev_norm)
            
            x = (1 - self.mix) * x + self.mix * rev

        # Load Labels (MIDI Semitones)
        # 0 = Unvoiced, >0 = Voiced Pitch
        if os.path.exists(pv_path):
            labels = np.loadtxt(pv_path)
            # We convert MIDI labels to Hz for consistent evaluation.
            # Hz = 440 * 2^((d-69)/12)
            # Mask unvoiced (0) to avoid log/exp errors
            mask = labels > 0
            labels_hz = np.zeros_like(labels)
            labels_hz[mask] = 440.0 * (2.0 ** ((labels[mask] - 69.0) / 12.0))
        else:
            labels_hz = np.array([])

        return x, labels_hz
