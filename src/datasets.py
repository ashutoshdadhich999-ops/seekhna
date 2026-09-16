"""Dataset wrapper for SpeechCommands: resample, mono, pad/trim, normalize."""

import torch
import torch.nn.functional as F
import torchaudio


class AudioDS(torch.utils.data.Dataset):
    def __init__(self, subset, target_len: int = 8000, target_sr: int = 16000):
        self.subset = subset
        self.target_len = target_len
        self.target_sr = target_sr

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        wav, sr0, *_ = self.subset[i]
        if sr0 != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr0, self.target_sr)
        wav = wav.mean(0)

        if wav.shape[0] < self.target_len:
            wav = F.pad(wav, (0, self.target_len - wav.shape[0]))
        else:
            wav = wav[: self.target_len]

        wav = (wav - wav.min()) / (wav.max() - wav.min() + 1e-8)
        return wav
