"""
Gaussian corruption process for audio -- ablation variant B.

Standard additive-Gaussian DDPM-style corruption, adapted to a 1D
waveform in [0, 1], using the same variance-preserving schedule as the
image branch (src/diffusion_image.py). Included purely as a controlled
comparison point for src/diffusion_audio_poisson.py: "why Poisson and
not the more common Gaussian choice?" is answered empirically by
scripts/run_noise_ablation.py, not asserted.
"""

import torch


class GaussianAudioDiffusion:
    def __init__(self, T: int, device: str = "cuda"):
        self.T = T
        beta = torch.linspace(1e-4, 0.02, T).to(device)
        alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(alpha, 0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        ab = self.alpha_bar[t].view(-1, 1)
        a = torch.sqrt(ab)
        b = torch.sqrt(1 - ab)
        noise = torch.randn_like(x0)
        xt = a * x0 + b * noise
        return xt.clamp(0, 1)

    def corrupt(self, x0: torch.Tensor, t: torch.Tensor, device: str = None):
        xt = self.q_sample(x0, t)
        return xt, xt - x0
