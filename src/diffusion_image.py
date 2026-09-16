"""
Forward diffusion process for the image (MNIST) branch, plus the
sinusoidal timestep embedding shared by the image models.
"""

import math
import torch
import torch.nn as nn


class Diffusion:
    """Standard DDPM-style forward (noising) process.

    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

    Note: `T` must always be passed explicitly and match the number of
    diffusion timesteps used elsewhere in the pipeline (e.g. TIMESTEPS
    in main.py). Relying on the constructor default is fragile if that
    constant ever changes.
    """

    def __init__(self, T: int, device: str = "cuda"):
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        b = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        return a * x0 + b * noise


class TimeEmb(nn.Module):
    """Sinusoidal timestep embedding (as in standard DDPM implementations)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], -1)
