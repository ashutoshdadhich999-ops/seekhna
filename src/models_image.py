"""
Image branch models: a spiking (SNN) residual denoiser and a matched
non-spiking (ANN) baseline with an identical parameter budget / topology,
used for a controlled ablation.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.diffusion_image import TimeEmb


class ResBlock(nn.Module):
    """Spiking residual block (LIF neurons) with a small internal
    unroll (`num_steps`) to obtain a rate-coded output."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid())
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        res = self.skip(x)
        h = self.norm1(self.conv1(x))
        temb = self.time_proj(t_emb)[:, :, None, None]  # computed once, not per-step

        outs = []
        for _ in range(self.num_steps):
            h_in = h + temb  # constant drive per step; LIF membrane state evolves internally
            spk, mem1 = self.lif1(h_in, mem1)
            h2 = self.norm2(self.conv2(spk))
            spk2, mem2 = self.lif2(h2, mem2)
            outs.append(spk2 + res)
        return torch.stack(outs).mean(0)


class NonSpikeResBlock(nn.Module):
    """Non-spiking counterpart of ResBlock (SiLU activations instead of LIF)."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        res = self.skip(x)
        h = self.norm1(self.conv1(x))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        return h + res


class StrongImgNet(nn.Module):
    """Spiking image denoiser."""

    def __init__(self, base_channels: int = 32, num_steps: int = 5, time_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmb(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.enc = nn.Conv2d(1, base_channels, 3, padding=1)
        self.b1 = ResBlock(base_channels, base_channels, time_dim, num_steps)
        self.b2 = ResBlock(base_channels, base_channels * 2, time_dim, num_steps)
        self.b3 = ResBlock(base_channels * 2, base_channels * 2, time_dim, num_steps)
        self.out = nn.Conv2d(base_channels * 2, 1, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(t)
        x = self.enc(x)
        x = self.b1(x, temb)
        x = self.b2(x, temb)
        x = self.b3(x, temb)
        return self.out(x)


class NonSpikeImgNet(nn.Module):
    """Non-spiking (ANN) image denoiser, matched topology to StrongImgNet."""

    def __init__(self, base_channels: int = 32, time_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmb(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.enc = nn.Conv2d(1, base_channels, 3, padding=1)
        self.b1 = NonSpikeResBlock(base_channels, base_channels, time_dim)
        self.b2 = NonSpikeResBlock(base_channels, base_channels * 2, time_dim)
        self.b3 = NonSpikeResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.out = nn.Conv2d(base_channels * 2, 1, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(t)
        x = self.enc(x)
        x = self.b1(x, temb)
        x = self.b2(x, temb)
        x = self.b3(x, temb)
        return self.out(x)
