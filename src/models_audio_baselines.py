"""
Stronger non-spiking baselines for the audio branch.

Rationale: comparing a spiking model only against a matched-topology ANN
(models_audio.py::NonSpikeAudioNet) risks the objection "the SNN only won
because the baseline was weak." These three architectures are established,
substantially stronger non-spiking waveform-processing designs, used here
purely as harder baselines -- if the spiking model remains competitive
against these, that is a much stronger result.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------- 1D U-Net ----

class UNet1DBlock(nn.Module):
    def __init__(self, ch, time_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, ch)
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()

    def forward(self, x, temb):
        res = x
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(temb)[:, :, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + res


class UNet1D(nn.Module):
    """Small 1D U-Net: downsample -> bottleneck -> upsample with skip
    connections, the standard architecture for waveform denoising/diffusion
    (e.g. DiffWave, WaveGrad use variants of this)."""

    def __init__(self, base_ch: int = 32, time_dim: int = 128, T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.stem = nn.Conv1d(1, base_ch, 7, padding=3)

        self.down1 = UNet1DBlock(base_ch, time_dim)
        self.pool1 = nn.Conv1d(base_ch, base_ch * 2, 4, stride=2, padding=1)

        self.down2 = UNet1DBlock(base_ch * 2, time_dim)
        self.pool2 = nn.Conv1d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)

        self.mid = UNet1DBlock(base_ch * 4, time_dim)

        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = UNet1DBlock(base_ch * 2, time_dim)

        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = UNet1DBlock(base_ch, time_dim)

        self.out = nn.Conv1d(base_ch, 1, 7, padding=3)

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))
        h0 = self.stem(x.unsqueeze(1))

        h1 = self.down1(h0, temb)
        h2_in = self.pool1(h1)

        h2 = self.down2(h2_in, temb)
        h3_in = self.pool2(h2)

        h3 = self.mid(h3_in, temb)

        u2 = self.up2(h3)
        u2 = u2 + h2  # skip connection
        u2 = self.dec2(u2, temb)

        u1 = self.up1(u2)
        u1 = u1 + h1  # skip connection
        u1 = self.dec1(u1, temb)

        return self.out(u1).squeeze(1)


# --------------------------------------------------- Dilated (WaveNet) CNN --

class DilatedResBlock(nn.Module):
    def __init__(self, ch, time_dim, dilation):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, ch)
        pad = dilation * 2  # kernel_size=5 -> pad = dilation*(k-1)/2
        self.conv = nn.Conv1d(ch, ch, 5, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.conv(x)
        h = h + self.time_proj(temb)[:, :, None]
        h = self.act(self.norm(h))
        return h + x


class DilatedCNN(nn.Module):
    """WaveNet-style stack of exponentially-dilated causal-ish convolutions,
    giving a large receptive field with few layers -- a strong, standard
    choice for raw-waveform modelling."""

    def __init__(self, ch: int = 64, time_dim: int = 128, T_audio: int = 40,
                 num_blocks: int = 6):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.input = nn.Conv1d(1, ch, 7, padding=3)
        self.blocks = nn.ModuleList([
            DilatedResBlock(ch, time_dim, dilation=2 ** i) for i in range(num_blocks)
        ])
        self.out = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))
        h = self.input(x.unsqueeze(1))
        for block in self.blocks:
            h = block(h, temb)
        return self.out(h).squeeze(1)


# ----------------------------------------------------------- Residual TCN --

class TCNBlock(nn.Module):
    """Temporal Convolutional Network block: dilated conv + weight norm +
    residual, the standard TCN building block (Bai et al., 2018)."""

    def __init__(self, ch, time_dim, dilation, kernel_size=3):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, ch)
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, temb):
        h = self.act(self.conv1(x))
        h = h + self.time_proj(temb)[:, :, None]
        h = self.dropout(h)
        h = self.act(self.conv2(h))
        return h + x


class ResidualTCN(nn.Module):
    def __init__(self, ch: int = 64, time_dim: int = 128, T_audio: int = 40,
                 num_blocks: int = 6):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.input = nn.Conv1d(1, ch, 7, padding=3)
        self.blocks = nn.ModuleList([
            TCNBlock(ch, time_dim, dilation=2 ** i) for i in range(num_blocks)
        ])
        self.out = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))
        h = self.input(x.unsqueeze(1))
        for block in self.blocks:
            h = block(h, temb)
        return self.out(h).squeeze(1)


BASELINES = {
    "unet": UNet1D,
    "dilated": DilatedCNN,
    "tcn": ResidualTCN,
}
