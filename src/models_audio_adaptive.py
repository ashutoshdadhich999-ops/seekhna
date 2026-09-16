"""
Adaptive-computation spiking block for audio.

=== Honest framing ===
A full continuous-time SNN (event-driven simulation, or an ODE-based LIF
solved with an adaptive-step integrator) is a substantially larger
undertaking than can be verified here without real event-based data and a
dedicated simulator. What IS implemented here is a discrete but
input-dependent step budget: instead of every sample running a fixed
`num_steps` unroll (models_audio.py), each sample learns to "halt" its
internal simulation early via an Adaptive Computation Time (ACT) style
mechanism (Graves, 2016). This is a genuine, defensible move *towards*
event-driven/continuous-time computation -- variable compute per input,
governed by the network itself -- without claiming to be a full
continuous-time solver.

=== Mechanism ===
At each of up to `max_steps` internal steps, a small linear "halting unit"
reads the current membrane potential and outputs a halting probability
p_i in [0,1]. Steps accumulate weighted output using the standard ACT
remainder trick (Graves, 2016) so the total weight sums to 1, and a small
ponder-cost penalty on the *expected* number of steps taken is exposed
for the training loss, encouraging fewer steps when the input doesn't
need them (e.g. low-noise timesteps).
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class AdaptiveConvResBlock1D(nn.Module):
    def __init__(self, ch: int, time_dim: int, max_steps: int = 8,
                 halt_threshold: float = 0.99):
        super().__init__()
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold

        self.time_proj = nn.Linear(time_dim, ch)
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, ch)
        self.lif1 = snn.Leaky(beta=0.92, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)
        self.norm2 = nn.GroupNorm(8, ch)
        self.lif2 = snn.Leaky(beta=0.92, spike_grad=surrogate.fast_sigmoid())

        # Halting unit: reads pooled membrane state -> scalar halting logit
        self.halt_proj = nn.Linear(ch, 1)

        self.last_ponder_cost = None  # exposed for an optional aux loss
        self.last_mean_steps = None   # exposed for logging / analysis

    def forward(self, x, temb):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        res = x
        h = self.norm1(self.conv1(x))
        temb_p = self.time_proj(temb)[:, :, None]

        B = x.shape[0]
        device = x.device
        halting_prob = torch.zeros(B, device=device)
        remainders = torch.ones(B, device=device)
        n_updates = torch.zeros(B, device=device)
        weighted_out = torch.zeros_like(x)

        for step in range(self.max_steps):
            h_in = h + temb_p
            spk, mem1 = self.lif1(h_in, mem1)
            h2 = self.norm2(self.conv2(spk))
            spk2, mem2 = self.lif2(h2, mem2)
            step_out = spk2 + res

            # halting probability from pooled membrane potential
            pooled = mem2.mean(dim=-1)  # (B, ch)
            p = torch.sigmoid(self.halt_proj(pooled)).squeeze(-1)  # (B,)

            still_running = (halting_prob < self.halt_threshold).float()
            is_last_step = torch.full_like(p, float(step == self.max_steps - 1))

            new_halted = still_running * (1 - is_last_step) * (
                (halting_prob + p) >= self.halt_threshold
            ).float()
            running = still_running * (1 - new_halted) * (1 - is_last_step)

            # weight for this step: p if still running/newly halting,
            # the remainder if forced to stop at max_steps
            update_weight = (
                p * running + remainders * (new_halted + is_last_step * still_running)
            )

            halting_prob = halting_prob + p * running
            remainders = remainders - p * running
            n_updates = n_updates + still_running

            weighted_out = weighted_out + update_weight.view(B, 1, 1) * step_out

        self.last_ponder_cost = n_updates.mean()
        self.last_mean_steps = n_updates.mean().item()
        return weighted_out


class AdaptiveStrongAudioNet(nn.Module):
    """Spiking audio denoiser with adaptive per-sample computation depth."""

    def __init__(self, channels: int = 64, time_dim: int = 128, max_steps: int = 8,
                 T_audio: int = 40):
        super().__init__()
        self.T_audio = T_audio
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, time_dim), nn.SiLU()
        )
        self.input = nn.Conv1d(1, channels, 7, padding=3)
        self.b1 = AdaptiveConvResBlock1D(channels, time_dim, max_steps)
        self.b2 = AdaptiveConvResBlock1D(channels, time_dim, max_steps)
        self.b3 = AdaptiveConvResBlock1D(channels, time_dim, max_steps)
        self.out = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, x, t):
        temb = self.time_mlp((t.float() / self.T_audio).unsqueeze(-1))
        h = self.input(x.unsqueeze(1))
        h = self.b1(h, temb)
        h = self.b2(h, temb)
        h = self.b3(h, temb)
        return self.out(h).squeeze(1)

    def mean_steps_taken(self):
        """Average number of internal steps used across the 3 blocks on
        the last forward pass -- the efficiency signal this architecture
        is meant to produce (lower = more adaptive/event-driven)."""
        vals = [b.last_mean_steps for b in (self.b1, self.b2, self.b3)
                if b.last_mean_steps is not None]
        return sum(vals) / len(vals) if vals else None
