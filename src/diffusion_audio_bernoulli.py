"""
Bernoulli spike corruption for audio -- ablation variant C.

Rate-coded Bernoulli spiking is the most common input-encoding scheme in
the SNN literature (each timestep/sample independently spikes with
probability equal to the underlying intensity). This gives a second,
very different "biologically inspired" corruption process to compare
against Poisson counts: same rate-coding idea, but capped at one event
per bin/step (Bernoulli) instead of allowing multiple (Poisson) -- the
two coincide only in the low-rate limit.
"""

import torch


class BernoulliAudioDiffusion:
    def __init__(self, T: int, device: str = "cuda", decay: float = 0.06,
                 max_rate: float = 0.9):
        self.T = T
        self.decay = decay
        self.max_rate = max_rate
        self.device = device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        t = t.view(-1, 1).float().to(self.device)
        gamma = torch.exp(-self.decay * t * 6 / self.T)

        p = (x0 * gamma * self.max_rate).clamp(1e-4, self.max_rate)
        spikes = torch.bernoulli(p)  # {0, 1} events, rate-coded like x0
        # A little jitter as t grows, mirroring the Poisson branch's
        # noise-floor behaviour, so the three corruption processes are
        # comparable in how "destroyed" the signal is by t = T.
        jitter = torch.randn_like(spikes) * 0.05 * (1 - gamma)
        xt = (spikes + jitter).clamp(0, 1)
        return xt

    def corrupt(self, x0: torch.Tensor, t: torch.Tensor, device: str = None):
        xt = self.q_sample(x0, t)
        return xt, xt - x0
