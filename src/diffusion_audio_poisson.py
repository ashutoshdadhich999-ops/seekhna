"""
Poisson Thinning Diffusion for audio (spike-count) signals.

=== Motivation ===
Gaussian DDPM combines variances, so its schedule appears via
sqrt(alpha_bar) -- variances add linearly, standard deviations don't.
A Poisson (count) process has a different, and cleaner, algebra: the
*rate* (intensity) is the additive/combinable quantity, via two classical
results in point-process theory:

  1. Superposition: if A ~ Poisson(a) and B ~ Poisson(b) are independent,
     A + B ~ Poisson(a + b).
  2. Thinning: if N ~ Poisson(lambda) and each event is independently kept
     with probability p, the kept count ~ Poisson(p * lambda), independent
     of the discarded count ~ Poisson((1-p) * lambda).

These two results let us build a genuine forward Markov chain
q(x_t | x_{t-1}) for a Poisson-valued signal, with a closed-form marginal
q(x_t | x_0) -- the audio analogue of the Gaussian DDPM forward process,
but using rate combination (no square root) instead of variance
combination.

=== Forward process ===
Let lambda_0 = x0 * scale be the clean signal's spike-count rate, and
mu = noise_floor_rate * scale be a fixed background ("dark count") rate
representing full corruption. Define a decreasing retain-schedule
alpha_bar_t in (0, 1] (alpha_bar_0 = 1, alpha_bar_T ~ 0), with per-step
ratio rho_t = alpha_bar_t / alpha_bar_{t-1}.

  q(x_t | x_{t-1}):
      thin x_{t-1}'s rate by rho_t (Thinning), and inject a fresh
      Poisson(mu * (1 - rho_t)) noise-floor contribution (Superposition):

          x_t ~ Poisson( rho_t * (x_{t-1} * scale) + (1 - rho_t) * mu ) / scale

  Marginal q(x_t | x_0) (telescoping the per-step ratios so the product
  rho_1 * rho_2 * ... * rho_t = alpha_bar_t):

          x_t ~ Poisson( alpha_bar_t * lambda_0 + (1 - alpha_bar_t) * mu ) / scale

This is verified numerically in scripts/verify_poisson_diffusion.py by
comparing many steps of q(x_t|x_{t-1}) against the closed-form marginal.

=== Reverse (generative) process ===
Training predicts the residual r = x_t - x0 (as before), which gives an
x0-estimate x0_hat = x_t - r_hat. An ancestral sampler can then be run
from pure noise-floor down to t=0:

  x_T ~ Poisson(mu) / scale                          (prior)
  for t = T .. 1:
      r_hat = model(x_t, t)
      x0_hat = clip(x_t - r_hat, 0, 1)
      lambda_{t-1} = alpha_bar_{t-1} * x0_hat * scale + (1 - alpha_bar_{t-1}) * mu
      x_{t-1} ~ Poisson(lambda_{t-1}) / scale

This makes the audio branch a genuine generative model (can sample from
noise) rather than only a single-shot denoiser.
"""

import torch


class PoissonDiffusion:
    def __init__(self, T: int, scale: float = 15.0, noise_floor_rate: float = 0.05,
                 device: str = "cuda"):
        """
        Args:
            T: number of diffusion steps.
            scale: count scale (x in [0,1] represents rate/scale spikes).
            noise_floor_rate: the rate (in [0,1] units) the signal decays
                towards as t -> T; analogous to the fixed-variance prior
                N(0, I) in Gaussian DDPM, but for a rate-based process this
                is a background firing rate rather than zero.
            device: torch device.
        """
        self.T = T
        self.scale = scale
        self.mu = noise_floor_rate * scale  # background rate, in count units

        # Retain-schedule: alpha_bar_t decreasing from ~1 to ~0.
        # (Linear decrease in log-space keeps per-step ratios rho_t well
        # behaved -- this plays the role beta/alpha_bar play in DDPM, but
        # note there is no square root here: see module docstring.)
        beta = torch.linspace(1e-3, 0.15, T).to(device)
        alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(alpha, 0)  # shape (T,)

    # -------------------------------------------------- closed-form q(x_t|x0)
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """Sample x_t given x0 directly from the closed-form marginal."""
        ab = self.alpha_bar[t].view(-1, 1)
        lam0 = x0 * self.scale
        rate = ab * lam0 + (1 - ab) * self.mu
        rate = rate.clamp(min=1e-6)
        counts = torch.poisson(rate)
        xt = (counts / self.scale).clamp(0, 1)
        return xt

    # -------------------------------------------- single-step q(x_t|x_{t-1})
    def q_step_counts(self, counts_prev: torch.Tensor, t: torch.Tensor):
        """One forward transition in RAW COUNT space, N_{t-1} -> N_t.

        IMPORTANT: correct thinning of an already-*realized* count requires
        Binomial(N_{t-1}, rho_t), not Poisson(rho_t * N_{t-1}). The latter
        looks similar but is wrong: it compounds two independent sources of
        Poisson randomness (Poisson-of-a-Poisson-outcome), which inflates
        variance at every step and makes a chained simulation diverge from
        the closed-form marginal. Binomial thinning of a Poisson count is
        the operation the thinning theorem is actually about, and it
        reproduces the closed-form marginal exactly when chained.
        """
        ab_t = self.alpha_bar[t].view(-1, 1)
        ab_prev = torch.where(t > 0, self.alpha_bar[(t - 1).clamp(min=0)],
                               torch.ones_like(self.alpha_bar[0])).view(-1, 1)
        rho = (ab_t / ab_prev.clamp(min=1e-8)).clamp(0, 1)

        n = counts_prev.round().clamp(min=0)
        rho_b = rho.expand_as(n).clamp(1e-6, 1 - 1e-6)
        thinned = torch.distributions.Binomial(total_count=n, probs=rho_b).sample()

        noise_rate = ((1 - rho) * self.mu).clamp(min=1e-6)
        noise = torch.poisson(noise_rate.expand_as(n))

        return thinned + noise

    def q_step(self, x_prev: torch.Tensor, t: torch.Tensor):
        """Convenience wrapper of q_step_counts operating on normalized
        ([0,1]-ish) signals instead of raw counts."""
        counts_prev = (x_prev * self.scale).clamp(min=0)
        counts_t = self.q_step_counts(counts_prev, t)
        return (counts_t / self.scale).clamp(0, 1)

    # ------------------------------------------------------------ training
    def corrupt(self, x0: torch.Tensor, t: torch.Tensor, device: str = None):
        """Interface matching the other corruption modules: returns
        (noisy, target) for residual-prediction training."""
        xt = self.q_sample(x0, t)
        return xt, xt - x0

    # ------------------------------------------------- reverse / generation
    @torch.no_grad()
    def sample(self, model, shape, device, num_samples: int = None):
        """Ancestral sampling: draws new waveforms starting from the
        noise-floor prior, using the model's residual predictions to walk
        the reverse chain down to t=0. This is what makes the audio branch
        a generative model, not just a denoiser."""
        model.eval()
        n = shape[0]
        x_t = (torch.poisson(torch.full(shape, self.mu, device=device)) / self.scale).clamp(0, 1)

        for step in reversed(range(self.T)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            r_hat = model(x_t, t)
            x0_hat = (x_t - r_hat).clamp(0, 1)

            if step == 0:
                x_t = x0_hat
                break

            ab_prev = self.alpha_bar[step - 1]
            rate = ab_prev * (x0_hat * self.scale) + (1 - ab_prev) * self.mu
            rate = rate.clamp(min=1e-6)
            x_t = (torch.poisson(rate) / self.scale).clamp(0, 1)

        return x_t
