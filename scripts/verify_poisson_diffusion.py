"""
Sanity check: chaining the single-step transition q(x_t|x_{t-1}) for T
steps should match the closed-form marginal q(x_t|x0) used for training.
This does NOT require any dataset or GPU -- it runs on synthetic rate
signals and is meant to be run once to confirm the math in
diffusion_audio_poisson.py is self-consistent.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.diffusion_audio_poisson import PoissonDiffusion


def main():
    torch.manual_seed(0)
    device = "cpu"
    T = 40
    diff = PoissonDiffusion(T=T, scale=15.0, noise_floor_rate=0.05, device=device)

    n = 20000
    x0 = torch.rand(n, 1, device=device)  # synthetic clean rates in [0,1]

    # --- Path A: closed-form marginal, direct sample at t = T-1
    t_final = torch.full((n,), T - 1, dtype=torch.long, device=device)
    x_closed = diff.q_sample(x0, t_final)

    # --- Path B: chain the transitions T times starting from deterministic x0
    # Step 0 is a direct Poisson draw (x0 is deterministic, not yet a random
    # count, so there's nothing to "thin" yet). Steps 1..T-1 use proper
    # Binomial thinning of the previously *realized* count.
    ab0 = diff.alpha_bar[0]
    rate0 = ab0 * (x0 * diff.scale) + (1 - ab0) * diff.mu
    counts_chained = torch.poisson(rate0.clamp(min=1e-6))

    for step in range(1, T):
        t = torch.full((n,), step, dtype=torch.long, device=device)
        counts_chained = diff.q_step_counts(counts_chained, t)

    x_chained = (counts_chained / diff.scale).clamp(0, 1)

    mean_closed = x_closed.mean().item()
    mean_chained = x_chained.mean().item()
    std_closed = x_closed.std().item()
    std_chained = x_chained.std().item()

    print(f"Closed-form marginal  q(x_T | x0):  mean={mean_closed:.5f}  std={std_closed:.5f}")
    print(f"Chained single-step   q(x_t|x_t-1): mean={mean_chained:.5f}  std={std_chained:.5f}")
    print(f"Mean absolute difference: {abs(mean_closed - mean_chained):.5f}")

    rel_diff = abs(mean_closed - mean_chained) / max(mean_closed, 1e-6)
    print(f"Relative difference: {rel_diff * 100:.2f}%")

    if rel_diff < 0.05:
        print("\n[PASS] Chained transitions are consistent with the closed-form marginal.")
    else:
        print("\n[FAIL] Chained transitions diverge from the closed-form marginal — "
              "check the schedule / rho_t derivation.")

    # Also check the boundary behaviour: alpha_bar_0 -> 1 (near-clean),
    # alpha_bar_{T-1} -> ~0 (near-noise-floor)
    print(f"\nalpha_bar[0]   = {diff.alpha_bar[0].item():.4f}  (should be close to 1, i.e. ~clean)")
    print(f"alpha_bar[T-1] = {diff.alpha_bar[T - 1].item():.4f}  (should be close to 0, i.e. ~noise floor)")


if __name__ == "__main__":
    main()
