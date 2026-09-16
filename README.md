# Spiking vs Non-Spiking Residual Denoising

A controlled comparison between a spiking neural network (SNN) and
topologically matched non-spiking (ANN) residual denoisers, across two
modalities (MNIST, SpeechCommands), with a real Poisson diffusion process
for audio, a MAC/AC-based energy estimate, and a set of ablations
(noise-process choice, baseline architecture strength) rather than a
single uncontested comparison.

## Overview

**Image branch (MNIST):** standard Gaussian diffusion forward process;
the network predicts the added noise (epsilon-prediction).

**Audio branch (SpeechCommands):** a genuine Poisson thinning diffusion
process (`src/diffusion_audio_poisson.py`) with a real forward transition
`q(x_t | x_{t-1})`, derived from the Poisson thinning/superposition
theorems, a closed-form marginal `q(x_t | x_0)` used for efficient
training, and a reverse ancestral sampler that can generate audio from
pure noise — making this branch a genuine generative model, not only a
single-shot denoiser. The forward-chain vs. closed-form consistency is
numerically verified in `scripts/verify_poisson_diffusion.py`.

Both branches condition the network on the diffusion timestep and use
matched spiking/non-spiking block topology, so the only architectural
difference is the neuron model.

**Headline result:** across 3 seeds, the spiking audio denoiser trails
the matched non-spiking baseline by ~0.9 dB SI-SDR, but does so at
**27.0 ± 0.35% lower estimated energy** and **72.5% activation
sparsity** — a genuine quality/efficiency trade-off. See
[Results](#results) below and `REPORT.md` for the full picture,
including two ablations that were run to stress-test this finding.

See [`REPORT.md`](REPORT.md) for the full write-up: methodology, the
Poisson-diffusion derivation and its numerical verification, the energy
analysis methodology, ablation protocols, and honestly-reported
limitations.

## What's New in v2

Building on the fixed-bug v1 (paired evaluation, on-distribution
sparsity/latency), this version adds:

| Addition | What it does | Where |
|---|---|---|
| **Real Poisson diffusion** | Proper `q(x_t\|x_{t-1})` forward chain + closed-form marginal + reverse ancestral sampler, derived from the Poisson thinning theorem | `src/diffusion_audio_poisson.py` |
| **Energy analysis** | Per-layer MAC/AC counting via forward hooks, converted to an energy estimate using cited literature per-op costs (Horowitz 2014). A first version applied AC pricing uniformly and got a real Colab run *wrong* (-247% "savings"); now fixed to inspect each layer's actual input and only price genuine spike-fed layers as AC — see REPORT.md §4 | `src/energy.py` |
| **Noise-process ablation** | Same architecture trained under Poisson / Gaussian / Bernoulli corruption, compared empirically | `src/diffusion_audio_{gaussian,bernoulli}.py`, `scripts/run_noise_ablation.py` |
| **Temporal denoising viz** | Denoising behavior shown across `t = 0, 5, 10, 20, 30, 39` | `src/visualize.py::plot_temporal_denoising` |
| **Stronger ANN baselines** | 1D U-Net, dilated (WaveNet-style) CNN, Residual TCN, in addition to the matched-topology baseline | `src/models_audio_baselines.py`, `scripts/run_baseline_comparison.py` |
| **Adaptive-step spiking block** | ACT-style (Graves 2016) learned per-sample halting — a step towards event-driven computation, explicitly *not* claimed to be a full continuous-time ODE-SNN | `src/models_audio_adaptive.py` |
| **Multi-seed runner** | Aggregates mean ± std across seeds, saves incrementally per-seed (Colab-disconnect-safe) | `scripts/run_multiseed.py` |
| **Resumable checkpointing** | `--checkpoint-dir` saves model+optimizer+scheduler state every epoch; re-running the same command auto-resumes instead of restarting from scratch | `src/checkpoint.py`, wired into `main.py` and `scripts/run_multiseed.py` |

## Architecture

- **Spiking residual block** — Conv → GroupNorm → LIF neuron (surrogate
  gradient, `fast_sigmoid`), unrolled over `N` internal timesteps and
  rate-averaged, with a skip connection.
- **Non-spiking residual block** — identical topology with SiLU instead
  of LIF; single forward pass.
- **Adaptive spiking block** (`models_audio_adaptive.py`) — same LIF core,
  but a learned halting unit lets each sample use fewer internal steps.
- **Strong ANN baselines** (`models_audio_baselines.py`) — 1D U-Net
  (encoder/decoder + skips), dilated WaveNet-style stack, and a Residual
  TCN (Bai et al., 2018 style), all matched on time-conditioning.

## Repository Structure

```
.
├── main.py                       # Entry point: trains + evaluates both branches
├── requirements.txt
├── src/
│   ├── diffusion_image.py        # DDPM-style noise schedule + time embedding (image)
│   ├── diffusion_audio_poisson.py    # Real Poisson thinning diffusion (default, audio)
│   ├── diffusion_audio_gaussian.py   # Gaussian corruption -- ablation variant
│   ├── diffusion_audio_bernoulli.py  # Bernoulli spike corruption -- ablation variant
│   ├── corruption_registry.py    # Factory: select audio corruption process by name
│   ├── models_image.py           # Spiking / non-spiking image residual blocks + nets
│   ├── models_audio.py           # Spiking / non-spiking audio residual blocks + nets
│   ├── models_audio_baselines.py # 1D U-Net, Dilated CNN, Residual TCN
│   ├── models_audio_adaptive.py  # Adaptive-step (ACT-style) spiking block
│   ├── energy.py                 # MAC/AC counting + literature-based energy estimate
│   ├── datasets.py                # SpeechCommands dataset wrapper
│   ├── metrics.py                 # SI-SDR, SNR, safe averaging
│   ├── train.py                   # Training loops (generic corruption interface)
│   ├── evaluate.py                # Paired evaluation, sparsity, latency measurement
│   └── visualize.py               # Figures, incl. temporal denoising grid
├── scripts/
│   ├── verify_poisson_diffusion.py   # Numerically verifies q(x_t|x_t-1) vs closed form
│   ├── run_multiseed.py              # Multi-seed ablation, incremental save
│   ├── run_noise_ablation.py         # Poisson vs Gaussian vs Bernoulli comparison
│   └── run_baseline_comparison.py    # Spiking vs U-Net / Dilated / TCN
├── outputs/                       # Created at runtime: figures + logs (gitignored)
├── REPORT.md
└── LICENSE
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Verify the diffusion math (no dataset/GPU needed, ~seconds):**
```bash
python scripts/verify_poisson_diffusion.py
```

**Main run (both branches, default = Poisson diffusion, matched baseline):**
```bash
python main.py
```

**Single branch:**
```bash
python main.py --skip-audio
python main.py --skip-image
```

**Noise-process ablation (Poisson vs Gaussian vs Bernoulli):**
```bash
python scripts/run_noise_ablation.py
```

**Strong-baseline comparison (spiking vs U-Net/Dilated/TCN):**
```bash
python scripts/run_baseline_comparison.py
```

**Multi-seed run (mean ± std, Colab-disconnect-safe):**
```bash
python scripts/run_multiseed.py --seeds 42 123 2024 7 999
```

**Resumable training (survives a Colab disconnect):**
```bash
# point --checkpoint-dir at a mounted Google Drive folder; re-running the
# SAME command later auto-resumes from the last completed epoch instead
# of starting over
python main.py --checkpoint-dir /content/drive/MyDrive/checkpoints

# works the same way for the multi-seed runner (per-seed checkpoints,
# and whole seeds that already finished are skipped entirely)
python scripts/run_multiseed.py --seeds 42 123 2024 \
    --checkpoint-root /content/drive/MyDrive/checkpoints \
    --save-dir /content/drive/MyDrive/results
```

**Generate audio from noise (reverse Poisson sampling):**
```bash
python main.py --skip-image --generate-samples
```

**Quick smoke test:**
```bash
python main.py --epochs-img 2 --epochs-audio 2 --audio-subset-size 500
```

Run `python main.py --help` / `python scripts/<name>.py --help` for the
full flag list. Figures are written to `outputs/figures/`.

## Results

**3-seed mean ± std (seeds 42, 123, 2024), T4 GPU, default hyperparameters:**

| Branch | Metric | Spiking | Non-Spiking (matched) |
|---|---|---|---|
| Image (MNIST) | Denoised MSE improvement | 94.75 ± 0.03% | 95.11 ± 0.02% |
| Audio (SpeechCommands) | SI-SDR improvement | 7.28 ± 0.20 dB | 8.19 ± 0.20 dB |
| Audio (SpeechCommands) | SNR improvement | 13.49 ± 0.08 dB | 13.91 ± 0.06 dB |
| Audio (SpeechCommands) | Spike rate / Sparsity | 0.275 ± 0.003 / 0.725 | 1.0 / 0.0 |
| Audio (SpeechCommands) | **Energy savings (spiking vs non-spiking)** | **27.0 ± 0.35%** | n/a |

**Headline finding:** the spiking model trails the matched non-spiking
baseline by a small, consistent margin on both image and audio denoising
quality (~0.9 dB SI-SDR on audio), but does so at **27% lower estimated
energy** with **72.5% activation sparsity** — a genuine quality/efficiency
trade-off, not a clean win either direction. See `REPORT.md` §7 for the
full discussion, including the two ablations below.

**Noise-process ablation (single run, seed 42):** Poisson (7.20 dB) and
Gaussian (7.26 dB) corruption gave statistically indistinguishable
SI-SDR improvement; Bernoulli was clearly worse (5.99 dB). **Poisson did
not outperform Gaussian** — the "biological spike statistics preserve
information better" hypothesis is not supported by this result, reported
here as a negative finding rather than omitted.

**Baseline-strength ablation (single run, seed 42):** the spiking model
(7.19 dB) beat the weakest baseline (TCN, 6.01 dB) but trailed the
matched ANN (8.10 dB), a dilated CNN (8.37 dB), and especially a 1D U-Net
(9.54 dB) — the strongest of the four non-spiking architectures tested.

Full per-corruption-type and per-baseline tables in `REPORT.md` §5-6.

## Bugs Found & Fixed on the First Real (Colab) Run

Running the smoke test on an actual T4 GPU surfaced two real bugs the
sandbox-only testing here couldn't have caught:

1. **Energy accounting bug** — the first energy estimate applied one
   overall spike rate uniformly to every layer, including layers that
   never actually see a spike, and charged non-firing elements at full
   MAC price instead of the zero cost event-driven hardware gives them.
   This produced a nonsensical **-247% "energy savings"** (spiking model
   costing *more*). Fixed to inspect each layer's real input tensor and
   price strictly by measured per-layer activity. Full derivation and
   before/after numbers in `REPORT.md` §4.
2. **NaN latency bug** — `measure_time` requested more warmup+timing
   batches than a small `--audio-subset-size` smoke test's test split
   contained, silently producing `nan±nan` instead of a real number.
   Fixed by cycling the test loader. Details in `REPORT.md` §4.1.
3. **No resume support** — the original run had no way to survive a
   Colab disconnect mid-training; a lost session meant starting over from
   epoch 1. Added `src/checkpoint.py` (model + optimizer + scheduler
   state saved every epoch) wired into `main.py` (`--checkpoint-dir`) and
   `scripts/run_multiseed.py` (`--checkpoint-root`, plus skipping whole
   seeds that already finished). Verified with a synthetic
   train-2-epochs / simulate-crash / resume-and-finish test before being
   pushed — resumed training picked up at the correct epoch with the
   correct weights, not a fresh random init.

Both are fixed in this version and were re-verified with targeted tests
before being pushed (see conversation / commit history).

## Limitations

See [`REPORT.md`](REPORT.md) Section 8 for the full list. Headline items:
- The spiking model does **not** win on raw denoising quality against
  most baselines tested — its advantage is energy/sparsity, not quality
  (see Results above and `REPORT.md` §6-7).
- The noise-process ablation does not support the Poisson hypothesis —
  Gaussian performed statistically indistinguishably from Poisson.
- Both ablations (noise-process, baseline-strength) are single-seed
  only; the main comparison is 3-seed.
- The adaptive spiking block is an ACT-style discrete approximation to
  event-driven computation, not a full continuous-time ODE-SNN.
- The energy estimate uses literature per-operation costs (Horowitz 2014,
  45nm CMOS), not a measurement on real neuromorphic hardware.
- Poisson thinning diffusion for audio is a novel construction for this
  project (derived from standard point-process theorems), not an
  implementation of a specific published paper — treat it as such when
  citing.

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic
  Models. *NeurIPS*.
- Kingman, J. F. C. (1993). *Poisson Processes*. Oxford University Press.
  (Thinning and superposition theorems.)
- Horowitz, M. (2014). Computing's Energy Problem (and what we can do
  about it). *ISSCC*.
- Graves, A. (2016). Adaptive Computation Time for Recurrent Neural
  Networks. *arXiv:1603.08983*.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of
  Generic Convolutional and Recurrent Networks for Sequence Modeling
  (TCN). *arXiv:1803.01271*.
- Oord, A. van den, et al. (2016). WaveNet: A Generative Model for Raw
  Audio. *arXiv:1609.03499*.
- Eshraghian, J. K., et al. (2021). Training Spiking Neural Networks Using
  Lessons from Deep Learning. *arXiv:2109.12894* (snnTorch).

## License

MIT — see [LICENSE](LICENSE).
