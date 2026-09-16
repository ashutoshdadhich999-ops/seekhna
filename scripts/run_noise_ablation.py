"""
Noise-type ablation: trains the SAME spiking audio architecture
(StrongAudioNet) under three different forward corruption processes --
Poisson (thinning diffusion), Gaussian (standard DDPM-style), and
Bernoulli (rate-coded spike corruption) -- and compares denoising
quality. Answers "why Poisson?" empirically instead of asserting it.

Resumable: pass --checkpoint-dir to save each corruption type's model
state every epoch, and --save-path results are written incrementally
after EACH corruption type finishes. Re-running the same command later
skips any corruption type whose result is already in --save-path, and
resumes any in-progress one from its last completed epoch -- safe to
call repeatedly (e.g. after a Colab disconnect) without redoing
finished work.

Usage (from repo root):
    python scripts/run_noise_ablation.py
    python scripts/run_noise_ablation.py --checkpoint-dir /content/drive/MyDrive/ckpt/noise_ablation
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchaudio

from src.datasets import AudioDS
from src.models_audio import StrongAudioNet
from src.corruption_registry import build_corruption, CORRUPTION_TYPES
from src.train import train_audio_model
from src.evaluate import evaluate_audio


def parse_args():
    p = argparse.ArgumentParser(description="Poisson vs Gaussian vs Bernoulli audio corruption ablation")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--audio-len", type=int, default=8000)
    p.add_argument("--audio-sr", type=int, default=16000)
    p.add_argument("--timesteps-audio", type=int, default=40)
    p.add_argument("--num-steps-audio", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--audio-subset-size", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=str, default="outputs/noise_ablation_results.json")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                    help="If set, saves each corruption type's model state every "
                         "epoch here and auto-resumes on restart.")
    return p.parse_args()


def load_existing_results(save_path):
    if os.path.exists(save_path):
        with open(save_path) as f:
            return json.load(f)
    return {}


def save_results(save_path, results):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tmp = save_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, save_path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    results = load_existing_results(args.save_path)
    if results:
        print(f"[Resume] Loaded existing results for: {list(results.keys())}")

    remaining = [n for n in CORRUPTION_TYPES if n not in results]
    if not remaining:
        print("[Skip] All corruption types already have saved results -- nothing to do.")
        print_summary(results)
        return

    os.makedirs("./data", exist_ok=True)
    base = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True)
    subset = Subset(base, range(min(args.audio_subset_size, len(base))))
    tr_size = int(0.85 * len(subset))
    tr_sub, te_sub = random_split(subset, [tr_size, len(subset) - tr_size])

    train_loader = DataLoader(AudioDS(tr_sub, args.audio_len, args.audio_sr),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(AudioDS(te_sub, args.audio_len, args.audio_sr),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    for name in remaining:
        print("\n" + "=" * 70)
        print(f"CORRUPTION TYPE: {name}")
        print("=" * 70)

        torch.manual_seed(args.seed)  # same init/data order for every corruption type
        corruption = build_corruption(name, T=args.timesteps_audio, device=device)

        model = StrongAudioNet(num_steps=args.num_steps_audio,
                                T_audio=args.timesteps_audio).to(device)

        ckpt_path = None
        if args.checkpoint_dir:
            ckpt_path = os.path.join(args.checkpoint_dir, f"model_{name}.pt")

        model = train_audio_model(model, f"Spiking Audio ({name})", train_loader,
                                   corruption, args.timesteps_audio, args.epochs,
                                   args.lr, device, checkpoint_path=ckpt_path)

        res = evaluate_audio(model, f"Spiking Audio ({name})", test_loader,
                              corruption, args.timesteps_audio, device, seed=args.seed)
        results[name] = res

        save_results(args.save_path, results)
        print(f"[Saved] {args.save_path} now has: {list(results.keys())}")

    print_summary(results)


def print_summary(results):
    print("\n" + "=" * 70)
    print("NOISE-TYPE ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Corruption':<12} {'MSE':>10} {'SI-SDR Imp (dB)':>18} {'SNR Imp (dB)':>15}")
    print("-" * 58)
    for name in CORRUPTION_TYPES:
        if name not in results:
            print(f"{name:<12} {'(not run)':>10}")
            continue
        r = results[name]
        print(f"{name:<12} {r['MSE']:>10.5f} {r['SI-SDR Imp']:>18.2f} {r['SNR Imp']:>15.2f}")

    complete = [n for n in CORRUPTION_TYPES if n in results]
    if complete:
        best = max(complete, key=lambda n: results[n]["SI-SDR Imp"])
        print(f"\nBest corruption type by SI-SDR improvement (of those run): {best}")
        print("(Report this honestly in REPORT.md whether or not it's Poisson --"
              " a negative result for the Poisson hypothesis is still a valid finding.)")


if __name__ == "__main__":
    main()
