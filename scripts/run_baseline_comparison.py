"""
Compares the spiking audio model against each strong non-spiking baseline
(1D U-Net, Dilated/WaveNet-style CNN, Residual TCN) in turn, in addition
to the topology-matched ablation baseline.

Resumable: pass --checkpoint-dir to save each model's state every epoch,
and --save-path results are written incrementally after EACH model
finishes. Re-running the same command later skips any model whose result
is already in --save-path, and resumes any in-progress one from its last
completed epoch -- safe to call repeatedly (e.g. after a Colab
disconnect) without redoing finished work.

Usage (from repo root):
    python scripts/run_baseline_comparison.py
    python scripts/run_baseline_comparison.py --checkpoint-dir /content/drive/MyDrive/ckpt/baselines
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
from src.models_audio import StrongAudioNet, NonSpikeAudioNet
from src.models_audio_baselines import BASELINES
from src.corruption_registry import build_corruption
from src.train import train_audio_model
from src.evaluate import evaluate_audio


def parse_args():
    p = argparse.ArgumentParser(description="Spiking vs strong ANN baselines (audio)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--audio-len", type=int, default=8000)
    p.add_argument("--audio-sr", type=int, default=16000)
    p.add_argument("--timesteps-audio", type=int, default=40)
    p.add_argument("--num-steps-audio", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--audio-subset-size", type=int, default=6000)
    p.add_argument("--corruption", type=str, default="poisson",
                    choices=["poisson", "gaussian", "bernoulli"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=str, default="outputs/baseline_comparison_results.json")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                    help="If set, saves each model's state every epoch here "
                         "and auto-resumes on restart.")
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

    all_names = ["spiking", "matched"] + list(BASELINES.keys())
    results = load_existing_results(args.save_path)
    if results:
        print(f"[Resume] Loaded existing results for: {list(results.keys())}")

    remaining = [n for n in all_names if n not in results]
    if not remaining:
        print("[Skip] All models already have saved results -- nothing to do.")
        print_summary(results, all_names)
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

    corruption = build_corruption(args.corruption, T=args.timesteps_audio, device=device)

    def ckpt_path_for(name):
        if not args.checkpoint_dir:
            return None
        return os.path.join(args.checkpoint_dir, f"model_{name}.pt")

    def build_model(name):
        if name == "spiking":
            return StrongAudioNet(num_steps=args.num_steps_audio,
                                   T_audio=args.timesteps_audio).to(device)
        if name == "matched":
            return NonSpikeAudioNet(T_audio=args.timesteps_audio).to(device)
        return BASELINES[name](T_audio=args.timesteps_audio).to(device)

    for name in remaining:
        print("\n" + "=" * 70)
        print(f"MODEL: {name}")
        print("=" * 70)

        torch.manual_seed(args.seed)
        model = build_model(name)
        display_name = "Spiking Audio" if name == "spiking" else f"Non-Spiking Audio ({name})"

        model = train_audio_model(model, display_name, train_loader, corruption,
                                   args.timesteps_audio, args.epochs, args.lr, device,
                                   checkpoint_path=ckpt_path_for(name))

        res = evaluate_audio(model, display_name, test_loader, corruption,
                              args.timesteps_audio, device, seed=args.seed)
        results[name] = res

        save_results(args.save_path, results)
        print(f"[Saved] {args.save_path} now has: {list(results.keys())}")

    print_summary(results, all_names)


def print_summary(results, all_names):
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'MSE':>10} {'SI-SDR Imp (dB)':>18} {'SNR Imp (dB)':>15}")
    print("-" * 66)
    for name in all_names:
        if name not in results:
            print(f"{name:<20} {'(not run)':>10}")
            continue
        r = results[name]
        print(f"{name:<20} {r['MSE']:>10.5f} {r['SI-SDR Imp']:>18.2f} {r['SNR Imp']:>15.2f}")


if __name__ == "__main__":
    main()
