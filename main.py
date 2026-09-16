"""
Spiking vs Non-Spiking Residual Denoising -- main experiment script.

Trains and evaluates the image (MNIST) and audio (SpeechCommands)
branches. The audio branch now uses a proper Poisson thinning diffusion
process by default (src/diffusion_audio_poisson.py) with a real
q(x_t|x_{t-1}) transition and a reverse ancestral sampler, and reports a
MAC/AC-based energy estimate alongside the usual quality metrics.

Usage:
    python main.py                              # full run, default config
    python main.py --skip-audio                 # image branch only
    python main.py --skip-image                 # audio branch only
    python main.py --corruption gaussian         # audio ablation: try Gaussian corruption
    python main.py --audio-arch unet             # audio ablation: stronger ANN baseline
    python main.py --epochs-img 2 --epochs-audio 2 --audio-subset-size 500   # smoke test
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import torchaudio

from src.diffusion_image import Diffusion
from src.models_image import StrongImgNet, NonSpikeImgNet
from src.models_audio import StrongAudioNet, NonSpikeAudioNet
from src.models_audio_baselines import BASELINES
from src.datasets import AudioDS
from src.corruption_registry import build_corruption
from src.train import train_img_model, train_audio_model
from src.evaluate import (
    eval_img_pair, evaluate_audio_pair, measure_sparsity, measure_time,
)
from src.visualize import (
    plot_image_samples, plot_audio_waveforms, plot_comparison_bars,
    plot_temporal_denoising, plot_loss_curves,
)
from src.energy import estimate_energy, real_world_savings


def parse_args():
    p = argparse.ArgumentParser(description="Spiking vs Non-Spiking Residual Denoising")
    p.add_argument("--skip-image", action="store_true")
    p.add_argument("--skip-audio", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs")

    # Image branch
    p.add_argument("--batch-size-img", type=int, default=64)
    p.add_argument("--epochs-img", type=int, default=12)
    p.add_argument("--timesteps-img", type=int, default=20)
    p.add_argument("--num-steps-img", type=int, default=5)
    p.add_argument("--base-channels-img", type=int, default=32)
    p.add_argument("--lr-img", type=float, default=2e-4)

    # Audio branch
    p.add_argument("--audio-len", type=int, default=8000)
    p.add_argument("--audio-sr", type=int, default=16000)
    p.add_argument("--timesteps-audio", type=int, default=40)
    p.add_argument("--num-steps-audio", type=int, default=8)
    p.add_argument("--epochs-audio", type=int, default=20)
    p.add_argument("--batch-size-audio", type=int, default=32)
    p.add_argument("--lr-audio", type=float, default=3e-4)
    p.add_argument("--audio-subset-size", type=int, default=6000)
    p.add_argument("--corruption", type=str, default="poisson",
                    choices=["poisson", "gaussian", "bernoulli"],
                    help="Audio forward corruption process (default: poisson, "
                         "the proper diffusion process with q(x_t|x_t-1)).")
    p.add_argument("--audio-arch", type=str, default="matched",
                    choices=["matched", "unet", "dilated", "tcn"],
                    help="Non-spiking baseline architecture: 'matched' is the "
                         "topology-matched ablation baseline; the others are "
                         "stronger established architectures.")
    p.add_argument("--max-rate-audio", type=float, default=0.9)
    p.add_argument("--noise-floor-rate", type=float, default=0.05)
    p.add_argument("--poisson-scale", type=float, default=15.0)
    p.add_argument("--generate-samples", action="store_true",
                    help="After training, run the reverse (ancestral) Poisson "
                         "sampler to generate audio from pure noise.")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                    help="If set, saves model+optimizer+scheduler state after "
                         "every epoch here, and auto-resumes from it if the run "
                         "is restarted (e.g. after a Colab disconnect). Point "
                         "this at a Google Drive path for persistence across "
                         "sessions, e.g. /content/drive/MyDrive/checkpoints.")

    return p.parse_args()


def run_image_branch(args, device):
    print("\n" + "=" * 70)
    print("PART 1: MNIST Image Denoising")
    print("=" * 70)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size_img, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size_img, shuffle=False,
                              num_workers=2, pin_memory=True)

    diff = Diffusion(T=args.timesteps_img, device=device)

    model_img = StrongImgNet(base_channels=args.base_channels_img,
                              num_steps=args.num_steps_img).to(device)
    model_ns_img = NonSpikeImgNet(base_channels=args.base_channels_img).to(device)

    ckpt_s = ckpt_ns = None
    history_path = None
    if args.checkpoint_dir:
        ckpt_s = os.path.join(args.checkpoint_dir, "model_img_spiking.pt")
        ckpt_ns = os.path.join(args.checkpoint_dir, "model_img_nonspiking.pt")
        history_path = os.path.join(args.checkpoint_dir, "loss_history_image.json")

    model_img = train_img_model(model_img, "Spiking Image", train_loader, diff,
                                 args.timesteps_img, args.epochs_img, args.lr_img, device,
                                 checkpoint_path=ckpt_s, history_path=history_path)
    model_ns_img = train_img_model(model_ns_img, "Non-Spiking Image", train_loader, diff,
                                    args.timesteps_img, args.epochs_img, args.lr_img, device,
                                    checkpoint_path=ckpt_ns, history_path=history_path)

    print("\n--- Image Results (paired, matched noise draws) ---")
    (img_s_noisy, img_s_den, img_s_imp), (img_ns_noisy, img_ns_den, img_ns_imp) = eval_img_pair(
        model_img, model_ns_img, "Spiking Image", "Non-Spiking Image",
        test_loader, diff, args.timesteps_img, device, seed=args.seed,
    )

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_image_samples(model_img, test_loader, diff, args.timesteps_img, device, fig_dir)
    if history_path is not None:
        plot_loss_curves(history_path, fig_dir, title="Image Branch Training Loss")

    return {
        "img_s_noisy": img_s_noisy, "img_s_den": img_s_den, "img_s_imp": img_s_imp,
        "img_ns_noisy": img_ns_noisy, "img_ns_den": img_ns_den, "img_ns_imp": img_ns_imp,
    }


def build_audio_baseline(args, device):
    if args.audio_arch == "matched":
        return NonSpikeAudioNet(T_audio=args.timesteps_audio).to(device)
    Cls = BASELINES[args.audio_arch]
    return Cls(T_audio=args.timesteps_audio).to(device)


def run_audio_branch(args, device):
    print("\n" + "=" * 70)
    print(f"PART 2: Audio Denoising (SpeechCommands) -- corruption='{args.corruption}', "
          f"baseline_arch='{args.audio_arch}'")
    print("=" * 70)

    os.makedirs("./data", exist_ok=True)
    base = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True)
    subset = Subset(base, range(min(args.audio_subset_size, len(base))))
    tr_size = int(0.85 * len(subset))
    tr_sub, te_sub = random_split(subset, [tr_size, len(subset) - tr_size])

    train_loader = DataLoader(AudioDS(tr_sub, args.audio_len, args.audio_sr),
                               batch_size=args.batch_size_audio, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(AudioDS(te_sub, args.audio_len, args.audio_sr),
                              batch_size=args.batch_size_audio, shuffle=False,
                              num_workers=2, pin_memory=True)

    corruption = build_corruption(args.corruption, T=args.timesteps_audio, device=device,
                                   max_rate=args.max_rate_audio,
                                   noise_floor_rate=args.noise_floor_rate,
                                   scale=args.poisson_scale)

    model_a = StrongAudioNet(num_steps=args.num_steps_audio, T_audio=args.timesteps_audio).to(device)
    model_ns_a = build_audio_baseline(args, device)

    ckpt_s = ckpt_ns = None
    history_path = None
    if args.checkpoint_dir:
        ckpt_s = os.path.join(args.checkpoint_dir, "model_audio_spiking.pt")
        ckpt_ns = os.path.join(args.checkpoint_dir, f"model_audio_nonspiking_{args.audio_arch}.pt")
        history_path = os.path.join(args.checkpoint_dir, "loss_history_audio.json")

    model_a = train_audio_model(model_a, "Spiking Audio", train_loader, corruption,
                                 args.timesteps_audio, args.epochs_audio, args.lr_audio, device,
                                 checkpoint_path=ckpt_s, history_path=history_path)
    model_ns_a = train_audio_model(model_ns_a, f"Non-Spiking Audio ({args.audio_arch})",
                                    train_loader, corruption, args.timesteps_audio,
                                    args.epochs_audio, args.lr_audio, device,
                                    checkpoint_path=ckpt_ns, history_path=history_path)

    print("\n--- Audio Results (paired, matched noise draws) ---")
    res_s, res_ns = evaluate_audio_pair(
        model_a, model_ns_a, "Spiking Audio", f"Non-Spiking Audio ({args.audio_arch})",
        test_loader, corruption, args.timesteps_audio, device, seed=args.seed,
    )

    spike_rate, sparsity = measure_sparsity(model_a, test_loader, corruption,
                                             args.timesteps_audio, device)
    (t_s, t_s_std), (t_ns, t_ns_std) = measure_time(model_a, model_ns_a, test_loader,
                                                      corruption, args.timesteps_audio, device)

    # -------- Energy estimate (MAC/AC-based, see src/energy.py) --------
    sample_x = next(iter(test_loader)).to(device)
    sample_t = torch.randint(0, args.timesteps_audio, (sample_x.size(0),), device=device)
    snn_energy = estimate_energy(model_a, sample_x, sample_t, spike_rate=spike_rate,
                                  is_spiking=True)
    ann_energy = estimate_energy(model_ns_a, sample_x, sample_t, is_spiking=False)
    energy_savings = real_world_savings(snn_energy, ann_energy)

    print("\n--- Energy Estimate (see REPORT.md for methodology/limitations) ---")
    print(f"Spiking model:     {snn_energy.snn_energy_pj / 1e6:.4f} uJ/sample "
          f"({snn_energy.total_acs:,} ACs @ measured spike rate {spike_rate:.4f})")
    print(f"Non-spiking model: {ann_energy.ann_energy_pj / 1e6:.4f} uJ/sample "
          f"({ann_energy.total_macs:,} MACs)")
    print(f"Estimated energy savings (spiking vs non-spiking): {energy_savings:.1f}%")

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_audio_waveforms(model_a, test_loader, corruption, args.timesteps_audio, device, fig_dir)
    plot_temporal_denoising(model_a, test_loader, corruption, args.timesteps_audio, device, fig_dir)
    if history_path is not None:
        plot_loss_curves(history_path, fig_dir, title="Audio Branch Training Loss")

    if args.generate_samples and args.corruption == "poisson":
        print("\nGenerating audio samples via reverse Poisson diffusion (ancestral sampling)...")
        gen = corruption.sample(model_a, (4, args.audio_len), device)
        torch.save(gen.cpu(), os.path.join(args.out_dir, "generated_samples.pt"))
        print(f"Saved 4 generated waveforms to {args.out_dir}/generated_samples.pt")

    return {
        "res_s": res_s, "res_ns": res_ns,
        "spike_rate": spike_rate, "sparsity": sparsity,
        "t_s": t_s, "t_s_std": t_s_std, "t_ns": t_ns, "t_ns_std": t_ns_std,
        "snn_energy_uj": snn_energy.snn_energy_pj / 1e6,
        "ann_energy_uj": ann_energy.ann_energy_pj / 1e6,
        "energy_savings_pct": energy_savings,
    }


def print_final_tables(img_results, audio_results):
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    if img_results:
        print("\n[IMAGE]")
        print(f"{'Model':<25} {'Noisy MSE':>12} {'Denoised MSE':>14} {'Improvement':>12}")
        print("-" * 65)
        print(f"{'Spiking':<25} {img_results['img_s_noisy']:>12.5f} "
              f"{img_results['img_s_den']:>14.5f} {img_results['img_s_imp']:>11.2f}%")
        print(f"{'Non-Spiking':<25} {img_results['img_ns_noisy']:>12.5f} "
              f"{img_results['img_ns_den']:>14.5f} {img_results['img_ns_imp']:>11.2f}%")

    if audio_results:
        res_s, res_ns = audio_results["res_s"], audio_results["res_ns"]
        print("\n[AUDIO]")
        print(f"{'Metric':<22} {'Spiking':>12} {'Non-Spiking':>14}")
        print("-" * 50)
        print(f"{'MSE (Denoised)':<22} {res_s['MSE']:>12.5f} {res_ns['MSE']:>14.5f}")
        print(f"{'SI-SDR Imp (dB)':<22} {res_s['SI-SDR Imp']:>12.2f} {res_ns['SI-SDR Imp']:>14.2f}")
        print(f"{'SNR Imp (dB)':<22} {res_s['SNR Imp']:>12.2f} {res_ns['SNR Imp']:>14.2f}")
        print(f"{'Spike Rate':<22} {audio_results['spike_rate']:>12.4f} {'1.0000':>14}")
        print(f"{'Sparsity':<22} {audio_results['sparsity']:>12.4f} {'0.0000':>14}")
        print(f"{'Time (ms/sample)':<22} "
              f"{audio_results['t_s']:>9.3f}\u00b1{audio_results['t_s_std']:.2f} "
              f"{audio_results['t_ns']:>10.3f}\u00b1{audio_results['t_ns_std']:.2f}")
        print(f"{'Energy (uJ/sample)':<22} {audio_results['snn_energy_uj']:>12.4f} "
              f"{audio_results['ann_energy_uj']:>14.4f}")
        print(f"{'Energy savings (%)':<22} {audio_results['energy_savings_pct']:>12.1f}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs(args.out_dir, exist_ok=True)

    img_results = run_image_branch(args, device) if not args.skip_image else None
    audio_results = run_audio_branch(args, device) if not args.skip_audio else None

    print_final_tables(img_results, audio_results)

    if img_results and audio_results:
        fig_dir = os.path.join(args.out_dir, "figures")
        plot_comparison_bars(
            img_results["img_s_imp"], img_results["img_ns_imp"],
            audio_results["res_s"]["SI-SDR Imp"], audio_results["res_ns"]["SI-SDR Imp"],
            audio_results["sparsity"], fig_dir,
        )

    print("\n" + "=" * 70)
    print(f"ALL DONE. Figures saved under: {os.path.join(args.out_dir, 'figures')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
