"""
Multi-seed ablation runner.

Runs the full image + audio pipeline (main.py's logic) across several
random seeds and aggregates results into mean +/- std, mirroring the
run_experiments.py pattern used for statistically reliable continual
learning results. Writes results incrementally after EACH seed so a
Colab disconnect only loses the current seed's run, not all of them.

Usage (from repo root):
    python scripts/run_multiseed.py --seeds 42 123 2024 7 999
    python scripts/run_multiseed.py --seeds 42 123 2024 --epochs-img 8 --epochs-audio 12
    python scripts/run_multiseed.py --seeds 42 123 2024 --save-dir /content/drive/MyDrive/aajki_results
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from main import run_image_branch, run_audio_branch


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed ablation runner")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--save-dir", type=str, default="outputs/multiseed")
    p.add_argument("--skip-image", action="store_true")
    p.add_argument("--skip-audio", action="store_true")

    # Pass-through knobs (kept minimal; extend as needed)
    p.add_argument("--epochs-img", type=int, default=12)
    p.add_argument("--epochs-audio", type=int, default=20)
    p.add_argument("--audio-subset-size", type=int, default=6000)
    p.add_argument("--corruption", type=str, default="poisson",
                    choices=["poisson", "gaussian", "bernoulli"])
    p.add_argument("--audio-arch", type=str, default="matched",
                    choices=["matched", "unet", "dilated", "tcn"])
    p.add_argument("--checkpoint-root", type=str, default=None,
                    help="If set, each seed gets its own checkpoint subfolder "
                         "under this path, and training auto-resumes per-seed "
                         "after a disconnect. Point this at a Google Drive path "
                         "for persistence, e.g. /content/drive/MyDrive/checkpoints.")

    return p.parse_args()


def make_run_args(cli_args, seed):
    """Build a main.py-style args namespace for one seed's run."""
    import argparse as _argparse
    checkpoint_dir = None
    if cli_args.checkpoint_root:
        checkpoint_dir = os.path.join(cli_args.checkpoint_root, f"seed_{seed}")
    ns = _argparse.Namespace(
        skip_image=cli_args.skip_image,
        skip_audio=cli_args.skip_audio,
        seed=seed,
        out_dir=f"outputs/multiseed/seed_{seed}",
        batch_size_img=64, epochs_img=cli_args.epochs_img, timesteps_img=20,
        num_steps_img=5, base_channels_img=32, lr_img=2e-4,
        audio_len=8000, audio_sr=16000, timesteps_audio=40, num_steps_audio=8,
        epochs_audio=cli_args.epochs_audio, batch_size_audio=32, lr_audio=3e-4,
        audio_subset_size=cli_args.audio_subset_size, corruption=cli_args.corruption,
        audio_arch=cli_args.audio_arch, max_rate_audio=0.9, noise_floor_rate=0.05,
        poisson_scale=15.0, generate_samples=False, checkpoint_dir=checkpoint_dir,
    )
    return ns


def summarize(results, keys):
    """mean/std across seeds for a list of scalar keys."""
    out = {}
    for k in keys:
        vals = [r[k] for r in results if r is not None and k in r]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return out


def main():
    cli_args = parse_args()
    os.makedirs(cli_args.save_dir, exist_ok=True)

    all_img_results = []
    all_audio_results = []

    for seed in cli_args.seeds:
        print("\n" + "#" * 70)
        print(f"# SEED {seed}")
        print("#" * 70)

        result_path = os.path.join(cli_args.save_dir, f"seed_{seed}_results.json")
        if os.path.exists(result_path):
            print(f"[Skip] {result_path} already exists -- this seed already "
                  f"finished a previous run. Delete the file to force a rerun.")
            with open(result_path) as f:
                saved = json.load(f)
            all_img_results.append(saved.get("image"))
            audio_reconstructed = None
            if saved.get("audio_res_s") is not None:
                audio_reconstructed = {
                    "res_s": saved["audio_res_s"], "res_ns": saved["audio_res_ns"],
                    **(saved.get("audio") or {}),
                }
            all_audio_results.append(audio_reconstructed)
            continue

        torch.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_args = make_run_args(cli_args, seed)
        os.makedirs(run_args.out_dir, exist_ok=True)

        img_res = run_image_branch(run_args, device) if not cli_args.skip_image else None
        audio_res = run_audio_branch(run_args, device) if not cli_args.skip_audio else None

        all_img_results.append(img_res)
        all_audio_results.append(audio_res)

        # Incremental save after EVERY seed -- protects against disconnects.
        with open(result_path, "w") as f:
            json.dump({"seed": seed, "image": img_res,
                       "audio": {k: v for k, v in (audio_res or {}).items()
                                 if k not in ("res_s", "res_ns")} if audio_res else None,
                       "audio_res_s": audio_res["res_s"] if audio_res else None,
                       "audio_res_ns": audio_res["res_ns"] if audio_res else None},
                      f, indent=2)
        print(f"\n[Saved] {cli_args.save_dir}/seed_{seed}_results.json")

    # -------- Aggregate --------
    img_summary = None
    if not cli_args.skip_image:
        img_summary = summarize(all_img_results, ["img_s_imp", "img_ns_imp"])

    audio_summary = None
    if not cli_args.skip_audio:
        flat = []
        for r in all_audio_results:
            if r is None:
                continue
            flat.append({
                "spiking_sdr_imp": r["res_s"]["SI-SDR Imp"],
                "nonspiking_sdr_imp": r["res_ns"]["SI-SDR Imp"],
                "spiking_snr_imp": r["res_s"]["SNR Imp"],
                "nonspiking_snr_imp": r["res_ns"]["SNR Imp"],
                "spike_rate": r["spike_rate"],
                "energy_savings_pct": r["energy_savings_pct"],
            })
        audio_summary = summarize(flat, list(flat[0].keys())) if flat else None

    summary = {"seeds": cli_args.seeds, "image": img_summary, "audio": audio_summary}
    with open(os.path.join(cli_args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY (mean \u00b1 std)")
    print("=" * 70)
    if img_summary:
        for k, v in img_summary.items():
            print(f"[IMAGE] {k}: {v['mean']:.4f} \u00b1 {v['std']:.4f}  (n={v['n']})")
    if audio_summary:
        for k, v in audio_summary.items():
            print(f"[AUDIO] {k}: {v['mean']:.4f} \u00b1 {v['std']:.4f}  (n={v['n']})")
    print(f"\nFull results saved under: {cli_args.save_dir}")


if __name__ == "__main__":
    main()
