"""
Checkpoint save/load utility for resumable training.

Saves model + optimizer + scheduler state + the epoch number after every
epoch, to a single file per model (overwritten each epoch, so it always
holds the LATEST progress). If training is interrupted (e.g. a Colab
disconnect) and restarted, `load_checkpoint` restores exactly where it
left off -- same weights, same optimizer momentum/Adam state, same LR
schedule position -- rather than restarting from a fresh random init.
"""

import os
import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "extra": extra or {},
    }, tmp_path)
    # Atomic-ish replace: write to a temp file first so a crash mid-save
    # (e.g. the exact moment of a disconnect) can't leave a corrupted
    # checkpoint behind.
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """Returns (last_completed_epoch, extra_dict) if a checkpoint exists,
    else None. `last_completed_epoch` is 0-indexed; training should resume
    at `last_completed_epoch + 1`."""
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", -1), ckpt.get("extra", {})
