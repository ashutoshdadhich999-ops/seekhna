"""Training loops for the image and audio denoisers, with resumable
checkpointing: pass `checkpoint_path` to save model+optimizer+scheduler
state after every epoch, and to automatically resume from it if it
already exists (e.g. after a Colab disconnect and re-run)."""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.checkpoint import save_checkpoint, load_checkpoint


def train_img_model(model, name, train_loader, diff, timesteps, epochs, lr, device,
                     checkpoint_path=None, history_path=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 0
    loss_history = []
    if checkpoint_path is not None:
        result = load_checkpoint(checkpoint_path, model, opt, sched, device=device)
        if result is not None:
            last_epoch, extra = result
            start_epoch = last_epoch + 1
            loss_history = extra.get("loss_history", [])
            print(f"[Resume] {name}: found checkpoint at epoch {last_epoch + 1}/{epochs}, "
                  f"resuming from epoch {start_epoch + 1}.")

    if start_epoch >= epochs:
        print(f"[Resume] {name}: already fully trained ({epochs} epochs) -- skipping training.")
        return model

    print(f"\nTraining {name}...")
    for ep in range(start_epoch, epochs):
        model.train()
        total = 0.0
        for x, _ in tqdm(train_loader, leave=False, desc=f"{name} Ep {ep + 1}"):
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            xt = diff.q_sample(x, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        epoch_loss = total / len(train_loader)
        print(f"Epoch {ep + 1}/{epochs}  Loss: {epoch_loss:.4f}")
        loss_history.append(epoch_loss)

        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, model, opt, sched, epoch=ep,
                             extra={"loss_history": loss_history})
        if history_path is not None:
            _save_loss_history(history_path, name, loss_history)
    return model


def train_audio_model(model, name, train_loader, corruption, T_audio, epochs, lr, device,
                       checkpoint_path=None, history_path=None):
    """
    Args:
        corruption: any object exposing `.corrupt(x0, t, device) -> (noisy, target)`,
            e.g. a PoissonDiffusion / GaussianAudioDiffusion / BernoulliAudioDiffusion
            instance from src/corruption_registry.py.
        checkpoint_path: if given, model+optimizer+scheduler state (plus loss
            history) is saved after every epoch, and training auto-resumes
            from it on restart.
        history_path: if given, per-epoch loss is additionally written to
            this JSON file after every epoch (independent of checkpointing),
            so a loss-curve plot can be made even without inspecting the
            checkpoint file directly.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 0
    loss_history = []
    if checkpoint_path is not None:
        result = load_checkpoint(checkpoint_path, model, opt, sched, device=device)
        if result is not None:
            last_epoch, extra = result
            start_epoch = last_epoch + 1
            loss_history = extra.get("loss_history", [])
            print(f"[Resume] {name}: found checkpoint at epoch {last_epoch + 1}/{epochs}, "
                  f"resuming from epoch {start_epoch + 1}.")

    if start_epoch >= epochs:
        print(f"[Resume] {name}: already fully trained ({epochs} epochs) -- skipping training.")
        return model

    print(f"\nTraining {name}...")
    for ep in range(start_epoch, epochs):
        model.train()
        total = 0.0
        for x0 in tqdm(train_loader, leave=False, desc=f"{name} Ep {ep + 1}"):
            x0 = x0.to(device)
            t = torch.randint(0, T_audio, (x0.size(0),), device=device)
            noisy, target = corruption.corrupt(x0, t, device)
            pred = model(noisy, t)
            loss = F.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        epoch_loss = total / len(train_loader)
        print(f"Epoch {ep + 1}/{epochs}  Loss: {epoch_loss:.5f}")
        loss_history.append(epoch_loss)

        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, model, opt, sched, epoch=ep,
                             extra={"loss_history": loss_history})
        if history_path is not None:
            _save_loss_history(history_path, name, loss_history)
    return model


def _save_loss_history(history_path, name, loss_history):
    """Append/update this model's loss history in a shared JSON file, so
    multiple models (e.g. spiking + non-spiking) can share one history
    file per run without clobbering each other."""
    import json
    data = {}
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data[name] = loss_history
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    tmp = history_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, history_path)
