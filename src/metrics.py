"""Evaluation metrics for the audio branch."""

import numpy as np
import torch


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = torch.sum(est * ref, dim=-1, keepdim=True)
    energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + eps
    proj = (dot / energy) * ref
    noise = est - proj
    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def snr_fn(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    noise = est - ref
    return 10 * torch.log10(
        (torch.sum(ref ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    )


def safe_mean(values) -> float:
    arr = np.array(values)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if len(finite) > 0 else 0.0
