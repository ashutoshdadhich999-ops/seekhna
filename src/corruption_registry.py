"""Factory for the audio corruption process, so the noise-type ablation
(scripts/run_noise_ablation.py) can swap corruption processes without
touching the model, training loop, or evaluation code -- all three
processes expose the same `.corrupt(x0, t, device) -> (noisy, target)`
interface.
"""

from src.diffusion_audio_poisson import PoissonDiffusion
from src.diffusion_audio_gaussian import GaussianAudioDiffusion
from src.diffusion_audio_bernoulli import BernoulliAudioDiffusion

CORRUPTION_TYPES = ("poisson", "gaussian", "bernoulli")


def build_corruption(name: str, T: int, device: str, max_rate: float = 0.9,
                      noise_floor_rate: float = 0.05, scale: float = 15.0):
    name = name.lower()
    if name == "poisson":
        return PoissonDiffusion(T=T, scale=scale, noise_floor_rate=noise_floor_rate,
                                 device=device)
    if name == "gaussian":
        return GaussianAudioDiffusion(T=T, device=device)
    if name == "bernoulli":
        return BernoulliAudioDiffusion(T=T, device=device, max_rate=max_rate)
    raise ValueError(f"Unknown corruption type '{name}'. Choose from {CORRUPTION_TYPES}.")
