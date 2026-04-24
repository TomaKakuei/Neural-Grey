"""Final color and exposure noise layer."""

from __future__ import annotations

import numpy as np

from .config import NoiseConfig


def generate_noise_intensity_ratio(
    rng: np.random.Generator,
    noise_config: NoiseConfig | None = None,
) -> float:
    """Sample the per-group noise strength ratio."""

    if noise_config is None:
        low, high = 0.5, 1.5
    else:
        low, high = noise_config.intensity_ratio_range
    return float(rng.uniform(low, high))


def apply_noise(
    image: np.ndarray,
    color_noise_sigma: float,
    exposure_noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one final, non-fixed noise realization to an RGB image."""

    h, w = image.shape[:2]
    if image.ndim == 2:
        image_rgb = np.repeat(image[:, :, None], 3, axis=2)
    else:
        image_rgb = image.astype(np.float32, copy=True)

    color = rng.normal(0.0, color_noise_sigma, size=(h, w, 3)).astype(
        np.float32, copy=False
    )
    exposure = np.float32(rng.normal(0.0, exposure_noise_sigma))
    noisy = image_rgb + color + exposure
    return np.clip(noisy, 0.0, 1.0).astype(np.float32, copy=False)


def apply_noise_fixed_ratio(
    image: np.ndarray,
    noise_config: NoiseConfig,
    rng: np.random.Generator,
    intensity_ratio: float = 1.0,
) -> np.ndarray:
    """Apply noise while keeping the group-level strength ratio fixed."""

    return apply_noise(
        image=image,
        color_noise_sigma=float(noise_config.color_noise_sigma * intensity_ratio),
        exposure_noise_sigma=float(noise_config.exposure_noise_sigma * intensity_ratio),
        rng=rng,
    )
