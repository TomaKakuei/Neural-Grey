"""Base image creation, light spot injection, and crop helpers."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import LightSpotConfig


def create_base_image(width: int, height: int, seed: int | None = None) -> np.ndarray:
    """Create the locked pre-noise base image: a pure white field."""

    del seed
    return np.ones((height, width), dtype=np.float32)


def generate_light_spots(
    width: int,
    height: int,
    config: LightSpotConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a smooth multiplicative light-spot field."""

    yy, xx = np.indices((height, width), dtype=np.float32)
    spots = np.ones((height, width), dtype=np.float32)

    for _ in range(int(config.num_spots)):
        x = np.float32(rng.uniform(*config.spot_position_range) * width)
        y = np.float32(rng.uniform(*config.spot_position_range) * height)
        sigma = np.float32(rng.uniform(*config.spot_size_range) * min(width, height))
        sigma = np.float32(max(float(sigma), 1.0))
        intensity = np.float32(rng.uniform(*config.spot_intensity_range))

        d2 = (xx - x) ** 2 + (yy - y) ** 2
        spot = np.exp(-0.5 * d2 / (sigma * sigma), dtype=np.float32)
        spots += (intensity - np.float32(1.0)) * spot

    return np.clip(spots, 0.0, 2.0).astype(np.float32, copy=False)


def lock_base_image(base_image: np.ndarray, light_spots: np.ndarray) -> np.ndarray:
    """Apply light spots once; no noise is added at this stage."""

    return np.clip(base_image * light_spots, 0.0, 1.5).astype(np.float32, copy=False)


def rotate_and_center_crop(
    image: np.ndarray,
    crop_width: int,
    crop_height: int,
    angle_deg: float,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> np.ndarray:
    """Rotate an image around its center and take a center crop."""

    h, w = image.shape[:2]
    if crop_width > w or crop_height > h:
        raise ValueError(
            f"Crop size {crop_width}x{crop_height} exceeds image size {w}x{h}."
        )

    center = (float((w - 1) * 0.5), float((h - 1) * 0.5))
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0).astype(np.float32)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
    )

    x0 = int(round((w - crop_width) * 0.5))
    y0 = int(round((h - crop_height) * 0.5))
    return rotated[y0 : y0 + crop_height, x0 : x0 + crop_width].astype(
        np.float32, copy=False
    )


def sample_actual_angle(
    base_angle_deg: float,
    jitter_range_deg: Tuple[float, float],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Sample one non-negative angle jitter and return jitter plus actual angle."""

    jitter = float(rng.uniform(*jitter_range_deg))
    return jitter, float(base_angle_deg + jitter)


def crop_at_angle(
    image: np.ndarray,
    crop_width: int,
    crop_height: int,
    angle: float,
    rng: np.random.Generator | None = None,
    angle_perturbation: float = 0.0,
) -> np.ndarray:
    """Compatibility wrapper around rotate_and_center_crop."""

    if rng is not None and angle_perturbation > 0.0:
        angle = float(angle) + float(rng.uniform(0.0, angle_perturbation))
    return rotate_and_center_crop(image, crop_width, crop_height, angle)
