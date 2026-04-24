"""Configuration for the vignette and aberration synthesis pipeline."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VignetteConfig:
    """Sensor-space vignette configuration."""

    width: int = 4000
    height: int = 4000
    focal_px: float = 2500.0
    x_elliptical_scale: float = 1.0

    center_shift_max_ratio: float = 0.05
    polynomial_curve_drift_range: Tuple[float, float] = (0.10, 0.10)
    polynomial_degree: int = 9
    corner_shift_range: Tuple[float, float,] = (-0.05, 0.05)


@dataclass
class AberrationConfig:
    """Lens aberration configuration."""

    astigmatism_strength: float = 0.02
    astigmatism_frequency: float = 10.0

    chromatic_strength_px: float = 1.75
    chromatic_frequency: float = 8.0
    chromatic_tangential_ratio: float = 0.35


@dataclass
class CropConfig:
    """Four-view crop configuration."""

    crop_width: int = 3000
    crop_height: int = 2000
    base_angles: Tuple[int, int, int, int] = (0, 90, 180, 270)
    angle_jitter_deg_range: Tuple[float, float] = (0.0, 7.0)


@dataclass
class NoiseConfig:
    """Final post-processing noise configuration."""

    color_noise_sigma: float = 0.02
    exposure_noise_sigma: float = 0.01
    intensity_ratio_range: Tuple[float, float] = (0.5, 1.5)


@dataclass
class LightSpotConfig:
    """Locked-base light spot configuration."""

    num_spots: int = 5
    spot_intensity_range: Tuple[float, float] = (0.8, 1.2)
    spot_size_range: Tuple[float, float] = (0.02, 0.08)
    spot_position_range: Tuple[float, float] = (0.1, 0.9)


@dataclass
class SynthesisConfig:
    """Top-level synthesis configuration."""

    output_dir: str = "vignette_aberration_dataset"
    num_groups: int = 400
    seed: int = 42
    visualize: bool = True

    vignette: VignetteConfig = field(default_factory=VignetteConfig)
    aberration: AberrationConfig = field(default_factory=AberrationConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    light_spot: LightSpotConfig = field(default_factory=LightSpotConfig)
