"""Deployable continuous focal-conditioned radial-gauge correction head.

The head is intentionally small and independent of Lensfun.  It preserves the
input estimator's angular log residual and changes only its radial median
profile.  One smooth focal law applies to every supported focal length; there
is no hard cutoff, abstention branch, database lookup, or test-time selector.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SENSOR_DIAGONAL_MM = 43.3
DEFAULT_D2_ENERGY_THRESHOLD = 0.015
DEFAULT_D2_SHRINKAGE_POWER = 6.0
DEFAULT_WIDE_FOCAL_SCALE = 1.0
DEFAULT_WIDE_COSINE_POWER = 7.5
DEFAULT_WIDE_BLEND = 0.80
DEFAULT_WIDE_TAPER_REFERENCE_DIAGONALS = 35.0 / DEFAULT_SENSOR_DIAGONAL_MM
DEFAULT_WIDE_TAPER_POWER = 8.0
DEFAULT_WIDE_DISAGREEMENT_THRESHOLD = 0.20
DEFAULT_WIDE_DISAGREEMENT_POWER = 8.0
DEFAULT_STANDARD_FOCAL_SCALE = 0.45
DEFAULT_STANDARD_COSINE_POWER = 6.0
DEFAULT_STANDARD_BLEND = 0.25
DEFAULT_STANDARD_ONSET_DIAGONALS = 42.0 / DEFAULT_SENSOR_DIAGONAL_MM
DEFAULT_STANDARD_ONSET_WIDTH_DIAGONALS = 2.0 / DEFAULT_SENSOR_DIAGONAL_MM
DEFAULT_STANDARD_TAPER_REFERENCE_DIAGONALS = 2.5
DEFAULT_STANDARD_TAPER_POWER = 20.0
DEFAULT_STANDARD_DISAGREEMENT_THRESHOLD = 0.05
DEFAULT_STANDARD_DISAGREEMENT_POWER = 4.0


def normalize(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field, dtype=np.float32)
    return value / np.float32(max(float(np.max(value)), 1e-8))


def d2_low_frequency_stabilize(
    field: np.ndarray,
    sigma: float = 10.0,
    *,
    energy_threshold: float = DEFAULT_D2_ENERGY_THRESHOLD,
    shrinkage_power: float = DEFAULT_D2_SHRINKAGE_POWER,
) -> np.ndarray:
    """Continuously shrink low-frequency non-D2 energy and retain the high pass."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if energy_threshold <= 0.0:
        raise ValueError("energy_threshold must be positive")
    if shrinkage_power <= 0.0:
        raise ValueError("shrinkage_power must be positive")
    # Kept local so the radial head remains importable in NumPy-only contexts.
    import cv2

    base = normalize(field)
    low = cv2.GaussianBlur(base.astype(np.float32), (0, 0), float(sigma))
    d2_low = 0.25 * (low + np.fliplr(low) + np.flipud(low) + np.flipud(np.fliplr(low)))
    correction = d2_low - low
    energy = float(np.mean(np.abs(correction)))
    weight = energy**shrinkage_power / (
        energy**shrinkage_power + float(energy_threshold) ** shrinkage_power
    )
    return normalize(base + np.float32(weight) * correction)


def radial_coordinates(shape) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx = np.float32((w - 1) * 0.5)
    cy = np.float32((h - 1) * 0.5)
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return radius / np.float32(max(float(np.max(radius)), 1e-6))


def radial_log_profile(log_field, radius, bins=48):
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    index = np.minimum(np.floor(radius * bins).astype(np.int32), bins - 1)
    profile = np.full(bins, np.nan, dtype=np.float32)
    for bin_index in range(bins):
        selected = index == bin_index
        if np.any(selected):
            profile[bin_index] = np.float32(np.median(log_field[selected]))
    valid = np.isfinite(profile)
    if not np.any(valid):
        raise ValueError("field has no finite radial-profile samples")
    profile[~valid] = np.interp(centers[~valid], centers[valid], profile[valid])
    rendered = np.interp(radius.ravel(), centers, profile).reshape(radius.shape)
    return profile.astype(np.float32), rendered.astype(np.float32)


def cosine_log_profile(
    focal_length_mm: float,
    bins: int = 48,
    sensor_diagonal_mm: float = DEFAULT_SENSOR_DIAGONAL_MM,
    focal_scale: float = DEFAULT_STANDARD_FOCAL_SCALE,
    cosine_power: float = DEFAULT_STANDARD_COSINE_POWER,
) -> np.ndarray:
    if focal_length_mm <= 0.0:
        raise ValueError("focal_length_mm must be positive")
    centers = (np.arange(bins, dtype=np.float32) + 0.5) / bins
    focal_normalized = focal_length_mm * focal_scale / (sensor_diagonal_mm * 0.5)
    # cos(arctan(r/f))**p = (1 + (r/f)**2)**(-p/2).
    return (-0.5 * cosine_power * np.log1p((centers / focal_normalized) ** 2)).astype(
        np.float32
    )


def apply_radial_gauge_head(
    field: np.ndarray,
    focal_length_mm: float,
    *,
    sensor_diagonal_mm: float = DEFAULT_SENSOR_DIAGONAL_MM,
    wide_focal_scale: float = DEFAULT_WIDE_FOCAL_SCALE,
    wide_cosine_power: float = DEFAULT_WIDE_COSINE_POWER,
    wide_blend: float = DEFAULT_WIDE_BLEND,
    wide_taper_reference_diagonals: float = DEFAULT_WIDE_TAPER_REFERENCE_DIAGONALS,
    wide_taper_power: float = DEFAULT_WIDE_TAPER_POWER,
    wide_disagreement_threshold: float = DEFAULT_WIDE_DISAGREEMENT_THRESHOLD,
    wide_disagreement_power: float = DEFAULT_WIDE_DISAGREEMENT_POWER,
    standard_focal_scale: float = DEFAULT_STANDARD_FOCAL_SCALE,
    standard_cosine_power: float = DEFAULT_STANDARD_COSINE_POWER,
    standard_blend: float = DEFAULT_STANDARD_BLEND,
    standard_onset_diagonals: float = DEFAULT_STANDARD_ONSET_DIAGONALS,
    standard_onset_width_diagonals: float = DEFAULT_STANDARD_ONSET_WIDTH_DIAGONALS,
    standard_taper_reference_diagonals: float = DEFAULT_STANDARD_TAPER_REFERENCE_DIAGONALS,
    standard_taper_power: float = DEFAULT_STANDARD_TAPER_POWER,
    standard_disagreement_threshold: float = DEFAULT_STANDARD_DISAGREEMENT_THRESHOLD,
    standard_disagreement_power: float = DEFAULT_STANDARD_DISAGREEMENT_POWER,
    bins: int = 48,
) -> np.ndarray:
    """Correct a normalized shading estimate using only known camera metadata.

    No image target, ground truth, Lensfun record, or test-time score is used.
    The field radial profile is continuously mixed with wide and standard
    analytic representatives.  Both weights depend on dimensionless focal
    metadata and capture--prior profile disagreement.  They are positive at
    every finite focal length; no case score or database profile is queried.
    """
    for name, value in (("wide_blend", wide_blend), ("standard_blend", standard_blend)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    positive = {
        "wide_taper_reference_diagonals": wide_taper_reference_diagonals,
        "wide_taper_power": wide_taper_power,
        "wide_disagreement_threshold": wide_disagreement_threshold,
        "wide_disagreement_power": wide_disagreement_power,
        "standard_onset_width_diagonals": standard_onset_width_diagonals,
        "standard_taper_reference_diagonals": standard_taper_reference_diagonals,
        "standard_taper_power": standard_taper_power,
        "standard_disagreement_threshold": standard_disagreement_threshold,
        "standard_disagreement_power": standard_disagreement_power,
    }
    for name, value in positive.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    base = normalize(field)
    radius = radial_coordinates(base.shape)
    base_log = np.log(np.maximum(base, np.float32(1e-4)))
    profile, rendered = radial_log_profile(base_log, radius, bins=bins)
    centers = (np.arange(bins) + 0.5) / bins
    center_offset = float(np.median(profile[centers <= 0.20]))
    profile = profile - np.float32(center_offset)
    rendered = rendered - np.float32(center_offset)
    wide_profile = cosine_log_profile(
        focal_length_mm,
        bins=bins,
        sensor_diagonal_mm=sensor_diagonal_mm,
        focal_scale=wide_focal_scale,
        cosine_power=wide_cosine_power,
    )
    standard_profile = cosine_log_profile(
        focal_length_mm,
        bins=bins,
        sensor_diagonal_mm=sensor_diagonal_mm,
        focal_scale=standard_focal_scale,
        cosine_power=standard_cosine_power,
    )
    wide_disagreement = float(np.mean(np.abs(profile - wide_profile)))
    standard_disagreement = float(np.mean(np.abs(profile - standard_profile)))
    wide_gate = wide_disagreement**wide_disagreement_power / (
        wide_disagreement**wide_disagreement_power
        + float(wide_disagreement_threshold) ** wide_disagreement_power
    )
    standard_gate = standard_disagreement**standard_disagreement_power / (
        standard_disagreement**standard_disagreement_power
        + float(standard_disagreement_threshold) ** standard_disagreement_power
    )
    focal_diagonals = float(focal_length_mm) / float(sensor_diagonal_mm)
    wide_weight = (
        float(wide_blend)
        * wide_gate
        / (
            1.0
            + (focal_diagonals / float(wide_taper_reference_diagonals))
            ** float(wide_taper_power)
        )
    )
    onset_argument = np.clip(
        (focal_diagonals - float(standard_onset_diagonals))
        / float(standard_onset_width_diagonals),
        -60.0,
        60.0,
    )
    onset = 1.0 / (1.0 + float(np.exp(-onset_argument)))
    standard_weight = (
        float(standard_blend)
        * standard_gate
        * onset
        / (
            1.0
            + (focal_diagonals / float(standard_taper_reference_diagonals))
            ** float(standard_taper_power)
        )
    )
    total_weight = wide_weight + standard_weight
    if total_weight > 1.0:
        wide_weight /= total_weight
        standard_weight /= total_weight
        total_weight = 1.0
    fused = (
        (1.0 - total_weight) * profile
        + wide_weight * wide_profile
        + standard_weight * standard_profile
    )
    rendered_fused = np.interp(radius.ravel(), centers, fused).reshape(radius.shape)
    angular = base_log - rendered
    return normalize(np.exp(np.clip(angular + rendered_fused, -8.0, 2.0)))


def apply_focal_conditioned_gauge(
    field: np.ndarray,
    focal_length_mm: float,
    *,
    d2_sigma: float = 10.0,
    d2_reference_long_edge: float = 256.0,
    **radial_head_options,
) -> np.ndarray:
    """Full promoted post-fit estimator: fixed-D2 followed by the radial head.

    Every focal length follows the same continuous rule.  ``d2_sigma`` is
    specified at ``d2_reference_long_edge`` and scales with image size, so the
    default has the same spatial support as sigma=10 on the locked 256-pixel
    development audit.
    """
    if d2_reference_long_edge <= 0.0:
        raise ValueError("d2_reference_long_edge must be positive")
    value = np.asarray(field)
    if value.ndim != 2:
        raise ValueError("field must be a two-dimensional map")
    scaled_sigma = float(d2_sigma) * max(value.shape) / float(d2_reference_long_edge)
    stabilized = d2_low_frequency_stabilize(value, sigma=scaled_sigma)
    return apply_radial_gauge_head(stabilized, focal_length_mm, **radial_head_options)
