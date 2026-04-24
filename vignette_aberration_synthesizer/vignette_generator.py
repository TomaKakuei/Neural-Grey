"""Vignette, astigmatism, and chromatic aberration generators."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import AberrationConfig, VignetteConfig


POLY_COEFF_NAMES = ("a", "b", "c", "d", "f", "g", "h", "i", "j", "k")


def generate_cos4_vignette(
    width: int,
    height: int,
    focal_px: float,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    x_elliptical_scale: float = 1.0,
) -> np.ndarray:
    """Generate a standard cos^4 vignette in sensor coordinates."""

    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = np.float32(width * 0.5 if center_x is None else center_x)
    cy = np.float32(height * 0.5 if center_y is None else center_y)
    dx = (xx - cx) * np.float32(x_elliptical_scale)
    dy = yy - cy
    radius = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    theta = np.arctan(radius / np.float32(focal_px))
    return (np.cos(theta, dtype=np.float32) ** np.float32(4.0)).astype(
        np.float32, copy=False
    )


def normalize_to_unit(value: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] for diverse synthetic mask strength."""

    v_min = np.float32(np.min(value))
    v_max = np.float32(np.max(value))
    return ((value - v_min) / np.float32(max(float(v_max - v_min), 1e-8))).astype(
        np.float32, copy=False
    )


def apply_corner_shifts(
    vignette: np.ndarray,
    corner_shifts: Tuple[float, float, float, float],
) -> np.ndarray:
    """Apply independent smooth intensity offsets at TL, TR, BL, and BR."""

    h, w = vignette.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    xn = xx / np.float32(max(w - 1, 1))
    yn = yy / np.float32(max(h - 1, 1))

    tl = (1.0 - xn) * (1.0 - yn)
    tr = xn * (1.0 - yn)
    bl = (1.0 - xn) * yn
    br = xn * yn
    shift = (
        np.float32(corner_shifts[0]) * tl
        + np.float32(corner_shifts[1]) * tr
        + np.float32(corner_shifts[2]) * bl
        + np.float32(corner_shifts[3]) * br
    )
    return vignette * (1.0 + shift)


def solve_small_linear_system(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tiny dense linear system without numpy.linalg."""

    n = int(rhs.shape[0])
    aug = [
        [float(matrix[row, col]) for col in range(n)] + [float(rhs[row])]
        for row in range(n)
    ]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-14:
            raise ValueError("Singular polynomial interpolation matrix.")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return np.asarray([aug[row][n] for row in range(n)], dtype=np.float64)


def interpolate_cos4_polynomial_coeffs(
    focal_px: float,
    max_radius_px: float,
    degree: int,
) -> np.ndarray:
    """Convert the cos4 radial curve into ascending polynomial coefficients."""

    n = int(degree) + 1
    k = np.arange(n, dtype=np.float64)
    nodes = 0.5 * (1.0 - np.cos(np.pi * k / float(degree)))
    radii = nodes * float(max_radius_px)
    values = np.cos(np.arctan(radii / float(focal_px))) ** 4.0
    vandermonde = np.stack([nodes**power for power in range(n)], axis=1)
    return solve_small_linear_system(vandermonde, values)


def apply_polynomial_curve_drift(
    rho_map: np.ndarray,
    focal_px: float,
    max_radius_px: float,
    degree: int,
    target_drift: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Approximate cos4 as a polynomial and perturb coefficients by curve drift."""

    degree = int(degree)
    if degree != 9:
        raise ValueError("The coefficient drift model expects polynomial_degree=9.")

    sample_rho = np.linspace(0.0, 1.0, 1024, dtype=np.float64)
    sample_radius = sample_rho * float(max_radius_px)
    sample_curve = np.cos(np.arctan(sample_radius / float(focal_px))) ** 4.0

    base_coeffs = interpolate_cos4_polynomial_coeffs(
        focal_px=focal_px,
        max_radius_px=max_radius_px,
        degree=degree,
    )
    coeff_scale = np.maximum(np.abs(base_coeffs), np.max(np.abs(base_coeffs)) * 0.03)
    raw_delta = rng.normal(0.0, 1.0, size=base_coeffs.shape) * coeff_scale

    raw_curve_delta = np.polynomial.polynomial.polyval(sample_rho, raw_delta)
    max_raw_drift = float(np.max(np.abs(raw_curve_delta)))
    if max_raw_drift < 1e-12:
        raw_delta[-1] = coeff_scale[-1]
        raw_curve_delta = np.polynomial.polynomial.polyval(sample_rho, raw_delta)
        max_raw_drift = float(np.max(np.abs(raw_curve_delta)))

    delta_coeffs = raw_delta * (float(target_drift) / max_raw_drift)
    drifted_coeffs = base_coeffs + delta_coeffs
    drifted_curve = np.polynomial.polynomial.polyval(sample_rho, drifted_coeffs)
    achieved_drift = float(np.max(np.abs(drifted_curve - sample_curve)))

    drifted_map = np.polynomial.polynomial.polyval(
        rho_map.astype(np.float64, copy=False),
        drifted_coeffs,
    )

    names = POLY_COEFF_NAMES[: degree + 1]
    metadata = {
        "model": "cos4_degree9_polynomial_coefficient_curve_drift",
        "coefficient_names": list(names),
        "target_curve_drift": float(target_drift),
        "achieved_curve_max_abs_drift": achieved_drift,
        "base_coefficients": {
            name: float(value) for name, value in zip(names, base_coeffs)
        },
        "delta_coefficients": {
            name: float(value) for name, value in zip(names, delta_coeffs)
        },
        "drifted_coefficients": {
            name: float(value) for name, value in zip(names, drifted_coeffs)
        },
    }
    return np.clip(drifted_map, 0.0, 1.0).astype(np.float32, copy=False), metadata


def create_diverse_vignette(
    config: VignetteConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create both the pure cos4 map and the perturbed synthesis/GT map."""

    w, h = config.width, config.height
    gt = generate_cos4_vignette(
        w,
        h,
        focal_px=config.focal_px,
        x_elliptical_scale=config.x_elliptical_scale,
    )

    direction = float(rng.uniform(0.0, 2.0 * np.pi))
    radius_px = float(rng.uniform(0.0, config.center_shift_max_ratio) * min(w, h))
    center_shift_x_px = radius_px * float(np.cos(direction))
    center_shift_y_px = radius_px * float(np.sin(direction))
    yy, xx = np.indices((h, w), dtype=np.float32)
    center_x = np.float32(w * 0.5 + center_shift_x_px)
    center_y = np.float32(h * 0.5 + center_shift_y_px)
    dx = (xx - center_x) * np.float32(config.x_elliptical_scale)
    dy = yy - center_y
    radius = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    max_radius = float(np.max(radius))
    rho = radius / np.float32(max(max_radius, 1e-6))

    target_curve_drift = float(rng.uniform(*config.polynomial_curve_drift_range))
    curve_drifted, polynomial_meta = apply_polynomial_curve_drift(
        rho,
        focal_px=config.focal_px,
        max_radius_px=max_radius,
        degree=config.polynomial_degree,
        target_drift=target_curve_drift,
        rng=rng,
    )
    corner_shifts = tuple(float(rng.uniform(*config.corner_shift_range)) for _ in range(4))
    perturbed = apply_corner_shifts(curve_drifted, corner_shifts)
    perturbed = np.clip(perturbed, 0.0, 1.0).astype(np.float32, copy=False)

    metadata = {
        "center_shift_x_px": center_shift_x_px,
        "center_shift_y_px": center_shift_y_px,
        "center_shift_x_ratio": center_shift_x_px / float(w),
        "center_shift_y_ratio": center_shift_y_px / float(h),
        "polynomial_curve_drift": polynomial_meta,
        "corner_shifts": [float(x) for x in corner_shifts],
    }
    return perturbed, gt, metadata


def generate_astigmatism(
    shape: Tuple[int, int],
    strength: float = 0.02,
    frequency: float = 10.0,
    angle: float = 0.0,
) -> np.ndarray:
    """Generate a high-frequency astigmatism gain residual."""

    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    xn = xx / np.float32(max(w - 1, 1)) - np.float32(0.5)
    yn = yy / np.float32(max(h - 1, 1)) - np.float32(0.5)

    angle_rad = np.float32(np.deg2rad(angle))
    cos_a = np.cos(angle_rad, dtype=np.float32)
    sin_a = np.sin(angle_rad, dtype=np.float32)
    xr = xn * cos_a - yn * sin_a
    yr = xn * sin_a + yn * cos_a

    carrier = np.sin(xr * frequency * 2.0 * np.pi, dtype=np.float32) * np.sin(
        yr * frequency * 2.0 * np.pi, dtype=np.float32
    )
    envelope = np.clip(np.sqrt(xn * xn + yn * yn, dtype=np.float32) * 2.0, 0.0, 1.0)
    return (carrier * envelope * np.float32(strength)).astype(np.float32, copy=False)


def generate_chromatic_displacement(
    shape: Tuple[int, int],
    strength_px: float,
    frequency: float,
    tangential_ratio: float,
) -> dict:
    """Generate R/B radial and tangential sub-pixel displacement fields."""

    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx = np.float32((w - 1) * 0.5)
    cy = np.float32((h - 1) * 0.5)
    dx = xx - cx
    dy = yy - cy
    radius = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    radius_safe = np.maximum(radius, np.float32(1e-6))
    r_norm = radius / np.float32(max(float(radius.max()), 1.0))

    radial = (r_norm**1.35) * (
        0.75 + 0.25 * np.sin(r_norm * frequency * 2.0 * np.pi, dtype=np.float32)
    )
    ux = dx / radius_safe
    uy = dy / radius_safe
    tx = -uy
    ty = ux

    mag = np.float32(strength_px) * radial
    tmag = np.float32(strength_px * tangential_ratio) * radial * np.sin(
        r_norm * (frequency * 0.5) * 2.0 * np.pi, dtype=np.float32
    )
    disp_x = mag * ux + tmag * tx
    disp_y = mag * uy + tmag * ty
    return {
        "chromatic_dx_r": disp_x.astype(np.float32, copy=False),
        "chromatic_dy_r": disp_y.astype(np.float32, copy=False),
        "chromatic_dx_b": (-disp_x).astype(np.float32, copy=False),
        "chromatic_dy_b": (-disp_y).astype(np.float32, copy=False),
    }


def create_composite_mask(
    vignette_config: VignetteConfig,
    aberration_config: AberrationConfig,
    rng: np.random.Generator,
) -> dict:
    """Create a sensor-space lens feature bundle for one group."""

    w, h = vignette_config.width, vignette_config.height
    vignette, vignette_gt, vignette_meta = create_diverse_vignette(vignette_config, rng)
    astigmatism_angle = float(rng.uniform(0.0, 180.0))
    astigmatism = generate_astigmatism(
        (h, w),
        strength=aberration_config.astigmatism_strength,
        frequency=aberration_config.astigmatism_frequency,
        angle=astigmatism_angle,
    )
    chromatic = generate_chromatic_displacement(
        (h, w),
        strength_px=aberration_config.chromatic_strength_px,
        frequency=aberration_config.chromatic_frequency,
        tangential_ratio=aberration_config.chromatic_tangential_ratio,
    )

    return {
        "vignette": vignette,
        "vignette_gt": vignette_gt,
        "astigmatism": astigmatism,
        **chromatic,
        "metadata": {
            **vignette_meta,
            "astigmatism_angle_deg": astigmatism_angle,
            "astigmatism_strength": float(aberration_config.astigmatism_strength),
            "chromatic_strength_px": float(aberration_config.chromatic_strength_px),
        },
    }


def rotate_mask(mask: dict, angle: float) -> dict:
    """Rotate all array-valued mask planes by the same angle."""

    result = {}
    for key, value in mask.items():
        if isinstance(value, np.ndarray):
            h, w = value.shape[:2]
            center = (float((w - 1) * 0.5), float((h - 1) * 0.5))
            matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            result[key] = cv2.warpAffine(
                value,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ).astype(np.float32, copy=False)
        else:
            result[key] = value
    return result
