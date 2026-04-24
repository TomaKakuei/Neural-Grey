"""
Realistic synthetic dataset generator for vignetting research.

Compared with the ideal generator, this version intentionally breaks several
assumptions:
1) Vignetting is not only clean circular cos^4 (adds azimuthal and local terms)
2) Lighting is not a simple linear gradient
3) Capture rotation is imperfect (angle/translation/scale jitter)
4) Sensor artifacts are present (PRNU, row noise, shot + read noise, black drift)

Outputs remain compatible with your current pipeline:
- ground_truth_vignetting.npy
- sim_capture_0.npy / 90 / 180 / 270
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RealisticConfig:
    width: int = 4000
    height: int = 4000
    seed: int = 42
    output_dir: str = "synthetic_realistic_output"
    generate_plot: bool = True
    preset: str = "hard"  # mild | hard | extreme

    # Base physical vignetting
    focal_px: float = 2500.0
    center_shift_x: float = 170.0
    center_shift_y: float = -220.0
    x_elliptical_scale: float = 1.08

    # Non-ideal vignetting terms
    azimuthal_amp: float = 0.08
    local_blob_count: int = 5
    local_blob_amp: float = 0.06
    local_blob_sigma_px: float = 450.0

    # Illumination bias complexity
    lighting_min: float = 0.50
    lighting_max: float = 1.20
    lighting_texture_amp: float = 0.08
    lighting_hotspot_amp: float = 0.22

    # Imperfect rotation + capture drift
    rotation_jitter_deg_std: float = 1.2
    shift_jitter_px_std: float = 8.0
    scale_jitter_std: float = 0.008
    exposure_jitter_std: float = 0.03

    # Sensor / RAW-like effects
    full_well_electrons: float = 8000.0
    read_noise_sigma: float = 0.008
    row_noise_sigma: float = 0.003
    prnu_sigma: float = 0.010
    black_level_drift_sigma: float = 0.004


def apply_preset(config: RealisticConfig) -> RealisticConfig:
    preset = config.preset.lower()
    if preset == "mild":
        config.azimuthal_amp = 0.05
        config.local_blob_amp = 0.035
        config.rotation_jitter_deg_std = 0.6
        config.shift_jitter_px_std = 4.0
        config.lighting_texture_amp = 0.04
        config.lighting_hotspot_amp = 0.14
        config.read_noise_sigma = 0.005
        config.row_noise_sigma = 0.002
    elif preset == "hard":
        pass
    elif preset == "extreme":
        config.azimuthal_amp = 0.12
        config.local_blob_amp = 0.09
        config.local_blob_count = 8
        config.rotation_jitter_deg_std = 2.2
        config.shift_jitter_px_std = 14.0
        config.scale_jitter_std = 0.014
        config.exposure_jitter_std = 0.05
        config.lighting_texture_amp = 0.12
        config.lighting_hotspot_amp = 0.30
        config.read_noise_sigma = 0.012
        config.row_noise_sigma = 0.005
        config.prnu_sigma = 0.014
    else:
        raise ValueError(f"Unknown preset: {config.preset}. Use mild|hard|extreme.")
    return config


def generate_ground_truth_vignetting(config: RealisticConfig, rng: np.random.Generator) -> np.ndarray:
    h, w = config.height, config.width
    yy, xx = np.indices((h, w), dtype=np.float32)

    cx = np.float32(w * 0.5 + config.center_shift_x)
    cy = np.float32(h * 0.5 + config.center_shift_y)
    dx = (xx - cx) * np.float32(config.x_elliptical_scale)
    dy = yy - cy

    r = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    theta = np.arctan2(dy, dx)
    base = np.cos(np.arctan(r / np.float32(config.focal_px)), dtype=np.float32) ** np.float32(4.0)

    phi1 = np.float32(rng.uniform(-np.pi, np.pi))
    phi2 = np.float32(rng.uniform(-np.pi, np.pi))
    azi = (
        1.0
        + np.float32(config.azimuthal_amp) * np.cos(2.0 * (theta - phi1), dtype=np.float32)
        + np.float32(0.55 * config.azimuthal_amp) * np.sin(3.0 * (theta - phi2), dtype=np.float32)
    )

    local = np.ones_like(base, dtype=np.float32)
    for _ in range(int(config.local_blob_count)):
        bx = np.float32(rng.uniform(0, w - 1))
        by = np.float32(rng.uniform(0, h - 1))
        amp = np.float32(rng.uniform(-config.local_blob_amp, config.local_blob_amp))
        sig = np.float32(rng.uniform(0.6, 1.4) * config.local_blob_sigma_px)
        d2 = (xx - bx) ** 2 + (yy - by) ** 2
        local *= 1.0 + amp * np.exp(-0.5 * d2 / (sig * sig), dtype=np.float32)

    v = base * azi * local
    v = np.clip(v, 0.02, 1.0).astype(np.float32, copy=False)
    v /= np.float32(max(float(v.max()), 1e-8))
    return v


def generate_realistic_lighting(config: RealisticConfig, rng: np.random.Generator) -> np.ndarray:
    h, w = config.height, config.width
    yy, xx = np.indices((h, w), dtype=np.float32)
    xn = (xx - np.float32((w - 1) * 0.5)) / np.float32(max((w - 1) * 0.5, 1.0))
    yn = (yy - np.float32((h - 1) * 0.5)) / np.float32(max((h - 1) * 0.5, 1.0))

    # Slanted gradient + low-order bowl distortion.
    ang = np.float32(rng.uniform(-np.pi, np.pi))
    grad = 0.22 * (np.cos(ang, dtype=np.float32) * xn + np.sin(ang, dtype=np.float32) * yn)
    bowl = 0.10 * (0.7 * xn * xn + 1.2 * yn * yn + 0.5 * xn * yn)

    # Two random hotspots/shadows.
    lighting = np.float32(0.92) + grad + bowl
    for _ in range(2):
        hx = np.float32(rng.uniform(-0.75, 0.75))
        hy = np.float32(rng.uniform(-0.75, 0.75))
        hs = np.float32(rng.uniform(0.18, 0.42))
        ha = np.float32(rng.uniform(-0.7, 1.0) * config.lighting_hotspot_amp)
        d2 = (xn - hx) ** 2 + (yn - hy) ** 2
        lighting += ha * np.exp(-0.5 * d2 / (hs * hs), dtype=np.float32)

    # Low-frequency texture (wall unevenness).
    tex = np.zeros_like(lighting, dtype=np.float32)
    for _ in range(5):
        fx = np.float32(rng.uniform(0.4, 2.2))
        fy = np.float32(rng.uniform(0.4, 2.2))
        ph = np.float32(rng.uniform(-np.pi, np.pi))
        coeff = np.float32(rng.uniform(-1.0, 1.0))
        tex += coeff * np.sin(2.0 * np.pi * (fx * xn + fy * yn) + ph, dtype=np.float32)
    lighting += np.float32(config.lighting_texture_amp / 5.0) * tex

    # Normalize to requested range.
    l_min = float(lighting.min())
    l_max = float(lighting.max())
    lighting = (lighting - np.float32(l_min)) / np.float32(max(l_max - l_min, 1e-8))
    target_span = np.float32(config.lighting_max - config.lighting_min)
    lighting = np.float32(config.lighting_min) + target_span * lighting
    return np.clip(lighting, config.lighting_min, config.lighting_max).astype(np.float32, copy=False)


def warp_lighting_nonideal(
    lighting: np.ndarray,
    base_angle_deg: float,
    config: RealisticConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    h, w = lighting.shape
    angle = float(base_angle_deg + rng.normal(0.0, config.rotation_jitter_deg_std))
    shift_x = float(rng.normal(0.0, config.shift_jitter_px_std))
    shift_y = float(rng.normal(0.0, config.shift_jitter_px_std))
    scale = float(1.0 + rng.normal(0.0, config.scale_jitter_std))

    center = (float((w - 1) * 0.5), float((h - 1) * 0.5))
    m = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    m[0, 2] += np.float32(shift_x)
    m[1, 2] += np.float32(shift_y)

    warped = cv2.warpAffine(
        lighting,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    info = {
        "effective_angle_deg": angle,
        "shift_x_px": shift_x,
        "shift_y_px": shift_y,
        "scale": scale,
    }
    return warped.astype(np.float32, copy=False), info


def simulate_nonideal_rotated_captures(
    vignette: np.ndarray,
    lighting: np.ndarray,
    config: RealisticConfig,
    rng: np.random.Generator,
) -> tuple[dict[int, np.ndarray], dict]:
    h, w = vignette.shape
    captures: dict[int, np.ndarray] = {}
    metadata = {"captures": {}}

    # Sensor fixed-pattern gain map (same for all captures).
    prnu = (1.0 + rng.normal(0.0, config.prnu_sigma, size=(h, w))).astype(np.float32, copy=False)
    prnu = np.clip(prnu, 0.9, 1.1)

    for k, nominal in enumerate((0, 90, 180, 270)):
        l_warp, warp_info = warp_lighting_nonideal(lighting, float(nominal), config, rng)
        exposure = float(1.0 + rng.normal(0.0, config.exposure_jitter_std))
        exposure = float(np.clip(exposure, 0.85, 1.15))

        ideal = np.clip(vignette * l_warp * np.float32(exposure), 0.0, 1.25)
        signal = ideal * prnu

        # Shot noise using Poisson approximation in electron domain.
        lam = np.clip(signal * np.float32(config.full_well_electrons), 0.0, None)
        shot = (rng.poisson(lam).astype(np.float32) - lam) / np.float32(config.full_well_electrons)

        read = rng.normal(0.0, config.read_noise_sigma, size=(h, w)).astype(np.float32, copy=False)
        row = rng.normal(0.0, config.row_noise_sigma, size=(h, 1)).astype(np.float32, copy=False)
        black = np.float32(rng.normal(0.0, config.black_level_drift_sigma))

        img = signal + shot + read + row + black
        captures[nominal] = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

        metadata["captures"][str(nominal)] = {
            "nominal_angle_deg": float(nominal),
            "exposure_scale": exposure,
            "black_level_drift": float(black),
            **warp_info,
        }

    return captures, metadata


def save_dataset(
    vignette: np.ndarray,
    lighting: np.ndarray,
    captures: dict[int, np.ndarray],
    metadata: dict,
    config: RealisticConfig,
) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    np.save(os.path.join(config.output_dir, "ground_truth_vignetting.npy"), vignette)
    np.save(os.path.join(config.output_dir, "canonical_lighting.npy"), lighting)
    for angle in (0, 90, 180, 270):
        np.save(os.path.join(config.output_dir, f"sim_capture_{angle}.npy"), captures[angle])

    full_meta = {
        "config": asdict(config),
        "notes": (
            "Dataset intentionally breaks ideal 4-rotation assumptions: "
            "non-pure-cos4 V, imperfect rotation, exposure drift, and sensor artifacts."
        ),
        **metadata,
    }
    with open(os.path.join(config.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)


def plot_overview(
    vignette: np.ndarray,
    lighting: np.ndarray,
    captures: dict[int, np.ndarray],
    config: RealisticConfig,
) -> str:
    out_png = os.path.join(config.output_dir, "realistic_overview.png")
    fig, axes = plt.subplots(1, 6, figsize=(30, 5), constrained_layout=True)
    items = [
        ("GT V", vignette),
        ("Canonical L", lighting),
        ("Capture 0", captures[0]),
        ("Capture 90", captures[90]),
        ("Capture 180", captures[180]),
        ("Capture 270", captures[270]),
    ]
    for ax, (title, img) in zip(axes, items):
        im = ax.imshow(img, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Realistic Vignetting Synthetic Set ({config.preset})", fontsize=13)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def print_summary(v: np.ndarray, l: np.ndarray, captures: dict[int, np.ndarray], config: RealisticConfig) -> None:
    print("=== Realistic Synthetic Dataset Generated ===")
    print(f"Output directory: {config.output_dir}")
    print(f"Preset: {config.preset}")
    print(f"Resolution: {config.width} x {config.height}")
    print(f"GT V: min={float(v.min()):.4f}, max={float(v.max()):.4f}, mean={float(v.mean()):.4f}")
    print(f"L:    min={float(l.min()):.4f}, max={float(l.max()):.4f}, mean={float(l.mean()):.4f}")
    for a in (0, 90, 180, 270):
        img = captures[a]
        print(f"Img {a:>3}: min={float(img.min()):.4f}, max={float(img.max()):.4f}, mean={float(img.mean()):.4f}")
    print("Saved files:")
    print(" - ground_truth_vignetting.npy")
    print(" - canonical_lighting.npy")
    print(" - sim_capture_0.npy / 90 / 180 / 270")
    print(" - metadata.json")
    if config.generate_plot:
        print(" - realistic_overview.png")


def parse_args() -> RealisticConfig:
    parser = argparse.ArgumentParser(description="Generate non-ideal realistic 4-rotation synthetic dataset.")
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="synthetic_realistic_output")
    parser.add_argument("--preset", type=str, default="hard", choices=("mild", "hard", "extreme"))
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--focal-px", type=float, default=2500.0)
    args = parser.parse_args()

    cfg = RealisticConfig(
        width=args.width,
        height=args.height,
        seed=args.seed,
        output_dir=args.output_dir,
        preset=args.preset,
        generate_plot=not args.skip_plot,
        focal_px=args.focal_px,
    )
    return apply_preset(cfg)


def main() -> None:
    config = parse_args()
    rng = np.random.default_rng(config.seed)

    vignette = generate_ground_truth_vignetting(config, rng)
    lighting = generate_realistic_lighting(config, rng)
    captures, metadata = simulate_nonideal_rotated_captures(vignette, lighting, config, rng)

    save_dataset(vignette, lighting, captures, metadata, config)
    print_summary(vignette, lighting, captures, config)

    if config.generate_plot:
        fig_path = plot_overview(vignette, lighting, captures, config)
        print(f"Overview figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
