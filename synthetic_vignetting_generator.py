"""
Synthetic RAW-like vignetting dataset generator for "Neural Grey".

This script creates:
1) Absolute ground-truth asymmetric vignetting V(x, y)
2) Uneven wall-lighting bias L(x, y)
3) Four simulated captures with rotated lighting and independent Gaussian noise:
   Img_0, Img_90, Img_180, Img_270

Outputs are saved as .npy arrays and a 1x5 visualization PNG.
"""

import argparse
import os
from dataclasses import dataclass

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SimulationConfig:
    width: int = 4000
    height: int = 4000
    focal_px: float = 2500.0
    center_shift_x: float = 150.0
    center_shift_y: float = -200.0
    x_elliptical_scale: float = 1.05
    light_max: float = 1.0
    light_min: float = 0.6
    noise_sigma: float = 0.03
    seed: int = 42
    output_dir: str = "synthetic_output"
    generate_plot: bool = True


def generate_asymmetric_vignetting(config: SimulationConfig) -> np.ndarray:
    """
    Generate asymmetric cosine-fourth-law vignetting:
      V(x,y) = cos^4( arctan( r_distorted / f ) )
    where r_distorted uses a decentered optical center and X-axis elliptical scaling.
    """
    h, w = config.height, config.width
    yy, xx = np.indices((h, w), dtype=np.float32)

    cx = (w / 2.0) + config.center_shift_x
    cy = (h / 2.0) + config.center_shift_y

    dx = (xx - cx) * np.float32(config.x_elliptical_scale)
    dy = (yy - cy)
    r_distorted = np.sqrt(dx * dx + dy * dy, dtype=np.float32)

    theta = np.arctan(r_distorted / np.float32(config.focal_px))
    vignette = np.cos(theta, dtype=np.float32) ** np.float32(4.0)

    return np.clip(vignette, 0.0, 1.0).astype(np.float32, copy=False)


def generate_diagonal_lighting_bias(config: SimulationConfig) -> np.ndarray:
    """
    Generate a diagonal gradient lighting bias:
      top-left ~ light_max, bottom-right ~ light_min
    """
    h, w = config.height, config.width
    yy, xx = np.indices((h, w), dtype=np.float32)

    denom = np.float32(max((w - 1) + (h - 1), 1))
    diagonal_norm = (xx + yy) / denom

    light_span = np.float32(config.light_max - config.light_min)
    lighting = np.float32(config.light_max) - light_span * diagonal_norm

    return np.clip(lighting, config.light_min, config.light_max).astype(np.float32, copy=False)


def simulate_rotated_captures(
    vignette: np.ndarray,
    lighting_bias: np.ndarray,
    config: SimulationConfig,
) -> dict[int, np.ndarray]:
    """
    Create 4 captures with rotated lighting and independent Gaussian noise.
    Lighting rotates; vignetting remains fixed in sensor coordinates.
    """
    rng = np.random.default_rng(config.seed)
    captures: dict[int, np.ndarray] = {}

    for k, angle in enumerate((0, 90, 180, 270)):
        lighting_rot = np.rot90(lighting_bias, k=k)
        noise = rng.normal(
            loc=0.0,
            scale=config.noise_sigma,
            size=vignette.shape,
        ).astype(np.float32, copy=False)

        img = vignette * lighting_rot + noise
        captures[angle] = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

    return captures


def save_outputs(vignette: np.ndarray, captures: dict[int, np.ndarray], config: SimulationConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    gt_path = os.path.join(config.output_dir, "ground_truth_vignetting.npy")
    np.save(gt_path, vignette)

    for angle, img in captures.items():
        out_path = os.path.join(config.output_dir, f"sim_capture_{angle}.npy")
        np.save(out_path, img)


def plot_overview(vignette: np.ndarray, captures: dict[int, np.ndarray], config: SimulationConfig) -> str:
    os.makedirs(config.output_dir, exist_ok=True)
    out_png = os.path.join(config.output_dir, "simulation_overview.png")

    figure_items = [
        ("Ground Truth V", vignette),
        ("Capture 0°", captures[0]),
        ("Capture 90°", captures[90]),
        ("Capture 180°", captures[180]),
        ("Capture 270°", captures[270]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
    for ax, (title, img) in zip(axes, figure_items):
        im = ax.imshow(img, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("Neural Grey Synthetic Dataset: Ground Truth + 4 Rotated Noisy Captures", fontsize=14)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def print_summary(vignette: np.ndarray, captures: dict[int, np.ndarray], config: SimulationConfig) -> None:
    print("=== Synthetic Dataset Generated ===")
    print(f"Output directory: {config.output_dir}")
    print(f"Resolution: {config.width} x {config.height}")
    print(
        "Ground Truth V stats: "
        f"min={float(vignette.min()):.4f}, max={float(vignette.max()):.4f}, mean={float(vignette.mean()):.4f}"
    )
    for angle in (0, 90, 180, 270):
        img = captures[angle]
        print(
            f"Capture {angle:>3}° stats: "
            f"min={float(img.min()):.4f}, max={float(img.max()):.4f}, mean={float(img.mean()):.4f}"
        )
    print("Saved files:")
    print(" - ground_truth_vignetting.npy")
    print(" - sim_capture_0.npy")
    print(" - sim_capture_90.npy")
    print(" - sim_capture_180.npy")
    print(" - sim_capture_270.npy")
    if config.generate_plot:
        print(" - simulation_overview.png")


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description="Generate synthetic asymmetric vignetting data with 4 rotated lighting captures."
    )
    parser.add_argument("--width", type=int, default=4000, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=4000, help="Sensor height in pixels.")
    parser.add_argument("--focal-px", type=float, default=2500.0, help="Simulated focal length in pixels.")
    parser.add_argument("--center-shift-x", type=float, default=150.0, help="Optical center X shift in pixels.")
    parser.add_argument("--center-shift-y", type=float, default=-200.0, help="Optical center Y shift in pixels.")
    parser.add_argument("--x-elliptical-scale", type=float, default=1.05, help="Scale factor for X distance in r.")
    parser.add_argument("--light-max", type=float, default=1.0, help="Maximum lighting bias value.")
    parser.add_argument("--light-min", type=float, default=0.6, help="Minimum lighting bias value.")
    parser.add_argument("--noise-sigma", type=float, default=0.03, help="Stddev of Gaussian read noise.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible noise.")
    parser.add_argument("--output-dir", type=str, default="synthetic_output", help="Output directory.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip matplotlib 1x5 overview figure generation.")

    args = parser.parse_args()
    return SimulationConfig(
        width=args.width,
        height=args.height,
        focal_px=args.focal_px,
        center_shift_x=args.center_shift_x,
        center_shift_y=args.center_shift_y,
        x_elliptical_scale=args.x_elliptical_scale,
        light_max=args.light_max,
        light_min=args.light_min,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
        output_dir=args.output_dir,
        generate_plot=not args.skip_plot,
    )


def main() -> None:
    config = parse_args()
    vignette = generate_asymmetric_vignetting(config)
    lighting = generate_diagonal_lighting_bias(config)
    captures = simulate_rotated_captures(vignette, lighting, config)
    save_outputs(vignette, captures, config)
    print_summary(vignette, captures, config)
    if config.generate_plot:
        png_path = plot_overview(vignette, captures, config)
        print(f"Overview figure saved to: {png_path}")
    else:
        print("Overview figure generation skipped (--skip-plot).")


if __name__ == "__main__":
    main()
