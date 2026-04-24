"""Command line entry point for the vignette/aberration synthesizer."""

from __future__ import annotations

import argparse

from .synthesizer import run_synthesis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark-compatible vignette/aberration datasets."
    )
    parser.add_argument("--num-groups", type=int, default=400)
    parser.add_argument("--output-dir", type=str, default="vignette_aberration_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=4000)
    parser.add_argument("--crop-width", type=int, default=3000)
    parser.add_argument("--crop-height", type=int, default=2000)
    parser.add_argument("--focal-px", type=float, default=2500.0)
    parser.add_argument(
        "--curve-drift",
        type=float,
        default=0.10,
        help="Fixed polynomial curve max drift, e.g. 0.10 for 10 percent.",
    )
    parser.add_argument(
        "--max-angle-jitter-deg",
        type=float,
        default=7.0,
        help="Maximum non-negative per-capture angle jitter in degrees.",
    )
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing completed groups and regenerate from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_synthesis(
        num_groups=args.num_groups,
        output_dir=args.output_dir,
        seed=args.seed,
        visualize=not args.skip_visualize,
        width=args.width,
        height=args.height,
        crop_width=args.crop_width,
        crop_height=args.crop_height,
        focal_px=args.focal_px,
        curve_drift=args.curve_drift,
        max_angle_jitter_deg=args.max_angle_jitter_deg,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
