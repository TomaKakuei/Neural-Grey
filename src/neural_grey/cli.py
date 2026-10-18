"""Fit a calibration, resume fitting, or correct later images."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from .training import FitConfig, FitSession
from .calibration import Calibration
from .data import load_captures, synthetic_case


def export(session, output):
    calibration = session.calibration()
    calibration.save(output / "calibration.npz")
    session.save(output / "training.pt")
    (output / "fit.json").write_text(
        json.dumps(
            dict(**calibration.metadata, history=session.history),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    preview = np.clip(calibration.shading * 255.0, 0.0, 255.0).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", preview)
    if not ok:
        raise RuntimeError("Could not encode shading preview")
    (output / "shading.png").write_bytes(encoded.tobytes())
    print(
        json.dumps(
            dict(
                status="complete" if session.complete else "paused",
                epochs=session.completed_epochs,
                output=str(output / "calibration.npz"),
            )
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="neural-grey", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "demo"):
        fit = commands.add_parser(name)
        if name == "fit":
            fit.add_argument(
                "captures",
                type=Path,
                help="(4,H,W) NPY, NPZ captures array, or four-NPY directory",
            )
        fit.add_argument("--output", type=Path, required=True)
        fit.add_argument("--focal-mm", type=float, default=None)
        fit.add_argument("--sensor-diag-mm", type=float, default=43.3)
        fit.add_argument("--epochs", type=int, default=200 if name == "fit" else 2)
        fit.add_argument(
            "--sample-count", type=int, default=240000 if name == "fit" else 1024
        )
        fit.add_argument(
            "--batch-size", type=int, default=8192 if name == "fit" else 256
        )
        fit.add_argument("--seed", type=int, default=42)
        fit.add_argument("--device", default="auto")
        fit.add_argument("--threads", type=int, default=4)
        fit.add_argument(
            "--stop-after",
            type=int,
            help="Pause after this many epochs; retain the full configured schedule",
        )
    resume = commands.add_parser("resume")
    resume.add_argument("checkpoint", type=Path)
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--device", default="auto")
    resume.add_argument("--threads", type=int, default=4)
    resume.add_argument("--stop-after", type=int)
    correction = commands.add_parser("correct")
    correction.add_argument("calibration", type=Path)
    correction.add_argument("image", type=Path, help="Linear grayscale or RGB NPY")
    correction.add_argument("--output", required=True, type=Path)
    correction.add_argument("--minimum-transmittance", type=float, default=0.05)
    raw = commands.add_parser("prepare-raw")
    raw.add_argument(
        "files",
        nargs=4,
        type=Path,
        help="Bayer RAW files in 0,90,180,270 degree capture order",
    )
    raw.add_argument("--output", required=True, type=Path)
    raw.add_argument("--long-edge", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        if args.command == "correct":
            if args.output.resolve() in (
                args.image.resolve(),
                args.calibration.resolve(),
            ):
                raise ValueError("Output must not overwrite an input")
            image = np.load(args.image, allow_pickle=False)
            corrected = Calibration.load(args.calibration).correct(
                image, minimum_transmittance=args.minimum_transmittance
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("wb") as stream:
                np.save(stream, corrected, allow_pickle=False)
            print(str(args.output))
            return
        if args.command == "prepare-raw":
            from .raw import load_linear_bayer_luminance, resize_long_edge

            if args.long_edge < 8:
                raise ValueError("--long-edge must be at least 8")
            if args.output.exists():
                raise FileExistsError("Choose a new capture output path")
            captures = np.stack(
                [
                    resize_long_edge(load_linear_bayer_luminance(p), args.long_edge)
                    for p in args.files
                ]
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("wb") as stream:
                np.savez_compressed(stream, captures=captures)
            print(str(args.output))
            return
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(1)
        args.output.mkdir(parents=True, exist_ok=True)
        if args.command == "resume":
            if (args.output / "training.pt").exists() and (
                args.output / "training.pt"
            ).resolve() != args.checkpoint.resolve():
                raise FileExistsError(
                    "Output belongs to a different run; choose an empty directory"
                )
            session = FitSession.resume(args.checkpoint, device=args.device)
        else:
            if any(
                (args.output / name).exists()
                for name in ("training.pt", "calibration.npz", "fit.json")
            ):
                raise FileExistsError(
                    "Output already contains a fit; use resume or choose a new directory"
                )
            if args.command == "demo":
                captures, truth = synthetic_case(seed=args.seed)
                np.savez_compressed(
                    args.output / "example.npz", captures=captures, ground_truth=truth
                )
                if args.focal_mm is None:
                    args.focal_mm = 28.0
            else:
                captures = load_captures(args.captures)
            config = FitConfig(
                epochs=args.epochs,
                sample_count=args.sample_count,
                batch_size=args.batch_size,
            )
            session = FitSession(
                captures,
                focal_length_mm=args.focal_mm,
                sensor_diagonal_mm=args.sensor_diag_mm,
                config=config,
                seed=args.seed,
                device=args.device,
            )
        session.run(stop_after=args.stop_after, checkpoint=args.output / "training.pt")
        export(session, args.output)
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"neural-grey: {exc}\n")
