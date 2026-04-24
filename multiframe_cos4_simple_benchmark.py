import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

import benchmark_real_rotated_sets as brs
from generate_gt_vignetting_maps import estimate_center


def fit_cos4(image_norm: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    cx_est, cy_est = estimate_center(image_norm)
    h, w = image_norm.shape
    half_diag_px = float(np.hypot((w - 1) * 0.5, (h - 1) * 0.5))
    yy, xx = np.indices(image_norm.shape, dtype=np.float32)
    r_norm = np.sqrt(
        ((xx - np.float32(cx_est)) / np.float32(half_diag_px)) ** 2
        + ((yy - np.float32(cy_est)) / np.float32(half_diag_px)) ** 2,
        dtype=np.float32,
    )

    f_grid = np.geomspace(0.15, 8.0, 4000).astype(np.float32)
    best = None
    best_pred = None
    for f_norm in f_grid:
        base = 1.0 / (1.0 + (r_norm / np.float32(f_norm)) ** 2) ** 2
        denom = float(np.sum(base * base))
        i0 = float(np.sum(image_norm * base) / max(denom, 1e-12))
        pred = np.clip(np.float32(i0) * base, 0.0, 1.0).astype(np.float32, copy=False)
        mse_img = float(np.mean((pred - image_norm) ** 2))
        if best is None or mse_img < best["fit_to_image_mse"]:
            best = {
                "f_norm": float(f_norm),
                "i0": i0,
                "fit_to_image_mse": mse_img,
            }
            best_pred = pred.copy()

    runtime_sec = float(time.perf_counter() - t0)
    metrics = brs.compute_metrics(gt, best_pred)
    summary = {
        "model": "cos4_simple_multiframe",
        "runtime_sec": runtime_sec,
        "shape": [int(h), int(w)],
        "v_metrics": metrics,
        "fit_to_image_mse": best["fit_to_image_mse"],
        "learned": {
            "center_px": [float(cx_est), float(cy_est)],
            "f_norm": best["f_norm"],
            "f_mm_equiv": float(best["f_norm"] * brs.SENSOR_DIAG_MM * 0.5),
            "i0": best["i0"],
        },
    }
    return best_pred, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Equalize a four-frame case by simple scalar brightness and fit a fused cos4 model.")
    parser.add_argument("--case-dir", type=Path, required=True, help="Prepared dataset directory.")
    parser.add_argument("--benchmark-summary", type=Path, default=Path("real_benchmark_outputs") / "benchmark_summary.csv")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--stat",
        type=str,
        default="p99.8",
        choices=["p99.8", "p95"],
        help="Per-frame scalar statistic used for brightness equalization.",
    )
    parser.add_argument(
        "--fuse",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="How to fuse the equalized frames before cos4 fitting.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="unity",
        choices=["unity", "median"],
        help="Brightness target after per-frame equalization: unity rescales each frame close to 1, median preserves the median raw scale.",
    )
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.case_dir / "manifest.json").read_text(encoding="utf-8"))
    gt = np.load(args.case_dir / "ground_truth_vignetting.npy").astype(np.float32, copy=False)

    captures = []
    stats = []
    for item in manifest["capture_files_sorted_to_angles"]:
        angle = int(item["angle"])
        capture = np.load(args.case_dir / f"sim_capture_{angle}.npy").astype(np.float32, copy=False)
        captures.append(capture)
        if args.stat == "p95":
            stats.append(float(np.percentile(capture, 95.0)))
        else:
            stats.append(float(np.percentile(capture, 99.8)))

    if args.target == "unity":
        target = 1.0
    else:
        target = float(np.median(np.asarray(stats, dtype=np.float32)))
    scales = [target / max(v, 1e-6) for v in stats]
    equalized = [np.clip(c * np.float32(s), 0.0, 1.0).astype(np.float32, copy=False) for c, s in zip(captures, scales)]
    stack = np.stack(equalized, axis=0).astype(np.float32, copy=False)
    if args.fuse == "median":
        fused = np.median(stack, axis=0).astype(np.float32, copy=False)
    else:
        fused = np.mean(stack, axis=0).astype(np.float32, copy=False)
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32, copy=False)

    pred, summary = fit_cos4(fused, gt)
    summary["equalization"] = {
        "stat": args.stat,
        "target": float(target),
        "per_frame_stats": stats,
        "scales": scales,
    }
    summary["fusion"] = args.fuse
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    brs.save_model_outputs(output_root, gt, pred)
    np.save(output_root / "fused_equalized_input.npy", fused)
    brs.save_png16(output_root / "fused_equalized_input.png", fused)
    brs.save_preview_gray(output_root / "fused_equalized_input_preview.png", fused)

    aux = {
        "case_id": manifest["case_id"],
        "display_name": manifest["display_name"],
        "model": summary["model"],
        "runtime_sec": summary["runtime_sec"],
        "v_fit_percent": summary["v_metrics"]["fit_percent"],
        "v_r2": summary["v_metrics"]["r2"],
        "v_rmse": summary["v_metrics"]["rmse"],
        "v_mae": summary["v_metrics"]["mae"],
        "equalization_stat": args.stat,
        "equalization_target": args.target,
        "fusion": args.fuse,
    }
    with (output_root / "summary_row.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aux.keys()))
        writer.writeheader()
        writer.writerow(aux)

    group_rows = []
    with args.benchmark_summary.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == manifest["case_id"]:
                group_rows.append(row)
    comparison = {
        "case_id": manifest["case_id"],
        "display_name": manifest["display_name"],
        "multiframe_cos4": aux,
        "group_benchmark_reference": group_rows,
    }
    (output_root / "comparison_against_group.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    gt_bgr = cv2.cvtColor(brs.normalize_for_preview(gt), cv2.COLOR_GRAY2BGR)
    fused_bgr = cv2.cvtColor(brs.normalize_for_preview(fused), cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(brs.normalize_for_preview(pred), cv2.COLOR_GRAY2BGR)
    grid = np.concatenate([gt_bgr, fused_bgr, pred_bgr], axis=1)
    (output_root / "preview_triptych.png").write_bytes(cv2.imencode(".png", grid)[1].tobytes())


if __name__ == "__main__":
    main()
