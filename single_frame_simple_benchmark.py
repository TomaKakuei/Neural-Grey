import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

import benchmark_real_rotated_sets as brs
import TriditionalMask as geo_v0
from generate_gt_vignetting_maps import estimate_center


def load_case(case_dir: Path) -> tuple[dict, np.ndarray]:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    gt = np.load(case_dir / "ground_truth_vignetting.npy").astype(np.float32, copy=False)
    return manifest, gt


def simple_normalize(image: np.ndarray) -> tuple[np.ndarray, dict]:
    scale_998 = float(max(np.percentile(image, 99.8), 1e-6))
    normalized = np.clip(image / np.float32(scale_998), 0.0, 1.0).astype(np.float32, copy=False)
    return normalized, {"type": "single_scalar_divide_by_p99.8", "scale_99p8": scale_998}


def run_cos4_simple(image_norm: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, dict]:
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
        "model": "cos4_simple_single",
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


def run_geometric_v0(image_norm: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    baseline = geo_v0.TraditionalPolynomialBaseline(image_norm, ground_truth=gt)
    baseline.find_optical_center()
    baseline.extract_radial_profile()
    baseline.fit_model()
    baseline.generate_2d_prediction()
    runtime_sec = float(time.perf_counter() - t0)
    pred = baseline.predicted_surface.astype(np.float32, copy=False)
    metrics = brs.compute_metrics(gt, pred)
    summary = {
        "model": "geometricV0_single",
        "runtime_sec": runtime_sec,
        "shape": [int(pred.shape[0]), int(pred.shape[1])],
        "v_metrics": metrics,
        "learned": {
            "optical_center_px": [int(baseline.optical_center[0]), int(baseline.optical_center[1])],
            "geometry": baseline.geometry,
            "poly_coeffs": [float(v) for v in baseline.poly_coeffs.tolist()],
        },
    }
    return pred, summary


def save_frame_visuals(frame_dir: Path, gt: np.ndarray, input_norm: np.ndarray, cos4_pred: np.ndarray, geo0_pred: np.ndarray) -> None:
    brs.save_preview_gray(frame_dir / "input_normalized_preview.png", input_norm)
    brs.save_png16(frame_dir / "input_normalized.png", input_norm)
    brs.save_preview_gray(frame_dir / "gt_preview.png", gt)

    labels_and_images = [
        ("GT", cv2.cvtColor(brs.normalize_for_preview(gt), cv2.COLOR_GRAY2BGR)),
        ("Input norm", cv2.cvtColor(brs.normalize_for_preview(input_norm), cv2.COLOR_GRAY2BGR)),
        ("Cos4 simple", cv2.cvtColor(brs.normalize_for_preview(cos4_pred), cv2.COLOR_GRAY2BGR)),
        ("Geometric V0", cv2.cvtColor(brs.normalize_for_preview(geo0_pred), cv2.COLOR_GRAY2BGR)),
    ]

    pad = 24
    label_h = 34
    cell_h = max(img.shape[0] for _, img in labels_and_images)
    cell_w = max(img.shape[1] for _, img in labels_and_images)
    canvas = np.full((pad * 2 + len(labels_and_images) * (cell_h + label_h + pad), cell_w + pad * 2, 3), 255, dtype=np.uint8)

    y = pad
    for label, img in labels_and_images:
        cv2.putText(canvas, label, (pad, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        canvas[y + label_h:y + label_h + img.shape[0], pad:pad + img.shape[1]] = img
        cv2.rectangle(canvas, (pad, y + label_h), (pad + img.shape[1], y + label_h + img.shape[0]), (200, 200, 200), 2)
        y += cell_h + label_h + pad

    (frame_dir / "preview_grid.png").write_bytes(cv2.imencode(".png", canvas)[1].tobytes())


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "frame_name",
        "angle",
        "source_file",
        "model",
        "runtime_sec",
        "v_fit_percent",
        "v_r2",
        "v_rmse",
        "v_mae",
        "normalization_type",
        "normalization_scale_99p8",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple single-frame cos4 and geometricV0 benchmarks for a prepared real case.")
    parser.add_argument("--case-dir", type=Path, required=True, help="Prepared dataset directory containing manifest.json and sim_capture_*.npy.")
    parser.add_argument("--benchmark-summary", type=Path, default=Path("real_benchmark_outputs") / "benchmark_summary.csv")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest, gt = load_case(args.case_dir)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    comparisons = []
    frame_summary = []
    for item in manifest["capture_files_sorted_to_angles"]:
        angle = int(item["angle"])
        source_file = str(item["source_file"])
        frame_name = Path(source_file).stem
        frame_dir = output_root / frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)

        capture = np.load(args.case_dir / f"sim_capture_{angle}.npy").astype(np.float32, copy=False)
        image_norm, norm_meta = simple_normalize(capture)
        (frame_dir / "normalization.json").write_text(json.dumps(norm_meta, indent=2), encoding="utf-8")

        cos4_pred, cos4_summary = run_cos4_simple(image_norm, gt)
        geo0_pred, geo0_summary = run_geometric_v0(image_norm, gt)

        for model_name, pred, summary in [
            ("cos4_simple_single", cos4_pred, cos4_summary),
            ("geometricV0_single", geo0_pred, geo0_summary),
        ]:
            model_dir = frame_dir / model_name
            brs.save_model_outputs(model_dir, gt, pred)
            (model_dir / "summary.json").write_text(json.dumps({**summary, "simple_normalization": norm_meta}, indent=2), encoding="utf-8")

            rows.append(
                {
                    "frame_name": frame_name,
                    "angle": angle,
                    "source_file": source_file,
                    "model": model_name,
                    "runtime_sec": summary["runtime_sec"],
                    "v_fit_percent": summary["v_metrics"]["fit_percent"],
                    "v_r2": summary["v_metrics"]["r2"],
                    "v_rmse": summary["v_metrics"]["rmse"],
                    "v_mae": summary["v_metrics"]["mae"],
                    "normalization_type": norm_meta["type"],
                    "normalization_scale_99p8": norm_meta["scale_99p8"],
                }
            )
            comparisons.append(
                {
                    "frame_name": frame_name,
                    "source_file": source_file,
                    "model": model_name,
                    "v_fit_percent": summary["v_metrics"]["fit_percent"],
                    "v_rmse": summary["v_metrics"]["rmse"],
                }
            )

        save_frame_visuals(frame_dir, gt, image_norm, cos4_pred, geo0_pred)
        frame_summary.append(
            {
                "frame_name": frame_name,
                "angle": angle,
                "source_file": source_file,
                "normalization": norm_meta,
                "cos4_simple_single": cos4_summary["v_metrics"],
                "geometricV0_single": geo0_summary["v_metrics"],
            }
        )

    write_summary_csv(output_root / "single_frame_summary.csv", rows)
    (output_root / "single_frame_summary.json").write_text(json.dumps(frame_summary, indent=2), encoding="utf-8")

    group_rows = []
    with args.benchmark_summary.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == manifest["case_id"]:
                group_rows.append(row)
    (output_root / "group_benchmark_reference.json").write_text(json.dumps(group_rows, indent=2), encoding="utf-8")

    comparison_table = {
        "case_id": manifest["case_id"],
        "display_name": manifest["display_name"],
        "group_benchmark_reference": group_rows,
        "single_frame_results": comparisons,
    }
    (output_root / "comparison_table.json").write_text(json.dumps(comparison_table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
