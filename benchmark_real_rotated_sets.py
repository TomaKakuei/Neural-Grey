import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np

from generate_gt_vignetting_maps import (
    load_linear_bayer_luminance,
    make_gt_map,
    resize_long_edge,
    save_png16,
    suppress_outliers,
)


ANGLES = (0, 90, 180, 270)
SENSOR_DIAG_MM = 43.27
DEFAULT_LONG_EDGE = 2000
CASE_SPECS = [
    {
        "case_id": "55mm_f12",
        "display_name": "55mm f1.2",
        "capture_dir": Path("test images") / "55mm f.12",
        "gt_file": Path("test images") / "GT" / "551.2GT.NEF",
        "gt_generated_stem": "551.2GT",
        "focal_mm": 55.0,
    },
    {
        "case_id": "85mm_f18",
        "display_name": "85mm f1.8",
        "capture_dir": Path("test images") / "80mm f.18",
        "gt_file": Path("test images") / "GT" / "851.8GT.NEF",
        "gt_generated_stem": "851.8GT",
        "focal_mm": 85.0,
    },
    {
        "case_id": "tram_28mm_f35",
        "display_name": "Tramlon 28mm f3.5",
        "capture_dir": Path("test images") / "Tramlon 28mm f3.5",
        "gt_file": Path("test images") / "GT" / "TRAM28mmGT.NEF",
        "gt_generated_stem": "TRAM28mmGT",
        "focal_mm": 28.0,
    },
    {
        "case_id": "tram_70mm_f35",
        "display_name": "Tramlon 70mm f3.5",
        "capture_dir": Path("test images") / "Tramlon 70mm f3.5",
        "gt_file": Path("test images") / "GT" / "TRAM70mmGT.NEF",
        "gt_generated_stem": "TRAM70mmGT",
        "focal_mm": 70.0,
    },
]


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = np.asarray(y_true, dtype=np.float32).ravel()
    yp = np.asarray(y_pred, dtype=np.float32).ravel()
    sse = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2 = float(1.0 - sse / max(sst, 1e-12))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    return {"r2": r2, "fit_percent": 100.0 * r2, "rmse": rmse, "mae": mae}


def normalize_for_preview(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return np.round(arr * 255.0).astype(np.uint8)


def save_preview_gray(path: Path, image_01: np.ndarray) -> None:
    path.write_bytes(cv2.imencode(".png", normalize_for_preview(image_01))[1].tobytes())


def save_preview_error(path: Path, error_01: np.ndarray) -> None:
    err = np.clip(np.asarray(error_01, dtype=np.float32), 0.0, 1.0)
    heat = cv2.applyColorMap(np.round(err * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    path.write_bytes(cv2.imencode(".png", heat)[1].tobytes())


def save_strip(path: Path, gt: np.ndarray, pred: np.ndarray, abs_err: np.ndarray) -> None:
    gt_u8 = cv2.cvtColor(normalize_for_preview(gt), cv2.COLOR_GRAY2BGR)
    pred_u8 = cv2.cvtColor(normalize_for_preview(pred), cv2.COLOR_GRAY2BGR)
    err_u8 = cv2.applyColorMap(np.round(np.clip(abs_err, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    strip = np.concatenate([gt_u8, pred_u8, err_u8], axis=1)
    path.write_bytes(cv2.imencode(".png", strip)[1].tobytes())


def ensure_gt_map(case_spec: dict, long_edge: int) -> tuple[np.ndarray, dict]:
    gt_generated_dir = Path("test images") / "GT" / "generated_gt_maps"
    npy_path = gt_generated_dir / f"{case_spec['gt_generated_stem']}_gt_vignetting.npy"
    json_path = gt_generated_dir / f"{case_spec['gt_generated_stem']}_gt_vignetting.json"
    if npy_path.exists():
        gt = np.load(npy_path).astype(np.float32, copy=False)
        meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
        return gt, meta
    return make_gt_map(case_spec["gt_file"], long_edge=long_edge, blur_sigma=None)


def preprocess_capture(raw_path: Path, long_edge: int) -> np.ndarray:
    image = load_linear_bayer_luminance(raw_path)
    image = resize_long_edge(image, long_edge=long_edge)
    image = suppress_outliers(image)
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def prepare_case(case_spec: dict, output_root: Path, long_edge: int) -> dict:
    case_root = output_root / case_spec["case_id"]
    prepared_dir = case_root / "prepared_dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(case_spec["capture_dir"].glob("*.NEF"))
    if len(raw_paths) != 4:
        raise RuntimeError(f"{case_spec['capture_dir']} must contain exactly 4 NEF files, got {len(raw_paths)}.")

    captures = []
    capture_items = []
    for angle, raw_path in zip(ANGLES, raw_paths):
        capture = preprocess_capture(raw_path, long_edge=long_edge)
        captures.append(capture)
        np.save(prepared_dir / f"sim_capture_{angle}.npy", capture)
        save_png16(prepared_dir / f"sim_capture_{angle}.png", capture)
        save_preview_gray(prepared_dir / f"sim_capture_{angle}_preview.png", capture)
        capture_items.append(
            {
                "angle": angle,
                "source_file": raw_path.name,
                "shape": [int(capture.shape[0]), int(capture.shape[1])],
                "min": float(np.min(capture)),
                "max": float(np.max(capture)),
                "mean": float(np.mean(capture)),
            }
        )

    gt, gt_meta = ensure_gt_map(case_spec, long_edge=long_edge)
    if gt.shape != captures[0].shape:
        raise RuntimeError(
            f"GT shape mismatch for {case_spec['case_id']}: capture {captures[0].shape}, gt {gt.shape}"
        )

    np.save(prepared_dir / "ground_truth_vignetting.npy", gt)
    save_png16(prepared_dir / "ground_truth_vignetting.png", gt)
    save_preview_gray(prepared_dir / "ground_truth_vignetting_preview.png", gt)

    manifest = {
        "case_id": case_spec["case_id"],
        "display_name": case_spec["display_name"],
        "capture_dir": str(case_spec["capture_dir"]),
        "gt_file": str(case_spec["gt_file"]),
        "gt_generated_stem": case_spec["gt_generated_stem"],
        "focal_mm": float(case_spec["focal_mm"]),
        "sensor_diag_mm": float(SENSOR_DIAG_MM),
        "long_edge": int(long_edge),
        "capture_files_sorted_to_angles": capture_items,
        "assumption": "Sorted NEF filenames are mapped to lighting angles 0/90/180/270 in ascending filename order.",
        "gt_metadata": gt_meta,
    }
    (prepared_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "case_root": case_root,
        "prepared_dir": prepared_dir,
        "captures": captures,
        "gt": gt,
        "manifest": manifest,
        "raw_paths": raw_paths,
    }


def save_model_outputs(model_dir: Path, gt: np.ndarray, pred_v: np.ndarray, extra: dict | None = None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    abs_err = np.abs(np.asarray(pred_v, dtype=np.float32) - np.asarray(gt, dtype=np.float32)).astype(np.float32, copy=False)
    np.save(model_dir / "predicted_v.npy", pred_v.astype(np.float32, copy=False))
    np.save(model_dir / "abs_error_vs_gt.npy", abs_err)
    save_png16(model_dir / "predicted_v.png", pred_v)
    save_preview_gray(model_dir / "predicted_v_preview.png", pred_v)
    save_preview_error(model_dir / "abs_error_vs_gt_preview.png", np.clip(abs_err * 4.0, 0.0, 1.0))
    save_strip(model_dir / "comparison_strip.png", gt, pred_v, np.clip(abs_err * 4.0, 0.0, 1.0))
    if extra:
        for name, value in extra.items():
            if isinstance(value, np.ndarray):
                np.save(model_dir / f"{name}.npy", value.astype(np.float32, copy=False))
                if value.ndim == 2:
                    save_preview_gray(model_dir / f"{name}_preview.png", value)


def run_masking_model(module, version_name: str, prepared_dir: Path, gt: np.ndarray, focal_mm: float, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = module.MaskingV12Analyzer()
    analyzer.load_from_synthetic_dir(str(prepared_dir))

    if version_name == "maskingV1.2":
        cfg = module.V12Config(epochs=600, batch_size=8192, sample_count=120000, hidden_dim=128)
        t0 = time.perf_counter()
        analyzer.fit(focal_length_mm=focal_mm, sensor_diag_mm=SENSOR_DIAG_MM, cfg=cfg, seed=42)
        runtime_sec = float(time.perf_counter() - t0)
    elif version_name == "maskingV1.3":
        cfg = module.V12Config(epochs=600, batch_size=8192, sample_count=240000, hidden_dim=96)
        t0 = time.perf_counter()
        analyzer.fit(
            focal_length_mm=focal_mm,
            sensor_diag_mm=SENSOR_DIAG_MM,
            cfg=cfg,
            seed=42,
            log_every=50,
            progress_json=str(output_dir / "progress.json"),
        )
        runtime_sec = float(time.perf_counter() - t0)
    elif version_name == "maskingV1.4":
        cfg = module.V12Config(epochs=600, batch_size=8192, sample_count=240000, hidden_dim=96)
        t0 = time.perf_counter()
        analyzer.fit(
            focal_length_mm=focal_mm,
            sensor_diag_mm=SENSOR_DIAG_MM,
            cfg=cfg,
            seed=42,
            log_every=50,
            progress_json=str(output_dir / "progress.json"),
        )
        runtime_sec = float(time.perf_counter() - t0)
    else:
        raise ValueError(version_name)

    report = analyzer.evaluate(gt_v=gt)
    pred_v, pred_l, pred_captures = analyzer.predict()

    capture_metrics = [compute_metrics(obs, pred) for obs, pred in zip(analyzer.captures, pred_captures)]
    save_model_outputs(output_dir, gt, pred_v, extra={"predicted_l": pred_l})
    for angle, pred_capture in zip(ANGLES, pred_captures):
        np.save(output_dir / f"pred_capture_{angle}.npy", pred_capture.astype(np.float32, copy=False))
        save_preview_gray(output_dir / f"pred_capture_{angle}_preview.png", pred_capture)

    summary = {
        "model": version_name,
        "runtime_sec": runtime_sec,
        "shape": [int(gt.shape[0]), int(gt.shape[1])],
        "focal_mm": float(focal_mm),
        "sensor_diag_mm": float(SENSOR_DIAG_MM),
        "config": dict(vars(cfg)),
        "v_metrics": report["v_metrics"],
        "i_metrics": capture_metrics,
        "i_mean": {
            "fit_percent": float(np.mean([m["fit_percent"] for m in capture_metrics])),
            "rmse": float(np.mean([m["rmse"] for m in capture_metrics])),
            "mae": float(np.mean([m["mae"] for m in capture_metrics])),
        },
        "learned": {
            "f_norm": float(analyzer.learned_f_norm),
            "f_mm": float(analyzer.learned_f_mm),
            "i0": float(analyzer.learned_i0),
            "center_px": [float(analyzer.learned_center_px[0]), float(analyzer.learned_center_px[1])],
            "ax": float(analyzer.learned_ax),
            "ay": float(analyzer.learned_ay),
            "phi_deg": float(analyzer.learned_phi_deg),
            "gains": [float(v) for v in analyzer.learned_gains],
            "biases": [float(v) for v in analyzer.learned_biases],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_geometric_model(module, version_name: str, captures: list[np.ndarray], gt: np.ndarray, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = module.TraditionalPolynomialBaseline(captures, ground_truth=gt)
    baseline.radial_model_type = "poly"
    baseline.lighting_mode = "mesh"
    baseline.pointwise_fit_enabled = True
    if version_name == "geometricV4":
        baseline.optimizer_mode = "bcd"
        baseline.residual_loss_type = "huber"
        baseline.channel_shared_geometry = True
        baseline.bcd_outer_iters = 6

    t0 = time.perf_counter()
    baseline.find_optical_center()
    baseline.extract_radial_profile()
    baseline.fit_model()
    baseline.generate_2d_prediction()
    runtime_sec = float(time.perf_counter() - t0)

    pred_v = baseline.predicted_surface.astype(np.float32, copy=False)
    v_metrics = compute_metrics(gt, pred_v)
    pred_captures = [baseline.predict_capture(k).astype(np.float32, copy=False) for k in range(len(captures))]
    capture_metrics = [compute_metrics(obs, pred) for obs, pred in zip(captures, pred_captures)]

    save_model_outputs(output_dir, gt, pred_v)
    for angle, pred_capture in zip(ANGLES, pred_captures):
        np.save(output_dir / f"pred_capture_{angle}.npy", pred_capture)
        save_preview_gray(output_dir / f"pred_capture_{angle}_preview.png", pred_capture)

    optical_center = baseline.optical_center
    summary = {
        "model": version_name,
        "runtime_sec": runtime_sec,
        "shape": [int(gt.shape[0]), int(gt.shape[1])],
        "v_metrics": v_metrics,
        "i_metrics": capture_metrics,
        "i_mean": {
            "fit_percent": float(np.mean([m["fit_percent"] for m in capture_metrics])),
            "rmse": float(np.mean([m["rmse"] for m in capture_metrics])),
            "mae": float(np.mean([m["mae"] for m in capture_metrics])),
        },
        "optical_center_px": [float(optical_center[0]), float(optical_center[1])] if optical_center is not None else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_global_summary(output_root: Path, rows: list[dict]) -> None:
    summary_json = output_root / "benchmark_summary.json"
    summary_csv = output_root / "benchmark_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "case_id",
        "display_name",
        "model",
        "runtime_sec",
        "v_fit_percent",
        "v_r2",
        "v_rmse",
        "i_fit_percent",
        "i_rmse",
        "i_mae",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "display_name": row["display_name"],
                    "model": row["model"],
                    "runtime_sec": row["runtime_sec"],
                    "v_fit_percent": row["v_metrics"]["fit_percent"],
                    "v_r2": row["v_metrics"]["r2"],
                    "v_rmse": row["v_metrics"]["rmse"],
                    "i_fit_percent": row["i_mean"]["fit_percent"],
                    "i_rmse": row["i_mean"]["rmse"],
                    "i_mae": row["i_mean"]["mae"],
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark real 4-shot RAW sets against GT on all local models.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("real_benchmark_outputs"),
        help="Directory for prepared datasets and benchmark results.",
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=DEFAULT_LONG_EDGE,
        help="Resize captures and GT so the long edge equals this value.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of case_ids to run.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of models: maskingV1.2 maskingV1.3 maskingV1.4 geometricV3 geometricV4",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-model summary.json when present.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    masking_v12 = load_module("masking_v12_file", Path("maskingV1.2.py"))
    masking_v13 = load_module("masking_v13_file", Path("maskingV1.3.py"))
    masking_v14 = load_module("masking_v14_file", Path("maskingV1.4.py"))
    geo_v3 = load_module("geo_v3_file", Path("TriditionalMaskV3.py"))
    geo_v4 = load_module("geo_v4_file", Path("TriditionalMaskV4.py"))

    wanted = None if not args.cases else set(args.cases)
    wanted_models = None if not args.models else set(args.models)
    rows = []

    for case_spec in CASE_SPECS:
        if wanted is not None and case_spec["case_id"] not in wanted:
            continue

        print(f"\n=== Preparing {case_spec['case_id']} ===")
        prepared = prepare_case(case_spec, output_root=output_root, long_edge=args.long_edge)
        case_root = prepared["case_root"]
        gt = prepared["gt"]
        captures = prepared["captures"]

        model_jobs = [
            ("maskingV1.2", lambda: run_masking_model(masking_v12, "maskingV1.2", prepared["prepared_dir"], gt, case_spec["focal_mm"], case_root / "maskingV1.2")),
            ("maskingV1.3", lambda: run_masking_model(masking_v13, "maskingV1.3", prepared["prepared_dir"], gt, case_spec["focal_mm"], case_root / "maskingV1.3")),
            ("maskingV1.4", lambda: run_masking_model(masking_v14, "maskingV1.4", prepared["prepared_dir"], gt, case_spec["focal_mm"], case_root / "maskingV1.4")),
            ("geometricV3", lambda: run_geometric_model(geo_v3, "geometricV3", captures, gt, case_root / "geometricV3")),
            ("geometricV4", lambda: run_geometric_model(geo_v4, "geometricV4", captures, gt, case_root / "geometricV4")),
        ]

        for model_name, runner in model_jobs:
            if wanted_models is not None and model_name not in wanted_models:
                continue

            summary_path = case_root / model_name / "summary.json"
            if args.skip_existing and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row = {
                    "case_id": case_spec["case_id"],
                    "display_name": case_spec["display_name"],
                    **summary,
                }
                rows.append(row)
                print(
                    f"[skip] {case_spec['case_id']} | {model_name} | "
                    f"existing V-fit={summary['v_metrics']['fit_percent']:.4f}%"
                )
                continue

            print(f"\n--- Running {model_name} on {case_spec['case_id']} ---")
            summary = runner()
            row = {
                "case_id": case_spec["case_id"],
                "display_name": case_spec["display_name"],
                **summary,
            }
            rows.append(row)
            print(
                f"[done] {case_spec['case_id']} | {model_name} | "
                f"V-fit={summary['v_metrics']['fit_percent']:.4f}% | "
                f"V-rmse={summary['v_metrics']['rmse']:.6f} | "
                f"I-fit={summary['i_mean']['fit_percent']:.4f}%"
            )

    write_global_summary(output_root, rows)
    print(f"\n[complete] Results written to {output_root}")


if __name__ == "__main__":
    main()
