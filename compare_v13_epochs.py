import argparse
import csv
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

import benchmark_real_rotated_sets as brs


CASES = [
    {
        "case_id": "85mm_f18",
        "display_name": "85mm f1.8",
        "prepared_dir": Path("real_benchmark_outputs") / "85mm_f18" / "prepared_dataset",
        "focal_mm": 85.0,
    },
    {
        "case_id": "55mm_f12",
        "display_name": "55mm f1.2",
        "prepared_dir": Path("real_benchmark_outputs") / "55mm_f12" / "prepared_dataset",
        "focal_mm": 55.0,
    },
]


FIELDNAMES = [
    "case_id",
    "display_name",
    "model",
    "epochs",
    "sample_count",
    "runtime_sec",
    "v_fit_percent",
    "v_r2",
    "v_rmse",
    "v_mae",
    "v_p95_abs_error",
    "v_max_abs_error",
    "learned_f_mm",
    "learned_center_x",
    "learned_center_y",
    "learned_ax",
    "learned_ay",
    "learned_phi_deg",
]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("masking_v13_epoch_compare", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_v_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    base = brs.compute_metrics(gt, pred)
    err = np.abs(np.asarray(gt, dtype=np.float32) - np.asarray(pred, dtype=np.float32))
    base["mae"] = float(np.mean(err))
    base["p95_abs_error"] = float(np.percentile(err, 95.0))
    base["max_abs_error"] = float(np.max(err))
    return base


def run_case(module, case: dict, args, output_root: Path) -> dict:
    trial_dir = output_root / case["case_id"] / f"maskingV1.3_e{args.epochs}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    summary_path = trial_dir / "summary.json"
    if args.resume and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    analyzer = module.MaskingV12Analyzer()
    analyzer.load_from_synthetic_dir(str(case["prepared_dir"]))
    gt = np.load(case["prepared_dir"] / "ground_truth_vignetting.npy").astype(np.float32, copy=False)
    cfg = module.V12Config(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        sample_count=(None if int(args.sample_count) < 0 else int(args.sample_count)),
        hidden_dim=int(args.hidden_dim),
    )

    t0 = time.perf_counter()
    analyzer.fit(
        focal_length_mm=float(case["focal_mm"]),
        sensor_diag_mm=brs.SENSOR_DIAG_MM,
        cfg=cfg,
        seed=int(args.seed),
        log_every=int(args.log_every),
        progress_json=str(trial_dir / "progress.json"),
    )
    runtime_sec = float(time.perf_counter() - t0)
    pred_v, _, _ = analyzer.predict()
    v_metrics = compute_v_metrics(gt, pred_v)

    row = {
        "case_id": case["case_id"],
        "display_name": case["display_name"],
        "model": "maskingV1.3",
        "epochs": int(args.epochs),
        "sample_count": int(args.sample_count),
        "runtime_sec": runtime_sec,
        "v_fit_percent": float(v_metrics["fit_percent"]),
        "v_r2": float(v_metrics["r2"]),
        "v_rmse": float(v_metrics["rmse"]),
        "v_mae": float(v_metrics["mae"]),
        "v_p95_abs_error": float(v_metrics["p95_abs_error"]),
        "v_max_abs_error": float(v_metrics["max_abs_error"]),
        "learned_f_mm": float(analyzer.learned_f_mm),
        "learned_center_x": float(analyzer.learned_center_px[0]),
        "learned_center_y": float(analyzer.learned_center_px[1]),
        "learned_ax": float(analyzer.learned_ax),
        "learned_ay": float(analyzer.learned_ay),
        "learned_phi_deg": float(analyzer.learned_phi_deg),
    }
    summary_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    del analyzer, pred_v, gt
    gc.collect()
    try:
        module.torch.cuda.empty_cache()
    except Exception:
        pass
    return row


def write_summary(output_root: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["case_id"])
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in FIELDNAMES})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare maskingV1.3 epoch count on prepared real datasets.")
    parser.add_argument("--output-root", type=Path, default=Path("maskingV13_epoch_compare"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--sample-count", type=int, default=240000)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--log-every", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    module = load_module(Path("maskingV1.3.py"))

    rows = []
    for case in CASES:
        print(f"\n=== {case['case_id']} | maskingV1.3 {args.epochs} epochs ===", flush=True)
        row = run_case(module, case, args, output_root)
        rows.append(row)
        write_summary(output_root, rows)
        print(
            f"[done] {case['case_id']} | V-fit={row['v_fit_percent']:.4f}% "
            f"RMSE={row['v_rmse']:.6f} f={row['learned_f_mm']:.2f}mm",
            flush=True,
        )
    write_summary(output_root, rows)


if __name__ == "__main__":
    main()
