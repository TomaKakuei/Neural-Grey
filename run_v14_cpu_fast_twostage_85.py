import argparse
import csv
import gc
import importlib.util
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import benchmark_real_rotated_sets as brs


CASE = {
    "case_id": "85mm_f18",
    "display_name": "85mm f1.8",
    "prepared_dir": Path("real_benchmark_outputs") / "85mm_f18" / "prepared_dataset",
    "focal_mm": 85.0,
}

FIELDNAMES = [
    "case_id",
    "display_name",
    "model",
    "experiment",
    "device",
    "stage1_epochs",
    "stage1_sample_count",
    "stage2_epochs",
    "stage2_sample_count",
    "batch_size",
    "hidden_dim",
    "torch_num_threads",
    "shuffle_every_epochs",
    "edge_weight_strength",
    "focal_prior_mode",
    "lambda_focal_prior",
    "lambda_center_prior",
    "lambda_anchor",
    "stage1_runtime_sec",
    "stage2_runtime_sec",
    "total_runtime_sec",
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
    "platform",
    "logical_cpus",
]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("masking_v14_cpu_fast_twostage", str(path))
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


def save_preview_gray(path: Path, image: np.ndarray) -> None:
    arr = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    path.write_bytes(brs.cv2.imencode(".png", np.round(arr * 255.0).astype(np.uint8))[1].tobytes())


def configure_cpu(module, threads: int, interop_threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    module.torch.cuda.is_available = lambda: False
    if interop_threads > 0:
        try:
            module.torch.set_num_interop_threads(int(interop_threads))
        except RuntimeError:
            pass
    if threads > 0:
        module.torch.set_num_threads(int(threads))


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-fast two-stage V1.4 test on 85mm f1.8.")
    parser.add_argument("--output-root", type=Path, default=Path("maskingV14_cpu_fast_twostage_85"))
    parser.add_argument("--stage1-epochs", type=int, default=150)
    parser.add_argument("--stage1-sample-count", type=int, default=60000)
    parser.add_argument("--stage2-epochs", type=int, default=300)
    parser.add_argument("--stage2-sample-count", type=int, default=240000)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--shuffle-every-epochs", type=int, default=5)
    parser.add_argument("--lambda-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-center-prior", type=float, default=0.0)
    parser.add_argument("--edge-weight-strength", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    module = load_module(Path("maskingV1.4.py"))
    configure_cpu(module, int(args.threads), int(args.interop_threads))

    analyzer = module.MaskingV12Analyzer()
    analyzer.load_from_synthetic_dir(str(CASE["prepared_dir"]))
    gt = np.load(CASE["prepared_dir"] / "ground_truth_vignetting.npy").astype(np.float32, copy=False)

    common_cfg = {
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "edge_weight_strength": float(args.edge_weight_strength),
        "edge_weight_power": 2.0,
        "focal_prior_mode": "free",
        "lambda_focal_prior": 0.0,
        "lambda_center_prior": float(args.lambda_center_prior),
        "lambda_anchor": float(args.lambda_anchor),
        "shuffle_every_epochs": int(args.shuffle_every_epochs),
    }
    cfg_stage1 = module.V12Config(
        epochs=int(args.stage1_epochs),
        sample_count=int(args.stage1_sample_count),
        **common_cfg,
    )
    cfg_stage2 = module.V12Config(
        epochs=int(args.stage2_epochs),
        sample_count=int(args.stage2_sample_count),
        **common_cfg,
    )

    print(
        "[run] V1.4 CPU fast two-stage 85mm | "
        f"threads={module.torch.get_num_threads()} batch={args.batch_size} "
        f"edge={float(args.edge_weight_strength):.3f} "
        f"center_prior={float(args.lambda_center_prior):.3f} "
        f"focal_prior=free anchor={float(args.lambda_anchor):.3f}",
        flush=True,
    )

    total_t0 = time.perf_counter()
    stage1_t0 = time.perf_counter()
    analyzer.fit(
        focal_length_mm=float(CASE["focal_mm"]),
        sensor_diag_mm=brs.SENSOR_DIAG_MM,
        cfg=cfg_stage1,
        seed=int(args.seed),
        log_every=int(args.log_every),
        progress_json=str(output_root / "stage1_progress.json"),
        warm_start=False,
    )
    stage1_runtime = float(time.perf_counter() - stage1_t0)

    stage2_t0 = time.perf_counter()
    analyzer.fit(
        focal_length_mm=float(CASE["focal_mm"]),
        sensor_diag_mm=brs.SENSOR_DIAG_MM,
        cfg=cfg_stage2,
        seed=int(args.seed) + 1,
        log_every=int(args.log_every),
        progress_json=str(output_root / "stage2_progress.json"),
        warm_start=True,
    )
    stage2_runtime = float(time.perf_counter() - stage2_t0)
    total_runtime = float(time.perf_counter() - total_t0)

    pred_v, pred_l, pred_captures = analyzer.predict()
    v_metrics = compute_v_metrics(gt, pred_v)
    abs_err = np.abs(gt - pred_v).astype(np.float32, copy=False)

    np.save(output_root / "predicted_v.npy", pred_v.astype(np.float32, copy=False))
    np.save(output_root / "predicted_l.npy", pred_l.astype(np.float32, copy=False))
    np.save(output_root / "abs_error_vs_gt.npy", abs_err)
    save_preview_gray(output_root / "predicted_v_preview.png", pred_v)
    save_preview_gray(output_root / "predicted_l_preview.png", pred_l)
    save_preview_gray(output_root / "abs_error_vs_gt_preview.png", np.clip(abs_err * 4.0, 0.0, 1.0))
    for angle, pred_capture in zip((0, 90, 180, 270), pred_captures):
        np.save(output_root / f"pred_capture_{angle}.npy", pred_capture.astype(np.float32, copy=False))

    row = {
        "case_id": CASE["case_id"],
        "display_name": CASE["display_name"],
        "model": "maskingV1.4",
        "experiment": "cpu_fast_two_stage_no_pivot_no_edge",
        "device": str(analyzer.device),
        "stage1_epochs": int(args.stage1_epochs),
        "stage1_sample_count": int(args.stage1_sample_count),
        "stage2_epochs": int(args.stage2_epochs),
        "stage2_sample_count": int(args.stage2_sample_count),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "torch_num_threads": int(module.torch.get_num_threads()),
        "shuffle_every_epochs": int(args.shuffle_every_epochs),
        "edge_weight_strength": float(args.edge_weight_strength),
        "focal_prior_mode": "free",
        "lambda_focal_prior": 0.0,
        "lambda_center_prior": float(args.lambda_center_prior),
        "lambda_anchor": float(args.lambda_anchor),
        "stage1_runtime_sec": stage1_runtime,
        "stage2_runtime_sec": stage2_runtime,
        "total_runtime_sec": total_runtime,
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
        "platform": platform.platform(),
        "logical_cpus": int(os.cpu_count() or 0),
        "stage1_config": asdict(cfg_stage1),
        "stage2_config": asdict(cfg_stage2),
    }

    (output_root / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow({key: row.get(key) for key in FIELDNAMES})

    print(json.dumps(row, indent=2), flush=True)

    del analyzer, pred_v, pred_l, pred_captures, gt
    gc.collect()


if __name__ == "__main__":
    main()
