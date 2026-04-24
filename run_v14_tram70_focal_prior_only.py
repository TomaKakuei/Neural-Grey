import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

import benchmark_real_rotated_sets as brs


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("masking_v14_tram70_focal_prior_only", str(path))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tram70 V1.4 with focal prior and no edge weighting.")
    parser.add_argument("--output-root", type=Path, default=Path("maskingV14_tram70_focal_prior_only_e300"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--sample-count", type=int, default=240000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lambda-focal-prior", type=float, default=0.15)
    parser.add_argument("--lambda-center-prior", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    if args.resume and summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
        return

    module = load_module(Path("maskingV1.4.py"))
    prepared_dir = Path("real_benchmark_outputs") / "tram_70mm_f35" / "prepared_dataset"
    gt = np.load(prepared_dir / "ground_truth_vignetting.npy").astype(np.float32, copy=False)

    analyzer = module.MaskingV12Analyzer()
    analyzer.load_from_synthetic_dir(str(prepared_dir))
    cfg = module.V12Config(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        sample_count=(None if int(args.sample_count) < 0 else int(args.sample_count)),
        hidden_dim=int(args.hidden_dim),
        edge_weight_strength=0.0,
        edge_weight_power=2.0,
        focal_prior_mode="prior",
        lambda_focal_prior=float(args.lambda_focal_prior),
        lambda_center_prior=float(args.lambda_center_prior),
    )

    t0 = time.perf_counter()
    analyzer.fit(
        focal_length_mm=70.0,
        sensor_diag_mm=brs.SENSOR_DIAG_MM,
        cfg=cfg,
        seed=int(args.seed),
        log_every=int(args.log_every),
        progress_json=str(output_root / "progress.json"),
    )
    runtime_sec = float(time.perf_counter() - t0)
    pred_v, _, _ = analyzer.predict()
    v_metrics = compute_v_metrics(gt, pred_v)

    row = {
        "case_id": "tram_70mm_f35",
        "display_name": "Tramlon 70mm f3.5",
        "model": "maskingV1.4",
        "experiment": "focal_prior_only_no_edge",
        "epochs": int(args.epochs),
        "edge_weight_strength": 0.0,
        "focal_prior_mode": "prior",
        "lambda_focal_prior": float(args.lambda_focal_prior),
        "lambda_center_prior": float(args.lambda_center_prior),
        "runtime_sec": runtime_sec,
        "v_metrics": v_metrics,
        "learned": {
            "f_mm": float(analyzer.learned_f_mm),
            "center_px": [float(analyzer.learned_center_px[0]), float(analyzer.learned_center_px[1])],
            "ax": float(analyzer.learned_ax),
            "ay": float(analyzer.learned_ay),
            "phi_deg": float(analyzer.learned_phi_deg),
        },
    }
    summary_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))

    del analyzer, pred_v, gt
    gc.collect()
    try:
        module.torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
