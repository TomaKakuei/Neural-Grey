import argparse
import csv
import importlib.util
import itertools
import json
import os
import random
import time
from dataclasses import asdict
from typing import Any

import numpy as np


DEFAULT_SPACE = {
    "hidden_dim": [96, 128, 160],
    "sample_count": [120000, 240000],
    "lr_net": [6e-4, 8e-4, 1e-3],
    "lr_phys": [4e-5, 8e-5, 1.2e-4],
    "lr_light": [3e-5, 6e-5, 1e-4],
    "lr_gain_bias": [3e-5, 6e-5, 1e-4],
    "lambda_phys": [1.0, 1.2, 1.5],
    "lambda_light": [0.03, 0.06, 0.10],
    "lambda_residual": [0.05, 0.08, 0.12],
    "lambda_anchor": [0.08, 0.12, 0.20],
    "lambda_gain_bias": [0.08, 0.15, 0.25],
    "lambda_warp": [0.20, 0.35, 0.50],
}


def load_module_from_path(file_path: str):
    spec = importlib.util.spec_from_file_location("maskingV12_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_space(space_json: str | None):
    if space_json is None:
        return {k: list(v) for k, v in DEFAULT_SPACE.items()}
    with open(space_json, "r", encoding="utf-8") as f:
        user_space = json.load(f)
    space = {k: list(v) for k, v in DEFAULT_SPACE.items()}
    for k, v in user_space.items():
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"Search space key '{k}' must be a non-empty list.")
        space[k] = v
    return space


def unique_trials_random(space: dict[str, list[Any]], n_trials: int, seed: int):
    rng = random.Random(seed)
    keys = sorted(space.keys())
    seen = set()
    trials = []
    attempts = 0
    max_attempts = max(5000, n_trials * 30)
    while len(trials) < n_trials and attempts < max_attempts:
        attempts += 1
        t = {k: rng.choice(space[k]) for k in keys}
        sig = tuple((k, t[k]) for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        trials.append(t)
    return trials


def trials_grid(space: dict[str, list[Any]], keys: list[str], max_trials: int | None):
    for k in keys:
        if k not in space:
            raise KeyError(f"Grid key '{k}' not in search space.")

    fixed = {k: space[k][0] for k in space.keys() if k not in keys}
    all_values = [space[k] for k in keys]

    out = []
    for vals in itertools.product(*all_values):
        t = dict(fixed)
        for k, v in zip(keys, vals):
            t[k] = v
        out.append(t)
        if max_trials is not None and len(out) >= max_trials:
            break
    return out


def make_cfg(module, trial: dict[str, Any], epochs: int, batch_size: int):
    cfg = module.V12Config(
        epochs=int(epochs),
        batch_size=int(batch_size),
        sample_count=(None if int(trial["sample_count"]) < 0 else int(trial["sample_count"])),
        hidden_dim=int(trial["hidden_dim"]),
        lr_net=float(trial["lr_net"]),
        lr_phys=float(trial["lr_phys"]),
        lr_light=float(trial["lr_light"]),
        lr_gain_bias=float(trial["lr_gain_bias"]),
        lambda_phys=float(trial["lambda_phys"]),
        lambda_light=float(trial["lambda_light"]),
        lambda_residual=float(trial["lambda_residual"]),
        lambda_anchor=float(trial["lambda_anchor"]),
        lambda_gain_bias=float(trial["lambda_gain_bias"]),
        lambda_warp=float(trial["lambda_warp"]),
    )
    return cfg


def objective_score(metrics: dict[str, Any], objective: str):
    if objective == "v_fit":
        vm = metrics.get("v_metrics")
        if vm is None:
            raise ValueError("Objective 'v_fit' requires ground-truth V (--gt-v).")
        return float(vm["fit_percent"])
    if objective == "v_r2":
        vm = metrics.get("v_metrics")
        if vm is None:
            raise ValueError("Objective 'v_r2' requires ground-truth V (--gt-v).")
        return float(vm["r2"])
    if objective == "i_fit":
        return float(metrics["i_mean"]["fit_percent"])
    raise ValueError(f"Unknown objective: {objective}")


def write_json(path: str, payload: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: str, rows: list[dict[str, Any]]):
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    cols = sorted(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_trial(
    module,
    trial_id: int,
    trial: dict[str, Any],
    synthetic_dir: str,
    gt_v_path: str | None,
    epochs: int,
    batch_size: int,
    focal_mm: float | None,
    sensor_diag_mm: float | None,
    seed: int,
    objective: str,
):
    analyzer = module.MaskingV12Analyzer()
    analyzer.load_from_synthetic_dir(synthetic_dir)

    h, w = analyzer.h, analyzer.w
    sensor_diag = float(np.hypot(h, w)) if sensor_diag_mm is None else float(sensor_diag_mm)
    focal = 0.62 * sensor_diag if focal_mm is None else float(focal_mm)

    cfg = make_cfg(module, trial, epochs=epochs, batch_size=batch_size)

    gt_v = None
    if gt_v_path is not None:
        gt_v = np.load(gt_v_path).astype(np.float32, copy=False)

    t0 = time.perf_counter()
    analyzer.fit(focal_length_mm=focal, sensor_diag_mm=sensor_diag, cfg=cfg, seed=seed)
    metrics = analyzer.evaluate(gt_v=gt_v)
    runtime = float(time.perf_counter() - t0)
    score = objective_score(metrics, objective)

    out = {
        "trial_id": int(trial_id),
        "score": float(score),
        "objective": objective,
        "runtime_sec": float(runtime),
        "device": str(analyzer.device),
        "v_fit_percent": None if metrics.get("v_metrics") is None else float(metrics["v_metrics"]["fit_percent"]),
        "v_r2": None if metrics.get("v_metrics") is None else float(metrics["v_metrics"]["r2"]),
        "v_rmse": None if metrics.get("v_metrics") is None else float(metrics["v_metrics"]["rmse"]),
        "i_fit_percent": float(metrics["i_mean"]["fit_percent"]),
        "i_rmse": float(metrics["i_mean"]["rmse"]),
        "config": asdict(cfg),
        "learned": {
            "f_norm": float(analyzer.learned_f_norm),
            "f_mm": float(analyzer.learned_f_mm),
            "i0": float(analyzer.learned_i0),
            "center_px": [float(analyzer.learned_center_px[0]), float(analyzer.learned_center_px[1])],
            "ax": float(analyzer.learned_ax),
            "ay": float(analyzer.learned_ay),
            "phi_deg": float(analyzer.learned_phi_deg),
        },
    }
    return out


def compact_row(result: dict[str, Any]):
    cfg = result["config"]
    return {
        "trial_id": result["trial_id"],
        "score": result["score"],
        "objective": result["objective"],
        "runtime_sec": result["runtime_sec"],
        "device": result["device"],
        "v_fit_percent": result["v_fit_percent"],
        "v_r2": result["v_r2"],
        "v_rmse": result["v_rmse"],
        "i_fit_percent": result["i_fit_percent"],
        "i_rmse": result["i_rmse"],
        "hidden_dim": cfg["hidden_dim"],
        "sample_count": cfg["sample_count"],
        "lr_net": cfg["lr_net"],
        "lr_phys": cfg["lr_phys"],
        "lr_light": cfg["lr_light"],
        "lr_gain_bias": cfg["lr_gain_bias"],
        "lambda_phys": cfg["lambda_phys"],
        "lambda_light": cfg["lambda_light"],
        "lambda_residual": cfg["lambda_residual"],
        "lambda_anchor": cfg["lambda_anchor"],
        "lambda_gain_bias": cfg["lambda_gain_bias"],
        "lambda_warp": cfg["lambda_warp"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep tool for maskingV1.2 (no code changes required).")
    parser.add_argument("--masking-file", type=str, default="maskingV1.2.py", help="Path to maskingV1.2.py")
    parser.add_argument("--synthetic-dir", type=str, required=True, help="Dataset folder containing sim_capture_*.npy")
    parser.add_argument("--gt-v", type=str, default=None, help="Path to ground_truth_vignetting.npy (required for v_fit/v_r2 objective)")
    parser.add_argument("--focal-mm", type=float, default=None, help="Optional fixed focal init")
    parser.add_argument("--sensor-diag-mm", type=float, default=None, help="Optional fixed sensor diagonal")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mode", type=str, default="random", choices=["random", "grid"])
    parser.add_argument("--trials", type=int, default=20, help="Number of random trials or max grid trials.")
    parser.add_argument("--grid-keys", type=str, default="lr_net,lambda_anchor,lambda_warp", help="Comma-separated keys for grid mode.")
    parser.add_argument("--space-json", type=str, default=None, help="Optional JSON to override DEFAULT_SPACE.")

    parser.add_argument("--objective", type=str, default="v_fit", choices=["v_fit", "v_r2", "i_fit"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-json", type=str, default="maskingV1.2_sweep_results.json")
    parser.add_argument("--out-csv", type=str, default="maskingV1.2_sweep_results.csv")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned trials, do not train.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.masking_file):
        raise FileNotFoundError(args.masking_file)
    if not os.path.isdir(args.synthetic_dir):
        raise FileNotFoundError(args.synthetic_dir)

    space = load_space(args.space_json)

    if args.mode == "random":
        trials = unique_trials_random(space, n_trials=int(args.trials), seed=int(args.seed))
    else:
        keys = [k.strip() for k in args.grid_keys.split(",") if k.strip()]
        trials = trials_grid(space, keys=keys, max_trials=int(args.trials))

    if len(trials) == 0:
        raise RuntimeError("No trials generated. Check search space settings.")

    print(f"[Sweep] mode={args.mode} objective={args.objective} trials={len(trials)}")
    for i, t in enumerate(trials[: min(5, len(trials))], start=1):
        print(f"  trial#{i}: {t}")
    if len(trials) > 5:
        print(f"  ... ({len(trials) - 5} more)")

    if args.dry_run:
        print("[Sweep] dry-run enabled, exiting without training.")
        return

    module = load_module_from_path(args.masking_file)
    results = []
    csv_rows = []
    started = time.perf_counter()

    for i, trial in enumerate(trials, start=1):
        print(f"\n[Sweep] Running trial {i}/{len(trials)}")
        result = run_trial(
            module=module,
            trial_id=i,
            trial=trial,
            synthetic_dir=args.synthetic_dir,
            gt_v_path=args.gt_v,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            focal_mm=args.focal_mm,
            sensor_diag_mm=args.sensor_diag_mm,
            seed=int(args.seed) + i,
            objective=args.objective,
        )
        results.append(result)
        csv_rows.append(compact_row(result))

        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        payload = {
            "objective": args.objective,
            "mode": args.mode,
            "total_trials": len(trials),
            "elapsed_sec": float(time.perf_counter() - started),
            "best": results_sorted[0],
            "top_k": results_sorted[: int(args.top_k)],
            "all_results": results,
        }
        write_json(args.out_json, payload)
        write_csv(args.out_csv, sorted(csv_rows, key=lambda x: x["score"], reverse=True))
        print(
            f"[Sweep] trial={i} score={result['score']:.4f} "
            f"v_fit={result['v_fit_percent']} i_fit={result['i_fit_percent']:.4f} "
            f"runtime={result['runtime_sec']:.2f}s"
        )

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    best = sorted(results, key=lambda x: x["score"], reverse=True)[0]
    print("\n[Sweep] Finished")
    print(f"Best score={best['score']:.4f} @ trial#{best['trial_id']}")
    print(f"Saved: {args.out_json}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
