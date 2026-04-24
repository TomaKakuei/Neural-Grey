import argparse
import csv
import importlib.util
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import benchmark_real_rotated_sets as brs


CASES = {
    "tram_70mm_f35": {
        "display_name": "Tramlon 70mm f3.5",
        "prepared_dir": Path("real_benchmark_outputs") / "tram_70mm_f35" / "prepared_dataset",
        "focal_mm": 70.0,
    },
    "85mm_f18": {
        "display_name": "85mm f1.8",
        "prepared_dir": Path("real_benchmark_outputs") / "85mm_f18" / "prepared_dataset",
        "focal_mm": 85.0,
    },
    "55mm_f12": {
        "display_name": "55mm f1.2",
        "prepared_dir": Path("real_benchmark_outputs") / "55mm_f12" / "prepared_dataset",
        "focal_mm": 55.0,
    },
}

FIELDNAMES = [
    "case_id",
    "display_name",
    "model",
    "device",
    "epochs",
    "sample_count",
    "batch_size",
    "hidden_dim",
    "torch_num_threads",
    "wall_runtime_sec",
    "process_cpu_sec",
    "cpu_sec_per_wall_sec",
    "sec_per_epoch",
    "estimated_600_epoch_wall_sec",
    "estimated_600_epoch_wall_min",
    "peak_working_set_mb",
    "final_working_set_mb",
    "v_fit_percent",
    "v_rmse",
    "v_mae",
    "v_p95_abs_error",
    "learned_f_mm",
    "python",
    "platform",
    "logical_cpus",
]


def windows_memory_mb() -> tuple[float | None, float | None]:
    if os.name != "nt":
        return None, None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
    kernel = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return None, None
    mb = 1024.0 * 1024.0
    return counters.PeakWorkingSetSize / mb, counters.WorkingSetSize / mb


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("masking_v13_cpu_bench", str(path))
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
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure CPU runtime probe for maskingV1.3.")
    parser.add_argument("--case", choices=sorted(CASES), default="85mm_f18")
    parser.add_argument("--output-root", type=Path, default=Path("maskingV13_cpu_benchmark"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--sample-count", type=int, default=240000)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--threads", type=int, default=0, help="0 keeps PyTorch's default CPU thread count.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output_root = args.output_root / args.case / f"e{args.epochs}_s{args.sample_count}_cpu"
    output_root.mkdir(parents=True, exist_ok=True)

    module = load_module(Path("maskingV1.3.py"))
    module.torch.cuda.is_available = lambda: False
    if int(args.threads) > 0:
        module.torch.set_num_threads(int(args.threads))

    case = CASES[args.case]
    analyzer = module.MaskingV12Analyzer()
    analyzer.load_from_synthetic_dir(str(case["prepared_dir"]))
    gt = np.load(case["prepared_dir"] / "ground_truth_vignetting.npy").astype(np.float32, copy=False)
    cfg = module.V12Config(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        sample_count=(None if int(args.sample_count) < 0 else int(args.sample_count)),
        hidden_dim=int(args.hidden_dim),
    )

    print(
        f"[cpu-bench] case={args.case} epochs={args.epochs} samples={args.sample_count} "
        f"batch={args.batch_size} torch_threads={module.torch.get_num_threads()}",
        flush=True,
    )

    wall_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    analyzer.fit(
        focal_length_mm=float(case["focal_mm"]),
        sensor_diag_mm=brs.SENSOR_DIAG_MM,
        cfg=cfg,
        seed=int(args.seed),
        log_every=int(args.log_every),
        progress_json=str(output_root / "progress.json"),
    )
    wall_runtime = float(time.perf_counter() - wall_t0)
    process_cpu = float(time.process_time() - cpu_t0)

    pred_v, _, _ = analyzer.predict()
    v_metrics = compute_v_metrics(gt, pred_v)
    peak_mb, final_mb = windows_memory_mb()
    sec_per_epoch = wall_runtime / max(int(args.epochs), 1)

    row = {
        "case_id": args.case,
        "display_name": case["display_name"],
        "model": "maskingV1.3",
        "device": str(analyzer.device),
        "epochs": int(args.epochs),
        "sample_count": int(args.sample_count),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "torch_num_threads": int(module.torch.get_num_threads()),
        "wall_runtime_sec": wall_runtime,
        "process_cpu_sec": process_cpu,
        "cpu_sec_per_wall_sec": process_cpu / max(wall_runtime, 1e-12),
        "sec_per_epoch": sec_per_epoch,
        "estimated_600_epoch_wall_sec": sec_per_epoch * 600.0,
        "estimated_600_epoch_wall_min": sec_per_epoch * 10.0,
        "peak_working_set_mb": peak_mb,
        "final_working_set_mb": final_mb,
        "v_fit_percent": float(v_metrics["fit_percent"]),
        "v_rmse": float(v_metrics["rmse"]),
        "v_mae": float(v_metrics["mae"]),
        "v_p95_abs_error": float(v_metrics["p95_abs_error"]),
        "learned_f_mm": float(analyzer.learned_f_mm),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": int(os.cpu_count() or 0),
    }

    (output_root / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    np.save(output_root / "predicted_v.npy", pred_v.astype(np.float32, copy=False))
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)

    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
