import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


CASES = [
    "tram_70mm_f35",
    "85mm_f18",
    "55mm_f12",
]

METHODS = [
    {
        "method": "DeVigNet single output-as-V",
        "kind": "devignet_diagnostic",
        "path": lambda case: Path("devignet_single_outputs") / case / "devignet_output_as_v.npy",
    },
    {
        "method": "DeVigNet fourblend output-as-V",
        "kind": "devignet_diagnostic",
        "path": lambda case: Path("devignet_fourblend_outputs") / case / "devignet_fourblend_output_as_v.npy",
    },
    {
        "method": "Fourblend input-as-V baseline",
        "kind": "baseline",
        "path": lambda case: Path("devignet_fourblend_outputs") / case / "devignet_fourblend_input_as_v.npy",
    },
    {
        "method": "maskingV1.3",
        "kind": "four_view_model",
        "path": lambda case: Path("real_benchmark_outputs") / case / "maskingV1.3" / "predicted_v.npy",
    },
    {
        "method": "maskingV1.2",
        "kind": "four_view_model",
        "path": lambda case: Path("real_benchmark_outputs") / case / "maskingV1.2" / "predicted_v.npy",
    },
    {
        "method": "geometricV4",
        "kind": "four_view_model",
        "path": lambda case: Path("real_benchmark_outputs") / case / "geometricV4" / "predicted_v.npy",
    },
    {
        "method": "geometricV3",
        "kind": "four_view_model",
        "path": lambda case: Path("real_benchmark_outputs") / case / "geometricV3" / "predicted_v.npy",
    },
]

FIELDNAMES = [
    "case_id",
    "method",
    "kind",
    "fit_percent",
    "rmse",
    "mae",
    "psnr_db",
    "ssim",
    "pred_file",
]


def compute_basic_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    err = gt - pred
    mse = float(np.mean(err * err))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(err)))
    sse = float(np.sum(err * err))
    sst = float(np.sum((gt - float(np.mean(gt))) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-12)
    psnr = float("inf") if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)
    return {
        "fit_percent": 100.0 * r2,
        "rmse": rmse,
        "mae": mae,
        "psnr_db": psnr,
    }


def ssim_01(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_gt = cv2.GaussianBlur(gt, (11, 11), 1.5)
    mu_pred = cv2.GaussianBlur(pred, (11, 11), 1.5)
    mu_gt_sq = mu_gt * mu_gt
    mu_pred_sq = mu_pred * mu_pred
    mu_cross = mu_gt * mu_pred
    sigma_gt_sq = cv2.GaussianBlur(gt * gt, (11, 11), 1.5) - mu_gt_sq
    sigma_pred_sq = cv2.GaussianBlur(pred * pred, (11, 11), 1.5) - mu_pred_sq
    sigma_cross = cv2.GaussianBlur(gt * pred, (11, 11), 1.5) - mu_cross
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_gt_sq + mu_pred_sq + c1) * (sigma_gt_sq + sigma_pred_sq + c2)
    return float(np.mean(numerator / np.clip(denominator, 1e-12, None)))


def main() -> None:
    output_root = Path("vmap_quality_comparison")
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for case in CASES:
        gt_path = Path("real_benchmark_outputs") / case / "prepared_dataset" / "ground_truth_vignetting.npy"
        gt = np.load(gt_path).astype(np.float32, copy=False)
        for spec in METHODS:
            pred_path = spec["path"](case)
            if not pred_path.exists():
                missing.append({"case_id": case, "method": spec["method"], "pred_file": str(pred_path)})
                continue
            pred = np.load(pred_path).astype(np.float32, copy=False)
            pred = np.clip(pred, 0.0, 1.0)
            metrics = compute_basic_metrics(gt, pred)
            row = {
                "case_id": case,
                "method": spec["method"],
                "kind": spec["kind"],
                "fit_percent": float(metrics["fit_percent"]),
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "psnr_db": float(metrics["psnr_db"]),
                "ssim": ssim_01(gt, pred),
                "pred_file": str(pred_path),
            }
            rows.append(row)

    rows_sorted = sorted(rows, key=lambda row: (row["case_id"], -row["psnr_db"]))
    with (output_root / "vmap_quality_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_sorted)
    (output_root / "vmap_quality_metrics.json").write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")
    (output_root / "missing.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")

    print("case_id,method,fit_percent,rmse,psnr_db,ssim")
    for row in rows_sorted:
        print(
            f"{row['case_id']},{row['method']},{row['fit_percent']:.4f},"
            f"{row['rmse']:.6f},{row['psnr_db']:.4f},{row['ssim']:.6f}"
        )


if __name__ == "__main__":
    main()
