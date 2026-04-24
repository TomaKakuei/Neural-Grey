import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import MLMask
import TriditionalMask


def load_dataset(dataset_dir: str):
    d = Path(dataset_dir)
    captures = [
        np.load(d / "sim_capture_0.npy").astype(np.float32, copy=False),
        np.load(d / "sim_capture_90.npy").astype(np.float32, copy=False),
        np.load(d / "sim_capture_180.npy").astype(np.float32, copy=False),
        np.load(d / "sim_capture_270.npy").astype(np.float32, copy=False),
    ]
    gt = np.load(d / "ground_truth_vignetting.npy").astype(np.float32, copy=False)
    avg = (captures[0] + captures[1] + captures[2] + captures[3]) * np.float32(0.25)
    return avg, gt


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    yt = y_true.ravel().astype(np.float64, copy=False)
    yp = y_pred.ravel().astype(np.float64, copy=False)
    sse = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-12)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {"r2": float(r2), "fit_percent": float(100.0 * r2), "rmse": rmse}


def run_traditional(avg: np.ndarray, gt: np.ndarray):
    t0 = time.perf_counter()
    model = TriditionalMask.TraditionalPolynomialBaseline(avg, ground_truth=gt)
    model.find_optical_center()
    model.extract_radial_profile()
    model.fit_model()
    model.generate_2d_prediction()
    eval_out = model.evaluate()
    dt = float(time.perf_counter() - t0)
    if eval_out is None:
        m = metrics(gt, model.predicted_surface)
    else:
        m = {"r2": float(eval_out[0]), "fit_percent": float(100.0 * eval_out[0]), "rmse": float(eval_out[1])}
    return m, dt


def run_mlmask(
    avg: np.ndarray,
    gt: np.ndarray,
    epochs: int = 600,
    batch_size: int = 8192,
    sample_count: int = 120000,
):
    t0 = time.perf_counter()
    analyzer = MLMask.VignetteAnalyzer("synthetic")
    analyzer.img_data = avg.astype(np.float32, copy=False)

    heavy_blur = cv2.GaussianBlur(analyzer.img_data, (101, 101), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(heavy_blur)
    analyzer.optical_center = max_loc
    analyzer.max_brightness = max_val

    analyzer.extract_radial_profile()

    h, w = analyzer.img_data.shape
    sensor_diag_mm = float(np.sqrt(h * h + w * w))
    focal_mm = 0.62 * sensor_diag_mm

    analyzer.fit_model_pinn(
        focal_length_mm=focal_mm,
        sensor_diag_mm=sensor_diag_mm,
        lambda_physics=1.0,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=1e-3,
        physics_lr_f=1e-4,
        hidden_dim=32,
        sample_count=int(sample_count),
        learnable_physics=True,
        seed=42,
    )

    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    cx, cy = analyzer.optical_center
    r = np.sqrt((x_idx - np.float32(cx)) ** 2 + (y_idx - np.float32(cy)) ** 2, dtype=np.float32)
    r_norm = np.clip(r / max(float(analyzer.max_radius_px), 1e-6), 0.0, 1.0).astype(np.float32, copy=False)

    pred = analyzer.predict_pinn(r_norm.reshape(-1)).reshape(h, w).astype(np.float32, copy=False)
    pred = np.clip(pred, 0.0, 1.0)
    peak = float(np.max(pred))
    if peak > 1e-6:
        pred = pred / np.float32(peak)

    m = metrics(gt, pred)
    dt = float(time.perf_counter() - t0)
    extra = {
        "device": str(analyzer.device),
        "learned_f_norm": float(analyzer.learned_f_norm),
        "learned_f_mm": float(analyzer.learned_f_mm),
        "learned_i0": float(analyzer.learned_i0),
    }
    return m, dt, extra


def main():
    parser = argparse.ArgumentParser(description="Compare MLMask vs Traditional on realistic synthetic dataset.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--sample-count", type=int, default=120000)
    parser.add_argument("--out-json", type=str, default="realistic_compare_results.json")
    args = parser.parse_args()

    avg, gt = load_dataset(args.dataset_dir)

    trad_m, trad_t = run_traditional(avg, gt)
    ml_m, ml_t, ml_extra = run_mlmask(
        avg,
        gt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
    )

    out = {
        "dataset_dir": args.dataset_dir,
        "shape": [int(avg.shape[0]), int(avg.shape[1])],
        "traditional": {
            **trad_m,
            "runtime_sec": float(trad_t),
        },
        "mlmask": {
            **ml_m,
            "runtime_sec": float(ml_t),
            **ml_extra,
        },
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
