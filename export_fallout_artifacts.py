import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

import TriditionalMask


def load_masking_v12_module(masking_file: str):
    spec = importlib.util.spec_from_file_location("maskingV12_mod", masking_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import masking module from: {masking_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_uint8(img: np.ndarray):
    arr = np.nan_to_num(img.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = (arr - lo) / (hi - lo)
    return np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)


def save_gray_jpeg(path: Path, img: np.ndarray):
    ok = cv2.imwrite(str(path), to_uint8(img), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Failed writing JPEG: {path}")


def radial_profile(img: np.ndarray, cx: float, cy: float):
    h, w = img.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((xx - np.float32(cx)) ** 2 + (yy - np.float32(cy)) ** 2, dtype=np.float32)
    r_int = rr.astype(np.int32)
    tbin = np.bincount(r_int.ravel(), weights=img.ravel().astype(np.float64))
    nbin = np.bincount(r_int.ravel())
    valid = nbin > 0
    prof = np.zeros_like(tbin, dtype=np.float64)
    prof[valid] = tbin[valid] / nbin[valid]
    radii = np.arange(len(prof), dtype=np.int32)
    return radii, prof


def r2_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    yt = y_true.ravel().astype(np.float64, copy=False)
    yp = y_pred.ravel().astype(np.float64, copy=False)
    sse = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-12)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return float(r2), rmse


def write_curve_csv(path: Path, radius_px, radius_norm, gt, geo, v12):
    header = "radius_px,radius_norm,gt_v,geometric_v,masking_v1_2_v"
    data = np.column_stack(
        [
            radius_px.astype(np.float64),
            radius_norm.astype(np.float64),
            gt.astype(np.float64),
            geo.astype(np.float64),
            v12.astype(np.float64),
        ]
    )
    np.savetxt(str(path), data, delimiter=",", header=header, comments="")


def process_case(
    case_dir: str,
    out_root: Path,
    masking_module,
    epochs: int,
    batch_size: int,
    sample_count: int,
    hidden_dim: int,
):
    case_path = Path(case_dir)
    case_name = case_path.name
    case_out = out_root / case_name
    ensure_dir(case_out)

    gt = np.load(case_path / "ground_truth_vignetting.npy").astype(np.float32, copy=False)
    avg, _ = TriditionalMask.load_rotated_average(str(case_path))

    # ---------- Geometric baseline ----------
    t0 = time.perf_counter()
    geo = TriditionalMask.TraditionalPolynomialBaseline(avg, ground_truth=gt)
    geo.find_optical_center()
    geo.extract_radial_profile()
    geo.fit_model()
    geo.generate_2d_prediction()
    geo_pred = geo.predicted_surface.astype(np.float32, copy=False)
    geo_time = float(time.perf_counter() - t0)
    geo_r2, geo_rmse = r2_rmse(gt, geo_pred)

    # ---------- maskingV1.2 ----------
    t1 = time.perf_counter()
    an = masking_module.MaskingV12Analyzer()
    an.load_from_synthetic_dir(str(case_path))
    h, w = an.h, an.w
    sensor_diag_mm = float(np.hypot(h, w))
    focal_mm = 0.62 * sensor_diag_mm

    cfg = masking_module.V12Config(
        epochs=int(epochs),
        batch_size=int(batch_size),
        sample_count=(None if int(sample_count) < 0 else int(sample_count)),
        hidden_dim=int(hidden_dim),
    )
    an.fit(
        focal_length_mm=focal_mm,
        sensor_diag_mm=sensor_diag_mm,
        cfg=cfg,
        seed=42,
    )
    v12_pred, l_map, i_maps = an.predict()
    v12_time = float(time.perf_counter() - t1)
    v12_r2, v12_rmse = r2_rmse(gt, v12_pred)

    # ---------- Curves ----------
    gy, gx = np.unravel_index(int(np.argmax(gt)), gt.shape)
    center_x = float(gx)
    center_y = float(gy)
    max_radius = float(np.hypot(max(center_x, w - 1 - center_x), max(center_y, h - 1 - center_y)))

    r_gt, p_gt = radial_profile(gt, center_x, center_y)
    r_geo, p_geo = radial_profile(geo_pred, center_x, center_y)
    r_v12, p_v12 = radial_profile(v12_pred, center_x, center_y)
    n = int(min(len(r_gt), len(r_geo), len(r_v12)))

    radius_px = r_gt[:n]
    radius_norm = radius_px.astype(np.float64) / max(max_radius, 1e-6)
    write_curve_csv(
        case_out / "fallout_curve.csv",
        radius_px=radius_px,
        radius_norm=radius_norm,
        gt=p_gt[:n],
        geo=p_geo[:n],
        v12=p_v12[:n],
    )

    # ---------- Images ----------
    save_gray_jpeg(case_out / "gt_v.jpg", gt)
    save_gray_jpeg(case_out / "geometric_v.jpg", geo_pred)
    save_gray_jpeg(case_out / "masking_v1_2_v.jpg", v12_pred)
    save_gray_jpeg(case_out / "masking_v1_2_lighting.jpg", l_map)
    save_gray_jpeg(case_out / "geometric_abs_error.jpg", np.abs(gt - geo_pred))
    save_gray_jpeg(case_out / "masking_v1_2_abs_error.jpg", np.abs(gt - v12_pred))

    strip = np.hstack([to_uint8(gt), to_uint8(geo_pred), to_uint8(v12_pred)])
    cv2.imwrite(
        str(case_out / "comparison_strip.jpg"),
        strip,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )

    summary = {
        "case": case_name,
        "shape": [int(gt.shape[0]), int(gt.shape[1])],
        "curve_center_px": [center_x, center_y],
        "geometric": {
            "r2": geo_r2,
            "fit_percent": 100.0 * geo_r2,
            "rmse": geo_rmse,
            "runtime_sec": geo_time,
        },
        "masking_v1_2": {
            "r2": v12_r2,
            "fit_percent": 100.0 * v12_r2,
            "rmse": v12_rmse,
            "runtime_sec": v12_time,
            "device": str(an.device),
        },
    }
    with open(case_out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export geometric and maskingV1.2 fallout artifacts to CSV/JPEG.")
    parser.add_argument(
        "--cases",
        type=str,
        nargs="+",
        default=[
            "synthetic_realistic_harder_01",
            "synthetic_realistic_harder_02",
            "synthetic_realistic_harder_03",
        ],
        help="Case directories containing sim_capture_*.npy and ground_truth_vignetting.npy",
    )
    parser.add_argument("--masking-file", type=str, default="maskingV1.2.py")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--sample-count", type=int, default=120000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default="fallout_artifacts_exports")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    module = load_masking_v12_module(args.masking_file)

    all_summary = []
    for case in args.cases:
        print(f"[Run] case={case}")
        s = process_case(
            case_dir=case,
            out_root=out_root,
            masking_module=module,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            sample_count=int(args.sample_count),
            hidden_dim=int(args.hidden_dim),
        )
        all_summary.append(s)

    with open(out_root / "summary_all_cases.json", "w", encoding="utf-8") as f:
        json.dump({"cases": all_summary}, f, indent=2)

    print(f"[Done] Exported artifacts to: {out_root}")


if __name__ == "__main__":
    main()
