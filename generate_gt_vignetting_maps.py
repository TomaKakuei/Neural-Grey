import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rawpy


def _decode_color_desc(color_desc):
    if isinstance(color_desc, bytes):
        desc = color_desc.decode("ascii", errors="ignore")
    else:
        desc = str(color_desc)
    return desc.replace("\x00", "")


def _channel_site_map(raw_pattern, color_desc):
    pattern = np.asarray(raw_pattern, dtype=np.int32)
    if pattern.shape != (2, 2):
        raise ValueError(f"Unsupported CFA pattern shape {pattern.shape}; expected Bayer 2x2 pattern.")

    desc = _decode_color_desc(color_desc)
    max_index = int(np.max(pattern))
    if len(desc) <= max_index:
        raise ValueError(
            f"color_desc length {len(desc)} is incompatible with raw_pattern max index {max_index}."
        )

    sites = {"R": [], "G": [], "B": []}
    for row in range(2):
        for col in range(2):
            chan_idx = int(pattern[row, col])
            chan_name = desc[chan_idx]
            if chan_name in sites:
                sites[chan_name].append((row, col, chan_idx))

    if len(sites["R"]) != 1 or len(sites["B"]) != 1 or len(sites["G"]) != 2:
        raise ValueError("Unsupported Bayer mapping; expected 1xR, 2xG, 1xB in the 2x2 tile.")
    return sites


def _channel_black_white_levels(raw, required_channels):
    black_levels = np.asarray(raw.black_level_per_channel, dtype=np.float32).reshape(-1)
    if black_levels.size == 1:
        black_levels = np.full((required_channels,), float(black_levels[0]), dtype=np.float32)
    elif black_levels.size < required_channels:
        raise ValueError(
            f"black_level_per_channel length {black_levels.size} is less than required {required_channels}."
        )

    cam_white = getattr(raw, "camera_white_level_per_channel", None)
    if cam_white is not None:
        white_levels = np.asarray(cam_white, dtype=np.float32).reshape(-1)
        if white_levels.size < required_channels:
            white_levels = np.full((required_channels,), 0.0, dtype=np.float32)
    else:
        white_levels = np.full((required_channels,), 0.0, dtype=np.float32)

    white_fallback = float(getattr(raw, "white_level", 0.0))
    if white_fallback <= 0.0 and np.any(white_levels[:required_channels] <= 0.0):
        raise ValueError("RAW metadata has no valid channel white level.")

    white_levels = white_levels[:required_channels].copy()
    invalid_white = white_levels <= 0.0
    if np.any(invalid_white):
        white_levels[invalid_white] = white_fallback
    return black_levels[:required_channels], white_levels


def _extract_linearized_channel(raw_image, row, col, chan_idx, black_levels, white_levels):
    plane = raw_image[row::2, col::2].astype(np.float32, copy=True)
    black = float(black_levels[chan_idx])
    white = float(white_levels[chan_idx])
    denom = white - black
    if denom <= 1e-12:
        raise ValueError(
            f"Invalid black/white levels for channel index {chan_idx}: black={black}, white={white}."
        )

    plane -= black
    plane /= denom
    np.clip(plane, 0.0, 1.0, out=plane)
    return plane


def load_linear_bayer_luminance(raw_path: Path) -> np.ndarray:
    with rawpy.imread(str(raw_path)) as raw:
        raw_image = raw.raw_image_visible
        raw_pattern = np.asarray(raw.raw_pattern, dtype=np.int32)
        site_map = _channel_site_map(raw_pattern, raw.color_desc)

        required_channels = int(np.max(raw_pattern)) + 1
        black_levels, white_levels = _channel_black_white_levels(raw, required_channels)

        r_site = site_map["R"][0]
        b_site = site_map["B"][0]
        g_site_1, g_site_2 = site_map["G"]

        plane_r = _extract_linearized_channel(raw_image, r_site[0], r_site[1], r_site[2], black_levels, white_levels)
        plane_b = _extract_linearized_channel(raw_image, b_site[0], b_site[1], b_site[2], black_levels, white_levels)
        plane_g1 = _extract_linearized_channel(
            raw_image, g_site_1[0], g_site_1[1], g_site_1[2], black_levels, white_levels
        )
        plane_g2 = _extract_linearized_channel(
            raw_image, g_site_2[0], g_site_2[1], g_site_2[2], black_levels, white_levels
        )

    plane_g = 0.5 * (plane_g1 + plane_g2)
    luminance = 0.2126 * plane_r + 0.7152 * plane_g + 0.0722 * plane_b
    return luminance.astype(np.float32, copy=False)


def resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = float(long_edge) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image.astype(np.float32, copy=False)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)


def suppress_outliers(image: np.ndarray) -> np.ndarray:
    base = image.astype(np.float32, copy=True)
    blurred = cv2.GaussianBlur(base, (5, 5), 0)
    diff = np.abs(base - blurred)
    threshold = np.maximum(0.40 * blurred, np.float32(0.01))
    mask = diff > threshold
    base[mask] = blurred[mask]
    return base.astype(np.float32, copy=False)


def estimate_center(img: np.ndarray) -> tuple[float, float]:
    blur = cv2.GaussianBlur(img, (0, 0), 25)
    q = float(np.percentile(blur, 65.0))
    w = np.clip(blur - q, 0.0, None)
    w = w * w
    s = float(np.sum(w))
    if s <= 1e-12:
        my, mx = np.unravel_index(int(np.argmax(blur)), blur.shape)
        return float(mx), float(my)
    yy, xx = np.indices(img.shape, dtype=np.float32)
    cx = float(np.sum(w * xx) / s)
    cy = float(np.sum(w * yy) / s)
    return cx, cy


def robust_center_peak(img: np.ndarray, center_xy: tuple[float, float]) -> float:
    h, w = img.shape
    cx, cy = center_xy
    radius = max(12, int(round(min(h, w) * 0.03)))
    x0 = max(0, int(round(cx)) - radius)
    x1 = min(w, int(round(cx)) + radius + 1)
    y0 = max(0, int(round(cy)) - radius)
    y1 = min(h, int(round(cy)) + radius + 1)
    patch = img[y0:y1, x0:x1].ravel()
    if patch.size == 0:
        return float(max(np.max(img), 1e-6))

    top_k = max(1, patch.size // 20)
    peak = float(np.mean(np.partition(patch, -top_k)[-top_k:]))
    return max(peak, 1e-6)


def save_png16(path: Path, image_01: np.ndarray) -> None:
    image_u16 = np.clip(np.round(image_01 * 65535.0), 0.0, 65535.0).astype(np.uint16)
    ok, encoded = cv2.imencode(".png", image_u16)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    path.write_bytes(encoded.tobytes())


def make_gt_map(raw_path: Path, long_edge: int, blur_sigma: float | None) -> tuple[np.ndarray, dict]:
    luminance = load_linear_bayer_luminance(raw_path)
    resized = resize_long_edge(luminance, long_edge=long_edge)
    cleaned = suppress_outliers(resized)

    auto_sigma = max(float(long_edge) * 0.012, 8.0)
    sigma = auto_sigma if blur_sigma is None else float(max(blur_sigma, 1.0))
    smooth = cv2.GaussianBlur(cleaned, (0, 0), sigmaX=sigma, sigmaY=sigma)

    center_xy = estimate_center(smooth)
    norm_peak = robust_center_peak(smooth, center_xy)
    gt = np.clip(smooth / np.float32(norm_peak), 0.0, 1.0).astype(np.float32, copy=False)

    metadata = {
        "source_file": raw_path.name,
        "source_shape": list(map(int, luminance.shape)),
        "output_shape": list(map(int, gt.shape)),
        "long_edge": int(long_edge),
        "blur_sigma_px": float(sigma),
        "optical_center_xy_px": [float(center_xy[0]), float(center_xy[1])],
        "normalization_peak": float(norm_peak),
        "min": float(np.min(gt)),
        "max": float(np.max(gt)),
        "mean": float(np.mean(gt)),
    }
    return gt, metadata


def process_one(raw_path: Path, output_dir: Path, long_edge: int, blur_sigma: float | None) -> dict:
    gt, metadata = make_gt_map(raw_path, long_edge=long_edge, blur_sigma=blur_sigma)
    stem = raw_path.stem

    npy_path = output_dir / f"{stem}_gt_vignetting.npy"
    png_path = output_dir / f"{stem}_gt_vignetting.png"
    json_path = output_dir / f"{stem}_gt_vignetting.json"

    np.save(npy_path, gt)
    save_png16(png_path, gt)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"[ok] {raw_path.name} -> {gt.shape[1]}x{gt.shape[0]} | "
        f"min={metadata['min']:.4f} mean={metadata['mean']:.4f} max={metadata['max']:.4f}"
    )
    return {
        "source_file": raw_path.name,
        "npy_path": str(npy_path),
        "png_path": str(png_path),
        "json_path": str(json_path),
        **metadata,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate normalized flat-field GT vignette maps from GT NEF files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("test images") / "GT",
        help="Directory containing GT NEF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input-dir>/generated_gt_maps.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.NEF",
        help="Filename glob for RAW files.",
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=2000,
        help="Resize output so the long edge equals this value.",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=None,
        help="Gaussian sigma in output pixels. Default is 1.2%% of long edge, with a floor of 8 px.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or (input_dir / "generated_gt_maps")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    raw_paths = sorted(input_dir.glob(args.pattern))
    if not raw_paths:
        raise FileNotFoundError(f"No files matched {args.pattern!r} in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for raw_path in raw_paths:
        summary.append(
            process_one(raw_path=raw_path, output_dir=output_dir, long_edge=args.long_edge, blur_sigma=args.blur_sigma)
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] Wrote {len(summary)} GT maps to {output_dir}")


if __name__ == "__main__":
    main()
