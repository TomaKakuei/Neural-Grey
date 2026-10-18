"""Optional Bayer RAW to linear luminance, matching the paper preprocessing."""

from pathlib import Path
import numpy as np
import cv2


def _decode_color_desc(color_desc):
    if isinstance(color_desc, bytes):
        desc = color_desc.decode("ascii", errors="ignore")
    else:
        desc = str(color_desc)
    return desc.replace("\x00", "")


def _channel_site_map(raw_pattern, color_desc):
    pattern = np.asarray(raw_pattern, dtype=np.int32)
    if pattern.shape != (2, 2):
        raise ValueError(
            f"Unsupported CFA pattern shape {pattern.shape}; expected Bayer 2x2 pattern."
        )

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
        raise ValueError(
            "Unsupported Bayer mapping; expected 1xR, 2xG, 1xB in the 2x2 tile."
        )
    return sites


def _channel_black_white_levels(raw, required_channels):
    black_levels = np.asarray(raw.black_level_per_channel, dtype=np.float32).reshape(-1)
    if black_levels.size == 1:
        black_levels = np.full(
            (required_channels,), float(black_levels[0]), dtype=np.float32
        )
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


def _extract_linearized_channel(
    raw_image, row, col, chan_idx, black_levels, white_levels
):
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
    import rawpy

    with rawpy.imread(str(raw_path)) as raw:
        raw_image = raw.raw_image_visible
        raw_pattern = np.asarray(raw.raw_pattern, dtype=np.int32)
        site_map = _channel_site_map(raw_pattern, raw.color_desc)

        required_channels = int(np.max(raw_pattern)) + 1
        black_levels, white_levels = _channel_black_white_levels(raw, required_channels)

        r_site = site_map["R"][0]
        b_site = site_map["B"][0]
        g_site_1, g_site_2 = site_map["G"]

        plane_r = _extract_linearized_channel(
            raw_image, r_site[0], r_site[1], r_site[2], black_levels, white_levels
        )
        plane_b = _extract_linearized_channel(
            raw_image, b_site[0], b_site[1], b_site[2], black_levels, white_levels
        )
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
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(
        np.float32, copy=False
    )
