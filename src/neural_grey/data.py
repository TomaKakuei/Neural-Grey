"""Capture I/O and a small independent analytic example generator."""

from pathlib import Path
import numpy as np

ANGLES = (0, 90, 180, 270)


def load_captures(path):
    path = Path(path)
    if path.is_dir():
        return np.stack(
            [
                np.load(path / f"sim_capture_{angle}.npy", allow_pickle=False)
                for angle in ANGLES
            ]
        )
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return z["captures"].copy()
    return np.load(path, allow_pickle=False)


def synthetic_case(height=48, width=64, seed=17):
    """Sensor-fixed analytic shading and scene-fixed light under four rolls.

    This runnable example is not a member of the paper's evaluation datasets.
    Ground truth is returned separately and is never supplied to the fitter.
    """
    if min(height, width) < 8:
        raise ValueError("Example dimensions must be at least 8")
    yy, xx = np.indices((height, width), dtype=np.float32)
    scale = np.hypot((width - 1) * 0.5, (height - 1) * 0.5)
    x, y = (xx - (width - 1) * 0.5) / scale, (yy - (height - 1) * 0.5) / scale
    radius2 = x * x + y * y
    shading = np.exp(-0.5 * radius2 - 0.12 * radius2**2 + 0.025 * x - 0.015 * y)
    shading = (shading / shading.max()).astype(np.float32)
    rng = np.random.default_rng(seed)
    frames = []
    for k, (rx, ry) in enumerate(((x, y), (-y, x), (-x, -y), (y, -x))):
        light = np.exp(0.12 * rx - 0.08 * ry + 0.025 * rx * ry)
        gain = (1.0, 1.01, 0.99, 1.005)[k]
        image = 0.65 * gain * shading * light + rng.normal(0.0, 0.0005, shading.shape)
        frames.append(np.clip(image, 0.0, 1.0).astype(np.float32))
    return np.stack(frames), shading
