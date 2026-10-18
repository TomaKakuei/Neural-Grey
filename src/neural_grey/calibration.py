"""Portable lens-shading maps and correction of later linear images."""

from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np


def valid_map(value):
    value = np.asarray(value)
    if (
        value.ndim != 2
        or min(value.shape) < 2
        or np.iscomplexobj(value)
        or not np.isfinite(value).all()
        or np.any(value <= 0)
    ):
        raise ValueError("Shading must be a finite, strictly positive 2D map")
    return value.astype(np.float32, copy=True)


def correct(image, shading, *, minimum_transmittance=0.05):
    """Divide linear intensities by shading; preserve range without clipping."""
    image = np.asarray(image)
    shading = valid_map(shading)
    if image.ndim not in (2, 3) or min(image.shape[:2]) < 2:
        raise ValueError("Expected H x W or H x W x channels")
    if (
        not np.issubdtype(image.dtype, np.number)
        or np.iscomplexobj(image)
        or not np.isfinite(image).all()
    ):
        raise ValueError("Image must contain finite real linear intensities")
    if not np.isfinite(minimum_transmittance) or not 0 < minimum_transmittance <= 1:
        raise ValueError("minimum_transmittance must be in (0, 1]")
    h, w = image.shape[:2]
    if not np.isclose(w / h, shading.shape[1] / shading.shape[0], rtol=0.01):
        raise ValueError(
            "Image and calibration must have the same sensor aspect ratio and crop"
        )
    if shading.shape != (h, w):
        shading = cv2.resize(shading, (w, h), interpolation=cv2.INTER_LINEAR)
    denominator = np.maximum(shading, minimum_transmittance)
    if image.ndim == 3:
        denominator = denominator[..., None]
    return image.astype(np.float32) / denominator


@dataclass
class Calibration:
    shading: np.ndarray
    metadata: dict
    base_joint: np.ndarray | None = None
    stabilized: np.ndarray | None = None

    def __post_init__(self):
        self.shading = valid_map(self.shading)

    def correct(self, image, *, minimum_transmittance=0.05):
        return correct(image, self.shading, minimum_transmittance=minimum_transmittance)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            shading=self.shading,
            metadata=np.array(json.dumps(self.metadata, allow_nan=False)),
            format_version=np.array(1),
        )
        for key in ("base_joint", "stabilized"):
            if getattr(self, key) is not None:
                arrays[key] = getattr(self, key)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if int(z["format_version"]) != 1:
                raise ValueError("Unsupported Neural Grey calibration file")
            return cls(
                z["shading"],
                json.loads(str(z["metadata"])),
                z["base_joint"].copy() if "base_joint" in z else None,
                z["stabilized"].copy() if "stabilized" in z else None,
            )
