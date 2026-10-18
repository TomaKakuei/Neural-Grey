"""Practical vignetting calibration from four casual camera rolls."""

from .training import FitConfig, FitSession
from .calibration import Calibration, correct


def calibrate(
    captures,
    *,
    focal_length_mm=None,
    sensor_diagonal_mm=43.3,
    config=None,
    seed=42,
    device="auto",
):
    """Fit one configuration from scratch and return its reusable shading map."""
    session = FitSession(
        captures,
        focal_length_mm=focal_length_mm,
        sensor_diagonal_mm=sensor_diagonal_mm,
        config=config,
        seed=seed,
        device=device,
    )
    session.run()
    return session.calibration()


__all__ = ["FitConfig", "FitSession", "Calibration", "calibrate", "correct"]
