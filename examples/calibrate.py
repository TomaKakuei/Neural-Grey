"""Small API example; replace synthetic_case with your four linear captures."""

from pathlib import Path
import numpy as np
from neural_grey import FitConfig, FitSession, Calibration
from neural_grey.data import synthetic_case

captures, _ = synthetic_case()
output = Path("outputs/api-example")
output.mkdir(parents=True, exist_ok=True)
session = FitSession(
    captures,
    focal_length_mm=28.0,
    config=FitConfig(epochs=2, sample_count=1024, batch_size=256),
)
session.run(checkpoint=output / "training.pt")
session.calibration().save(output / "calibration.npz")
loaded = Calibration.load(output / "calibration.npz")
np.save(output / "corrected.npy", loaded.correct(captures[0]))
print(output)
