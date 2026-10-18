import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import torch
from neural_grey import Calibration, FitConfig, FitSession
from neural_grey.cli import main
from neural_grey.data import load_captures, synthetic_case
from neural_grey.preprocessing import CaptureSet


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cv2.setNumThreads(1)

    def test_resume_preserves_training_and_export(self):
        captures, _ = synthetic_case(height=24, width=32)
        config = FitConfig(epochs=2, sample_count=512, batch_size=128)
        full = FitSession(captures, config=config, device="cpu", focal_length_mm=28.0)
        full.run(log_every=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.pt"
            interrupted = FitSession(
                captures, config=config, device="cpu", focal_length_mm=28.0
            )
            initial = interrupted.model.head_v.weight.detach().clone()
            interrupted.run(stop_after=1, checkpoint=path, log_every=0)
            self.assertFalse(interrupted.complete)
            self.assertFalse(torch.equal(initial, interrupted.model.head_v.weight))
            resumed = FitSession.resume(path, device="cpu").run(log_every=0)
            self.assertTrue(resumed.complete)
            self.assertEqual(resumed.updates, full.updates)
            for key, value in full.model.state_dict().items():
                torch.testing.assert_close(
                    value, resumed.model.state_dict()[key], atol=0, rtol=0
                )
            np.testing.assert_array_equal(
                full.calibration().shading, resumed.calibration().shading
            )
            final = resumed.calibration()
            final.save(Path(temporary) / "calibration.npz")
            loaded = Calibration.load(Path(temporary) / "calibration.npz")
            np.testing.assert_array_equal(loaded.shading, final.shading)
            self.assertEqual(loaded.metadata, final.metadata)

    def test_inputs_and_configuration_are_checked(self):
        captures, _ = synthetic_case()
        for bad in (
            captures[:3],
            np.zeros_like(captures),
            captures + 2,
            captures.astype(complex),
            captures * np.nan,
        ):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                CaptureSet(bad)
        for kwargs in (
            dict(epochs=0),
            dict(batch_size=0),
            dict(lr_net=-1),
            dict(lambda_phys=-1),
            dict(lambda_light=float("nan")),
        ):
            with self.assertRaises(ValueError):
                FitConfig(**kwargs)
        with self.assertRaises(ValueError):
            FitSession(captures, focal_length_mm=0)

    def test_missing_focal_metadata_uses_stabilized_field(self):
        captures, _ = synthetic_case(height=24, width=32)
        session = FitSession(
            captures, config=FitConfig(epochs=1, sample_count=256), device="cpu"
        )
        session.run(log_every=0)
        result = session.calibration()
        self.assertEqual(result.metadata["selector"], "d2_only")
        np.testing.assert_array_equal(result.shading, result.stabilized)

    def test_capture_loader_uses_only_observations(self):
        captures, truth = synthetic_case()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "captures.npz"
            np.savez(path, captures=captures, ground_truth=np.full_like(truth, np.nan))
            np.testing.assert_array_equal(load_captures(path), captures)
            for angle, image in zip((0, 90, 180, 270), captures):
                np.save(Path(temporary) / f"sim_capture_{angle}.npy", image)
            np.testing.assert_array_equal(load_captures(temporary), captures)

    def test_cli_demo_resume_and_correction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "run"
            main(["demo", "--device", "cpu", "--output", str(out), "--stop-after", "1"])
            self.assertFalse(json.loads((out / "fit.json").read_text())["complete"])
            main(
                [
                    "resume",
                    str(out / "training.pt"),
                    "--device",
                    "cpu",
                    "--output",
                    str(out),
                ]
            )
            self.assertTrue(json.loads((out / "fit.json").read_text())["complete"])
            shading = Calibration.load(out / "calibration.npz").shading
            np.save(root / "image.npy", shading)
            main(
                [
                    "correct",
                    str(out / "calibration.npz"),
                    str(root / "image.npy"),
                    "--output",
                    str(root / "corrected.npy"),
                ]
            )
            np.testing.assert_allclose(np.load(root / "corrected.npy"), 1.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
