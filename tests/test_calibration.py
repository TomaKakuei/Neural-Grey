import unittest
import numpy as np
from neural_grey import correct
from neural_grey.raw import (
    _channel_site_map,
    _channel_black_white_levels,
    _extract_linearized_channel,
)


class CalibrationTests(unittest.TestCase):
    def test_linear_rgb_correction_preserves_unclipped_intensity(self):
        field = np.linspace(0.3, 1.0, 96, dtype=np.float32).reshape(8, 12)
        scene = np.ones((8, 12, 3), dtype=np.float32) * [0.2, 1.0, 1.3]
        observed = scene * field[..., None]
        np.testing.assert_allclose(correct(observed, field), scene, atol=2e-7)

    def test_floor_limits_amplification(self):
        field = np.ones((8, 12), dtype=np.float32) * 0.001
        np.testing.assert_allclose(correct(np.ones_like(field), field), 20.0)

    def test_aspect_ratio_and_invalid_maps(self):
        with self.assertRaises(ValueError):
            correct(np.ones((8, 8)), np.ones((8, 12)))
        for bad in (np.zeros((8, 12)), np.full((8, 12), np.nan)):
            with self.assertRaises(ValueError):
                correct(np.ones((8, 12)), bad)

    def test_raw_channel_levels_and_bayer_mapping(self):
        class Sensor:
            black_level_per_channel = [64.0, 65.0, 66.0, 67.0]
            camera_white_level_per_channel = [1024.0, 1024.0, 0.0, 1024.0]
            white_level = 1024.0

        black, white = _channel_black_white_levels(Sensor(), 4)
        self.assertEqual(float(white[2]), 1024.0)
        sites = _channel_site_map([[0, 1], [3, 2]], b"RGBG")
        self.assertEqual(len(sites["G"]), 2)
        raw = np.ones((8, 12), dtype=np.float32) * 544.0
        plane = _extract_linearized_channel(raw, 0, 0, 0, black, white)
        np.testing.assert_array_equal(plane, np.full((4, 6), 0.5))


if __name__ == "__main__":
    unittest.main()
