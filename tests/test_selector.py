import unittest

import numpy as np

from neural_grey.selector import (
    apply_focal_conditioned_gauge,
    apply_radial_gauge_head,
    d2_low_frequency_stabilize,
    normalize,
    radial_coordinates,
)


class RadialGaugeHeadTest(unittest.TestCase):
    def setUp(self):
        radius = radial_coordinates((63, 91))
        theta = np.arctan2(*np.indices(radius.shape, dtype=np.float32))
        self.field = normalize(
            np.exp(-0.42 * radius**2 - 0.08 * radius**4 + 0.025 * np.cos(2 * theta))
        )

    def test_long_lens_uses_small_continuous_update(self):
        baseline = normalize(self.field)
        actual = apply_radial_gauge_head(self.field, 200.0)
        self.assertGreater(float(np.max(np.abs(actual - baseline))), 0.0)
        self.assertLess(float(np.max(np.abs(actual - baseline))), 0.02)

    def test_scale_invariant_and_normalized(self):
        first = apply_radial_gauge_head(self.field, 28.0)
        second = apply_radial_gauge_head(7.3 * self.field, 28.0)
        np.testing.assert_allclose(first, second, atol=2e-6, rtol=2e-6)
        self.assertAlmostEqual(float(np.max(first)), 1.0, places=6)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertTrue(np.all(first > 0.0))

    def test_zero_blend_is_identity(self):
        actual = apply_radial_gauge_head(
            self.field, 20.0, wide_blend=0.0, standard_blend=0.0
        )
        np.testing.assert_allclose(actual, normalize(self.field), atol=2e-6, rtol=2e-6)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_radial_gauge_head(self.field, 0.0)
        with self.assertRaises(ValueError):
            apply_radial_gauge_head(self.field, 28.0, wide_blend=1.1)

    def test_full_head_long_lens_remains_close_to_d2(self):
        stabilized = d2_low_frequency_stabilize(
            self.field, sigma=10.0 * max(self.field.shape) / 256.0
        )
        actual = apply_focal_conditioned_gauge(self.field, 200.0)
        self.assertGreater(float(np.max(np.abs(actual - stabilized))), 0.0)
        self.assertLess(float(np.max(np.abs(actual - stabilized))), 0.02)

    def test_full_head_rejects_invalid_reference_size(self):
        with self.assertRaises(ValueError):
            apply_focal_conditioned_gauge(self.field, 28.0, d2_reference_long_edge=0.0)

    def test_invalid_continuous_taper_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_radial_gauge_head(
                self.field, 28.0, wide_taper_reference_diagonals=0.0
            )
        with self.assertRaises(ValueError):
            apply_radial_gauge_head(self.field, 28.0, standard_taper_power=0.0)

    def test_invalid_d2_shrinkage_is_rejected(self):
        with self.assertRaises(ValueError):
            d2_low_frequency_stabilize(self.field, energy_threshold=0.0)


if __name__ == "__main__":
    unittest.main()
