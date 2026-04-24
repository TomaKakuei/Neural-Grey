import unittest
from unittest.mock import patch

import numpy as np

from MLMask import preprocess_rotated_raw_bayer


class FakeRaw:
    def __init__(
        self,
        raw_image_visible,
        raw_pattern,
        color_desc,
        black_level_per_channel,
        camera_white_level_per_channel,
        white_level,
    ):
        self.raw_image_visible = raw_image_visible
        self.raw_pattern = raw_pattern
        self.color_desc = color_desc
        self.black_level_per_channel = black_level_per_channel
        self.camera_white_level_per_channel = camera_white_level_per_channel
        self.white_level = white_level

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def make_raw_image(pattern, r_plane, g1_plane, g2_plane, b_plane):
    h, w = r_plane.shape
    raw = np.zeros((h * 2, w * 2), dtype=np.uint16)

    color_sites = {}
    for row in range(2):
        for col in range(2):
            color_sites[int(pattern[row, col])] = (row, col)

    b_site = color_sites[2]
    g1_site = color_sites[1]
    g2_site = color_sites[3]
    r_site = color_sites[0]

    raw[b_site[0]::2, b_site[1]::2] = b_plane
    raw[g1_site[0]::2, g1_site[1]::2] = g1_plane
    raw[g2_site[0]::2, g2_site[1]::2] = g2_plane
    raw[r_site[0]::2, r_site[1]::2] = r_plane
    return raw


class TestPreprocessRotatedRawBayer(unittest.TestCase):
    def _build_fake_paths(self, count):
        return [f"shot_{idx}.dng" for idx in range(count)]

    def test_requires_exactly_four_paths(self):
        with self.assertRaises(ValueError):
            preprocess_rotated_raw_bayer(["a.dng", "b.dng", "c.dng"])

    def test_channel_extraction_averaging_and_normalization(self):
        pattern = np.array([[2, 3], [1, 0]], dtype=np.int32)
        color_desc = b"RGBG"
        black = [0, 0, 0, 0]
        white = [1000, 1000, 1000, 1000]

        paths = self._build_fake_paths(4)
        raws = {}
        r_frames = []
        g_frames = []
        b_frames = []
        for k, path in enumerate(paths):
            r = np.array([[100 + 10 * k, 200 + 5 * k], [300 + 3 * k, 400 + 7 * k]], dtype=np.float32)
            g1 = np.array([[40 + 2 * k, 80 + 3 * k], [120 + 4 * k, 160 + 5 * k]], dtype=np.float32)
            g2 = np.array([[60 + 6 * k, 100 + 7 * k], [140 + 8 * k, 180 + 9 * k]], dtype=np.float32)
            b = np.array([[500 + 9 * k, 450 + 8 * k], [350 + 7 * k, 250 + 6 * k]], dtype=np.float32)

            raw_img = make_raw_image(pattern, r.astype(np.uint16), g1.astype(np.uint16), g2.astype(np.uint16), b.astype(np.uint16))
            raws[path] = FakeRaw(raw_img, pattern, color_desc, black, white, 1000)

            r_frames.append(r / 1000.0)
            g_frames.append((0.5 * (g1 + g2)) / 1000.0)
            b_frames.append(b / 1000.0)

        expected_r = np.mean(np.stack(r_frames, axis=0), axis=0)
        expected_g = np.mean(np.stack(g_frames, axis=0), axis=0)
        expected_b = np.mean(np.stack(b_frames, axis=0), axis=0)
        expected_r /= np.max(expected_r)
        expected_g /= np.max(expected_g)
        expected_b /= np.max(expected_b)

        with patch("MLMask.os.path.isfile", return_value=True), patch("MLMask.rawpy.imread", side_effect=lambda p: raws[p]):
            i_r, i_g, i_b = preprocess_rotated_raw_bayer(paths)

        self.assertEqual(i_r.shape, (2, 2))
        self.assertEqual(i_g.shape, (2, 2))
        self.assertEqual(i_b.shape, (2, 2))
        self.assertEqual(i_r.dtype, np.float32)
        self.assertEqual(i_g.dtype, np.float32)
        self.assertEqual(i_b.dtype, np.float32)

        np.testing.assert_allclose(i_r, expected_r.astype(np.float32), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(i_g, expected_g.astype(np.float32), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(i_b, expected_b.astype(np.float32), rtol=1e-6, atol=1e-6)

        self.assertAlmostEqual(float(np.max(i_r)), 1.0, places=6)
        self.assertAlmostEqual(float(np.max(i_g)), 1.0, places=6)
        self.assertAlmostEqual(float(np.max(i_b)), 1.0, places=6)
        self.assertGreaterEqual(float(np.min(i_r)), 0.0)
        self.assertGreaterEqual(float(np.min(i_g)), 0.0)
        self.assertGreaterEqual(float(np.min(i_b)), 0.0)

    def test_shape_mismatch_raises(self):
        pattern = np.array([[2, 3], [1, 0]], dtype=np.int32)
        color_desc = b"RGBG"
        black = [0, 0, 0, 0]
        white = [1000, 1000, 1000, 1000]

        paths = self._build_fake_paths(4)
        raw_a = make_raw_image(
            pattern,
            np.array([[100, 200], [300, 400]], dtype=np.uint16),
            np.array([[10, 20], [30, 40]], dtype=np.uint16),
            np.array([[11, 21], [31, 41]], dtype=np.uint16),
            np.array([[500, 600], [700, 800]], dtype=np.uint16),
        )
        raw_b = make_raw_image(
            pattern,
            np.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]], dtype=np.uint16),
            np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint16),
            np.array([[11, 21, 31], [41, 51, 61], [71, 81, 91]], dtype=np.uint16),
            np.array([[500, 600, 700], [800, 900, 950], [960, 970, 980]], dtype=np.uint16),
        )

        raws = {
            paths[0]: FakeRaw(raw_a, pattern, color_desc, black, white, 1000),
            paths[1]: FakeRaw(raw_b, pattern, color_desc, black, white, 1000),
            paths[2]: FakeRaw(raw_a, pattern, color_desc, black, white, 1000),
            paths[3]: FakeRaw(raw_a, pattern, color_desc, black, white, 1000),
        }

        with patch("MLMask.os.path.isfile", return_value=True), patch("MLMask.rawpy.imread", side_effect=lambda p: raws[p]):
            with self.assertRaises(ValueError):
                preprocess_rotated_raw_bayer(paths)

    def test_pattern_mismatch_raises(self):
        pattern_a = np.array([[2, 3], [1, 0]], dtype=np.int32)
        pattern_b = np.array([[0, 1], [3, 2]], dtype=np.int32)
        color_desc = b"RGBG"
        black = [0, 0, 0, 0]
        white = [1000, 1000, 1000, 1000]

        raw_img = make_raw_image(
            pattern_a,
            np.array([[100, 200], [300, 400]], dtype=np.uint16),
            np.array([[10, 20], [30, 40]], dtype=np.uint16),
            np.array([[11, 21], [31, 41]], dtype=np.uint16),
            np.array([[500, 600], [700, 800]], dtype=np.uint16),
        )

        paths = self._build_fake_paths(4)
        raws = {
            paths[0]: FakeRaw(raw_img, pattern_a, color_desc, black, white, 1000),
            paths[1]: FakeRaw(raw_img, pattern_b, color_desc, black, white, 1000),
            paths[2]: FakeRaw(raw_img, pattern_a, color_desc, black, white, 1000),
            paths[3]: FakeRaw(raw_img, pattern_a, color_desc, black, white, 1000),
        }

        with patch("MLMask.os.path.isfile", return_value=True), patch("MLMask.rawpy.imread", side_effect=lambda p: raws[p]):
            with self.assertRaises(ValueError):
                preprocess_rotated_raw_bayer(paths)

    def test_white_level_fallback_when_camera_white_missing(self):
        pattern = np.array([[2, 3], [1, 0]], dtype=np.int32)
        color_desc = b"RGBG"
        black = [100, 100, 100, 100]

        paths = self._build_fake_paths(4)
        raws = {}
        r_frames = []
        g_frames = []
        b_frames = []
        for path in paths:
            r = np.array([[220, 320], [520, 620]], dtype=np.float32)
            g1 = np.array([[180, 280], [380, 480]], dtype=np.float32)
            g2 = np.array([[200, 300], [400, 500]], dtype=np.float32)
            b = np.array([[350, 450], [650, 750]], dtype=np.float32)
            raw_img = make_raw_image(pattern, r.astype(np.uint16), g1.astype(np.uint16), g2.astype(np.uint16), b.astype(np.uint16))
            raws[path] = FakeRaw(
                raw_img,
                pattern,
                color_desc,
                black,
                camera_white_level_per_channel=None,
                white_level=1000,
            )

            r_frames.append((r - 100.0) / 900.0)
            g_frames.append((0.5 * (g1 + g2) - 100.0) / 900.0)
            b_frames.append((b - 100.0) / 900.0)

        expected_r = np.mean(np.stack(r_frames, axis=0), axis=0)
        expected_g = np.mean(np.stack(g_frames, axis=0), axis=0)
        expected_b = np.mean(np.stack(b_frames, axis=0), axis=0)
        expected_r = np.clip(expected_r, 0.0, 1.0)
        expected_g = np.clip(expected_g, 0.0, 1.0)
        expected_b = np.clip(expected_b, 0.0, 1.0)
        expected_r /= np.max(expected_r)
        expected_g /= np.max(expected_g)
        expected_b /= np.max(expected_b)

        with patch("MLMask.os.path.isfile", return_value=True), patch("MLMask.rawpy.imread", side_effect=lambda p: raws[p]):
            i_r, i_g, i_b = preprocess_rotated_raw_bayer(paths)

        np.testing.assert_allclose(i_r, expected_r.astype(np.float32), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(i_g, expected_g.astype(np.float32), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(i_b, expected_b.astype(np.float32), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
