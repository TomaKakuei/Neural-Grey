"""Four-roll preparation, fusion and coordinate sampling."""

from typing import Optional
import numpy as np
import cv2
import torch


class CaptureSet:
    def __init__(self, captures):
        array = np.asarray(captures)
        if array.ndim != 3 or array.shape[0] != 4 or min(array.shape[1:]) < 8:
            raise ValueError(
                "Expected four equally sized linear images: (4, H, W), H/W >= 8"
            )
        if (
            not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)
            or not np.isfinite(array).all()
        ):
            raise ValueError("Captures must be finite real intensities")
        if array.min() < 0 or array.max() > 1:
            raise ValueError(
                "Captures must be black-subtracted linear intensities in [0, 1]"
            )
        if np.any(np.percentile(array, 95, axis=(1, 2)) <= 1e-6):
            raise ValueError("Every capture must contain usable nonzero illumination")
        caps = [np.asarray(c, dtype=np.float32).copy() for c in array]
        shapes = {c.shape for c in caps}
        if len(shapes) != 1:
            raise ValueError("All 4 captures must have identical shape.")
        h, w = caps[0].shape

        self.raw_captures = [np.clip(c, 0.0, 1.0) for c in caps]

        # Step 1: exposure normalization by robust high percentile.
        p95 = np.array(
            [np.percentile(c, 95.0) for c in self.raw_captures], dtype=np.float32
        )
        target = float(np.median(p95))
        exp_norm = []
        for c, p in zip(self.raw_captures, p95):
            s = target / max(float(p), 1e-6)
            exp_norm.append(
                np.clip(c * np.float32(s), 0.0, 1.0).astype(np.float32, copy=False)
            )

        # Step 2: conservative subpixel registration for fusion only.
        # We only accept tiny shifts; large estimates are likely caused by rotated lighting pattern.
        ref = cv2.GaussianBlur(exp_norm[0], (0, 0), 12).astype(np.float32)
        reg_for_fusion = [exp_norm[0]]
        max_shift = 2.5
        for i in range(1, 4):
            src = cv2.GaussianBlur(exp_norm[i], (0, 0), 12).astype(np.float32)
            (dx, dy), _ = cv2.phaseCorrelate(ref, src)
            if np.isfinite(dx) and np.isfinite(dy) and np.hypot(dx, dy) <= max_shift:
                reg_for_fusion.append(
                    self._shift_image(exp_norm[i], dx=-float(dx), dy=-float(dy))
                )
            else:
                reg_for_fusion.append(exp_norm[i])

        # Step 3: robust fusion target for V anchor.
        stack = np.stack(reg_for_fusion, axis=0).astype(np.float32, copy=False)
        fused = self._robust_fuse(stack)

        # Keep decomposition captures unshifted (only exposure-normalized).
        self.captures = [np.clip(c, 0.0, 1.0) for c in exp_norm]
        self.v_fused = np.clip(fused, 0.0, 1.0).astype(np.float32, copy=False)
        self.max_brightness = float(max(np.percentile(stack, 99.8), 1e-6))
        self.h, self.w = h, w

    @staticmethod
    def _estimate_center(img: np.ndarray):
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

    @staticmethod
    def _shift_image(img: np.ndarray, dx: float, dy: float):
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return cv2.warpAffine(
            img,
            m,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _robust_fuse(captures: np.ndarray):
        # captures shape: [4,H,W], trimmed mean drops min/max per pixel.
        s = np.sort(captures, axis=0)
        return np.mean(s[1:3], axis=0).astype(np.float32, copy=False)

    def _estimate_simple_cos4_prior(self):
        img = np.clip(
            self.v_fused / np.float32(max(self.max_brightness, 1e-6)), 0.0, 1.0
        ).astype(np.float32, copy=False)
        cx_est, cy_est = self._estimate_center(img)
        h, w = img.shape
        half_diag_px = float(np.hypot((w - 1) * 0.5, (h - 1) * 0.5))
        yy, xx = np.indices(img.shape, dtype=np.float32)
        r_norm = np.sqrt(
            ((xx - np.float32(cx_est)) / np.float32(max(half_diag_px, 1e-6))) ** 2
            + ((yy - np.float32(cy_est)) / np.float32(max(half_diag_px, 1e-6))) ** 2,
            dtype=np.float32,
        )

        f_grid = np.geomspace(0.15, 8.0, 512).astype(np.float32)
        best_f = float(f_grid[0])
        best_i0 = 1.0
        best_mse = float("inf")
        for f_norm in f_grid:
            base = 1.0 / (1.0 + (r_norm / np.float32(f_norm)) ** 2) ** 2
            denom = float(np.sum(base * base))
            i0 = float(np.sum(img * base) / max(denom, 1e-12))
            pred = np.clip(np.float32(i0) * base, 0.0, 1.0).astype(
                np.float32, copy=False
            )
            mse = float(np.mean((pred - img) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_f = float(f_norm)
                best_i0 = float(i0)

        cx0, cy0 = self.img_center_px
        s = max(float(self.sensor_half_diag_px), 1e-6)
        cx_norm = np.clip((cx_est - cx0) / s, -0.78, 0.78)
        cy_norm = np.clip((cy_est - cy0) / s, -0.78, 0.78)
        return {
            "f_norm": best_f,
            "i0": best_i0,
            "cx_norm": float(cx_norm),
            "cy_norm": float(cy_norm),
            "fit_to_anchor_mse": best_mse,
        }

    def _build_training_arrays(self, sample_count: Optional[int], seed: int):
        h, w = self.h, self.w
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        half_diag_px = float(np.hypot(cx, cy))

        yy, xx = np.indices((h, w), dtype=np.float32)
        x_norm = ((xx - np.float32(cx)) / np.float32(max(half_diag_px, 1e-6))).ravel()
        y_norm = ((yy - np.float32(cy)) / np.float32(max(half_diag_px, 1e-6))).ravel()

        y4 = np.stack([c.ravel() for c in self.captures], axis=1).astype(np.float32)
        y4 = np.clip(y4 / np.float32(self.max_brightness), 0.0, 1.0)
        vf = np.clip(
            self.v_fused.ravel() / np.float32(self.max_brightness), 0.0, 1.0
        ).astype(np.float32)

        n = x_norm.shape[0]
        n_use = n if sample_count is None else min(int(sample_count), n)
        if n_use < n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=n_use, replace=False)
            x_norm = x_norm[idx]
            y_norm = y_norm[idx]
            y4 = y4[idx]
            vf = vf[idx]
        xy = np.column_stack((x_norm, y_norm)).astype(np.float32)

        self.sensor_half_diag_px = half_diag_px
        self.img_center_px = (cx, cy)
        return xy, y4, vf

    def _full_xy(self):
        cx = (self.w - 1) * 0.5
        cy = (self.h - 1) * 0.5
        scale = max(float(np.hypot(cx, cy)), 1e-6)
        yy, xx = np.indices((self.h, self.w), dtype=np.float32)
        x = (xx - np.float32(cx)) / np.float32(scale)
        y = (yy - np.float32(cy)) / np.float32(scale)
        return x.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def _rotated_coords(x: torch.Tensor, y: torch.Tensor):
        # For np.rot90 convention:
        # k=0: (x,y)
        # k=1: (-y, x)
        # k=2: (-x,-y)
        # k=3: ( y,-x)
        return (
            (x, y),
            (-y, x),
            (-x, -y),
            (y, -x),
        )
