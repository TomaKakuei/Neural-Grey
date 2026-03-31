"""
Traditional Mask V2 (commented copy)

This version extends V1 to multi-frame joint modeling.
Main upgrades:
1) Support 4-frame stack input (0/90/180/270) with compatibility for single frame.
2) Add per-frame nuisance parameters:
   gain, bias, tx, ty, theta, scale.
3) Add shared geometry and deformable radius skeleton.
4) Build a joint residual forward path for deterministic fitting.

Notes:
- Kept backward-compatible interfaces and legacy loaders.
- Still no ML framework; NumPy-only classical pipeline.
"""
import argparse
from pathlib import Path
import numpy as np


class TraditionalPolynomialBaseline:
    """
    Deterministic geometric baseline with backward compatibility:
    - Supports single-frame input and multi-frame stack input.
    - Shared geometric vignette field + per-frame nuisance parameters.
    - Elliptical radius with low-DOF deformable correction skeleton.
    """

    def __init__(
        self,
        data_matrix,
        ground_truth=None,
        seed=42,
        max_fit_samples=200_000,
        radial_bins=320,
        deformation_clip=0.25,
    ):
        self.seed = int(seed)
        self.max_fit_samples = int(max_fit_samples)
        self.radial_bins = int(radial_bins)
        self.deformation_clip = float(deformation_clip)

        self.captures = self._coerce_captures(data_matrix)
        self.num_captures = len(self.captures)
        if self.num_captures < 1:
            raise RuntimeError("At least one capture must be provided.")
        self.h, self.w = self.captures[0].shape
        self._validate_capture_shapes(self.captures)

        # Legacy single-image handle (now a robust stack reference image when multi-frame input is provided).
        self.img_data = self._build_reference_image()
        self.ground_truth = None if ground_truth is None else np.asarray(ground_truth, dtype=np.float32)
        if self.ground_truth is not None and self.ground_truth.shape != (self.h, self.w):
            raise RuntimeError(
                f"ground_truth shape mismatch: expected {(self.h, self.w)}, got {self.ground_truth.shape}"
            )

        self.optical_center = None
        self.max_brightness = 0.0
        self.radial_profile = None
        self.popt = None  # legacy: [a, b, c]
        self.poly_coeffs = None  # full: [c0, c2, c4, c6]
        self.predicted_surface = None
        self.max_r = None
        self.geometry = None  # dict(cx, cy, sx, sy, phi)
        self.shared_geometry = None
        self.illumination_coeffs = None  # low-order 2D bias: [1, x, y, x2, y2, xy]
        self.deform_coeffs = np.zeros(9, dtype=np.float64)

        self.frame_params = []
        self.initialize_frame_params()

        self._fit_idx = None
        self._fit_x = None
        self._fit_y = None
        self._fit_xn = None
        self._fit_yn = None
        self._fit_i = None

    @staticmethod
    def _coerce_captures(data_matrix):
        if isinstance(data_matrix, (list, tuple)):
            return [np.asarray(frame, dtype=np.float32) for frame in data_matrix]

        arr = np.asarray(data_matrix, dtype=np.float32)
        if arr.ndim == 2:
            return [arr]
        if arr.ndim != 3:
            raise RuntimeError("data_matrix must be 2D, list/tuple of 2D, or a 3D stack.")

        # Support [N, H, W] and [H, W, N].
        if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
            return [arr[i, :, :].astype(np.float32, copy=False) for i in range(arr.shape[0])]
        if arr.shape[2] <= arr.shape[0] and arr.shape[2] <= arr.shape[1]:
            return [arr[:, :, i].astype(np.float32, copy=False) for i in range(arr.shape[2])]
        raise RuntimeError("Unable to infer capture axis from 3D input stack.")

    def _validate_capture_shapes(self, captures):
        shape0 = captures[0].shape
        if captures[0].ndim != 2:
            raise RuntimeError(f"Each capture must be 2D, got shape {shape0} at frame 0.")
        for k, frame in enumerate(captures):
            if frame.ndim != 2:
                raise RuntimeError(f"Capture {k} must be 2D, got ndim={frame.ndim}.")
            if frame.shape != shape0:
                raise RuntimeError(
                    f"All captures must have same shape. frame 0={shape0}, frame {k}={frame.shape}."
                )
            if not np.all(np.isfinite(frame)):
                raise RuntimeError(f"Capture {k} contains NaN/Inf.")

    def _build_reference_image(self):
        if self.num_captures == 1:
            return self.captures[0].astype(np.float32, copy=False)

        ref = np.zeros((self.h, self.w), dtype=np.float32)
        for frame in self.captures:
            lo = float(np.percentile(frame, 2.0))
            hi = float(np.percentile(frame, 99.8))
            denom = max(hi - lo, 1e-6)
            ref += np.clip((frame - lo) / denom, 0.0, 1.0).astype(np.float32, copy=False)
        ref /= np.float32(self.num_captures)
        return ref

    def initialize_frame_params(self):
        self.frame_params = []
        for _ in range(self.num_captures):
            self.frame_params.append(
                {
                    "gain": 1.0,
                    "bias": 0.0,
                    "tx": 0.0,
                    "ty": 0.0,
                    "theta": 0.0,
                    "scale": 1.0,
                }
            )
        return self.frame_params

    def unpack_frame_params(self, frame_index, frame_params=None):
        if frame_index < 0 or frame_index >= self.num_captures:
            raise RuntimeError(f"frame_index out of range: {frame_index}")

        source = self.frame_params if frame_params is None else frame_params
        if isinstance(source, dict):
            p = source
        elif isinstance(source, (list, tuple)):
            if len(source) <= frame_index:
                raise RuntimeError(f"frame_params length {len(source)} < required index {frame_index + 1}")
            p = source[frame_index]
        else:
            raise RuntimeError("frame_params must be None, dict, list, or tuple.")

        required = ("gain", "bias", "tx", "ty", "theta", "scale")
        missing = [key for key in required if key not in p]
        if missing:
            raise RuntimeError(f"frame_params missing keys: {missing}")

        return (
            float(p["gain"]),
            float(p["bias"]),
            float(p["tx"]),
            float(p["ty"]),
            float(p["theta"]),
            float(p["scale"]),
        )

    def apply_similarity_transform(self, x, y, tx, ty, theta, scale):
        """Map frame pixel coordinates into shared-field coordinates."""
        cx_ref = np.float32((self.w - 1) * 0.5)
        cy_ref = np.float32((self.h - 1) * 0.5)
        xr = x - cx_ref
        yr = y - cy_ref
        c = np.float32(np.cos(theta))
        s = np.float32(np.sin(theta))
        xs = np.float32(scale) * (c * xr - s * yr)
        ys = np.float32(scale) * (s * xr + c * yr)
        xw = xs + cx_ref + np.float32(tx)
        yw = ys + cy_ref + np.float32(ty)
        return xw, yw

    @staticmethod
    def _poly_eval(r_norm, coeffs):
        r2 = r_norm * r_norm
        r4 = r2 * r2
        r6 = r4 * r2
        return coeffs[0] + coeffs[1] * r2 + coeffs[2] * r4 + coeffs[3] * r6

    @staticmethod
    def _solve_linear_system(matrix, vector):
        """Small dense linear solver via Gaussian elimination (no LAPACK dependency)."""
        n = len(vector)
        aug = [list(map(float, row)) + [float(vector[i])] for i, row in enumerate(matrix)]

        for i in range(n):
            pivot = i
            max_abs = abs(aug[i][i])
            for j in range(i + 1, n):
                v = abs(aug[j][i])
                if v > max_abs:
                    max_abs = v
                    pivot = j
            if max_abs < 1e-14:
                return None
            if pivot != i:
                aug[i], aug[pivot] = aug[pivot], aug[i]

            div = aug[i][i]
            for k in range(i, n + 1):
                aug[i][k] /= div

            for j in range(n):
                if j == i:
                    continue
                factor = aug[j][i]
                if factor == 0.0:
                    continue
                for k in range(i, n + 1):
                    aug[j][k] -= factor * aug[i][k]

        return np.array([aug[i][n] for i in range(n)], dtype=np.float64)

    @staticmethod
    def _compute_max_r(h, w, cx, cy, sx, sy, phi):
        corners_x = np.array([0.0, w - 1.0, 0.0, w - 1.0], dtype=np.float32)
        corners_y = np.array([0.0, 0.0, h - 1.0, h - 1.0], dtype=np.float32)
        dx = corners_x - np.float32(cx)
        dy = corners_y - np.float32(cy)
        c = np.float32(np.cos(phi))
        s = np.float32(np.sin(phi))
        xr = c * dx + s * dy
        yr = -s * dx + c * dy
        rc = np.sqrt((np.float32(sx) * xr) ** 2 + (np.float32(sy) * yr) ** 2, dtype=np.float32)
        return float(max(np.max(rc), 1e-6))

    @staticmethod
    def _elliptical_r_norm(x, y, cx, cy, sx, sy, phi, max_r):
        dx = x - np.float32(cx)
        dy = y - np.float32(cy)
        c = np.float32(np.cos(phi))
        s = np.float32(np.sin(phi))
        xr = c * dx + s * dy
        yr = -s * dx + c * dy
        r = np.sqrt((np.float32(sx) * xr) ** 2 + (np.float32(sy) * yr) ** 2, dtype=np.float32)
        return r / np.float32(max_r)

    @staticmethod
    def _deformation_field(xn, yn, coeffs):
        """
        Low-order deformation basis:
        [x, y, x^2, y^2, xy, x^3, x^2y, xy^2, y^3]
        """
        c = np.asarray(coeffs, dtype=np.float32).ravel()
        if c.size != 9:
            raise RuntimeError(f"deform coeff size must be 9, got {c.size}")

        x = np.asarray(xn, dtype=np.float32)
        y = np.asarray(yn, dtype=np.float32)
        delta = (
            c[0] * x
            + c[1] * y
            + c[2] * (x * x)
            + c[3] * (y * y)
            + c[4] * (x * y)
            + c[5] * (x * x * x)
            + c[6] * (x * x * y)
            + c[7] * (x * y * y)
            + c[8] * (y * y * y)
        )
        return delta.astype(np.float32, copy=False)

    def _deformable_r_norm(self, x, y, cx, cy, sx, sy, phi, deform_coeffs, max_r):
        r_e = self._elliptical_r_norm(x, y, cx, cy, sx, sy, phi, max_r)
        nx = np.float32(max((self.w - 1) * 0.5, 1.0))
        ny = np.float32(max((self.h - 1) * 0.5, 1.0))
        xn = (x - np.float32(cx)) / nx
        yn = (y - np.float32(cy)) / ny

        coeffs = self.deform_coeffs if deform_coeffs is None else deform_coeffs
        delta = self._deformation_field(xn, yn, coeffs)
        delta = np.clip(delta, -self.deformation_clip, self.deformation_clip)
        r_d = r_e * (1.0 + delta)
        r_d = np.maximum(r_d, 0.0)
        return r_d.astype(np.float32, copy=False)

    @staticmethod
    def _bilinear_sample(surface, x, y, fill_value=0.0):
        h, w = surface.shape
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
        out = np.full_like(x, np.float32(fill_value), dtype=np.float32)
        if not np.any(valid):
            return out

        xv = x[valid]
        yv = y[valid]
        x0v = x0[valid]
        y0v = y0[valid]
        x1v = x1[valid]
        y1v = y1[valid]

        wa = (x1v - xv) * (y1v - yv)
        wb = (xv - x0v) * (y1v - yv)
        wc = (x1v - xv) * (yv - y0v)
        wd = (xv - x0v) * (yv - y0v)
        out[valid] = (
            wa * surface[y0v, x0v]
            + wb * surface[y0v, x1v]
            + wc * surface[y1v, x0v]
            + wd * surface[y1v, x1v]
        ).astype(np.float32, copy=False)
        return out

    def _shared_field_from_coords(self, x, y, geometry=None, deform_coeffs=None):
        geom = self.shared_geometry if geometry is None else geometry
        if geom is None:
            geom = self.geometry
        if geom is None or self.poly_coeffs is None or self.max_r is None:
            raise RuntimeError("Shared field is not ready. Run fit_model() first.")

        r_norm = self._deformable_r_norm(
            x=x,
            y=y,
            cx=geom["cx"],
            cy=geom["cy"],
            sx=geom["sx"],
            sy=geom["sy"],
            phi=geom["phi"],
            deform_coeffs=self.deform_coeffs if deform_coeffs is None else deform_coeffs,
            max_r=max(self.max_r, 1e-6),
        )
        r_norm = np.clip(r_norm, 0.0, 1.0)
        v = self._poly_eval(r_norm, self.poly_coeffs)
        return np.clip(v, 0.0, 1.0).astype(np.float32, copy=False)

    def _prepare_fit_samples(self):
        if self._fit_x is not None:
            return
        rng = np.random.default_rng(self.seed)
        n_all = self.h * self.w
        n_use = min(self.max_fit_samples, n_all)
        if n_use < n_all:
            idx = rng.choice(n_all, size=n_use, replace=False)
        else:
            idx = np.arange(n_all, dtype=np.int64)
        self._fit_idx = idx

        if self.num_captures == 1:
            flat = self.captures[0].ravel()
            fit_i = flat[idx].astype(np.float32, copy=False)
            scale = float(np.percentile(fit_i, 99.8))
            scale = max(scale, 1e-6)
            fit_i = np.clip(fit_i / scale, 0.0, 1.0)
            self.max_brightness = scale
        else:
            fit_stack = []
            scales = []
            for frame in self.captures:
                f = frame.ravel()[idx].astype(np.float32, copy=False)
                lo = float(np.percentile(f, 2.0))
                hi = float(np.percentile(f, 99.8))
                denom = max(hi - lo, 1e-6)
                fit_stack.append(np.clip((f - lo) / denom, 0.0, 1.0))
                scales.append(denom)
            fit_i = np.mean(np.stack(fit_stack, axis=0), axis=0).astype(np.float32, copy=False)
            self.max_brightness = float(max(np.mean(scales), 1e-6))

        self._fit_x = (idx % self.w).astype(np.float32, copy=False)
        self._fit_y = (idx // self.w).astype(np.float32, copy=False)
        cx_ref = np.float32((self.w - 1) * 0.5)
        cy_ref = np.float32((self.h - 1) * 0.5)
        sx_ref = np.float32(max((self.w - 1) * 0.5, 1.0))
        sy_ref = np.float32(max((self.h - 1) * 0.5, 1.0))
        self._fit_xn = (self._fit_x - cx_ref) / sx_ref
        self._fit_yn = (self._fit_y - cy_ref) / sy_ref
        self._fit_i = fit_i

    def find_optical_center(self):
        img = self.img_data
        ds = max(1, int(np.ceil(max(self.h, self.w) / 512.0)))
        img_ds = img[::ds, ::ds]
        yy, xx = np.indices(img_ds.shape, dtype=np.float32)

        floor = float(np.percentile(img_ds, 50.0))
        weights = np.clip(img_ds - floor, 0.0, None)
        weights *= weights
        wsum = float(np.sum(weights))

        if wsum > 1e-8:
            cx_w = float(np.sum(weights * xx) / wsum) * ds
            cy_w = float(np.sum(weights * yy) / wsum) * ds
        else:
            my, mx = np.unravel_index(int(np.argmax(img_ds)), img_ds.shape)
            cx_w = float(mx * ds)
            cy_w = float(my * ds)

        top_th = float(np.percentile(img_ds, 99.2))
        top_mask = img_ds >= top_th
        if np.any(top_mask):
            cx_t = float(np.mean(xx[top_mask])) * ds
            cy_t = float(np.mean(yy[top_mask])) * ds
            cx = 0.65 * cx_w + 0.35 * cx_t
            cy = 0.65 * cy_w + 0.35 * cy_t
        else:
            cx, cy = cx_w, cy_w

        cx = float(np.clip(cx, 0.0, self.w - 1.0))
        cy = float(np.clip(cy, 0.0, self.h - 1.0))
        self.optical_center = (int(round(cx)), int(round(cy)))
        self.max_brightness = float(max(np.percentile(img, 99.8), 1e-6))
        print(f"[-] Robust Center Found at: (X:{self.optical_center[0]}, Y:{self.optical_center[1]})")

    def extract_radial_profile(self):
        if self.optical_center is None:
            self.find_optical_center()
        cx, cy = self.optical_center
        y, x = np.indices((self.h, self.w), dtype=np.float32)
        r = np.sqrt((x - np.float32(cx)) ** 2 + (y - np.float32(cy)) ** 2, dtype=np.float32)

        r_flat = r.ravel()
        intensity_flat = np.clip(self.img_data.ravel() / max(self.max_brightness, 1e-6), 0.0, 1.0)

        r_int = r_flat.astype(np.int32)
        tbin = np.bincount(r_int, weights=intensity_flat)
        nr = np.bincount(r_int)
        valid = nr > 0
        radial_profile_y = tbin[valid] / nr[valid]
        radial_profile_x = np.arange(len(radial_profile_y), dtype=np.float32)
        self.radial_profile = (radial_profile_x, radial_profile_y)
        self._prepare_fit_samples()

    @staticmethod
    def _eval_illumination(xn, yn, coeffs):
        return (
            coeffs[0]
            + coeffs[1] * xn
            + coeffs[2] * yn
            + coeffs[3] * (xn * xn)
            + coeffs[4] * (yn * yn)
            + coeffs[5] * (xn * yn)
        )

    def _fit_illumination_coeffs(self, target_ratio):
        x = self._fit_xn.astype(np.float64, copy=False)
        y = self._fit_yn.astype(np.float64, copy=False)
        t = np.clip(target_ratio.astype(np.float64, copy=False), 0.6, 1.4)

        x0 = np.ones_like(x)
        x1 = x
        x2 = y
        x3 = x * x
        x4 = y * y
        x5 = x * y
        basis = [x0, x1, x2, x3, x4, x5]

        m = []
        for bi in basis:
            row = []
            for bj in basis:
                row.append(float(np.sum(bi * bj)))
            m.append(row)
        b = [float(np.sum(bi * t)) for bi in basis]

        coeffs = self._solve_linear_system(m, b)
        if coeffs is None:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        coeffs[0] = max(coeffs[0], 1e-6)
        return coeffs

    def _fit_poly_for_geometry(self, cx, cy, sx, sy, phi, fit_i):
        max_r = self._compute_max_r(self.h, self.w, cx, cy, sx, sy, phi)
        r_norm = self._deformable_r_norm(
            self._fit_x,
            self._fit_y,
            cx,
            cy,
            sx,
            sy,
            phi,
            self.deform_coeffs,
            max_r,
        )
        r_norm = np.clip(r_norm, 0.0, 1.0)

        bins = self.radial_bins
        b = np.minimum((r_norm * (bins - 1)).astype(np.int32), bins - 1)
        cnt = np.bincount(b, minlength=bins).astype(np.float64)
        sum_r = np.bincount(b, weights=r_norm, minlength=bins).astype(np.float64)
        sum_y = np.bincount(b, weights=fit_i, minlength=bins).astype(np.float64)

        valid = cnt >= 8.0
        if np.count_nonzero(valid) < 24:
            return None, max_r, float("inf"), float("inf")

        r_mean = (sum_r[valid] / cnt[valid]).astype(np.float64)
        y_mean = (sum_y[valid] / cnt[valid]).astype(np.float64)

        r2 = r_mean * r_mean
        r4 = r2 * r2
        r6 = r4 * r2

        x0 = np.ones_like(r2)
        x1 = r2
        x2 = r4
        x3 = r6

        # Normal equations: (X^T X) c = X^T y
        m00 = float(np.sum(x0 * x0))
        m01 = float(np.sum(x0 * x1))
        m02 = float(np.sum(x0 * x2))
        m03 = float(np.sum(x0 * x3))
        m11 = float(np.sum(x1 * x1))
        m12 = float(np.sum(x1 * x2))
        m13 = float(np.sum(x1 * x3))
        m22 = float(np.sum(x2 * x2))
        m23 = float(np.sum(x2 * x3))
        m33 = float(np.sum(x3 * x3))

        b0 = float(np.sum(x0 * y_mean))
        b1 = float(np.sum(x1 * y_mean))
        b2 = float(np.sum(x2 * y_mean))
        b3 = float(np.sum(x3 * y_mean))

        mat = [
            [m00, m01, m02, m03],
            [m01, m11, m12, m13],
            [m02, m12, m22, m23],
            [m03, m13, m23, m33],
        ]
        vec = [b0, b1, b2, b3]
        coeffs = self._solve_linear_system(mat, vec)
        if coeffs is None:
            return None, max_r, float("inf"), float("inf")

        pred = np.clip(self._poly_eval(r_norm, coeffs), 0.0, 1.0).astype(np.float32, copy=False)
        mse = float(np.mean((pred - fit_i) ** 2))

        # Keep physically plausible monotonic falloff.
        g = np.linspace(0.0, 1.0, 128, dtype=np.float64)
        curve = self._poly_eval(g, coeffs)
        mono_penalty = float(np.mean(np.maximum(np.diff(curve), 0.0) ** 2))
        edge_penalty = float(max(curve[-1] - curve[0], 0.0) ** 2)
        loss = mse + 0.4 * mono_penalty + 0.2 * edge_penalty
        return coeffs, max_r, loss, mse

    def _search_geometry(self, fit_i, init_geom, stages, seed_offset):
        rng = np.random.default_rng(self.seed + seed_offset)
        bounds = {
            "cx_min": 0.0,
            "cx_max": self.w - 1.0,
            "cy_min": 0.0,
            "cy_max": self.h - 1.0,
            "s_min": 0.70,
            "s_max": 1.35,
            "phi_min": -0.8,
            "phi_max": 0.8,
        }

        coeffs, max_r, best_loss, best_mse = self._fit_poly_for_geometry(*init_geom, fit_i=fit_i)
        if coeffs is None:
            coeffs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            max_r = self._compute_max_r(self.h, self.w, *init_geom)
            best_loss = float("inf")
            best_mse = float("inf")
        best = [*init_geom]
        best_coeffs = coeffs
        best_max_r = max_r

        for stage in stages:
            for _ in range(stage["n"]):
                cx = float(np.clip(best[0] + rng.normal(0.0, stage["std_cx"]), bounds["cx_min"], bounds["cx_max"]))
                cy = float(np.clip(best[1] + rng.normal(0.0, stage["std_cy"]), bounds["cy_min"], bounds["cy_max"]))
                sx = float(np.clip(best[2] * np.exp(rng.normal(0.0, stage["std_s"])), bounds["s_min"], bounds["s_max"]))
                sy = float(np.clip(best[3] * np.exp(rng.normal(0.0, stage["std_s"])), bounds["s_min"], bounds["s_max"]))
                phi = float(np.clip(best[4] + rng.normal(0.0, stage["std_phi"]), bounds["phi_min"], bounds["phi_max"]))
                cand_coeffs, cand_max_r, cand_loss, cand_mse = self._fit_poly_for_geometry(cx, cy, sx, sy, phi, fit_i=fit_i)
                if cand_coeffs is None:
                    continue
                if cand_loss < best_loss:
                    best = [cx, cy, sx, sy, phi]
                    best_coeffs = cand_coeffs
                    best_max_r = cand_max_r
                    best_loss = cand_loss
                    best_mse = cand_mse

        return best, best_coeffs, best_max_r, best_loss, best_mse

    def _update_frame_gain_bias(self):
        if self._fit_x is None or self._fit_i is None or self._fit_idx is None:
            return
        if self.num_captures < 1:
            return

        for k in range(self.num_captures):
            _, _, tx, ty, theta, scale = self.unpack_frame_params(k)
            xw, yw = self.apply_similarity_transform(self._fit_x, self._fit_y, tx, ty, theta, scale)
            v = self._shared_field_from_coords(xw, yw).astype(np.float64, copy=False)

            obs = self.captures[k].ravel()[self._fit_idx].astype(np.float64, copy=False)
            if obs.size == 0:
                continue
            lo = float(np.percentile(obs, 2.0))
            hi = float(np.percentile(obs, 99.8))
            denom = max(hi - lo, 1e-6)
            obs_n = np.clip((obs - lo) / denom, 0.0, 1.0)

            v_mean = float(np.mean(v))
            o_mean = float(np.mean(obs_n))
            var_v = float(np.mean((v - v_mean) ** 2))
            if var_v < 1e-12:
                gain = 1.0
                bias = 0.0
            else:
                cov = float(np.mean((v - v_mean) * (obs_n - o_mean)))
                gain = cov / var_v
                bias = o_mean - gain * v_mean

            self.frame_params[k]["gain"] = float(np.clip(gain, 0.2, 5.0))
            self.frame_params[k]["bias"] = float(np.clip(bias, -0.5, 0.5))

    def fit_model(self):
        if self._fit_i is None:
            self._prepare_fit_samples()
        if self.optical_center is None:
            self.find_optical_center()

        cx0, cy0 = float(self.optical_center[0]), float(self.optical_center[1])
        stage_coarse = [
            {"n": 80, "std_cx": 0.22 * self.w, "std_cy": 0.22 * self.h, "std_s": 0.12, "std_phi": 0.30},
            {"n": 120, "std_cx": 0.10 * self.w, "std_cy": 0.10 * self.h, "std_s": 0.08, "std_phi": 0.16},
        ]
        stage_refine = [
            {"n": 120, "std_cx": 0.05 * self.w, "std_cy": 0.05 * self.h, "std_s": 0.05, "std_phi": 0.08},
            {"n": 120, "std_cx": 0.02 * self.w, "std_cy": 0.02 * self.h, "std_s": 0.03, "std_phi": 0.04},
            {"n": 100, "std_cx": 0.01 * self.w, "std_cy": 0.01 * self.h, "std_s": 0.02, "std_phi": 0.02},
        ]

        init = (cx0, cy0, 1.0, 1.0, 0.0)
        best1, coeffs1, max_r1, _, mse1 = self._search_geometry(self._fit_i, init, stage_coarse, seed_offset=7)

        # One-pass illumination-bias compensation: I = V * L
        r1 = self._deformable_r_norm(
            self._fit_x, self._fit_y, best1[0], best1[1], best1[2], best1[3], best1[4], self.deform_coeffs, max_r1
        )
        v1 = np.clip(self._poly_eval(np.clip(r1, 0.0, 1.0), coeffs1), 1e-3, 1.0)
        ratio = np.clip(self._fit_i / v1.astype(np.float32, copy=False), 0.6, 1.4)
        illum = self._fit_illumination_coeffs(ratio)
        illum_map = self._eval_illumination(self._fit_xn, self._fit_yn, illum)
        illum_map = np.clip(illum_map, 0.6, 1.4).astype(np.float32, copy=False)
        corrected_i = np.clip(self._fit_i / illum_map, 0.0, 1.0)

        best2, coeffs2, max_r2, _, mse2 = self._search_geometry(
            corrected_i,
            tuple(best1),
            stage_refine,
            seed_offset=71,
        )
        best = best2
        best_coeffs = coeffs2
        best_max_r = max_r2
        best_mse = mse2
        self.illumination_coeffs = illum.astype(np.float64, copy=False)

        self.geometry = {
            "cx": float(best[0]),
            "cy": float(best[1]),
            "sx": float(best[2]),
            "sy": float(best[3]),
            "phi": float(best[4]),
        }
        self.shared_geometry = dict(self.geometry)
        self.optical_center = (int(round(best[0])), int(round(best[1])))
        self.poly_coeffs = best_coeffs.astype(np.float64, copy=False)
        self.popt = self.poly_coeffs[1:].copy()  # legacy compatibility
        self.max_r = float(best_max_r)
        if self.num_captures > 1:
            self._update_frame_gain_bias()
        print(
            "[-] Geometric Fit: "
            f"cx={self.geometry['cx']:.1f}, cy={self.geometry['cy']:.1f}, "
            f"sx={self.geometry['sx']:.4f}, sy={self.geometry['sy']:.4f}, "
            f"phi={self.geometry['phi']:.4f} rad | "
            f"poly(c0,a,b,c)=({self.poly_coeffs[0]:.4f},{self.poly_coeffs[1]:.4f},"
            f"{self.poly_coeffs[2]:.4f},{self.poly_coeffs[3]:.4f}) | "
            f"sample_mse(raw={mse1:.6e}, corrected={best_mse:.6e})"
        )

    def generate_2d_prediction(self, chunk_rows=256):
        if self.shared_geometry is None or self.poly_coeffs is None:
            raise RuntimeError("fit_model() must run before generate_2d_prediction().")

        cx = np.float32(self.shared_geometry["cx"])
        cy = np.float32(self.shared_geometry["cy"])
        sx = np.float32(self.shared_geometry["sx"])
        sy = np.float32(self.shared_geometry["sy"])
        phi = np.float32(self.shared_geometry["phi"])
        max_r = np.float32(max(self.max_r, 1e-6))

        out = np.empty((self.h, self.w), dtype=np.float32)
        x = np.arange(self.w, dtype=np.float32)[None, :]

        for y0 in range(0, self.h, chunk_rows):
            y1 = min(y0 + chunk_rows, self.h)
            y = np.arange(y0, y1, dtype=np.float32)[:, None]
            xx = np.broadcast_to(x, (y1 - y0, self.w))
            yy = np.broadcast_to(y, (y1 - y0, self.w))
            r_norm = self._deformable_r_norm(
                xx,
                yy,
                cx,
                cy,
                sx,
                sy,
                phi,
                self.deform_coeffs,
                max_r,
            )
            r_norm = np.clip(r_norm, 0.0, 1.0)
            pred = self._poly_eval(r_norm, self.poly_coeffs).astype(np.float32, copy=False)
            np.clip(pred, 0.0, 1.0, out=pred)
            out[y0:y1, :] = pred

        peak = float(np.max(out))
        if peak > 1e-6:
            out /= np.float32(peak)
        np.clip(out, 0.0, 1.0, out=out)
        self.predicted_surface = out

    def predict_capture(self, frame_index, chunk_rows=256):
        if self.predicted_surface is None:
            self.generate_2d_prediction(chunk_rows=chunk_rows)

        gain, bias, tx, ty, theta, scale = self.unpack_frame_params(frame_index)
        out = np.empty((self.h, self.w), dtype=np.float32)
        x = np.arange(self.w, dtype=np.float32)[None, :]

        for y0 in range(0, self.h, chunk_rows):
            y1 = min(y0 + chunk_rows, self.h)
            y = np.arange(y0, y1, dtype=np.float32)[:, None]
            xx = np.broadcast_to(x, (y1 - y0, self.w))
            yy = np.broadcast_to(y, (y1 - y0, self.w))
            xw, yw = self.apply_similarity_transform(xx, yy, tx, ty, theta, scale)
            v = self._shared_field_from_coords(xw, yw)
            out[y0:y1, :] = (gain * v + bias).astype(np.float32, copy=False)
        return out

    def joint_residual(self, frame_params=None, shared_surface=None, stride=1, return_per_frame=False):
        """
        Joint residual skeleton for:
            I_k(x, y) ~= g_k * V(W_k(x, y)) + b_k
        """
        if self.num_captures < 1:
            raise RuntimeError("No captures available.")
        if self.shared_geometry is None or self.poly_coeffs is None:
            raise RuntimeError("fit_model() must run before joint_residual().")

        stride = int(max(1, stride))
        yy, xx = np.indices((self.h, self.w), dtype=np.float32)
        if stride > 1:
            yy = yy[::stride, ::stride]
            xx = xx[::stride, ::stride]

        if shared_surface is not None:
            shared_surface = np.asarray(shared_surface, dtype=np.float32)
            if shared_surface.shape != (self.h, self.w):
                raise RuntimeError(
                    f"shared_surface shape mismatch: expected {(self.h, self.w)}, got {shared_surface.shape}"
                )

        residuals = []
        residual_maps = []
        for k in range(self.num_captures):
            gain, bias, tx, ty, theta, scale = self.unpack_frame_params(k, frame_params)
            xw, yw = self.apply_similarity_transform(xx, yy, tx, ty, theta, scale)
            if shared_surface is None:
                v = self._shared_field_from_coords(xw, yw)
            else:
                v = self._bilinear_sample(shared_surface, xw, yw, fill_value=0.0)
            pred = gain * v + bias

            obs = self.captures[k]
            if stride > 1:
                obs = obs[::stride, ::stride]
            res = (obs - pred).astype(np.float32, copy=False)
            residuals.append(res.ravel())
            residual_maps.append(res)

        all_res = np.concatenate(residuals, axis=0) if residuals else np.empty((0,), dtype=np.float32)
        if return_per_frame:
            return all_res, residual_maps
        return all_res

    def evaluate(self):
        if self.ground_truth is None:
            return None
        if self.predicted_surface is None:
            raise RuntimeError("generate_2d_prediction() must run before evaluate().")

        y_true = self.ground_truth.ravel().astype(np.float64, copy=False)
        y_pred = self.predicted_surface.ravel().astype(np.float64, copy=False)
        sse = float(np.sum((y_true - y_pred) ** 2))
        sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - (sse / max(sst, 1e-12))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        print("\n=== Traditional Baseline Evaluation ===")
        print(f"R-squared (Fit to GT): {r2 * 100:.2f}%")
        print(f"RMSE: {rmse:.4f}")
        return r2, rmse


def load_rotated_stack(case_dir):
    case_path = Path(case_dir)
    captures = []
    for angle in (0, 90, 180, 270):
        frame_path = case_path / f"sim_capture_{angle}.npy"
        captures.append(np.load(frame_path).astype(np.float32, copy=False))
    gt = np.load(case_path / "ground_truth_vignetting.npy").astype(np.float32, copy=False)

    if len(captures) != 4:
        raise RuntimeError(f"Expected 4 captures, got {len(captures)}")
    shape0 = captures[0].shape
    for i, frame in enumerate(captures):
        if frame.ndim != 2:
            raise RuntimeError(f"Capture sim_capture_{(i * 90) % 360}.npy must be 2D.")
        if frame.shape != shape0:
            raise RuntimeError("All captures must have identical shape.")
    if gt.shape != shape0:
        raise RuntimeError(f"GT shape mismatch: expected {shape0}, got {gt.shape}")
    return captures, gt


def load_rotated_average(case_dir):
    # Compatibility wrapper: keep legacy API, build from the new stack loader.
    captures, gt = load_rotated_stack(case_dir)
    avg = np.mean(np.stack(captures, axis=0), axis=0).astype(np.float32, copy=False)
    return avg, gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run improved traditional geometric vignetting baseline (V2).")
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default="synthetic_output",
        help="Directory containing sim_capture_{0,90,180,270}.npy and ground_truth_vignetting.npy",
    )
    args = parser.parse_args()

    run_mode = "stack"
    try:
        input_data, gt_vignette = load_rotated_stack(args.synthetic_dir)
        print("[-] Loaded 4-frame rotated stack.")
    except FileNotFoundError:
        try:
            input_data, gt_vignette = load_rotated_average(args.synthetic_dir)
            run_mode = "average_compat"
            print("[!] Stack loading failed, fallback to legacy average mode.")
        except FileNotFoundError:
            run_mode = "dummy_single"
            print("[!] Synthetic data not found. Creating a dummy matrix for testing...")
            input_data = np.ones((1024, 1024), dtype=np.float32) * 0.8
            gt_vignette = np.ones((1024, 1024), dtype=np.float32)

    baseline = TraditionalPolynomialBaseline(input_data, ground_truth=gt_vignette)
    baseline.find_optical_center()
    baseline.extract_radial_profile()
    baseline.fit_model()
    baseline.generate_2d_prediction()
    baseline.evaluate()
    if baseline.num_captures > 1:
        residual = baseline.joint_residual(stride=max(1, int(np.ceil(max(baseline.h, baseline.w) / 256.0))))
        mean_abs = float(np.mean(np.abs(residual))) if residual.size > 0 else 0.0
        print(
            "[-] Joint residual ready "
            f"(mode={run_mode}, samples={residual.size}, mean_abs={mean_abs:.6e})"
        )

