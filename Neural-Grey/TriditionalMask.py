"""
Traditional Mask V1 (commented copy)

This file is the initial deterministic geometric baseline.
Main idea:
1) Estimate optical center from intensity distribution.
2) Build elliptical normalized radius field.
3) Fit even-order radial polynomial for shading falloff.
4) Reconstruct a smooth vignette surface and evaluate with GT.

Notes:
- This is non-learning and fully CPU deterministic.
- Keep this version as a simple reference baseline.
"""
import argparse
from pathlib import Path
import numpy as np


class TraditionalPolynomialBaseline:
    """
    Improved traditional geometric baseline:
    - Robust optical-center estimate (not raw max pixel).
    - Elliptical radial geometry (center + anisotropy + rotation).
    - Radial polynomial fit (1 + a r^2 + b r^4 + c r^6) on binned profile.

    This stays non-ML and deterministic, but is much more stable on asymmetric cases.
    """

    def __init__(
        self,
        data_matrix,
        ground_truth=None,
        seed=42,
        max_fit_samples=200_000,
        radial_bins=320,
    ):
        self.img_data = np.asarray(data_matrix, dtype=np.float32)
        self.ground_truth = None if ground_truth is None else np.asarray(ground_truth, dtype=np.float32)
        self.h, self.w = self.img_data.shape
        self.seed = int(seed)
        self.max_fit_samples = int(max_fit_samples)
        self.radial_bins = int(radial_bins)

        self.optical_center = None
        self.max_brightness = 0.0
        self.radial_profile = None
        self.popt = None  # legacy: [a, b, c]
        self.poly_coeffs = None  # full: [c0, c2, c4, c6]
        self.predicted_surface = None
        self.max_r = None
        self.geometry = None  # dict(cx, cy, sx, sy, phi)
        self.illumination_coeffs = None  # low-order 2D bias: [1, x, y, x2, y2, xy]

        self._fit_x = None
        self._fit_y = None
        self._fit_xn = None
        self._fit_yn = None
        self._fit_i = None

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

        flat = self.img_data.ravel()
        fit_i = flat[idx].astype(np.float32, copy=False)
        scale = float(np.percentile(fit_i, 99.8))
        scale = max(scale, 1e-6)
        fit_i = np.clip(fit_i / scale, 0.0, 1.0)

        self.max_brightness = scale
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
        r_norm = self._elliptical_r_norm(self._fit_x, self._fit_y, cx, cy, sx, sy, phi, max_r)
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
        r1 = self._elliptical_r_norm(self._fit_x, self._fit_y, *best1, max_r1)
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
        self.optical_center = (int(round(best[0])), int(round(best[1])))
        self.poly_coeffs = best_coeffs.astype(np.float64, copy=False)
        self.popt = self.poly_coeffs[1:].copy()  # legacy compatibility
        self.max_r = float(best_max_r)
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
        if self.geometry is None or self.poly_coeffs is None:
            raise RuntimeError("fit_model() must run before generate_2d_prediction().")

        cx = np.float32(self.geometry["cx"])
        cy = np.float32(self.geometry["cy"])
        sx = np.float32(self.geometry["sx"])
        sy = np.float32(self.geometry["sy"])
        phi = np.float32(self.geometry["phi"])
        max_r = np.float32(max(self.max_r, 1e-6))
        c = np.float32(np.cos(phi))
        s = np.float32(np.sin(phi))

        out = np.empty((self.h, self.w), dtype=np.float32)
        x = np.arange(self.w, dtype=np.float32)[None, :]

        for y0 in range(0, self.h, chunk_rows):
            y1 = min(y0 + chunk_rows, self.h)
            y = np.arange(y0, y1, dtype=np.float32)[:, None]
            dx = x - cx
            dy = y - cy
            xr = c * dx + s * dy
            yr = -s * dx + c * dy
            r = np.sqrt((sx * xr) ** 2 + (sy * yr) ** 2, dtype=np.float32)
            r_norm = np.clip(r / max_r, 0.0, 1.0)
            pred = self._poly_eval(r_norm, self.poly_coeffs).astype(np.float32, copy=False)
            np.clip(pred, 0.0, 1.0, out=pred)
            out[y0:y1, :] = pred

        peak = float(np.max(out))
        if peak > 1e-6:
            out /= np.float32(peak)
        np.clip(out, 0.0, 1.0, out=out)
        self.predicted_surface = out

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


def load_rotated_average(case_dir):
    case_path = Path(case_dir)
    captures = []
    for angle in (0, 90, 180, 270):
        captures.append(np.load(case_path / f"sim_capture_{angle}.npy").astype(np.float32, copy=False))
    avg = (captures[0] + captures[1] + captures[2] + captures[3]) * np.float32(0.25)
    gt = np.load(case_path / "ground_truth_vignetting.npy").astype(np.float32, copy=False)
    return avg, gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run improved traditional geometric vignetting baseline.")
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default="synthetic_output",
        help="Directory containing sim_capture_{0,90,180,270}.npy and ground_truth_vignetting.npy",
    )
    args = parser.parse_args()

    try:
        input_data, gt_vignette = load_rotated_average(args.synthetic_dir)
    except FileNotFoundError:
        print("[!] Synthetic data not found. Creating a dummy matrix for testing...")
        input_data = np.ones((1024, 1024), dtype=np.float32) * 0.8
        gt_vignette = np.ones((1024, 1024), dtype=np.float32)

    baseline = TraditionalPolynomialBaseline(input_data, ground_truth=gt_vignette)
    baseline.find_optical_center()
    baseline.extract_radial_profile()
    baseline.fit_model()
    baseline.generate_2d_prediction()
    baseline.evaluate()

