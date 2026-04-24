import argparse
from pathlib import Path
import numpy as np


class TraditionalPolynomialBaseline:
    """
    Deterministic geometric baseline with backward compatibility:
    - Supports single-frame input and multi-frame stack input.
    - Shared geometric vignette field + per-frame nuisance parameters.
    - Elliptical radius with low-DOF deformable correction skeleton.
    - Pointwise fitting with configurable radial model and lighting field.
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

        # New configurable pointwise model options.
        self.pointwise_fit_enabled = True
        self.search_budget_scale = 1.0
        self.residual_loss_type = "l2"  # "l2" or "huber"
        self.robust_huber_delta = 0.05
        self.pointwise_alternations = 2

        self.radial_model_type = "poly"  # "poly" or "spline_like"
        self.radial_num_knots = 9
        self.radial_ridge_lambda = 1e-4
        self.radial_mono_penalty = 0.4
        self.radial_edge_penalty = 0.2
        self.radial_coeffs = None

        self.lighting_mode = "mesh"  # "mesh" or "poly2"
        self.lighting_mesh_shape = (10, 10)
        self.lighting_clip = (0.7, 1.3)
        self.lighting_smooth_lambda = 0.35
        self.lighting_smooth_steps = 8
        self.lighting_anchor_lambda = 0.05
        self.lighting_penalty_weight = 0.02
        self.lighting_mesh = self.initialize_lighting_mesh()

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

    def _coords_to_normalized(self, x, y):
        """Convert image coordinates into normalized coordinates in [-1, 1]-like range."""
        cx_ref = np.float32((self.w - 1) * 0.5)
        cy_ref = np.float32((self.h - 1) * 0.5)
        sx_ref = np.float32(max((self.w - 1) * 0.5, 1.0))
        sy_ref = np.float32(max((self.h - 1) * 0.5, 1.0))
        xn = (x - cx_ref) / sx_ref
        yn = (y - cy_ref) / sy_ref
        return xn.astype(np.float32, copy=False), yn.astype(np.float32, copy=False)

    def _build_radial_basis(self, r_norm, mode=None):
        """
        Build radial basis matrix.

        poly:
            [1, r^2, r^4, r^6, r^8]
        spline_like:
            triangular hat basis over uniformly spaced knots in [0, 1]
        """
        used_mode = self.radial_model_type if mode is None else mode
        r = np.clip(np.asarray(r_norm, dtype=np.float32).reshape(-1), 0.0, 1.0)

        if used_mode == "poly":
            r2 = r * r
            r4 = r2 * r2
            r6 = r4 * r2
            r8 = r4 * r4
            basis = np.stack([np.ones_like(r), r2, r4, r6, r8], axis=1)
            return basis.astype(np.float64, copy=False)

        if used_mode == "spline_like":
            k = int(max(4, self.radial_num_knots))
            knots = np.linspace(0.0, 1.0, k, dtype=np.float32)
            if k <= 1:
                return np.ones((r.size, 1), dtype=np.float64)
            spacing = float(max(knots[1] - knots[0], 1e-6))
            basis = []
            for kj in knots:
                b = np.maximum(1.0 - np.abs(r - kj) / spacing, 0.0)
                basis.append(b)
            return np.stack(basis, axis=1).astype(np.float64, copy=False)

        raise RuntimeError(f"Unsupported radial model type: {used_mode}")

    def _eval_radial_curve(self, r_norm, coeffs=None, mode=None):
        """Evaluate radial curve for points at normalized radius r_norm."""
        used_mode = self.radial_model_type if mode is None else mode
        c = self.radial_coeffs if coeffs is None else np.asarray(coeffs, dtype=np.float64)
        if c is None:
            raise RuntimeError("Radial coefficients are not initialized.")

        r = np.clip(np.asarray(r_norm, dtype=np.float32), 0.0, 1.0)
        shape = r.shape
        rf = r.reshape(-1)

        if used_mode == "poly" and c.size == 4:
            out = self._poly_eval(rf, c).astype(np.float32, copy=False)
        else:
            basis = self._build_radial_basis(rf, mode=used_mode)
            out = np.zeros((basis.shape[0],), dtype=np.float64)
            n = int(min(basis.shape[1], c.size))
            for j in range(n):
                out += basis[:, j] * float(c[j])
            out = out.astype(np.float32, copy=False)
        return out.reshape(shape)

    def evaluate_vignette_on_points(self, r_norm, coeffs, mode=None):
        """Compatibility wrapper to evaluate vignette radial field on sampled points."""
        return self._eval_radial_curve(r_norm, coeffs=coeffs, mode=mode)

    def compute_pointwise_residual(self, observed, predicted, loss_type=None):
        """
        Compute pointwise residual and scalar loss.
        The interface keeps a hook for robust losses.
        """
        used_loss = self.residual_loss_type if loss_type is None else loss_type
        obs = np.asarray(observed, dtype=np.float32).reshape(-1)
        pred = np.asarray(predicted, dtype=np.float32).reshape(-1)
        residual = obs - pred

        if used_loss == "huber":
            delta = float(max(self.robust_huber_delta, 1e-6))
            ar = np.abs(residual)
            quad = np.minimum(ar, delta)
            lin = ar - quad
            loss = float(np.mean(0.5 * quad * quad + delta * lin))
        else:
            loss = float(np.mean(residual * residual))
        return residual.astype(np.float32, copy=False), loss

    def _solve_ridge(self, basis, target, ridge_lambda):
        b = np.asarray(basis, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if b.shape[0] != y.size:
            raise RuntimeError(f"Basis/target mismatch: {b.shape[0]} vs {y.size}")

        n_basis = int(b.shape[1])
        gram = np.zeros((n_basis, n_basis), dtype=np.float64)
        rhs = np.zeros((n_basis,), dtype=np.float64)
        for i in range(n_basis):
            bi = b[:, i]
            rhs[i] = float(np.sum(bi * y))
            for j in range(i, n_basis):
                bj = b[:, j]
                g = float(np.sum(bi * bj))
                gram[i, j] = g
                gram[j, i] = g

        lam = float(max(ridge_lambda, 0.0))
        if lam > 0.0:
            gram = gram + lam * np.eye(gram.shape[0], dtype=np.float64)
        coeffs = self._solve_linear_system(gram.tolist(), rhs.tolist())
        if coeffs is None:
            # Add stronger diagonal damping if the system is near-singular.
            gram2 = gram + 1e-6 * np.eye(gram.shape[0], dtype=np.float64)
            coeffs = self._solve_linear_system(gram2.tolist(), rhs.tolist())
            if coeffs is None:
                coeffs = np.zeros((gram.shape[0],), dtype=np.float64)
        return coeffs.astype(np.float64, copy=False)

    @staticmethod
    def _enforce_monotone_nonincreasing(values):
        """Simple post-hoc monotone projection via cumulative minimum."""
        v = np.asarray(values, dtype=np.float64).copy()
        if v.size == 0:
            return v
        return np.minimum.accumulate(v)

    def _radial_physical_penalty(self, coeffs, mode=None):
        g = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        curve = self._eval_radial_curve(g, coeffs=coeffs, mode=mode).astype(np.float64, copy=False)
        d = np.diff(curve)
        mono = float(np.mean(np.maximum(d, 0.0) ** 2))
        edge = float(max(curve[-1] - curve[0], 0.0) ** 2)
        return mono, edge

    def _fit_radial_curve_direct(self, r_norm, observed, lighting=None, mode=None):
        """
        Direct pointwise fit of radial curve from samples (no radial-bin averaging).
        """
        used_mode = self.radial_model_type if mode is None else mode
        r = np.clip(np.asarray(r_norm, dtype=np.float32).reshape(-1), 0.0, 1.0)
        obs = np.asarray(observed, dtype=np.float32).reshape(-1)
        if lighting is None:
            l = np.ones_like(obs, dtype=np.float32)
        else:
            l = np.clip(np.asarray(lighting, dtype=np.float32).reshape(-1), 1e-3, 10.0)
        if obs.size != r.size or l.size != r.size:
            raise RuntimeError("Pointwise arrays must have same length in _fit_radial_curve_direct.")

        target = np.clip(obs / l, 0.0, 1.5).astype(np.float64, copy=False)
        basis = self._build_radial_basis(r, mode=used_mode)
        coeffs = self._solve_ridge(basis, target, self.radial_ridge_lambda)

        # Monotonic post-hoc projection to keep physically plausible falloff.
        grid = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        curve = self._eval_radial_curve(grid, coeffs=coeffs, mode=used_mode).astype(np.float64, copy=False)
        curve = np.clip(curve, 1e-4, 2.0)
        curve = self._enforce_monotone_nonincreasing(curve)
        if curve[0] > 1e-6:
            curve /= curve[0]
        bg = self._build_radial_basis(grid, mode=used_mode)
        coeffs = self._solve_ridge(bg, curve, self.radial_ridge_lambda)

        # Re-scale curve amplitude to match target in least-squares sense.
        pred = self._eval_radial_curve(r, coeffs=coeffs, mode=used_mode).astype(np.float64, copy=False)
        denom = float(np.sum(pred * pred))
        if denom > 1e-12:
            scale = float(np.clip(float(np.sum(target * pred)) / denom, 0.2, 5.0))
            coeffs = coeffs * scale
        return coeffs.astype(np.float64, copy=False)

    def initialize_lighting_mesh(self, mesh_shape=None, fill_value=1.0):
        """Initialize low-resolution smooth lighting mesh."""
        shape = self.lighting_mesh_shape if mesh_shape is None else mesh_shape
        mh, mw = int(shape[0]), int(shape[1])
        if mh < 2 or mw < 2:
            raise RuntimeError(f"lighting mesh shape must be >=2x2, got {shape}")
        return np.full((mh, mw), np.float32(fill_value), dtype=np.float32)

    def eval_lighting_mesh(self, xn, yn, mesh):
        """
        Evaluate lighting mesh at normalized coordinates via bilinear interpolation.
        xn, yn are typically in roughly [-1, 1].
        """
        if mesh is None:
            raise RuntimeError("Lighting mesh is None.")
        m = np.asarray(mesh, dtype=np.float32)
        mh, mw = m.shape

        x = np.asarray(xn, dtype=np.float32)
        y = np.asarray(yn, dtype=np.float32)
        shape = x.shape
        xf = x.reshape(-1)
        yf = y.reshape(-1)

        u = np.clip((xf + 1.0) * 0.5, 0.0, 1.0) * np.float32(mw - 1)
        v = np.clip((yf + 1.0) * 0.5, 0.0, 1.0) * np.float32(mh - 1)
        out = self._bilinear_sample(m, u, v, fill_value=1.0)
        out = np.clip(out, self.lighting_clip[0], self.lighting_clip[1])
        return out.reshape(shape).astype(np.float32, copy=False)

    def _lighting_mesh_smoothness(self, mesh):
        if mesh is None:
            return 0.0
        m = np.asarray(mesh, dtype=np.float32)
        dx = m[:, 1:] - m[:, :-1]
        dy = m[1:, :] - m[:-1, :]
        return float(np.mean(dx * dx) + np.mean(dy * dy))

    def fit_lighting_mesh(self, xn, yn, target_ratio, mesh=None):
        """
        Fit lighting mesh from ratio target with smooth regularization.
        """
        cur = self.initialize_lighting_mesh() if mesh is None else np.asarray(mesh, dtype=np.float32).copy()
        mh, mw = cur.shape
        x = np.asarray(xn, dtype=np.float32).reshape(-1)
        y = np.asarray(yn, dtype=np.float32).reshape(-1)
        t = np.clip(
            np.asarray(target_ratio, dtype=np.float32).reshape(-1),
            self.lighting_clip[0],
            self.lighting_clip[1],
        )
        if not (x.size == y.size == t.size):
            raise RuntimeError("xn, yn, target_ratio size mismatch in fit_lighting_mesh().")

        u = np.clip((x + 1.0) * 0.5, 0.0, 1.0) * np.float32(mw - 1)
        v = np.clip((y + 1.0) * 0.5, 0.0, 1.0) * np.float32(mh - 1)
        x0 = np.floor(u).astype(np.int32)
        y0 = np.floor(v).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, mw - 1)
        y1 = np.clip(y0 + 1, 0, mh - 1)

        wx = u - x0.astype(np.float32)
        wy = v - y0.astype(np.float32)
        wa = (1.0 - wx) * (1.0 - wy)
        wb = wx * (1.0 - wy)
        wc = (1.0 - wx) * wy
        wd = wx * wy

        num = np.zeros_like(cur, dtype=np.float32)
        den = np.zeros_like(cur, dtype=np.float32)
        np.add.at(num, (y0, x0), wa * t)
        np.add.at(num, (y0, x1), wb * t)
        np.add.at(num, (y1, x0), wc * t)
        np.add.at(num, (y1, x1), wd * t)
        np.add.at(den, (y0, x0), wa)
        np.add.at(den, (y0, x1), wb)
        np.add.at(den, (y1, x0), wc)
        np.add.at(den, (y1, x1), wd)

        data_mesh = np.where(den > 1e-6, num / den, cur)
        out = data_mesh.astype(np.float32, copy=False)

        for _ in range(self.lighting_smooth_steps):
            nbr_sum = np.zeros_like(out)
            nbr_cnt = np.zeros_like(out)
            nbr_sum[:, :-1] += out[:, 1:]
            nbr_cnt[:, :-1] += 1.0
            nbr_sum[:, 1:] += out[:, :-1]
            nbr_cnt[:, 1:] += 1.0
            nbr_sum[:-1, :] += out[1:, :]
            nbr_cnt[:-1, :] += 1.0
            nbr_sum[1:, :] += out[:-1, :]
            nbr_cnt[1:, :] += 1.0
            smooth = np.where(nbr_cnt > 0, nbr_sum / np.maximum(nbr_cnt, 1.0), out)
            out = (out + self.lighting_smooth_lambda * smooth) / (1.0 + self.lighting_smooth_lambda)
            out = (out + self.lighting_anchor_lambda * 1.0) / (1.0 + self.lighting_anchor_lambda)

        out = np.clip(out, self.lighting_clip[0], self.lighting_clip[1]).astype(np.float32, copy=False)
        return out

    def _eval_lighting_field_points(self, xn, yn, mesh=None, coeffs=None, mode=None):
        """Evaluate lighting field at normalized points for current lighting mode."""
        used_mode = self.lighting_mode if mode is None else mode
        if used_mode == "mesh":
            src_mesh = self.lighting_mesh if mesh is None else mesh
            if src_mesh is None:
                src_mesh = self.initialize_lighting_mesh()
            return self.eval_lighting_mesh(xn, yn, src_mesh)

        src_coeffs = self.illumination_coeffs if coeffs is None else coeffs
        if src_coeffs is None:
            src_coeffs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        out = self._eval_illumination(xn, yn, src_coeffs)
        return np.clip(out, self.lighting_clip[0], self.lighting_clip[1]).astype(np.float32, copy=False)

    def _fit_pointwise_fields_for_geometry(self, cx, cy, sx, sy, phi, fit_i):
        """
        Fit radial + lighting fields directly on sampled 2D points for one geometry candidate.
        """
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

        lighting = np.ones_like(fit_i, dtype=np.float32)
        mesh = None
        illum_coeffs = None
        if self.lighting_mode == "mesh":
            mesh = self.initialize_lighting_mesh() if self.lighting_mesh is None else self.lighting_mesh.copy()
            lighting = self._eval_lighting_field_points(self._fit_xn, self._fit_yn, mesh=mesh, mode="mesh")
        else:
            illum_coeffs = (
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                if self.illumination_coeffs is None
                else self.illumination_coeffs.copy()
            )
            lighting = self._eval_lighting_field_points(self._fit_xn, self._fit_yn, coeffs=illum_coeffs, mode="poly2")

        coeffs = None
        pred = None
        for _ in range(max(1, int(self.pointwise_alternations))):
            coeffs = self._fit_radial_curve_direct(r_norm, fit_i, lighting=lighting, mode=self.radial_model_type)
            radial = self.evaluate_vignette_on_points(r_norm, coeffs=coeffs, mode=self.radial_model_type)
            radial = np.clip(radial, 1e-4, 2.0)

            ratio = np.clip(fit_i / radial, self.lighting_clip[0], self.lighting_clip[1])
            if self.lighting_mode == "mesh":
                mesh = self.fit_lighting_mesh(self._fit_xn, self._fit_yn, ratio, mesh=mesh)
                lighting = self._eval_lighting_field_points(self._fit_xn, self._fit_yn, mesh=mesh, mode="mesh")
            else:
                illum_coeffs = self._fit_illumination_coeffs(ratio)
                lighting = self._eval_lighting_field_points(self._fit_xn, self._fit_yn, coeffs=illum_coeffs, mode="poly2")

            pred = np.clip(radial * lighting, 0.0, 2.0)

        if pred is None:
            pred = np.ones_like(fit_i, dtype=np.float32)
        _, data_loss = self.compute_pointwise_residual(fit_i, pred, loss_type=self.residual_loss_type)
        mono_pen, edge_pen = self._radial_physical_penalty(coeffs, mode=self.radial_model_type)
        smooth_pen = self._lighting_mesh_smoothness(mesh) if self.lighting_mode == "mesh" else 0.0
        total_loss = (
            data_loss
            + self.radial_mono_penalty * mono_pen
            + self.radial_edge_penalty * edge_pen
            + self.lighting_penalty_weight * smooth_pen
        )

        aux = {"mesh": mesh, "illum_coeffs": illum_coeffs}
        return coeffs, max_r, float(total_loss), float(data_loss), aux

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
        active_coeffs = self.radial_coeffs if self.radial_coeffs is not None else self.poly_coeffs
        if geom is None or active_coeffs is None or self.max_r is None:
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
        v = self._eval_radial_curve(r_norm, coeffs=active_coeffs, mode=self.radial_model_type)
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

        if self.pointwise_fit_enabled:
            coeffs, max_r, best_loss, best_mse, best_aux = self._fit_pointwise_fields_for_geometry(
                *init_geom,
                fit_i=fit_i,
            )
        else:
            coeffs, max_r, best_loss, best_mse = self._fit_poly_for_geometry(*init_geom, fit_i=fit_i)
            best_aux = {"mesh": None, "illum_coeffs": self.illumination_coeffs}

        if coeffs is None:
            coeffs = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            max_r = self._compute_max_r(self.h, self.w, *init_geom)
            best_loss = float("inf")
            best_mse = float("inf")
            best_aux = {"mesh": None, "illum_coeffs": self.illumination_coeffs}
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
                if self.pointwise_fit_enabled:
                    cand_coeffs, cand_max_r, cand_loss, cand_mse, cand_aux = self._fit_pointwise_fields_for_geometry(
                        cx, cy, sx, sy, phi, fit_i=fit_i
                    )
                else:
                    cand_coeffs, cand_max_r, cand_loss, cand_mse = self._fit_poly_for_geometry(
                        cx, cy, sx, sy, phi, fit_i=fit_i
                    )
                    cand_aux = {"mesh": None, "illum_coeffs": self.illumination_coeffs}
                if cand_coeffs is None:
                    continue
                if cand_loss < best_loss:
                    best = [cx, cy, sx, sy, phi]
                    best_coeffs = cand_coeffs
                    best_max_r = cand_max_r
                    best_loss = cand_loss
                    best_mse = cand_mse
                    best_aux = cand_aux

        return best, best_coeffs, best_max_r, best_loss, best_mse, best_aux

    def _update_frame_gain_bias(self):
        if self._fit_x is None or self._fit_i is None or self._fit_idx is None:
            return
        if self.num_captures < 1:
            return

        for k in range(self.num_captures):
            _, _, tx, ty, theta, scale = self.unpack_frame_params(k)
            xw, yw = self.apply_similarity_transform(self._fit_x, self._fit_y, tx, ty, theta, scale)
            v = self._shared_field_from_coords(xw, yw).astype(np.float64, copy=False)
            xn, yn = self._coords_to_normalized(xw, yw)
            l = self._eval_lighting_field_points(xn, yn).astype(np.float64, copy=False)
            vl = np.clip(v * l, 1e-6, 5.0)

            obs = self.captures[k].ravel()[self._fit_idx].astype(np.float64, copy=False)
            if obs.size == 0:
                continue
            lo = float(np.percentile(obs, 2.0))
            hi = float(np.percentile(obs, 99.8))
            denom = max(hi - lo, 1e-6)
            obs_n = np.clip((obs - lo) / denom, 0.0, 1.0)

            v_mean = float(np.mean(vl))
            o_mean = float(np.mean(obs_n))
            var_v = float(np.mean((vl - v_mean) ** 2))
            if var_v < 1e-12:
                gain = 1.0
                bias = 0.0
            else:
                cov = float(np.mean((vl - v_mean) * (obs_n - o_mean)))
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
        budget = float(max(self.search_budget_scale, 0.05))
        def _ns(v):
            return int(max(8, round(v * budget)))

        if self.pointwise_fit_enabled:
            stage_coarse = [
                {"n": _ns(32), "std_cx": 0.22 * self.w, "std_cy": 0.22 * self.h, "std_s": 0.12, "std_phi": 0.30},
                {"n": _ns(48), "std_cx": 0.10 * self.w, "std_cy": 0.10 * self.h, "std_s": 0.08, "std_phi": 0.16},
            ]
            stage_refine = [
                {"n": _ns(48), "std_cx": 0.05 * self.w, "std_cy": 0.05 * self.h, "std_s": 0.05, "std_phi": 0.08},
                {"n": _ns(48), "std_cx": 0.02 * self.w, "std_cy": 0.02 * self.h, "std_s": 0.03, "std_phi": 0.04},
                {"n": _ns(32), "std_cx": 0.01 * self.w, "std_cy": 0.01 * self.h, "std_s": 0.02, "std_phi": 0.02},
            ]
        else:
            stage_coarse = [
                {"n": _ns(80), "std_cx": 0.22 * self.w, "std_cy": 0.22 * self.h, "std_s": 0.12, "std_phi": 0.30},
                {"n": _ns(120), "std_cx": 0.10 * self.w, "std_cy": 0.10 * self.h, "std_s": 0.08, "std_phi": 0.16},
            ]
            stage_refine = [
                {"n": _ns(120), "std_cx": 0.05 * self.w, "std_cy": 0.05 * self.h, "std_s": 0.05, "std_phi": 0.08},
                {"n": _ns(120), "std_cx": 0.02 * self.w, "std_cy": 0.02 * self.h, "std_s": 0.03, "std_phi": 0.04},
                {"n": _ns(100), "std_cx": 0.01 * self.w, "std_cy": 0.01 * self.h, "std_s": 0.02, "std_phi": 0.02},
            ]

        init = (cx0, cy0, 1.0, 1.0, 0.0)
        best1, coeffs1, max_r1, _, mse1, aux1 = self._search_geometry(self._fit_i, init, stage_coarse, seed_offset=7)
        best2, coeffs2, max_r2, _, mse2, aux2 = self._search_geometry(
            self._fit_i,
            tuple(best1),
            stage_refine,
            seed_offset=71,
        )
        best = best2
        best_coeffs = coeffs2
        best_max_r = max_r2
        best_mse = mse2
        best_aux = aux2 if aux2 is not None else aux1

        self.geometry = {
            "cx": float(best[0]),
            "cy": float(best[1]),
            "sx": float(best[2]),
            "sy": float(best[3]),
            "phi": float(best[4]),
        }
        self.shared_geometry = dict(self.geometry)
        self.optical_center = (int(round(best[0])), int(round(best[1])))
        self.radial_coeffs = best_coeffs.astype(np.float64, copy=False)
        self.poly_coeffs = self.radial_coeffs.copy()  # legacy compatibility alias
        if self.poly_coeffs.size >= 4:
            self.popt = self.poly_coeffs[1:4].copy()
        else:
            self.popt = None
        self.max_r = float(best_max_r)

        if best_aux is not None:
            if best_aux.get("mesh", None) is not None:
                self.lighting_mesh = np.asarray(best_aux["mesh"], dtype=np.float32)
            if best_aux.get("illum_coeffs", None) is not None:
                self.illumination_coeffs = np.asarray(best_aux["illum_coeffs"], dtype=np.float64)

        # Ensure fallback illumination state is available.
        if self.illumination_coeffs is None:
            self.illumination_coeffs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if self.num_captures > 1:
            self._update_frame_gain_bias()
        print(
            "[-] Geometric Fit: "
            f"cx={self.geometry['cx']:.1f}, cy={self.geometry['cy']:.1f}, "
            f"sx={self.geometry['sx']:.4f}, sy={self.geometry['sy']:.4f}, "
            f"phi={self.geometry['phi']:.4f} rad | "
            f"radial_mode={self.radial_model_type}, lighting_mode={self.lighting_mode} | "
            f"radial_coeff_dim={self.radial_coeffs.size} | "
            f"sample_mse(raw={mse1:.6e}, corrected={best_mse:.6e})"
        )

    def generate_2d_prediction(self, chunk_rows=256, include_lighting=False):
        if self.shared_geometry is None or self.radial_coeffs is None:
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
            pred = self._eval_radial_curve(r_norm, coeffs=self.radial_coeffs, mode=self.radial_model_type)
            if include_lighting:
                xn, yn = self._coords_to_normalized(xx, yy)
                lit = self._eval_lighting_field_points(xn, yn)
                pred = pred * lit
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
            xn, yn = self._coords_to_normalized(xw, yw)
            l = self._eval_lighting_field_points(xn, yn)
            out[y0:y1, :] = (gain * (v * l) + bias).astype(np.float32, copy=False)
        return out

    def joint_residual(self, frame_params=None, shared_surface=None, stride=1, return_per_frame=False):
        """
        Joint residual skeleton for:
            I_k(x, y) ~= g_k * V(W_k(x, y)) + b_k
        """
        if self.num_captures < 1:
            raise RuntimeError("No captures available.")
        if self.shared_geometry is None or self.radial_coeffs is None:
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
            xn, yn = self._coords_to_normalized(xw, yw)
            l = self._eval_lighting_field_points(xn, yn)
            pred = gain * (v * l) + bias

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
    parser = argparse.ArgumentParser(description="Run improved traditional geometric vignetting baseline (V3).")
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default="synthetic_output",
        help="Directory containing sim_capture_{0,90,180,270}.npy and ground_truth_vignetting.npy",
    )
    parser.add_argument(
        "--radial-model",
        type=str,
        default="poly",
        choices=["poly", "spline_like"],
        help="Radial model type for pointwise fitting.",
    )
    parser.add_argument(
        "--lighting-mode",
        type=str,
        default="mesh",
        choices=["mesh", "poly2"],
        help="Lighting field mode.",
    )
    parser.add_argument(
        "--radial-fallback-binned",
        action="store_true",
        help="Use legacy radial-bin polynomial geometry fit instead of pointwise main path.",
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
    baseline.radial_model_type = args.radial_model
    baseline.lighting_mode = args.lighting_mode
    baseline.pointwise_fit_enabled = not args.radial_fallback_binned
    if baseline.lighting_mode == "mesh" and baseline.lighting_mesh is None:
        baseline.lighting_mesh = baseline.initialize_lighting_mesh()
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
