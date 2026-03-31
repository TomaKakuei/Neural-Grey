"""
maskingV1.3 (commented copy)

Purpose:
- Enhanced version of maskingV1.2 with stronger optimization/config controls.

Core modeling:
1) Shared physics-inspired vignetting field.
2) Illumination compensation and frame-wise nuisance parameter handling.
3) Multi-term training objective and evaluation/export path.

Usage notes:
- This copy is for organized reading and version tracking in Neural-Grey.
- Original source file is kept unchanged.
"""
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


if torch is not None:
    class AsymmetricVignettingPINNV12(nn.Module):
        """
        maskingV1.3
        - V(x,y): physics-informed asymmetric vignetting field
        - L(x,y): canonical illumination polynomial
        - I_k(x,y) = gain_k * V(x,y) * L(R_k[x,y]) + bias_k
          where R_k are 0/90/180/270 coordinate transforms.
        """

        def __init__(
            self,
            hidden_dim: int = 128,
            init_f_norm: float = 1.0,
            f_norm_min: float = 0.1,
            f_norm_max: float = 10.0,
            center_bound: float = 0.85,
        ):
            """Initialize model/analyzer state and default hyper-parameters."""
            super().__init__()
            self.f_norm_min = float(f_norm_min)
            self.f_norm_max = float(f_norm_max)
            self.center_bound = float(center_bound)

            self.f_raw = nn.Parameter(torch.tensor([self._inverse_softplus_scalar(init_f_norm)], dtype=torch.float32))
            self.i0_raw = nn.Parameter(torch.tensor([3.8], dtype=torch.float32))
            self.cx_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            self.cy_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            self.ax_raw = nn.Parameter(torch.tensor([self._inverse_softplus_scalar(1.0)], dtype=torch.float32))
            self.ay_raw = nn.Parameter(torch.tensor([self._inverse_softplus_scalar(1.0)], dtype=torch.float32))
            self.phi_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

            self.trunk = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.head_v = nn.Linear(hidden_dim, 1)

            # log L = c1*x + c2*y + c3*x*y + c4*x^2 + c5*y^2
            self.light_coef = nn.Parameter(torch.zeros((5,), dtype=torch.float32))
            self.light_trunk = nn.Sequential(
                nn.Linear(2, 48),
                nn.Tanh(),
                nn.Linear(48, 48),
                nn.Tanh(),
            )
            self.light_head = nn.Linear(48, 1)

            # Per-capture exposure and black drift correction.
            self.gain_raw = nn.Parameter(torch.zeros((4,), dtype=torch.float32))
            self.bias_raw = nn.Parameter(torch.zeros((4,), dtype=torch.float32))
            # Per-capture small geometric correction on illumination coordinates.
            self.warp_tx_raw = nn.Parameter(torch.zeros((4,), dtype=torch.float32))
            self.warp_ty_raw = nn.Parameter(torch.zeros((4,), dtype=torch.float32))
            self.warp_rot_raw = nn.Parameter(torch.zeros((4,), dtype=torch.float32))
            self.warp_scale_raw = nn.Parameter(torch.zeros((4,), dtype=torch.float32))

        @staticmethod
        def _inverse_softplus_scalar(x: float) -> float:
            """Convert a positive scalar to pre-softplus space for stable parameter initialization."""
            x = float(max(x, 1e-6))
            if x > 20.0:
                return x
            return float(np.log(np.expm1(x)))

        def physics_params(self):
            """Decode constrained physical optics parameters used by the vignette forward model."""
            f_norm = F.softplus(self.f_raw) + 1e-6
            f_norm = torch.clamp(f_norm, min=self.f_norm_min, max=self.f_norm_max)
            i0 = torch.clamp(torch.sigmoid(self.i0_raw), min=1e-4, max=1.0 - 1e-4)

            cx = self.center_bound * torch.tanh(self.cx_raw)
            cy = self.center_bound * torch.tanh(self.cy_raw)
            ax = torch.clamp(F.softplus(self.ax_raw) + 1e-6, min=0.45, max=2.2)
            ay = torch.clamp(F.softplus(self.ay_raw) + 1e-6, min=0.45, max=2.2)
            phi = np.pi * torch.tanh(self.phi_raw)
            return f_norm, i0, cx, cy, ax, ay, phi

        def capture_params(self):
            """Decode per-capture gain and bias parameters with bounded ranges."""
            gains = torch.exp(torch.clamp(self.gain_raw, min=-0.25, max=0.25))
            biases = 0.08 * torch.tanh(self.bias_raw)
            return gains, biases

        def capture_warp_params(self):
            """Decode per-capture illumination coordinate warp parameters (tx, ty, rot, scale)."""
            # tx/ty in normalized coordinates, rot in radians, scale around 1.
            tx = 0.012 * torch.tanh(self.warp_tx_raw)
            ty = 0.012 * torch.tanh(self.warp_ty_raw)
            rot = 0.11 * torch.tanh(self.warp_rot_raw)
            scale = 1.0 + 0.03 * torch.tanh(self.warp_scale_raw)
            return tx, ty, rot, scale

        def forward_v(self, x: torch.Tensor, y: torch.Tensor):
            """Evaluate vignette field V and return learned output, physics prior, and residual branch."""
            f_norm, i0, cx, cy, ax, ay, phi = self.physics_params()

            dx = x - cx
            dy = y - cy
            c = torch.cos(phi)
            s = torch.sin(phi)
            x_rot = c * dx + s * dy
            y_rot = -s * dx + c * dy

            r_ell = torch.sqrt((x_rot / ax) ** 2 + (y_rot / ay) ** 2 + 1e-12)
            theta = torch.atan2(y_rot, x_rot) / np.pi

            v_phys = i0 * torch.cos(torch.atan(r_ell / f_norm)) ** 4
            v_phys = torch.clamp(v_phys, 0.0, 1.0)

            feat = torch.cat([x, y, r_ell, theta], dim=1)
            h = self.trunk(feat)
            dv = 0.10 * torch.tanh(self.head_v(h))
            v_pred = torch.clamp(v_phys * (1.0 + dv), 0.0, 1.0)
            return v_pred, v_phys, dv

        def light_log(self, x: torch.Tensor, y: torch.Tensor):
            """Evaluate log illumination field at normalized coordinates and remove global scale ambiguity."""
            c1, c2, c3, c4, c5 = torch.unbind(self.light_coef)
            poly = c1 * x + c2 * y + c3 * x * y + c4 * (x * x) + c5 * (y * y)
            h = self.light_trunk(torch.cat([x, y], dim=1))
            residual = 0.25 * torch.tanh(self.light_head(h))
            log_l = poly + residual
            # Center log-light each call to remove scale ambiguity with V.
            return log_l - torch.mean(log_l)

        def light_from_xy(self, x: torch.Tensor, y: torch.Tensor):
            """Evaluate illumination field L(x,y) in linear domain by exponentiating log-light."""
            return torch.exp(self.light_log(x, y))
else:
    class AsymmetricVignettingPINNV12:
        def __init__(self, *args, **kwargs):
            """Initialize model/analyzer state and default hyper-parameters."""
            raise ImportError("PyTorch is required for AsymmetricVignettingPINNV12.")


@dataclass
class V12Config:
    """Training hyper-parameter container for masking v1.x experiments."""
    epochs: int = 600
    batch_size: int = 8192
    sample_count: Optional[int] = 240000
    hidden_dim: int = 96
    lr_net: float = 1e-3
    lr_phys: float = 1.2e-4
    lr_light: float = 6e-5
    lr_gain_bias: float = 1e-4
    lambda_phys: float = 1.0
    lambda_light: float = 0.1
    lambda_residual: float = 0.08
    lambda_anchor: float = 0.2
    lambda_gain_bias: float = 0.15
    lambda_warp: float = 0.2


class MaskingV12Analyzer:
    """End-to-end data loader, trainer, predictor, and evaluator for masking v1.x."""
    def __init__(self):
        """Initialize model/analyzer state and default hyper-parameters."""
        self.raw_captures = None
        self.captures = None
        self.v_fused = None
        self.max_brightness = 1.0
        self.h = None
        self.w = None

        self.model = None
        self.device = None
        self.runtime_sec = None
        self.training_history = []

        self.sensor_half_diag_mm = None
        self.sensor_half_diag_px = None
        self.img_center_px = None

        self.learned_f_norm = None
        self.learned_f_mm = None
        self.learned_i0 = None
        self.learned_center_px = None
        self.learned_ax = None
        self.learned_ay = None
        self.learned_phi_deg = None
        self.learned_gains = None
        self.learned_biases = None
        self.learned_warp = None

    @staticmethod
    def _estimate_center(img: np.ndarray):
        """Estimate bright-region center robustly using blurred intensity weighting."""
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
        """Apply subpixel translation with reflected border handling."""
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
        """Fuse rotated captures with trimmed mean to suppress outlier frames/pixels."""
        # captures shape: [4,H,W], trimmed mean drops min/max per pixel.
        s = np.sort(captures, axis=0)
        return np.mean(s[1:3], axis=0).astype(np.float32, copy=False)

    def load_from_synthetic_dir(self, root_dir: str):
        """Load 4 rotated captures and run normalization/registration/fusion preprocessing."""
        caps = []
        for angle in (0, 90, 180, 270):
            p = os.path.join(root_dir, f"sim_capture_{angle}.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            caps.append(np.load(p).astype(np.float32))

        shapes = {c.shape for c in caps}
        if len(shapes) != 1:
            raise ValueError("All 4 captures must have identical shape.")
        h, w = caps[0].shape
        if h != w:
            raise ValueError(f"maskingV1.3 expects square images for exact 90deg coordinate transforms, got {h}x{w}.")

        self.raw_captures = [np.clip(c, 0.0, 1.0) for c in caps]

        # Step 1: exposure normalization by robust high percentile.
        p95 = np.array([np.percentile(c, 95.0) for c in self.raw_captures], dtype=np.float32)
        target = float(np.median(p95))
        exp_norm = []
        for c, p in zip(self.raw_captures, p95):
            s = target / max(float(p), 1e-6)
            exp_norm.append(np.clip(c * np.float32(s), 0.0, 1.0).astype(np.float32, copy=False))

        # Step 2: conservative subpixel registration for fusion only.
        # We only accept tiny shifts; large estimates are likely caused by rotated lighting pattern.
        ref = cv2.GaussianBlur(exp_norm[0], (0, 0), 12).astype(np.float32)
        reg_for_fusion = [exp_norm[0]]
        max_shift = 2.5
        for i in range(1, 4):
            src = cv2.GaussianBlur(exp_norm[i], (0, 0), 12).astype(np.float32)
            (dx, dy), _ = cv2.phaseCorrelate(ref, src)
            if np.isfinite(dx) and np.isfinite(dy) and np.hypot(dx, dy) <= max_shift:
                reg_for_fusion.append(self._shift_image(exp_norm[i], dx=-float(dx), dy=-float(dy)))
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

    def _build_training_arrays(self, sample_count: Optional[int], seed: int):
        """Build normalized coordinate samples and aligned training targets for optimization."""
        h, w = self.h, self.w
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        half_diag_px = float(np.hypot(cx, cy))

        yy, xx = np.indices((h, w), dtype=np.float32)
        x_norm = ((xx - np.float32(cx)) / np.float32(max(half_diag_px, 1e-6))).ravel()
        y_norm = ((yy - np.float32(cy)) / np.float32(max(half_diag_px, 1e-6))).ravel()

        y4 = np.stack([c.ravel() for c in self.captures], axis=1).astype(np.float32)
        y4 = np.clip(y4 / np.float32(self.max_brightness), 0.0, 1.0)
        vf = np.clip(self.v_fused.ravel() / np.float32(self.max_brightness), 0.0, 1.0).astype(np.float32)

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
        """Return full-image normalized coordinate grids in canonical orientation."""
        cx = (self.w - 1) * 0.5
        cy = (self.h - 1) * 0.5
        scale = max(float(np.hypot(cx, cy)), 1e-6)
        yy, xx = np.indices((self.h, self.w), dtype=np.float32)
        x = (xx - np.float32(cx)) / np.float32(scale)
        y = (yy - np.float32(cy)) / np.float32(scale)
        return x.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def _rotated_coords(x: torch.Tensor, y: torch.Tensor):
        """Return coordinates corresponding to 0/90/180/270-degree rotation convention."""
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

    def fit(
        self,
        focal_length_mm: float,
        sensor_diag_mm: float,
        cfg: V12Config,
        seed: int = 42,
        log_every: int = 100,
        progress_json: Optional[str] = None,
    ):
        """Train the decomposition model with multi-term losses and record learned physical/nuisance parameters."""
        if torch is None:
            raise ImportError("PyTorch is required.")
        if self.captures is None:
            raise RuntimeError("No captures loaded.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        xy_np, y_np, vf_np = self._build_training_arrays(cfg.sample_count, seed=seed)
        x_all = torch.from_numpy(xy_np[:, 0:1]).to(device)
        y_all = torch.from_numpy(xy_np[:, 1:2]).to(device)
        y4_all = torch.from_numpy(y_np).to(device)
        vf_all = torch.from_numpy(vf_np[:, None]).to(device)
        n = x_all.shape[0]

        self.sensor_half_diag_mm = float(sensor_diag_mm) * 0.5
        init_f_norm = float(focal_length_mm) / max(self.sensor_half_diag_mm, 1e-6)
        model = AsymmetricVignettingPINNV12(hidden_dim=cfg.hidden_dim, init_f_norm=init_f_norm).to(device)

        # Initialize learnable optical center from robust fused target.
        cxf, cyf = self._estimate_center(self.v_fused)
        cx0, cy0 = self.img_center_px
        s = max(float(self.sensor_half_diag_px), 1e-6)
        cx_norm = np.clip((cxf - cx0) / s, -0.78, 0.78)
        cy_norm = np.clip((cyf - cy0) / s, -0.78, 0.78)
        with torch.no_grad():
            model.cx_raw.fill_(float(np.arctanh(cx_norm / model.center_bound)))
            model.cy_raw.fill_(float(np.arctanh(cy_norm / model.center_bound)))

        net_params = list(model.trunk.parameters()) + list(model.head_v.parameters())
        phys_params = [model.f_raw, model.i0_raw, model.cx_raw, model.cy_raw, model.ax_raw, model.ay_raw, model.phi_raw]
        light_params = [model.light_coef] + list(model.light_trunk.parameters()) + list(model.light_head.parameters())
        gain_bias_params = [model.gain_raw, model.bias_raw]
        warp_params = [model.warp_tx_raw, model.warp_ty_raw, model.warp_rot_raw, model.warp_scale_raw]

        optimizer = torch.optim.Adam(
            [
                {"params": net_params, "lr": cfg.lr_net},
                {"params": phys_params, "lr": cfg.lr_phys},
                {"params": light_params, "lr": cfg.lr_light},
                {"params": gain_bias_params, "lr": cfg.lr_gain_bias},
                {"params": warp_params, "lr": cfg.lr_gain_bias},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_net * 0.1)

        self.training_history.clear()
        t0 = time.perf_counter()

        print("\n=== maskingV1.3 Training ===")
        print(
            f"[-] Device={device} | Samples={n} | Epochs={cfg.epochs} | Batch={cfg.batch_size} | "
            f"Hidden={cfg.hidden_dim}"
        )
        print("[-] Upgrades: exposure norm + subpixel center alignment + trimmed fusion anchor + learnable center + I=V*L decomposition.")

        log_every = max(1, int(log_every))

        for epoch in range(1, cfg.epochs + 1):
            perm = torch.randperm(n, device=device)
            s_l = s_ld = s_lp = s_ll = s_lr = s_la = s_lgb = s_lw = 0.0
            nb = 0

            for sidx in range(0, n, cfg.batch_size):
                p = perm[sidx:sidx + cfg.batch_size]
                xb = x_all[p]
                yb = y_all[p]
                tgt = y4_all[p]
                v_anchor = vf_all[p]

                v_pred, v_phys, dv = model.forward_v(xb, yb)
                gains, biases = model.capture_params()
                tx, ty, rot, scale = model.capture_warp_params()

                i_parts = []
                for k, (rx, ry) in enumerate(self._rotated_coords(xb, yb)):
                    cr = torch.cos(rot[k])
                    sr = torch.sin(rot[k])
                    xw = cr * rx + sr * ry
                    yw = -sr * rx + cr * ry
                    xw = xw / scale[k] + tx[k]
                    yw = yw / scale[k] + ty[k]
                    lk = model.light_from_xy(xw, yw)
                    ik = torch.clamp(gains[k] * v_pred * lk + biases[k], 0.0, 1.0)
                    i_parts.append(ik)
                i_pred = torch.cat(i_parts, dim=1)

                loss_data = F.mse_loss(i_pred, tgt)
                loss_phys = F.mse_loss(v_pred, v_phys)
                # local light regularization on current batch canonical coords
                log_l = model.light_log(xb, yb)
                loss_light = torch.mean(log_l ** 2)
                loss_res = torch.mean(dv ** 2)
                loss_anchor = F.mse_loss(v_pred, v_anchor)
                loss_gb = torch.mean((gains - 1.0) ** 2) + torch.mean(biases ** 2)
                loss_warp = (
                    torch.mean(tx ** 2)
                    + torch.mean(ty ** 2)
                    + torch.mean(rot ** 2)
                    + torch.mean((scale - 1.0) ** 2)
                )

                loss = (
                    loss_data
                    + cfg.lambda_phys * loss_phys
                    + cfg.lambda_light * loss_light
                    + cfg.lambda_residual * loss_res
                    + cfg.lambda_anchor * loss_anchor
                    + cfg.lambda_gain_bias * loss_gb
                    + cfg.lambda_warp * loss_warp
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                s_l += float(loss.item())
                s_ld += float(loss_data.item())
                s_lp += float(loss_phys.item())
                s_ll += float(loss_light.item())
                s_lr += float(loss_res.item())
                s_la += float(loss_anchor.item())
                s_lgb += float(loss_gb.item())
                s_lw += float(loss_warp.item())
                nb += 1

            scheduler.step()

            avg_l = s_l / max(nb, 1)
            avg_ld = s_ld / max(nb, 1)
            avg_lp = s_lp / max(nb, 1)
            avg_ll = s_ll / max(nb, 1)
            avg_lr = s_lr / max(nb, 1)
            avg_la = s_la / max(nb, 1)
            avg_lgb = s_lgb / max(nb, 1)
            avg_lw = s_lw / max(nb, 1)

            if epoch == 1 or epoch % log_every == 0 or epoch == cfg.epochs:
                f_norm, i0, cx, cy, ax, ay, phi = model.physics_params()
                gains, biases = model.capture_params()
                tx, ty, rot, scale = model.capture_warp_params()
                f_mm = float(f_norm.item()) * self.sensor_half_diag_mm
                elapsed = float(time.perf_counter() - t0)
                eta_sec = (elapsed / max(epoch, 1)) * max(cfg.epochs - epoch, 0)
                print(
                    f"Epoch {epoch:4d}/{cfg.epochs} | "
                    f"L={avg_l:.6e} Ld={avg_ld:.6e} Lp={avg_lp:.6e} "
                    f"Ll={avg_ll:.6e} Lr={avg_lr:.6e} La={avg_la:.6e} "
                    f"Lgb={avg_lgb:.6e} Lw={avg_lw:.6e} | "
                    f"f~{f_mm:.2f}mm i0={float(i0.item()):.4f} "
                    f"cx={float(cx.item()):.3f} cy={float(cy.item()):.3f} "
                    f"ax={float(ax.item()):.3f} ay={float(ay.item()):.3f} "
                    f"phi={float(phi.item()*180.0/np.pi):.2f}deg | "
                    f"g={','.join([f'{float(v):.3f}' for v in gains.detach().cpu().numpy()])} "
                    f"tx={','.join([f'{float(v):.4f}' for v in tx.detach().cpu().numpy()])} | "
                    f"elapsed={elapsed:.1f}s eta={eta_sec:.1f}s"
                )

                if progress_json:
                    progress = {
                        "epoch": int(epoch),
                        "epochs": int(cfg.epochs),
                        "elapsed_sec": elapsed,
                        "eta_sec": eta_sec,
                        "loss": {
                            "total": float(avg_l),
                            "data": float(avg_ld),
                            "phys": float(avg_lp),
                            "light": float(avg_ll),
                            "residual": float(avg_lr),
                            "anchor": float(avg_la),
                            "gain_bias": float(avg_lgb),
                            "warp": float(avg_lw),
                        },
                        "learned": {
                            "f_mm": float(f_mm),
                            "i0": float(i0.item()),
                            "cx_norm": float(cx.item()),
                            "cy_norm": float(cy.item()),
                            "ax": float(ax.item()),
                            "ay": float(ay.item()),
                            "phi_deg": float(phi.item() * 180.0 / np.pi),
                            "gains": [float(v) for v in gains.detach().cpu().numpy().tolist()],
                            "biases": [float(v) for v in biases.detach().cpu().numpy().tolist()],
                            "tx": [float(v) for v in tx.detach().cpu().numpy().tolist()],
                            "ty": [float(v) for v in ty.detach().cpu().numpy().tolist()],
                        },
                        "device": str(device),
                    }
                    with open(progress_json, "w", encoding="utf-8") as f:
                        json.dump(progress, f, indent=2)

            self.training_history.append(
                (
                    epoch,
                    avg_l,
                    avg_ld,
                    avg_lp,
                    avg_ll,
                    avg_lr,
                    avg_la,
                    avg_lgb,
                    avg_lw,
                )
            )

        self.runtime_sec = float(time.perf_counter() - t0)
        self.model = model

        f_norm, i0, cx, cy, ax, ay, phi = model.physics_params()
        gains, biases = model.capture_params()
        tx, ty, rot, scale = model.capture_warp_params()
        self.learned_f_norm = float(f_norm.item())
        self.learned_f_mm = self.learned_f_norm * self.sensor_half_diag_mm
        self.learned_i0 = float(i0.item())
        self.learned_ax = float(ax.item())
        self.learned_ay = float(ay.item())
        self.learned_phi_deg = float(phi.item() * 180.0 / np.pi)
        self.learned_gains = [float(v) for v in gains.detach().cpu().numpy().tolist()]
        self.learned_biases = [float(v) for v in biases.detach().cpu().numpy().tolist()]
        self.learned_warp = {
            "tx": [float(v) for v in tx.detach().cpu().numpy().tolist()],
            "ty": [float(v) for v in ty.detach().cpu().numpy().tolist()],
            "rot_deg": [float(v * 180.0 / np.pi) for v in rot.detach().cpu().numpy().tolist()],
            "scale": [float(v) for v in scale.detach().cpu().numpy().tolist()],
        }

        cx0, cy0 = self.img_center_px
        scale = self.sensor_half_diag_px
        self.learned_center_px = (float(cx0 + cx.item() * scale), float(cy0 + cy.item() * scale))

        print(
            f"[-] Done in {self.runtime_sec:.2f}s | "
            f"f~{self.learned_f_mm:.2f}mm | center=({self.learned_center_px[0]:.2f},{self.learned_center_px[1]:.2f}) | "
            f"ax={self.learned_ax:.3f} ay={self.learned_ay:.3f} phi={self.learned_phi_deg:.2f}deg"
        )

    def _predict_v_map(self, chunk: int = 300000):
        """Predict full-resolution vignette map V with chunked inference."""
        x_map, y_map = self._full_xy()
        flat_x = x_map.ravel()
        flat_y = y_map.ravel()
        out = np.empty_like(flat_x, dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, flat_x.shape[0], chunk):
                tx = torch.from_numpy(flat_x[i:i + chunk, None]).to(self.device)
                ty = torch.from_numpy(flat_y[i:i + chunk, None]).to(self.device)
                vp, _, _ = self.model.forward_v(tx, ty)
                out[i:i + chunk] = vp.squeeze(1).cpu().numpy()

        v = out.reshape(self.h, self.w)
        v = np.clip(v, 0.0, 1.0).astype(np.float32, copy=False)
        vmax = float(np.max(v))
        if vmax > 1e-6:
            v = v / np.float32(vmax)
        return v.astype(np.float32, copy=False)

    def _predict_l_map(self):
        """Predict full-resolution illumination map L in canonical coordinates."""
        x_map, y_map = self._full_xy()
        tx = torch.from_numpy(x_map.reshape(-1, 1)).to(self.device)
        ty = torch.from_numpy(y_map.reshape(-1, 1)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            l = self.model.light_from_xy(tx, ty).squeeze(1).cpu().numpy().reshape(self.h, self.w).astype(np.float32)
        return np.clip(l, 1e-6, 1e6).astype(np.float32, copy=False)

    def predict(self):
        """Generate V, L, and four reconstructed capture predictions using learned parameters."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        v = self._predict_v_map()
        l = self._predict_l_map()
        gains = np.array(self.learned_gains, dtype=np.float32)
        biases = np.array(self.learned_biases, dtype=np.float32)
        warp = self.learned_warp

        i_maps = []
        cx = (self.w - 1) * 0.5
        cy = (self.h - 1) * 0.5
        px_scale = max(float(self.sensor_half_diag_px), 1e-6)
        for k in range(4):
            lk = np.rot90(l, k).astype(np.float32, copy=False)
            rot_deg = float(warp["rot_deg"][k])
            sc = float(warp["scale"][k])
            tx_px = float(warp["tx"][k]) * px_scale
            ty_px = float(warp["ty"][k]) * px_scale
            m = cv2.getRotationMatrix2D((cx, cy), rot_deg, sc).astype(np.float32)
            m[0, 2] += np.float32(tx_px)
            m[1, 2] += np.float32(ty_px)
            lk_w = cv2.warpAffine(
                lk,
                m,
                (self.w, self.h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ).astype(np.float32, copy=False)
            i = gains[k] * v * lk_w + biases[k]
            i_maps.append(np.clip(i, 0.0, 1.0).astype(np.float32, copy=False))
        return v, l, i_maps

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
        """Compute R2-based fit percentage and RMSE between two maps."""
        yt = y_true.ravel().astype(np.float32)
        yp = y_pred.ravel().astype(np.float32)
        sse = float(np.sum((yt - yp) ** 2))
        sst = float(np.sum((yt - float(np.mean(yt))) ** 2))
        r2 = float(1.0 - sse / max(sst, 1e-12))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        return {"r2": r2, "fit_percent": 100.0 * r2, "rmse": rmse}

    def evaluate(self, gt_v: Optional[np.ndarray] = None):
        """Evaluate reconstructed captures (and optional GT vignette) and return summary metrics."""
        v, l, i_maps = self.predict()
        out = {"i_metrics": [], "i_mean": None, "v_metrics": None}
        i_fits = []
        i_rmses = []
        for k in range(4):
            m = self._metrics(self.captures[k], i_maps[k])
            out["i_metrics"].append(m)
            i_fits.append(m["fit_percent"])
            i_rmses.append(m["rmse"])
        out["i_mean"] = {
            "fit_percent": float(np.mean(i_fits)),
            "rmse": float(np.mean(i_rmses)),
        }
        if gt_v is not None:
            out["v_metrics"] = self._metrics(np.asarray(gt_v, dtype=np.float32), v.astype(np.float32))
        return out


def main():
    """CLI entry: parse arguments, train model, evaluate, and optionally export JSON summary."""
    parser = argparse.ArgumentParser(description="maskingV1.3: robust preprocessing + decomposition PINN (best-default config)")
    parser.add_argument("--synthetic-dir", type=str, required=True)
    parser.add_argument("--gt-v", type=str, default=None)
    parser.add_argument("--focal-mm", type=float, default=None)
    parser.add_argument("--sensor-diag-mm", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--sample-count", type=int, default=240000)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--progress-json", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    if torch is None:
        print("[Error] PyTorch not installed.")
        raise SystemExit(1)

    analyzer = MaskingV12Analyzer()
    analyzer.load_from_synthetic_dir(args.synthetic_dir)

    h, w = analyzer.h, analyzer.w
    sensor_diag = float(np.hypot(h, w)) if args.sensor_diag_mm is None else float(args.sensor_diag_mm)
    focal = 0.62 * sensor_diag if args.focal_mm is None else float(args.focal_mm)

    cfg = V12Config(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        sample_count=(None if int(args.sample_count) < 0 else int(args.sample_count)),
        hidden_dim=int(args.hidden_dim),
    )

    analyzer.fit(
        focal_length_mm=focal,
        sensor_diag_mm=sensor_diag,
        cfg=cfg,
        seed=42,
        log_every=int(args.log_every),
        progress_json=args.progress_json,
    )

    gt_v = None
    if args.gt_v is not None:
        gt_v = np.load(args.gt_v).astype(np.float32)
    report = analyzer.evaluate(gt_v=gt_v)

    print("[Eval-I] Per capture:")
    for i, m in enumerate(report["i_metrics"]):
        ang = (0, 90, 180, 270)[i]
        print(f"  - {ang:>3}deg: Fit={m['fit_percent']:.4f}% | R2={m['r2']:.6f} | RMSE={m['rmse']:.6f}")
    print(
        f"[Eval-I-Mean] Fit={report['i_mean']['fit_percent']:.4f}% | "
        f"RMSE={report['i_mean']['rmse']:.6f}"
    )
    if report["v_metrics"] is not None:
        vm = report["v_metrics"]
        print(f"[Eval-V] Fit={vm['fit_percent']:.4f}% | R2={vm['r2']:.6f} | RMSE={vm['rmse']:.6f}")

    summary = {
        "synthetic_dir": args.synthetic_dir,
        "shape": [int(h), int(w)],
        "config": asdict(cfg),
        "runtime_sec": float(analyzer.runtime_sec),
        "device": str(analyzer.device),
        "learned": {
            "f_norm": float(analyzer.learned_f_norm),
            "f_mm": float(analyzer.learned_f_mm),
            "i0": float(analyzer.learned_i0),
            "center_px": [float(analyzer.learned_center_px[0]), float(analyzer.learned_center_px[1])],
            "ax": float(analyzer.learned_ax),
            "ay": float(analyzer.learned_ay),
            "phi_deg": float(analyzer.learned_phi_deg),
            "gains": [float(v) for v in analyzer.learned_gains],
            "biases": [float(v) for v in analyzer.learned_biases],
            "warp": analyzer.learned_warp,
        },
        "metrics": report,
    }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Saved] {args.out_json}")


if __name__ == "__main__":
    main()

