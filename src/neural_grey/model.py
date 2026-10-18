"""Neural Grey physical-neural field model from the paper production path."""

from typing import Optional
import numpy as np
import torch
from torch import nn


class ShadingModel(nn.Module):
    """Bounded shading and illumination fields with four-capture nuisance parameters."""

    def __init__(
        self,
        hidden_dim: int = 128,
        init_f_norm: float = 1.0,
        f_norm_min: float = 0.1,
        f_norm_max: float = 10.0,
        center_bound: float = 0.85,
        disable_prior: bool = False,
        disable_neural: bool = False,
        stop_gradient_wls: bool = False,
        disable_trust_network: bool = False,
    ):
        super().__init__()
        self.disable_prior = disable_prior
        self.disable_neural = disable_neural
        self.stop_gradient_wls = stop_gradient_wls
        self.disable_trust_network = disable_trust_network
        self.f_norm_min = float(f_norm_min)
        self.f_norm_max = float(f_norm_max)
        self.center_bound = float(center_bound)

        self.base_f_norm = float(init_f_norm)
        self.base_cx = 0.0
        self.base_cy = 0.0
        self.base_ax = 1.0
        self.base_ay = 1.0
        self.base_phi = 0.0
        self.delta_f_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.i0_raw = nn.Parameter(torch.tensor([6.0], dtype=torch.float32))
        self.delta_cx_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.delta_cy_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.delta_ax_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.delta_ay_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.delta_phi_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.harm2c_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.harm2s_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.harm4c_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.harm4s_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.p_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        self.trunk = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head_v = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head_v.weight)
        nn.init.zeros_(self.head_v.bias)

        self.trust_trunk = nn.Sequential(
            nn.Linear(7, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
        )
        self.trust_head = nn.Linear(48, 1)
        nn.init.zeros_(self.trust_head.weight)
        nn.init.zeros_(self.trust_head.bias)
        self.solver_gate_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

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
        x = float(max(x, 1e-6))
        if x > 20.0:
            return x
        return float(np.log(np.expm1(x)))

    def physics_params(self):
        device = self.delta_f_raw.device
        base_f = torch.tensor([self.base_f_norm], dtype=torch.float32, device=device)
        base_cx = torch.tensor([self.base_cx], dtype=torch.float32, device=device)
        base_cy = torch.tensor([self.base_cy], dtype=torch.float32, device=device)
        base_ax = torch.tensor([self.base_ax], dtype=torch.float32, device=device)
        base_ay = torch.tensor([self.base_ay], dtype=torch.float32, device=device)
        base_phi = torch.tensor([self.base_phi], dtype=torch.float32, device=device)

        f_scale = torch.exp(0.95 * torch.tanh(self.delta_f_raw))
        f_norm = torch.clamp(
            base_f * f_scale,
            min=self.f_norm_min,
            max=self.f_norm_max,
        )
        i0 = torch.clamp(torch.sigmoid(self.i0_raw), min=1e-4, max=1.0 - 1e-4)

        cx = torch.clamp(
            base_cx + 0.28 * torch.tanh(self.delta_cx_raw),
            min=-self.center_bound,
            max=self.center_bound,
        )
        cy = torch.clamp(
            base_cy + 0.28 * torch.tanh(self.delta_cy_raw),
            min=-self.center_bound,
            max=self.center_bound,
        )
        ax = torch.clamp(
            base_ax * torch.exp(0.34 * torch.tanh(self.delta_ax_raw)),
            min=0.50,
            max=2.10,
        )
        ay = torch.clamp(
            base_ay * torch.exp(0.34 * torch.tanh(self.delta_ay_raw)),
            min=0.50,
            max=2.10,
        )
        phi = base_phi + 0.60 * np.pi * torch.tanh(self.delta_phi_raw)
        return f_norm, i0, cx, cy, ax, ay, phi

    def set_priors(
        self,
        f_norm: float,
        cx: float,
        cy: float,
        ax: float = 1.0,
        ay: float = 1.0,
        phi: float = 0.0,
    ):
        self.base_f_norm = float(f_norm)
        self.base_cx = float(cx)
        self.base_cy = float(cy)
        self.base_ax = float(ax)
        self.base_ay = float(ay)
        self.base_phi = float(phi)

    def capture_params(self):
        gains = torch.exp(torch.clamp(self.gain_raw, min=-0.25, max=0.25))
        biases = 0.08 * torch.tanh(self.bias_raw)
        return gains, biases

    def capture_warp_params(self):
        # tx/ty in normalized coordinates, rot in radians, scale around 1.
        tx = 0.012 * torch.tanh(self.warp_tx_raw)
        ty = 0.012 * torch.tanh(self.warp_ty_raw)
        rot = 0.11 * torch.tanh(self.warp_rot_raw)
        scale = 1.0 + 0.03 * torch.tanh(self.warp_scale_raw)
        return tx, ty, rot, scale

    @staticmethod
    def _solver_basis(x: torch.Tensor, y: torch.Tensor):
        ones = torch.ones_like(x)
        return torch.cat([ones, x, y, x * y, x * x, y * y], dim=1)

    @staticmethod
    def _feature_stack(
        x: torch.Tensor, y: torch.Tensor, r_ell: torch.Tensor, theta: torch.Tensor
    ):
        theta_rad = theta * np.pi
        return torch.cat(
            [
                x,
                y,
                r_ell,
                torch.cos(theta_rad),
                torch.sin(theta_rad),
                torch.cos(2.0 * theta_rad),
                torch.sin(2.0 * theta_rad),
            ],
            dim=1,
        )

    def _weighted_solver_log_residual(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        r_ell: torch.Tensor,
        theta: torch.Tensor,
        v_phys: torch.Tensor,
        v_anchor: torch.Tensor,
    ):
        feat = self._feature_stack(x, y, r_ell, theta)
        trust_h = self.trust_trunk(feat)
        trust_weights = torch.sigmoid(self.trust_head(trust_h))
        if getattr(self, "disable_trust_network", False):
            trust_weights = torch.ones_like(trust_weights)

        edge_emphasis = torch.clamp((r_ell - 0.18) / 0.82, min=0.0, max=1.0) ** 2
        weights = (0.04 + 0.96 * trust_weights) * (0.15 + 0.85 * edge_emphasis)

        target = torch.log(torch.clamp(v_anchor, min=1e-4, max=1.0))
        target = target - torch.log(torch.clamp(v_phys, min=1e-4, max=1.0))
        basis = self._solver_basis(x, y)
        sqrt_w = torch.sqrt(torch.clamp(weights, min=1e-6))
        lhs = basis * sqrt_w
        rhs = target * sqrt_w
        coeffs = torch.linalg.lstsq(lhs, rhs).solution
        log_corr = basis @ coeffs
        log_corr = log_corr - torch.mean(log_corr)
        return log_corr, weights

    def forward_v(
        self, x: torch.Tensor, y: torch.Tensor, v_anchor: Optional[torch.Tensor] = None
    ):
        f_norm, i0, cx, cy, ax, ay, phi = self.physics_params()

        dx = x - cx
        dy = y - cy
        c = torch.cos(phi)
        s = torch.sin(phi)
        x_rot = c * dx + s * dy
        y_rot = -s * dx + c * dy

        r_ell = torch.sqrt((x_rot / ax) ** 2 + (y_rot / ay) ** 2 + 1e-12)
        theta = torch.atan2(y_rot, x_rot) / np.pi

        p = 4.0 + 2.0 * torch.tanh(self.p_raw)
        v_phys = i0 * torch.cos(torch.atan(r_ell / f_norm)) ** p
        v_phys = torch.clamp(v_phys, 0.0, 1.0)

        if self.disable_prior:
            v_phys = torch.ones_like(v_phys)

        if v_anchor is not None:
            solver_log, solver_weights = self._weighted_solver_log_residual(
                x, y, r_ell, theta, v_phys, v_anchor
            )
            solver_gate = torch.tanh(self.solver_gate_raw)
            solver_log = torch.clamp(solver_gate * solver_log, min=-0.30, max=0.30)
            if getattr(self, "stop_gradient_wls", False):
                solver_log = solver_log.detach()
                solver_weights = solver_weights.detach()
        else:
            solver_log = torch.zeros_like(v_phys)
            solver_weights = torch.ones_like(v_phys)

        v_solver = torch.clamp(v_phys * torch.exp(solver_log), 0.0, 1.0)

        feat = self._feature_stack(x, y, r_ell, theta)
        h = self.trunk(feat)
        theta_rad = theta * np.pi
        outer_env = torch.clamp((r_ell - 0.10) / 0.95, min=0.0, max=1.0) ** 1.6
        harm = (
            0.16 * torch.tanh(self.harm2c_raw) * torch.cos(2.0 * theta_rad)
            + 0.16 * torch.tanh(self.harm2s_raw) * torch.sin(2.0 * theta_rad)
            + 0.10 * torch.tanh(self.harm4c_raw) * torch.cos(4.0 * theta_rad)
            + 0.10 * torch.tanh(self.harm4s_raw) * torch.sin(4.0 * theta_rad)
        )
        dv = 0.18 * torch.tanh(self.head_v(h)) + outer_env * harm
        if getattr(self, "disable_neural", False):
            dv = torch.zeros_like(dv)
        v_pred = torch.clamp(v_phys * (1.0 + dv), 0.0, 1.0)
        return v_pred, v_phys, v_solver, dv, solver_log, solver_weights

    def light_log(self, x: torch.Tensor, y: torch.Tensor):
        c1, c2, c3, c4, c5 = torch.unbind(self.light_coef)
        poly = c1 * x + c2 * y + c3 * x * y + c4 * (x * x) + c5 * (y * y)
        h = self.light_trunk(torch.cat([x, y], dim=1))
        residual = 0.25 * torch.tanh(self.light_head(h))
        log_l = poly + residual
        # Center log-light each call to remove scale ambiguity with V.
        return log_l - torch.mean(log_l)

    def light_from_xy(self, x: torch.Tensor, y: torch.Tensor):
        return torch.exp(self.light_log(x, y))
