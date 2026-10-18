"""Joint per-configuration fitting, using the paper's production losses."""

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model import ShadingModel
from .preprocessing import CaptureSet


@dataclass
class FitConfig:
    epochs: int = 200
    batch_size: int = 8192
    sample_count: int = 240000
    hidden_dim: int = 96
    lr_net: float = 1e-3
    lr_phys: float = 1.2e-4
    lr_light: float = 6e-5
    lr_gain_bias: float = 1e-4
    lambda_phys: float = 1.0
    lambda_light: float = 0.1
    lambda_residual: float = 3.0
    lambda_anchor: float = 0.2
    lambda_gain_bias: float = 0.15
    lambda_warp: float = 0.2
    lambda_focal_prior: float = 0.15
    lambda_center_prior: float = 0.05
    edge_weight_strength: float = 1.25
    edge_weight_power: float = 2.0
    anchor_edge_power: float = 3.0
    anchor_robust_beta: float = 0.03
    # Retain the production auxiliary branch and loss accounting. It is
    # detached, has constant trust weights, and never changes the output map.
    solver_edge_power: float = 2.4
    lambda_solver: float = 0.0
    lambda_solver_gate: float = 0.0
    lambda_trust: float = 0.02

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.sample_count, self.hidden_dim) < 1:
            raise ValueError(
                "Epochs, batch size, sample count and width must be positive"
            )
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Fit configuration must be finite")
        if any(
            v < 0
            for k, v in values.items()
            if k.startswith(("lambda_", "edge_", "anchor_edge", "solver_edge"))
        ):
            raise ValueError("Loss weights and edge exponents must be nonnegative")
        if (
            min(
                self.lr_net,
                self.lr_phys,
                self.lr_light,
                self.lr_gain_bias,
                self.anchor_robust_beta,
            )
            <= 0
        ):
            raise ValueError("Learning rates and anchor beta must be positive")


def loss_terms(model, xb, yb, target, anchor, edge_weight, prior, config):
    """Four-capture reconstruction and the registered regularization terms."""
    c = config
    solver_edge = (torch.clamp(edge_weight - 1.0, min=0.0) + 1.0).clamp_min(
        1.0
    ) ** c.solver_edge_power
    anchor_edge = (torch.clamp(edge_weight - 1.0, min=0.0) + 1.0).clamp_min(
        1.0
    ) ** c.anchor_edge_power
    anchor_edge = anchor_edge / anchor_edge.mean().clamp_min(1e-6)
    vp, physical, solver, residual, solver_log, solver_weights = model.forward_v(
        xb, yb, v_anchor=anchor
    )
    gains, biases = model.capture_params()
    tx, ty, rotation, scale = model.capture_warp_params()
    images = []
    for k, (rx, ry) in enumerate(CaptureSet._rotated_coords(xb, yb)):
        cr, sr = torch.cos(rotation[k]), torch.sin(rotation[k])
        xw = (cr * rx + sr * ry) / scale[k] + tx[k]
        yw = (-sr * rx + cr * ry) / scale[k] + ty[k]
        illumination = model.light_from_xy(xw, yw)
        images.append((gains[k] * vp * illumination + biases[k]).clamp(0.0, 1.0))
    predicted = torch.cat(images, dim=1)
    f, _, cx, cy, _, _, _ = model.physics_params()
    terms = {
        "data": torch.mean(
            edge_weight * torch.mean((predicted - target) ** 2, dim=1, keepdim=True)
        ),
        "phys": F.mse_loss(vp, physical),
        "light": torch.mean(model.light_log(xb, yb) ** 2),
        "residual": torch.mean(residual**2),
        "anchor": torch.mean(
            anchor_edge
            * F.smooth_l1_loss(vp, anchor, reduction="none", beta=c.anchor_robust_beta)
        ),
        "solver": torch.mean(solver_edge * (solver - anchor) ** 2),
        "solver_gate": torch.mean(solver_edge * solver_log**2),
        "trust": torch.mean(solver_edge * (solver_weights - 0.5) ** 2),
        "gain_bias": torch.mean((gains - 1.0) ** 2) + torch.mean(biases**2),
        "warp": torch.mean(tx**2)
        + torch.mean(ty**2)
        + torch.mean(rotation**2)
        + torch.mean((scale - 1.0) ** 2),
        "focal_prior": torch.mean((f / prior[0].clamp_min(1e-6) - 1.0) ** 2),
        "center_prior": (cx - prior[1]) ** 2 + (cy - prior[2]) ** 2,
    }
    # Preserve production summation order for bounded extraction parity checks.
    loss = terms["data"]
    for name in (
        "phys",
        "light",
        "residual",
        "anchor",
        "solver",
        "solver_gate",
        "trust",
        "gain_bias",
        "warp",
        "focal_prior",
        "center_prior",
    ):
        loss = loss + getattr(c, "lambda_" + name) * terms[name]
    return loss, terms


class FitSession:
    """Fit one optical configuration. Ground truth is never a training input."""

    def __init__(
        self,
        captures,
        *,
        focal_length_mm=None,
        sensor_diagonal_mm=43.3,
        config=None,
        seed=42,
        device="auto",
    ):
        self.config = config or FitConfig()
        if focal_length_mm is not None and (
            not np.isfinite(focal_length_mm) or focal_length_mm <= 0
        ):
            raise ValueError("Focal length must be positive and finite")
        if not np.isfinite(sensor_diagonal_mm) or sensor_diagonal_mm <= 0:
            raise ValueError("Sensor diagonal must be positive and finite")
        self.focal_length_mm = (
            None if focal_length_mm is None else float(focal_length_mm)
        )
        self.sensor_diagonal_mm = float(sensor_diagonal_mm)
        self.seed = int(seed)
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.data = CaptureSet(captures)
        xy, target, fused = self.data._build_training_arrays(
            self.config.sample_count, self.seed
        )
        self.x = torch.from_numpy(xy[:, 0:1]).to(self.device)
        self.y = torch.from_numpy(xy[:, 1:2]).to(self.device)
        self.target = torch.from_numpy(target).to(self.device)
        self.anchor = torch.from_numpy(fused[:, None]).to(self.device)
        # The production joint fit uses an image-coordinate nominal focal
        # scale (0.62*image diagonal); physical mm metadata enters the selector.
        self.model = ShadingModel(
            hidden_dim=self.config.hidden_dim,
            init_f_norm=1.24,
            stop_gradient_wls=True,
            disable_trust_network=True,
        ).to(self.device)
        self.initial_prior = self.data._estimate_simple_cos4_prior()
        p = self.initial_prior
        self.model.set_priors(p["f_norm"], p["cx_norm"], p["cy_norm"])
        self.prior = torch.tensor(
            [p["f_norm"], p["cx_norm"], p["cy_norm"]],
            dtype=torch.float32,
            device=self.device,
        )
        radius = torch.sqrt(torch.clamp(self.x**2 + self.y**2, min=0.0))
        self.edge_weight = (
            1.0
            + self.config.edge_weight_strength
            * radius.clamp(0.0, 1.0) ** self.config.edge_weight_power
        )
        self.edge_weight = self.edge_weight / self.edge_weight.mean().clamp_min(1e-6)
        m, c = self.model, self.config
        physical = [
            m.delta_f_raw,
            m.i0_raw,
            m.delta_cx_raw,
            m.delta_cy_raw,
            m.delta_ax_raw,
            m.delta_ay_raw,
            m.delta_phi_raw,
            m.harm2c_raw,
            m.harm2s_raw,
            m.harm4c_raw,
            m.harm4s_raw,
            m.p_raw,
        ]
        self.optimizer = torch.optim.Adam(
            [
                dict(
                    params=list(m.trunk.parameters()) + list(m.head_v.parameters()),
                    lr=c.lr_net,
                ),
                dict(params=physical, lr=c.lr_phys),
                dict(
                    params=list(m.trust_trunk.parameters())
                    + list(m.trust_head.parameters())
                    + [m.solver_gate_raw],
                    lr=c.lr_net,
                ),
                dict(
                    params=[m.light_coef]
                    + list(m.light_trunk.parameters())
                    + list(m.light_head.parameters()),
                    lr=c.lr_light,
                ),
                dict(params=[m.gain_raw, m.bias_raw], lr=c.lr_gain_bias),
                dict(
                    params=[
                        m.warp_tx_raw,
                        m.warp_ty_raw,
                        m.warp_rot_raw,
                        m.warp_scale_raw,
                    ],
                    lr=c.lr_gain_bias,
                ),
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=c.epochs, eta_min=c.lr_net * 0.1
        )
        self.completed_epochs, self.updates = 0, 0
        self.history = []
        self.runtime_seconds = 0.0

    def run(self, *, stop_after=None, checkpoint=None, log_every=20):
        """Run additional epochs, optionally pausing at an epoch boundary."""
        if stop_after is not None and stop_after < 1:
            raise ValueError("stop_after must be positive")
        remaining = self.config.epochs - self.completed_epochs
        count = remaining if stop_after is None else min(stop_after, remaining)
        for _ in range(count):
            started = time.perf_counter()
            self.model.train()
            order = torch.randperm(len(self.x), device=self.device)
            accumulated, batches = {}, 0
            for start in range(0, len(order), self.config.batch_size):
                ids = order[start : start + self.config.batch_size]
                loss, terms = loss_terms(
                    self.model,
                    self.x[ids],
                    self.y[ids],
                    self.target[ids],
                    self.anchor[ids],
                    self.edge_weight[ids],
                    self.prior,
                    self.config,
                )
                if not torch.isfinite(loss).all():
                    raise RuntimeError("Nonfinite training loss; update not committed")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.model.parameters()
                ):
                    raise RuntimeError("Nonfinite gradient; update not committed")
                self.optimizer.step()
                for key, value in dict(total=loss, **terms).items():
                    accumulated[key] = accumulated.get(key, 0.0) + float(value.detach())
                batches += 1
                self.updates += 1
            self.scheduler.step()
            self.completed_epochs += 1
            self.runtime_seconds += time.perf_counter() - started
            self.history.append(
                dict(
                    epoch=self.completed_epochs,
                    updates=self.updates,
                    loss={k: v / batches for k, v in accumulated.items()},
                )
            )
            if log_every and (
                self.completed_epochs == 1
                or self.completed_epochs % log_every == 0
                or self.complete
            ):
                print(
                    f'Epoch {self.completed_epochs}/{self.config.epochs}: loss={accumulated["total"]/batches:.6g}',
                    flush=True,
                )
            if checkpoint is not None:
                self.save(checkpoint)
        return self

    @property
    def complete(self):
        return self.completed_epochs >= self.config.epochs

    @torch.no_grad()
    def base_map(self, chunk=65536):
        if chunk < 1:
            raise ValueError("chunk must be positive")
        self.model.eval()
        x, y = self.data._full_xy()
        result = np.empty(x.size, dtype=np.float32)
        for start in range(0, x.size, chunk):
            tx = torch.from_numpy(x.ravel()[start : start + chunk, None]).to(
                self.device
            )
            ty = torch.from_numpy(y.ravel()[start : start + chunk, None]).to(
                self.device
            )
            result[start : start + chunk] = (
                self.model.forward_v(tx, ty)[0].squeeze(1).cpu().numpy()
            )
        result = result.reshape(x.shape).clip(0.0, 1.0)
        return result / np.float32(max(float(result.max()), 1e-6))

    def calibration(self):
        from .calibration import Calibration
        from .selector import d2_low_frequency_stabilize, apply_radial_gauge_head

        base = self.base_map()
        stabilized = d2_low_frequency_stabilize(
            base, sigma=10.0 * max(base.shape) / 256.0
        )
        final = (
            stabilized
            if self.focal_length_mm is None
            else apply_radial_gauge_head(
                stabilized,
                self.focal_length_mm,
                sensor_diagonal_mm=self.sensor_diagonal_mm,
            )
        )
        metadata = dict(
            focal_length_mm=self.focal_length_mm,
            sensor_diagonal_mm=self.sensor_diagonal_mm,
            config=asdict(self.config),
            seed=self.seed,
            completed_epochs=self.completed_epochs,
            updates=self.updates,
            complete=self.complete,
            runtime_seconds=self.runtime_seconds,
            selector="d2_only" if self.focal_length_mm is None else "continuous_focal",
            device=str(self.device),
        )
        return Calibration(final, metadata, base, stabilized)

    def save(self, path):
        """Save inputs, model, optimizer, scheduler and RNG for self-contained resume."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            format_version=1,
            config=asdict(self.config),
            seed=self.seed,
            focal_length_mm=self.focal_length_mm,
            sensor_diagonal_mm=self.sensor_diagonal_mm,
            captures=torch.from_numpy(np.stack(self.data.raw_captures)),
            model={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            completed_epochs=self.completed_epochs,
            updates=self.updates,
            history=self.history,
            runtime_seconds=self.runtime_seconds,
            cpu_rng=torch.get_rng_state(),
            cuda_rng=(
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []
            ),
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def resume(cls, path, *, device="auto"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported Neural Grey training checkpoint")
        session = cls(
            payload["captures"].numpy(),
            focal_length_mm=payload["focal_length_mm"],
            sensor_diagonal_mm=payload["sensor_diagonal_mm"],
            config=FitConfig(**payload["config"]),
            seed=payload["seed"],
            device=device,
        )
        session.model.load_state_dict(payload["model"], strict=True)
        session.optimizer.load_state_dict(payload["optimizer"])
        session.scheduler.load_state_dict(payload["scheduler"])
        for key in ("completed_epochs", "updates", "history", "runtime_seconds"):
            setattr(session, key, payload[key])
        torch.set_rng_state(payload["cpu_rng"])
        if session.device.type == "cuda" and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return session
