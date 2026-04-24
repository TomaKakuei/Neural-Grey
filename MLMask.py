import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from typing import Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

if torch is not None:
    def _inverse_softplus_scalar(x):
        # Stable inverse for softplus(y)=x. For large x, softplus(y)~y.
        x = float(max(x, 1e-6))
        if x > 20.0:
            return x
        return float(np.log(np.expm1(x)))

    class PINNVignettingMLP(nn.Module):
        def __init__(
            self,
            hidden_dim=32,
            learnable_physics=True,
            init_f_norm=1.0,
            init_i0=1.0,
            f_norm_min=0.1,
            f_norm_max=10.0,
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.learnable_physics = learnable_physics
            self.f_norm_min = float(f_norm_min)
            self.f_norm_max = float(f_norm_max)

            if learnable_physics:
                init_f_clip = float(np.clip(init_f_norm, self.f_norm_min, self.f_norm_max))
                init_f_raw = _inverse_softplus_scalar(init_f_clip)
                init_i0_clip = float(np.clip(init_i0, 1e-4, 1.0 - 1e-4))
                init_i0_raw = np.log(init_i0_clip / (1.0 - init_i0_clip))
                self.f_raw = nn.Parameter(torch.tensor([init_f_raw], dtype=torch.float32))
                self.i0_raw = nn.Parameter(torch.tensor([init_i0_raw], dtype=torch.float32))
            else:
                self.register_buffer("f_const", torch.tensor([float(np.clip(init_f_norm, self.f_norm_min, self.f_norm_max))], dtype=torch.float32))
                self.register_buffer("i0_const", torch.tensor([float(np.clip(init_i0, 1e-4, 1.0 - 1e-4))], dtype=torch.float32))

        def forward(self, r_norm):
            # Output in [0, 1] for relative illumination.
            return torch.sigmoid(self.net(r_norm))

        def get_physics_params(self):
            if self.learnable_physics:
                f_norm = F.softplus(self.f_raw) + 1e-6
                f_norm = torch.clamp(f_norm, min=self.f_norm_min, max=self.f_norm_max)
                i0 = torch.clamp(torch.sigmoid(self.i0_raw), min=1e-4, max=1.0 - 1e-4)
                return f_norm, i0
            return self.f_const, self.i0_const
else:
    class PINNVignettingMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PINNVignettingMLP.")


def _decode_color_desc(color_desc):
    if isinstance(color_desc, bytes):
        desc = color_desc.decode("ascii", errors="ignore")
    else:
        desc = str(color_desc)
    return desc.replace("\x00", "")


def _channel_site_map(raw_pattern, color_desc):
    pattern = np.asarray(raw_pattern, dtype=np.int32)
    if pattern.shape != (2, 2):
        raise ValueError(f"Unsupported CFA pattern shape {pattern.shape}; expected Bayer 2x2 pattern.")

    desc = _decode_color_desc(color_desc)
    max_index = int(np.max(pattern))
    if len(desc) <= max_index:
        raise ValueError(
            f"color_desc length {len(desc)} is incompatible with raw_pattern max index {max_index}."
        )

    sites = {"R": [], "G": [], "B": []}
    for row in range(2):
        for col in range(2):
            chan_idx = int(pattern[row, col])
            chan_name = desc[chan_idx]
            if chan_name in sites:
                sites[chan_name].append((row, col, chan_idx))

    if len(sites["R"]) != 1 or len(sites["B"]) != 1 or len(sites["G"]) != 2:
        raise ValueError(
            "Unsupported Bayer mapping in raw_pattern/color_desc; expected 1xR, 2xG, 1xB in 2x2 tile."
        )
    return sites


def _extract_linearized_channel(raw_image, row, col, chan_idx, black_levels, white_levels):
    plane = raw_image[row::2, col::2].astype(np.float32, copy=True)
    black = float(black_levels[chan_idx])
    white = float(white_levels[chan_idx])
    denom = white - black
    if denom <= 1e-12:
        raise ValueError(
            f"Invalid black/white levels for channel index {chan_idx}: black={black}, white={white}."
        )

    plane -= black
    plane /= denom
    np.clip(plane, 0.0, 1.0, out=plane)
    return plane


def _channel_black_white_levels(raw, required_channels):
    black_levels = np.asarray(raw.black_level_per_channel, dtype=np.float32).reshape(-1)
    if black_levels.size == 1:
        black_levels = np.full((required_channels,), float(black_levels[0]), dtype=np.float32)
    elif black_levels.size < required_channels:
        raise ValueError(
            f"black_level_per_channel length {black_levels.size} is less than required {required_channels}."
        )

    cam_white = getattr(raw, "camera_white_level_per_channel", None)
    if cam_white is not None:
        white_levels = np.asarray(cam_white, dtype=np.float32).reshape(-1)
        if white_levels.size < required_channels:
            white_levels = np.full((required_channels,), 0.0, dtype=np.float32)
    else:
        white_levels = np.full((required_channels,), 0.0, dtype=np.float32)

    white_fallback = float(getattr(raw, "white_level", 0.0))
    if white_fallback <= 0.0 and np.any(white_levels[:required_channels] <= 0.0):
        raise ValueError("RAW metadata has no valid camera/channel white level.")

    white_levels = white_levels[:required_channels].copy()
    invalid_white = white_levels <= 0.0
    if np.any(invalid_white):
        white_levels[invalid_white] = white_fallback
    return black_levels[:required_channels], white_levels


def preprocess_rotated_raw_bayer(raw_paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load 4 physically-rotated RAW captures, extract Bayer channels directly (no demosaic),
    average pixelwise per channel, and return normalized R/G/B ground-truth planes in [0, 1].

    Notes:
    - Input images are not digitally rotated/aligned by design.
    - Returned arrays are native CFA-plane size: (H/2, W/2).
    """
    if not isinstance(raw_paths, Sequence):
        raise TypeError("raw_paths must be a sequence of 4 RAW file paths.")
    if len(raw_paths) != 4:
        raise ValueError(f"Exactly 4 RAW paths are required, got {len(raw_paths)}.")

    for path in raw_paths:
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Invalid RAW path entry: {path!r}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"RAW file not found: {path}")

    sum_r = None
    sum_g = None
    sum_b = None
    ref_shape = None
    ref_pattern = None
    site_map = None

    for path in raw_paths:
        with rawpy.imread(path) as raw:
            raw_image = raw.raw_image_visible
            raw_pattern = np.asarray(raw.raw_pattern, dtype=np.int32)

            if ref_shape is None:
                ref_shape = tuple(raw_image.shape)
                ref_pattern = raw_pattern.copy()
                site_map = _channel_site_map(raw_pattern, raw.color_desc)
            else:
                if tuple(raw_image.shape) != ref_shape:
                    raise ValueError(
                        f"RAW visible shape mismatch: expected {ref_shape}, got {tuple(raw_image.shape)} for {path}."
                    )
                if not np.array_equal(raw_pattern, ref_pattern):
                    raise ValueError(f"RAW Bayer pattern mismatch for {path}.")

            required_channels = int(np.max(raw_pattern)) + 1
            black_levels, white_levels = _channel_black_white_levels(raw, required_channels)

            r_site = site_map["R"][0]
            b_site = site_map["B"][0]
            g_site_1, g_site_2 = site_map["G"]

            plane_r = _extract_linearized_channel(raw_image, r_site[0], r_site[1], r_site[2], black_levels, white_levels)
            plane_b = _extract_linearized_channel(raw_image, b_site[0], b_site[1], b_site[2], black_levels, white_levels)
            plane_g1 = _extract_linearized_channel(
                raw_image, g_site_1[0], g_site_1[1], g_site_1[2], black_levels, white_levels
            )
            plane_g2 = _extract_linearized_channel(
                raw_image, g_site_2[0], g_site_2[1], g_site_2[2], black_levels, white_levels
            )
            plane_g = 0.5 * (plane_g1 + plane_g2)

            if sum_r is None:
                sum_r = plane_r
                sum_g = plane_g
                sum_b = plane_b
            else:
                sum_r += plane_r
                sum_g += plane_g
                sum_b += plane_b

    mean_r = sum_r / 4.0
    mean_g = sum_g / 4.0
    mean_b = sum_b / 4.0

    eps = np.float32(1e-12)
    i_r = (mean_r / max(float(np.max(mean_r)), float(eps))).astype(np.float32, copy=False)
    i_g = (mean_g / max(float(np.max(mean_g)), float(eps))).astype(np.float32, copy=False)
    i_b = (mean_b / max(float(np.max(mean_b)), float(eps))).astype(np.float32, copy=False)
    return i_r, i_g, i_b


class VignetteAnalyzer:
    def __init__(self, raw_path):
        self.raw_path = raw_path
        self.img_data = None
        self.optical_center = None
        self.max_brightness = 0.0
        self.radial_profile = None
        self.fallout_01ev_radius = None

        self.r_flat_px = None
        self.i_flat_norm = None
        self.max_radius_px = None

        self.model = None
        self.device = None
        self.sensor_half_diag_mm = None
        self.learned_f_norm = None
        self.learned_f_mm = None
        self.learned_i0 = None
        self.training_history = []
        self.lambda_physics = None

    def load_and_preprocess(self):
        print(f"[-] Loading RAW: {self.raw_path}")

        valid_exts = (".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".3fr", ".iiq")
        if not self.raw_path.lower().endswith(valid_exts):
            print("[!] Warning: File extension may not be standard. Attempting anyway...")

        try:
            with rawpy.imread(self.raw_path) as raw:
                rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        except Exception as exc:
            print(f"[!] Error reading RAW file: {exc}")
            sys.exit(1)

        h, w = gray.shape
        ratio = max(w, h) / min(w, h)
        is_landscape = w > h

        if abs(ratio - 1.5) < 0.05:
            print("[-] Detected Aspect Ratio: 3:2")
            target_size = (3000, 2000) if is_landscape else (2000, 3000)
        elif abs(ratio - 1.333) < 0.05:
            print("[-] Detected Aspect Ratio: 4:3")
            target_size = (4000, 3000) if is_landscape else (3000, 4000)
        else:
            print(f"[-] Detected Non-standard Aspect Ratio: {ratio:.2f}")
            scale = 4000 / max(w, h)
            target_size = (int(w * scale), int(h * scale))

        print(f"[-] Resizing from {w}x{h} to {target_size}")
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

        print("[-] Cleaning noise (outliers > 40%)...")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = np.abs(gray - blurred)
        mask = diff > (0.40 * blurred)
        gray[mask] = blurred[mask]
        self.img_data = gray

        heavy_blur = cv2.GaussianBlur(gray, (101, 101), 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(heavy_blur)
        self.optical_center = max_loc
        self.max_brightness = max_val
        print(f"[-] Optical Center Found: {self.optical_center}, Max Brightness: {self.max_brightness}")

    def extract_radial_profile(self):
        h, w = self.img_data.shape
        cx, cy = self.optical_center
        y, x = np.indices((h, w))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        r_flat = r.ravel().astype(np.float32)
        intensity_flat = self.img_data.ravel().astype(np.float32)
        intensity_flat_norm = intensity_flat / max(self.max_brightness, 1e-6)
        intensity_flat_norm = np.clip(intensity_flat_norm, 0.0, 1.0)

        self.r_flat_px = r_flat
        self.i_flat_norm = intensity_flat_norm
        self.max_radius_px = float(np.max(r_flat))

        r_int = r_flat.astype(int)
        tbin = np.bincount(r_int, intensity_flat_norm)
        nr = np.bincount(r_int)
        nr[nr == 0] = 1

        radial_profile_y = tbin / nr
        radial_profile_x = np.arange(len(radial_profile_y), dtype=np.float32)
        self.radial_profile = (radial_profile_x, radial_profile_y)

        threshold = 2 ** (-0.1)
        indices = np.where(radial_profile_y < threshold)[0]

        if len(indices) > 0:
            self.fallout_01ev_radius = int(indices[0])
            print(f"[-] 0.1 EV Falloff detected at radius: {self.fallout_01ev_radius} px")
        else:
            print("\n[!] CRITICAL: Vignetting is too light (< 0.1 EV falloff).")
            print("[!] No center filter needed. Process terminated to prevent invalid fabrication.")
            sys.exit(1)

    @staticmethod
    def analytical_cos4_torch(r_norm, f_norm, i0):
        f_safe = torch.clamp(f_norm, min=0.1, max=10.0)
        theta = torch.atan(r_norm / f_safe)
        return i0 * torch.cos(theta) ** 4

    @staticmethod
    def analytical_cos4_numpy(r_norm, f_norm, i0):
        f_safe = np.clip(f_norm, 0.1, 10.0)
        theta = np.arctan(r_norm / f_safe)
        return i0 * np.cos(theta) ** 4

    def fit_model_pinn(
        self,
        focal_length_mm,
        sensor_diag_mm,
        lambda_physics=1.0,
        epochs=600,
        batch_size=8192,
        lr=1e-3,
        physics_lr_f=1e-4,
        hidden_dim=32,
        sample_count=120000,
        learnable_physics=True,
        seed=42,
    ):
        if torch is None:
            print("[!] PyTorch is required for PINN training. Install torch and rerun.")
            sys.exit(1)

        print("\n=== PINN Training (Data + Physics) ===")
        rng = np.random.default_rng(seed)
        n_total = self.r_flat_px.shape[0]
        n_use = min(sample_count, n_total)
        select_idx = rng.choice(n_total, size=n_use, replace=False)

        r_train_px = self.r_flat_px[select_idx]
        y_train = self.i_flat_norm[select_idx]
        r_train_norm = r_train_px / max(self.max_radius_px, 1e-6)
        r_train_norm = np.clip(r_train_norm, 0.0, 1.0)

        x_tensor = torch.from_numpy(r_train_norm[:, None]).float()
        y_tensor = torch.from_numpy(y_train[:, None]).float()
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"[-] Training device: {device}")

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
        )

        sensor_half_diag_mm = float(sensor_diag_mm) / 2.0
        self.sensor_half_diag_mm = sensor_half_diag_mm
        init_f_norm = float(focal_length_mm) / max(sensor_half_diag_mm, 1e-6)

        model = PINNVignettingMLP(
            hidden_dim=hidden_dim,
            learnable_physics=learnable_physics,
            init_f_norm=init_f_norm,
            init_i0=1.0,
        ).to(device)

        if learnable_physics:
            optim_groups = [{"params": model.net.parameters(), "lr": lr}]
            if hasattr(model, "f_raw"):
                optim_groups.append({"params": [model.f_raw], "lr": physics_lr_f})
            if hasattr(model, "i0_raw"):
                optim_groups.append({"params": [model.i0_raw], "lr": lr})
            optimizer = torch.optim.Adam(optim_groups)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.lambda_physics = float(lambda_physics)
        self.training_history.clear()

        for epoch in range(1, epochs + 1):
            loss_sum = 0.0
            data_loss_sum = 0.0
            phys_loss_sum = 0.0
            n_batches = 0

            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                f_norm, i0 = model.get_physics_params()
                theory = self.analytical_cos4_torch(xb, f_norm, i0)

                loss_data = F.mse_loss(pred, yb)
                loss_physics = F.mse_loss(pred, theory)
                loss = loss_data + self.lambda_physics * loss_physics

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item())
                data_loss_sum += float(loss_data.item())
                phys_loss_sum += float(loss_physics.item())
                n_batches += 1

            avg_loss = loss_sum / max(n_batches, 1)
            avg_data_loss = data_loss_sum / max(n_batches, 1)
            avg_phys_loss = phys_loss_sum / max(n_batches, 1)
            self.training_history.append((epoch, avg_loss, avg_data_loss, avg_phys_loss))

            if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
                f_norm_now, i0_now = model.get_physics_params()
                f_mm_now = float(f_norm_now.item()) * sensor_half_diag_mm
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"L_total={avg_loss:.6e} | "
                    f"L_data={avg_data_loss:.6e} | "
                    f"L_phys={avg_phys_loss:.6e} | "
                    f"f_norm={float(f_norm_now.item()):.4f} | "
                    f"f~{f_mm_now:.3f} mm | "
                    f"I0={float(i0_now.item()):.4f}"
                )

        self.model = model
        f_norm_final, i0_final = model.get_physics_params()
        self.learned_f_norm = float(f_norm_final.item())
        self.learned_f_mm = self.learned_f_norm * sensor_half_diag_mm
        self.learned_i0 = float(i0_final.item())
        print(
            f"[-] PINN training complete. "
            f"Learned f_norm={self.learned_f_norm:.4f}, "
            f"f~{self.learned_f_mm:.3f} mm, I0={self.learned_i0:.4f}"
        )

    def predict_pinn(self, r_norm):
        if self.model is None:
            raise RuntimeError("PINN model not trained yet.")
        if torch is None:
            raise RuntimeError("PyTorch unavailable.")
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = torch.from_numpy(r_norm[:, None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            y = self.model(x).squeeze(1).cpu().numpy()
        return y

    def validation_plot(self, output_png, sensor_diag_mm, scatter_count=25000):
        if self.model is None:
            raise RuntimeError("PINN model not trained yet.")

        rng = np.random.default_rng(123)
        n_total = self.r_flat_px.shape[0]
        n_scatter = min(scatter_count, n_total)
        idx = rng.choice(n_total, size=n_scatter, replace=False)
        r_scatter_norm = self.r_flat_px[idx] / max(self.max_radius_px, 1e-6)
        y_scatter = self.i_flat_norm[idx]

        r_eval = np.linspace(0.0, 1.0, 800, dtype=np.float32)
        pinn_eval = self.predict_pinn(r_eval)
        analytic_eval = self.analytical_cos4_numpy(
            r_eval,
            f_norm=self.learned_f_norm,
            i0=self.learned_i0,
        )
        pinn_vs_analytic_mse = float(np.mean((pinn_eval - analytic_eval) ** 2))

        plt.figure(figsize=(10, 6))
        plt.scatter(
            r_scatter_norm,
            y_scatter,
            s=2,
            alpha=0.08,
            color="#4C72B0",
            label="Raw noisy pixels (scatter)",
        )
        plt.plot(r_eval, analytic_eval, "k--", linewidth=2.0, label="Analytical Cos^4 law")
        plt.plot(r_eval, pinn_eval, color="#C44E52", linewidth=2.0, label="PINN prediction")
        plt.xlabel("Normalized radius r / r_max")
        plt.ylabel("Relative illumination I")
        plt.title("Vignetting Verification: Raw Data vs Analytical vs PINN")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png, dpi=200)
        plt.close()

        print(f"[-] Validation plot saved: {output_png}")
        print(f"[-] PINN vs Analytical MSE: {pinn_vs_analytic_mse:.6e}")
        return pinn_vs_analytic_mse

    def geometric_mapping_pinn(self, focal_length_mm, sensor_diag_mm, filter_total_diameter_mm):
        print("\n=== Geometric Mapping Report ===")
        sensor_half_diag = sensor_diag_mm / 2.0
        theta_max_rad = np.arctan(sensor_half_diag / focal_length_mm)

        filter_radius_max = filter_total_diameter_mm / 2.0
        r_filter_covered_mm = focal_length_mm * np.tan(theta_max_rad)
        percent_covered = min((r_filter_covered_mm / filter_radius_max) * 100.0, 100.0)

        print(f"[-] Physical Filter Max Radius: {filter_radius_max:.2f} mm")
        print(f"[-] Sensor Data covers {percent_covered:.1f}% of the Filter Radius.")

        target_r_mm = np.arange(0, filter_radius_max, 0.1, dtype=np.float32)
        target_theta = np.arctan(target_r_mm / focal_length_mm)

        c_scale = np.tan(theta_max_rad) / max(self.max_radius_px, 1e-6)
        target_r_px = np.tan(target_theta) / c_scale
        target_r_norm = target_r_px / max(self.max_radius_px, 1e-6)

        target_r_norm_clip = np.clip(target_r_norm, 0.0, 1.0).astype(np.float32)
        pinn_curve = self.predict_pinn(target_r_norm_clip)

        analytical_curve = self.analytical_cos4_numpy(
            target_r_norm_clip,
            f_norm=self.learned_f_norm,
            i0=self.learned_i0,
        )
        return target_r_mm, pinn_curve, analytical_curve


if __name__ == "__main__":
    RAW_FILE = "test_vignette.ARW"
    FOCAL_LEN = 47.0
    SENSOR_DIAG = 150.0
    FILTER_DIA = 77.0

    # PINN hyperparameters (kept lightweight for local runtime).
    LAMBDA_PHYSICS = 1.0
    EPOCHS = 600
    BATCH_SIZE = 8192
    LR = 1e-3
    SAMPLE_COUNT = 120000
    HIDDEN_DIM = 32
    LEARNABLE_PHYSICS = True

    if not os.path.exists(RAW_FILE):
        print(f"[Error] File not found: {RAW_FILE}")
        print("Please provide a valid RAW file path.")
        sys.exit(1)

    analyzer = VignetteAnalyzer(RAW_FILE)
    analyzer.load_and_preprocess()
    analyzer.extract_radial_profile()
    analyzer.fit_model_pinn(
        focal_length_mm=FOCAL_LEN,
        sensor_diag_mm=SENSOR_DIAG,
        lambda_physics=LAMBDA_PHYSICS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        hidden_dim=HIDDEN_DIM,
        sample_count=SAMPLE_COUNT,
        learnable_physics=LEARNABLE_PHYSICS,
    )

    mse_val = analyzer.validation_plot(
        output_png="pinn_validation.png",
        sensor_diag_mm=SENSOR_DIAG,
        scatter_count=25000,
    )

    r_mm, curve_pinn, curve_analytic = analyzer.geometric_mapping_pinn(
        focal_length_mm=FOCAL_LEN,
        sensor_diag_mm=SENSOR_DIAG,
        filter_total_diameter_mm=FILTER_DIA,
    )

    output_data = np.column_stack((r_mm, curve_pinn, curve_analytic))
    csv_filename = "filter_profile_pinn.csv"
    np.savetxt(
        csv_filename,
        output_data,
        delimiter=",",
        header="Radius_mm,PINN_Curve,Analytical_Cos4_Curve",
        comments="",
    )

    print(f"\n[+] Success! PINN fabrication data exported to {csv_filename}")
    print(f"[+] PINN vs analytical MSE = {mse_val:.6e}")
    print("[+] Validation plot generated: pinn_validation.png")
