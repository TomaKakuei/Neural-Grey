"""Main vignette and aberration synthesis pipeline."""

from __future__ import annotations

import json
import gc
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .base_processor import (
    create_base_image,
    generate_light_spots,
    lock_base_image,
    rotate_and_center_crop,
    sample_actual_angle,
)
from .config import SynthesisConfig
from .noise_injector import apply_noise_fixed_ratio, generate_noise_intensity_ratio
from .vignette_generator import create_composite_mask


@dataclass
class SynthesisResult:
    """In-memory result for one generated group."""

    group_id: int
    images_0: np.ndarray
    images_90: np.ndarray
    images_180: np.ndarray
    images_270: np.ndarray
    gt: np.ndarray
    metadata: dict


class VignetteAberrationSynthesizer:
    """Generate four-angle vignette/aberration groups with exact GT."""

    def __init__(self, config: SynthesisConfig):
        self.config = config

    def synthesize_single_group(self, group_id: int) -> SynthesisResult:
        """Synthesize one complete group using a deterministic group seed."""

        group_seed = int(self.config.seed + group_id)
        group_rng = np.random.default_rng(group_seed)

        base = create_base_image(
            self.config.vignette.width,
            self.config.vignette.height,
            seed=group_seed,
        )
        light_spots = generate_light_spots(
            self.config.vignette.width,
            self.config.vignette.height,
            self.config.light_spot,
            group_rng,
        )
        locked_base = lock_base_image(base, light_spots)
        composite_mask = create_composite_mask(
            self.config.vignette,
            self.config.aberration,
            group_rng,
        )

        pre_noise: Dict[int, np.ndarray] = {}
        angle_records = {}
        for base_angle in self.config.crop.base_angles:
            jitter_deg, actual_angle_deg = sample_actual_angle(
                float(base_angle),
                self.config.crop.angle_jitter_deg_range,
                group_rng,
            )
            image = self._render_angle(
                locked_base,
                composite_mask,
                actual_angle_deg,
            )
            pre_noise[int(base_angle)] = image
            angle_records[str(base_angle)] = {
                "base_angle_deg": float(base_angle),
                "jitter_deg": float(jitter_deg),
                "actual_angle_deg": float(actual_angle_deg),
            }

        noise_ratio = generate_noise_intensity_ratio(group_rng, self.config.noise)
        final_images = {
            angle: apply_noise_fixed_ratio(
                image,
                self.config.noise,
                group_rng,
                intensity_ratio=noise_ratio,
            )
            for angle, image in pre_noise.items()
        }

        canonical_gt = rotate_and_center_crop(
            composite_mask["vignette"],
            self.config.crop.crop_width,
            self.config.crop.crop_height,
            0.0,
        )

        metadata = {
            "group_id": int(group_id),
            "seed": group_seed,
            "base_source": "pure_white",
            "noise_intensity_ratio": float(noise_ratio),
            "angles": angle_records,
            "gt_semantics": (
                "Single canonical perturbed vignette only: includes center shift, "
                "polynomial curve coefficient drift, and corner drift; excludes crop "
                "angle, light spots, astigmatism, chromatic aberration, and final noise."
            ),
            "lens_perturbations": composite_mask["metadata"],
            "files": {
                "0": "sim_capture_0.npy",
                "90": "sim_capture_90.npy",
                "180": "sim_capture_180.npy",
                "270": "sim_capture_270.npy",
                "gt": "ground_truth_vignetting.npy",
            },
        }

        return SynthesisResult(
            group_id=int(group_id),
            images_0=final_images[0],
            images_90=final_images[90],
            images_180=final_images[180],
            images_270=final_images[270],
            gt=canonical_gt,
            metadata=metadata,
        )

    def _render_angle(
        self,
        locked_base: np.ndarray,
        composite_mask: dict,
        actual_angle_deg: float,
    ) -> np.ndarray:
        """Apply the angle-bound lens mask and aberrations before noise."""

        cw = self.config.crop.crop_width
        ch = self.config.crop.crop_height
        base_crop = rotate_and_center_crop(locked_base, cw, ch, actual_angle_deg)
        vignette = rotate_and_center_crop(
            composite_mask["vignette"], cw, ch, actual_angle_deg
        )
        astigmatism = rotate_and_center_crop(
            composite_mask["astigmatism"], cw, ch, actual_angle_deg
        )

        signal = base_crop * vignette
        signal = signal * (1.0 + astigmatism)
        signal = np.clip(signal, 0.0, 1.0).astype(np.float32, copy=False)
        rgb = np.repeat(signal[:, :, None], 3, axis=2)

        displacement = {
            key: rotate_and_center_crop(value, cw, ch, actual_angle_deg)
            for key, value in composite_mask.items()
            if key.startswith("chromatic_")
        }
        return self._apply_chromatic_displacement(rgb, displacement)

    @staticmethod
    def _apply_chromatic_displacement(
        image: np.ndarray,
        displacement: dict,
    ) -> np.ndarray:
        """Warp R/B channels by the generated chromatic displacement fields."""

        h, w = image.shape[:2]
        yy, xx = np.indices((h, w), dtype=np.float32)

        out = image.astype(np.float32, copy=True)
        map_x_r = xx - displacement["chromatic_dx_r"]
        map_y_r = yy - displacement["chromatic_dy_r"]
        map_x_b = xx - displacement["chromatic_dx_b"]
        map_y_b = yy - displacement["chromatic_dy_b"]

        out[:, :, 0] = cv2.remap(
            out[:, :, 0],
            map_x_r,
            map_y_r,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        out[:, :, 2] = cv2.remap(
            out[:, :, 2],
            map_x_b,
            map_y_b,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    def save_result(self, result: SynthesisResult, output_dir: str | Path) -> None:
        """Save one group using the benchmark-compatible file contract."""

        group_dir = Path(output_dir) / f"group_{result.group_id:04d}"
        group_dir.mkdir(parents=True, exist_ok=True)

        np.save(group_dir / "sim_capture_0.npy", result.images_0)
        np.save(group_dir / "sim_capture_90.npy", result.images_90)
        np.save(group_dir / "sim_capture_180.npy", result.images_180)
        np.save(group_dir / "sim_capture_270.npy", result.images_270)
        np.save(group_dir / "ground_truth_vignetting.npy", result.gt)
        for legacy_name in (
            "ground_truth_vignetting_0.npy",
            "ground_truth_vignetting_90.npy",
            "ground_truth_vignetting_180.npy",
            "ground_truth_vignetting_270.npy",
        ):
            legacy_path = group_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

        with (group_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(result.metadata, f, indent=2)

    def synthesize_stream(
        self,
        output_dir: str | Path | None = None,
        resume: bool = True,
    ) -> List[dict]:
        """Generate and save groups one-by-one."""

        output_path = Path(output_dir or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_groups = int(self.config.num_groups)
        manifests: List[dict] = []

        if resume:
            manifests, completed_group_ids = self._load_completed_manifests(output_path)
            completed_group_set = set(completed_group_ids)
            if completed_group_ids:
                print(
                    f"resuming: found {len(completed_group_ids)}/{total_groups} completed groups",
                    flush=True,
                )
        else:
            completed_group_set = set()
            completed_group_ids = []

        self.write_progress(
            output_path,
            status="running",
            current_group_id=None,
            completed_group_ids=completed_group_ids,
            message="resume scan complete" if resume else "fresh run started",
        )
        self.write_summary(output_path, manifests)

        run_started_at = time.time()
        for group_id in range(int(self.config.num_groups)):
            if group_id in completed_group_set:
                continue

            group_path = output_path / f"group_{group_id:04d}"
            self.write_progress(
                output_path,
                status="running",
                current_group_id=group_id,
                completed_group_ids=completed_group_set,
                message=f"generating group {group_id + 1}/{total_groups}",
            )
            print(
                f"generating group {group_id + 1}/{total_groups} -> {group_path}",
                flush=True,
            )

            group_started_at = time.time()
            result = self.synthesize_single_group(group_id)
            metadata = result.metadata
            self.save_result(result, output_path)
            manifests.append(metadata)
            completed_group_set.add(group_id)
            del result
            gc.collect()

            if self.config.visualize and group_id < 5:
                vis_dir = output_path / "visualizations"
                vis_dir.mkdir(exist_ok=True)
                self.visualize_saved_group(
                    output_path / f"group_{group_id:04d}",
                    vis_dir / f"group_{group_id:04d}.png",
                )

            elapsed = time.time() - group_started_at
            total_elapsed = time.time() - run_started_at
            self.write_summary(output_path, manifests)
            self.write_progress(
                output_path,
                status="running",
                current_group_id=None,
                completed_group_ids=completed_group_set,
                message=(
                    f"completed group {group_id:04d} in {elapsed:.1f}s; "
                    f"{len(completed_group_set)}/{total_groups} done; "
                    f"elapsed {total_elapsed:.1f}s"
                ),
                last_group_id=group_id,
                last_group_seconds=elapsed,
            )
            self._append_progress_log(
                output_path,
                (
                    f"{self._timestamp()} completed group {group_id:04d} "
                    f"({len(completed_group_set)}/{total_groups}) "
                    f"in {elapsed:.1f}s"
                ),
            )
            print(
                f"completed {len(completed_group_set)}/{total_groups} groups "
                f"({group_id:04d}) in {elapsed:.1f}s",
                flush=True,
            )

        self.write_summary(output_path, manifests)
        self.write_progress(
            output_path,
            status="done",
            current_group_id=None,
            completed_group_ids=completed_group_set,
            message=f"finished {len(completed_group_set)}/{total_groups} groups",
        )
        return manifests

    def synthesize_batch(self, num_groups: int) -> List[SynthesisResult]:
        """Compatibility method that returns all results in memory."""

        return [self.synthesize_single_group(i) for i in range(int(num_groups))]

    def save_results(self, results: List[SynthesisResult], output_dir: str | Path) -> None:
        """Compatibility method for saving an in-memory batch."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifests = []
        for result in results:
            self.save_result(result, output_path)
            manifests.append(result.metadata)
        self.write_summary(output_path, manifests)

    def write_summary(self, output_path: Path, manifests: List[dict]) -> None:
        """Write root-level metadata for the generated dataset."""

        sorted_manifests = sorted(manifests, key=lambda item: int(item["group_id"]))
        summary = {
            "num_groups": len(sorted_manifests),
            "num_groups_target": int(self.config.num_groups),
            "seed": int(self.config.seed),
            "output_contract": "benchmark_compatible_group_dirs",
            "config": asdict(self.config),
            "groups": [
                {
                    "group_id": item["group_id"],
                    "seed": item["seed"],
                    "dir": f"group_{item['group_id']:04d}",
                }
                for item in sorted_manifests
            ],
        }
        self._atomic_write_json(output_path / "summary.json", summary)

    def write_progress(
        self,
        output_path: Path,
        status: str,
        current_group_id: Optional[int],
        completed_group_ids: Sequence[int],
        message: str,
        last_group_id: Optional[int] = None,
        last_group_seconds: Optional[float] = None,
    ) -> None:
        """Write a small monitoring file that can be read while the run is live."""

        completed_sorted = sorted({int(group_id) for group_id in completed_group_ids})
        completed_lookup = set(completed_sorted)
        total_groups = max(int(self.config.num_groups), 1)
        next_group_id = next(
            (group_id for group_id in range(int(self.config.num_groups)) if group_id not in completed_lookup),
            None,
        )
        progress = {
            "status": status,
            "message": message,
            "updated_at": self._timestamp(),
            "seed": int(self.config.seed),
            "output_dir": str(output_path),
            "target_groups": int(self.config.num_groups),
            "completed_groups": len(completed_sorted),
            "remaining_groups": int(self.config.num_groups) - len(completed_sorted),
            "percent_complete": round(100.0 * len(completed_sorted) / total_groups, 2),
            "current_group_id": current_group_id,
            "last_completed_group_id": last_group_id,
            "last_group_seconds": None if last_group_seconds is None else round(float(last_group_seconds), 3),
            "next_group_id": next_group_id,
            "completed_group_ids": completed_sorted,
        }
        self._atomic_write_json(output_path / "progress.json", progress)

    def _load_completed_manifests(self, output_path: Path) -> tuple[List[dict], List[int]]:
        """Load all fully written groups already on disk and clear partial leftovers."""

        manifests: List[dict] = []
        completed_group_ids: List[int] = []
        for group_id in range(int(self.config.num_groups)):
            group_dir = output_path / f"group_{group_id:04d}"
            if self._group_is_complete(group_dir):
                manifest_path = group_dir / "manifest.json"
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifests.append(json.load(f))
                completed_group_ids.append(group_id)
            elif group_dir.exists():
                shutil.rmtree(group_dir)
        manifests.sort(key=lambda item: int(item["group_id"]))
        return manifests, completed_group_ids

    @staticmethod
    def _group_is_complete(group_dir: Path) -> bool:
        """Check whether a group directory has the expected benchmark files."""

        required = (
            "sim_capture_0.npy",
            "sim_capture_90.npy",
            "sim_capture_180.npy",
            "sim_capture_270.npy",
            "ground_truth_vignetting.npy",
            "manifest.json",
        )
        return group_dir.is_dir() and all((group_dir / name).exists() for name in required)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _append_progress_log(output_path: Path, line: str) -> None:
        with (output_path / "progress.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def visualize_result(self, result: SynthesisResult, save_path: str | Path) -> None:
        """Save a compact visual overview for human preview."""

        def preview(image: np.ndarray, max_edge: int = 900) -> np.ndarray:
            h, w = image.shape[:2]
            scale = min(1.0, float(max_edge) / float(max(h, w)))
            if scale >= 1.0:
                return image
            size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        items = [
            ("0 deg", result.images_0),
            ("90 deg", result.images_90),
            ("180 deg", result.images_180),
            ("270 deg", result.images_270),
            ("GT cos4", result.gt),
        ]
        self._write_preview_canvas(
            [(title, preview(image)) for title, image in items],
            save_path,
        )

    def visualize_saved_group(self, group_dir: str | Path, save_path: str | Path) -> None:
        """Create a preview PNG from saved arrays with low peak memory."""

        def load_preview(path: Path, max_edge: int = 900) -> np.ndarray:
            arr = np.load(path, mmap_mode="r")
            h, w = arr.shape[:2]
            scale = min(1.0, float(max_edge) / float(max(h, w)))
            step = max(1, int(np.floor(1.0 / max(scale, 1e-6))))
            sampled = np.asarray(arr[::step, ::step])
            if sampled.shape[0] > max_edge or sampled.shape[1] > max_edge:
                size = (
                    min(max_edge, sampled.shape[1]),
                    min(max_edge, sampled.shape[0]),
                )
                sampled = cv2.resize(sampled, size, interpolation=cv2.INTER_AREA)
            return sampled.astype(np.float32, copy=False)

        group_path = Path(group_dir)
        items = [
            ("0 deg", group_path / "sim_capture_0.npy"),
            ("90 deg", group_path / "sim_capture_90.npy"),
            ("180 deg", group_path / "sim_capture_180.npy"),
            ("270 deg", group_path / "sim_capture_270.npy"),
            ("GT cos4", group_path / "ground_truth_vignetting.npy"),
        ]
        self._write_preview_canvas(
            [(title, load_preview(path)) for title, path in items],
            save_path,
        )

    @staticmethod
    def _write_preview_canvas(items: list[tuple[str, np.ndarray]], save_path: str | Path) -> None:
        """Write a small overview panel with OpenCV only."""

        panel_w, panel_h, title_h, gap = 480, 320, 34, 10
        canvas_h = title_h + panel_h
        canvas_w = len(items) * panel_w + (len(items) - 1) * gap
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        for idx, (title, image) in enumerate(items):
            image = np.clip(image, 0.0, 1.0)
            if image.ndim == 2:
                image = np.repeat(image[:, :, None], 3, axis=2)
            image_u8 = (image * 255.0 + 0.5).astype(np.uint8, copy=False)
            resized = cv2.resize(image_u8, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

            x0 = idx * (panel_w + gap)
            canvas[title_h : title_h + panel_h, x0 : x0 + panel_w] = resized
            cv2.putText(
                canvas,
                title,
                (x0 + 12, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        # Arrays are RGB; OpenCV writes BGR.
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(save_path), canvas_bgr)
        if not ok:
            raise RuntimeError(f"Failed to write preview image: {save_path}")


def run_synthesis(
    num_groups: int = 400,
    output_dir: str = "vignette_aberration_dataset",
    seed: int = 42,
    visualize: bool = True,
    width: int = 4000,
    height: int = 4000,
    crop_width: int = 3000,
    crop_height: int = 2000,
    focal_px: float = 2500.0,
    curve_drift: float = 0.10,
    max_angle_jitter_deg: float = 7.0,
    resume: bool = True,
) -> List[dict]:
    """Run the full pipeline and stream results to disk."""

    config = SynthesisConfig(
        output_dir=output_dir,
        num_groups=int(num_groups),
        seed=int(seed),
        visualize=bool(visualize),
    )
    config.vignette.width = int(width)
    config.vignette.height = int(height)
    config.vignette.focal_px = float(focal_px)
    config.vignette.polynomial_curve_drift_range = (float(curve_drift), float(curve_drift))
    config.crop.crop_width = int(crop_width)
    config.crop.crop_height = int(crop_height)
    config.crop.angle_jitter_deg_range = (0.0, float(max_angle_jitter_deg))

    synthesizer = VignetteAberrationSynthesizer(config)
    print(f"starting synthesis: {num_groups} groups -> {output_dir}", flush=True)
    manifests = synthesizer.synthesize_stream(output_dir, resume=resume)
    print(f"done: wrote {len(manifests)} groups to {output_dir}", flush=True)
    return manifests


if __name__ == "__main__":
    run_synthesis(num_groups=400)
