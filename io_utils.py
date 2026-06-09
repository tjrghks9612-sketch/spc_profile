from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from area_calculation import CapMetrics
from image_processing import save_image
from plotting import plot_3d_surface, plot_cap_highlight, plot_cap_top_view, plot_detection_overlay, plot_profiles
from profile_extraction import ProfileResult
from surface_model import SurfaceGrid


def create_output_dir(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    output_root = root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _profile_to_frame(profile: ProfileResult) -> pd.DataFrame:
    depth_um = np.clip(profile.height_um - profile.z_um, 0.0, None)
    return pd.DataFrame(
        {
            profile.coordinate_name: profile.coord_um,
            "z_um": depth_um,
            "depth_from_apex_um": depth_um,
            "height_from_baseline_um": profile.z_um,
            "r_normalized": np.abs(profile.coord_um) / (profile.cd_um * 0.5),
        }
    )


def build_summary(
    horizontal: ProfileResult,
    vertical: ProfileResult,
    surface: SurfaceGrid,
    cap: CapMetrics,
    params: dict[str, Any],
) -> dict[str, float]:
    return {
        "CD_x_um": horizontal.cd_um,
        "CD_y_um": vertical.cd_um,
        "H_x_um": horizontal.height_um,
        "H_y_um": vertical.height_um,
        "H_global_um": surface.H_global_um,
        "H_surface_max_um": surface.H_surface_max_um,
        "cap_depth_um": float(params["cap_depth_um"]),
        "z_cut_um": cap.z_cut_um,
        "cap_curved_surface_area_um2": cap.cap_curved_surface_area_um2,
        "cap_projected_area_um2": cap.cap_projected_area_um2,
        "grid_resolution": float(params["grid_resolution"]),
        "pixel_size_um": float(params["pixel_size_um"]),
        "threshold_sensitivity": float(params["threshold_sensitivity"]),
        "smoothing_strength": float(params["smoothing_strength"]),
        "morph_strength": float(params["morph_strength"]),
        "sanity_check_x_rmse_um": surface.sanity_check_x_rmse_um,
        "sanity_check_y_rmse_um": surface.sanity_check_y_rmse_um,
    }


def save_outputs(
    output_dir: str | Path,
    horizontal_image: np.ndarray,
    vertical_image: np.ndarray,
    horizontal: ProfileResult,
    vertical: ProfileResult,
    surface: SurfaceGrid,
    cap: CapMetrics,
    params: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizontal_overlay = plot_detection_overlay(horizontal_image, horizontal)
    vertical_overlay = plot_detection_overlay(vertical_image, vertical)
    save_image(output_dir / "horizontal_detection_overlay.png", cv2.cvtColor(horizontal_overlay, cv2.COLOR_RGB2BGR))
    save_image(output_dir / "vertical_detection_overlay.png", cv2.cvtColor(vertical_overlay, cv2.COLOR_RGB2BGR))

    _profile_to_frame(horizontal).to_csv(output_dir / "horizontal_profile.csv", index=False)
    _profile_to_frame(vertical).to_csv(output_dir / "vertical_profile.csv", index=False)

    np.savez_compressed(
        output_dir / "surface_grid.npz",
        X=surface.X,
        Y=surface.Y,
        Z=surface.Z,
        Z_depth_um=surface.Z,
        Z_height_from_baseline_um=surface.Z_height_um,
        valid_mask=surface.valid_mask,
        cap_mask=cap.cap_mask,
    )

    summary = build_summary(horizontal, vertical, surface, cap, params)
    pd.DataFrame([summary]).to_csv(output_dir / "result_summary.csv", index=False)

    plot_profiles(horizontal, vertical, output_dir / "profiles.png")
    plot_3d_surface(surface, output_dir / "3d_surface.png")
    plot_cap_highlight(surface, cap, output_dir / "cap_highlighted_3d_surface.png")
    plot_cap_top_view(surface, cap, output_dir / "cap_top_view.png")
    return output_dir
