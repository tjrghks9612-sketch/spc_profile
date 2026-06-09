from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from image_processing import DetectionError, save_image
from plotting import plot_detection_overlay, plot_single_batch_cd_depth, plot_single_profile_cd_depth
from profile_extraction import ProfileResult


@dataclass
class SingleDepthCDResult:
    depth_um: np.ndarray
    left_x_um: np.ndarray
    right_x_um: np.ndarray
    cd_um: np.ndarray
    requested_max_depth_um: float
    effective_max_depth_um: float
    depth_step_um: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "depth_from_apex_um": self.depth_um,
                "left_x_um": self.left_x_um,
                "right_x_um": self.right_x_um,
                "CD_um": self.cd_um,
            }
        )


@dataclass
class SingleTaperResult:
    offset_um: np.ndarray
    side: np.ndarray
    x_um: np.ndarray
    depth_from_apex_um: np.ndarray
    slope_um_per_um: np.ndarray
    taper_angle_deg: np.ndarray
    taper_from_vertical_deg: np.ndarray
    step_um: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "offset_from_center_um": self.offset_um,
                "side": self.side,
                "x_um": self.x_um,
                "depth_from_apex_um": self.depth_from_apex_um,
                "slope_um_per_um": self.slope_um_per_um,
                "taper_angle_deg": self.taper_angle_deg,
                "taper_from_vertical_deg": self.taper_from_vertical_deg,
            }
        )


def _profile_depth_from_apex(profile: ProfileResult) -> tuple[np.ndarray, np.ndarray]:
    coord = profile.coord_um.astype(np.float64)
    depth = profile.height_um - profile.z_um.astype(np.float64)
    valid = np.isfinite(coord) & np.isfinite(depth)
    coord = coord[valid]
    depth = depth[valid]
    if coord.size < 8:
        raise DetectionError("depth-CD 계산에 필요한 profile point 수가 부족합니다.")

    order = np.argsort(coord)
    coord = coord[order]
    depth = np.clip(depth[order], 0.0, None)
    depth = depth - float(np.nanmin(depth))
    return coord, depth


def _side_crossing(x_outward: np.ndarray, depth_outward: np.ndarray, target_depth: float) -> float:
    if x_outward.size == 0:
        raise DetectionError("depth-CD 계산 중 한쪽 profile segment가 비어 있습니다.")
    if x_outward.size == 1 or target_depth <= 0:
        return float(x_outward[0])

    monotonic_depth = np.maximum.accumulate(depth_outward.astype(np.float64))
    if target_depth >= monotonic_depth[-1]:
        return float(x_outward[-1])

    idx = int(np.searchsorted(monotonic_depth, target_depth, side="right"))
    idx = max(1, min(idx, monotonic_depth.size - 1))
    d0 = float(monotonic_depth[idx - 1])
    d1 = float(monotonic_depth[idx])
    x0 = float(x_outward[idx - 1])
    x1 = float(x_outward[idx])
    if d1 <= d0 + 1e-12:
        return x1

    t = (float(target_depth) - d0) / (d1 - d0)
    return x0 + t * (x1 - x0)


def compute_cd_by_depth(
    profile: ProfileResult,
    max_depth_um: float,
    depth_step_um: float,
) -> SingleDepthCDResult:
    if max_depth_um <= 0:
        raise ValueError("max_depth_um은 0보다 커야 합니다.")
    if depth_step_um <= 0:
        raise ValueError("depth_step_um은 0보다 커야 합니다.")

    coord, depth = _profile_depth_from_apex(profile)
    min_depth = float(np.nanmin(depth))
    apex_candidates = np.where(depth <= min_depth + max(1e-9, profile.height_um * 1e-4))[0]
    if apex_candidates.size:
        apex_idx = int(apex_candidates[np.argmin(np.abs(coord[apex_candidates]))])
    else:
        apex_idx = int(np.nanargmin(depth))

    apex_x = float(coord[apex_idx])
    left_x = np.concatenate([[apex_x], coord[:apex_idx][::-1]])
    left_depth = np.concatenate([[0.0], depth[:apex_idx][::-1]])
    right_x = np.concatenate([[apex_x], coord[apex_idx + 1 :]])
    right_depth = np.concatenate([[0.0], depth[apex_idx + 1 :]])
    if left_x.size < 2 or right_x.size < 2:
        raise DetectionError("apex 좌우 profile point가 부족해 depth별 CD를 계산할 수 없습니다.")

    effective_max = min(float(max_depth_um), float(profile.height_um), float(max(np.nanmax(depth), 0.0)))
    if effective_max <= 0:
        raise DetectionError("유효한 depth 범위가 없습니다.")

    depths = np.arange(0.0, effective_max + depth_step_um * 0.5, float(depth_step_um), dtype=np.float64)
    if depths.size == 0 or not np.isclose(depths[-1], effective_max):
        depths = np.append(depths, effective_max)
    depths = np.clip(depths, 0.0, effective_max)
    depths = np.unique(np.round(depths, 10))

    out_left = []
    out_right = []
    out_cd = []
    for depth_value in depths:
        lx = _side_crossing(left_x, left_depth, float(depth_value))
        rx = _side_crossing(right_x, right_depth, float(depth_value))
        if rx < lx:
            lx, rx = rx, lx
        out_left.append(lx)
        out_right.append(rx)
        out_cd.append(max(0.0, rx - lx))

    return SingleDepthCDResult(
        depth_um=depths.astype(np.float32),
        left_x_um=np.asarray(out_left, dtype=np.float32),
        right_x_um=np.asarray(out_right, dtype=np.float32),
        cd_um=np.asarray(out_cd, dtype=np.float32),
        requested_max_depth_um=float(max_depth_um),
        effective_max_depth_um=float(effective_max),
        depth_step_um=float(depth_step_um),
    )


def _side_offsets(limit_um: float, step_um: float) -> np.ndarray:
    if limit_um <= 0:
        return np.asarray([], dtype=np.float64)
    offsets = np.arange(float(step_um), limit_um + step_um * 0.5, float(step_um), dtype=np.float64)
    offsets = offsets[offsets <= limit_um + 1e-9]
    if offsets.size == 0:
        offsets = np.asarray([limit_um], dtype=np.float64)
    return np.unique(np.round(offsets, 10))


def compute_taper_by_offset(profile: ProfileResult, taper_step_um: float) -> SingleTaperResult:
    if taper_step_um <= 0:
        raise ValueError("taper_step_um은 0보다 커야 합니다.")

    coord, depth = _profile_depth_from_apex(profile)
    if coord.size < 8:
        raise DetectionError("taper 계산에 필요한 profile point 수가 부족합니다.")

    gradient = np.gradient(depth, coord)
    rows: list[tuple[float, str, float, float, float, float, float]] = []
    side_specs = [
        ("left", -1.0, abs(float(np.nanmin(coord)))),
        ("right", 1.0, abs(float(np.nanmax(coord)))),
    ]
    for side, sign, limit_um in side_specs:
        for offset in _side_offsets(limit_um, float(taper_step_um)):
            x = sign * float(offset)
            sampled_depth = float(np.interp(x, coord, depth))
            slope = float(np.interp(x, coord, gradient))
            angle = float(np.degrees(np.arctan(abs(slope))))
            from_vertical = float(max(0.0, 90.0 - angle))
            rows.append((float(offset), side, x, sampled_depth, slope, angle, from_vertical))

    if not rows:
        raise DetectionError("taper를 계산할 좌우 offset 범위가 없습니다.")

    return SingleTaperResult(
        offset_um=np.asarray([row[0] for row in rows], dtype=np.float32),
        side=np.asarray([row[1] for row in rows], dtype=object),
        x_um=np.asarray([row[2] for row in rows], dtype=np.float32),
        depth_from_apex_um=np.asarray([row[3] for row in rows], dtype=np.float32),
        slope_um_per_um=np.asarray([row[4] for row in rows], dtype=np.float32),
        taper_angle_deg=np.asarray([row[5] for row in rows], dtype=np.float32),
        taper_from_vertical_deg=np.asarray([row[6] for row in rows], dtype=np.float32),
        step_um=float(taper_step_um),
    )


def save_single_profile_outputs(
    output_dir: str | Path,
    image: np.ndarray,
    profile: ProfileResult,
    cd_result: SingleDepthCDResult,
    params: dict[str, Any],
    taper_result: SingleTaperResult | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = plot_detection_overlay(image, profile)
    save_image(output_dir / "single_detection_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    depth = np.clip(profile.height_um - profile.z_um, 0.0, None)
    depth = depth - float(np.nanmin(depth))
    pd.DataFrame(
        {
            profile.coordinate_name: profile.coord_um,
            "depth_from_apex_um": depth,
            "height_from_baseline_um": profile.z_um,
        }
    ).to_csv(output_dir / "single_profile.csv", index=False)

    cd_result.to_frame().to_csv(output_dir / "cd_by_depth.csv", index=False)
    if taper_result is not None:
        taper_result.to_frame().to_csv(output_dir / "taper_by_offset.csv", index=False)
    summary = {
        "CD_um": profile.cd_um,
        "H_um": profile.height_um,
        "requested_max_depth_um": cd_result.requested_max_depth_um,
        "effective_max_depth_um": cd_result.effective_max_depth_um,
        "depth_step_um": cd_result.depth_step_um,
        "depth_count": float(cd_result.depth_um.size),
        "taper_step_um": taper_result.step_um if taper_result is not None else np.nan,
        "left_taper_angle_mean_deg": _mean_taper_angle(taper_result, "left"),
        "right_taper_angle_mean_deg": _mean_taper_angle(taper_result, "right"),
        "pixel_size_um": float(params["pixel_size_um"]),
        "threshold_sensitivity": float(params["threshold_sensitivity"]),
        "smoothing_strength": float(params["smoothing_strength"]),
        "morph_strength": float(params["morph_strength"]),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "single_result_summary.csv", index=False)
    plot_single_profile_cd_depth(profile, cd_result, output_dir / "single_profile_cd_depth.png", taper_result=taper_result)
    return output_dir


def _safe_stem(name: str, index: int) -> str:
    stem = Path(name).stem or f"image_{index + 1}"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    return safe[:80] or f"image_{index + 1}"


def _graph_raw_data_frame(image_name: str, profile: ProfileResult, cd_result: SingleDepthCDResult) -> pd.DataFrame:
    depth = np.clip(profile.height_um - profile.z_um, 0.0, None)
    depth = depth - float(np.nanmin(depth))
    profile_rows = pd.DataFrame(
        {
            "image_name": image_name,
            "graph": "top_boundary_profile",
            "x_axis": profile.coordinate_name,
            "y_axis": "depth_from_apex_um",
            "x_value": profile.coord_um,
            "y_value": depth,
        }
    )
    cd_rows = pd.DataFrame(
        {
            "image_name": image_name,
            "graph": "cd_by_depth",
            "x_axis": "CD_um",
            "y_axis": "depth_from_apex_um",
            "x_value": cd_result.cd_um,
            "y_value": cd_result.depth_um,
        }
    )
    return pd.concat([profile_rows, cd_rows], ignore_index=True)


def _mean_taper_angle(taper_result: SingleTaperResult | None, side: str) -> float:
    if taper_result is None:
        return float("nan")
    mask = taper_result.side == side
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(taper_result.taper_angle_deg[mask]))


def save_single_batch_outputs(
    output_dir: str | Path,
    results,
    params: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "single_overlays"
    profile_dir = output_dir / "single_profiles"
    cd_dir = output_dir / "cd_by_depth"
    taper_dir = output_dir / "taper_by_offset"
    plot_dir = output_dir / "single_plots"
    raw_data_dir = output_dir / "single_graph_raw_data"
    for directory in [overlay_dir, profile_dir, cd_dir, taper_dir, plot_dir, raw_data_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    profile_raw_frames = []
    cd_raw_frames = []
    taper_raw_frames = []
    for index, result in enumerate(results):
        name = result.item.name
        stem = _safe_stem(name, index)
        roi = result.roi
        if result.profile is None or result.cd_result is None:
            summary_rows.append(
                {
                    "image": name,
                    "status": "FAILED" if result.error else "NOT_ANALYZED",
                    "error": result.error,
                    "CD_um": np.nan,
                    "H_um": np.nan,
                    "effective_max_depth_um": np.nan,
                    "depth_step_um": float(params["depth_step_um"]),
                    "depth_count": 0,
                    "taper_step_um": float(params.get("taper_step_um", np.nan)),
                    "left_taper_angle_mean_deg": np.nan,
                    "right_taper_angle_mean_deg": np.nan,
                    "roi_x": roi[0] if roi else np.nan,
                    "roi_y": roi[1] if roi else np.nan,
                    "roi_w": roi[2] if roi else np.nan,
                    "roi_h": roi[3] if roi else np.nan,
                }
            )
            continue

        profile = result.profile
        cd_result = result.cd_result
        taper_result = getattr(result, "taper_result", None)
        overlay = plot_detection_overlay(result.item.image, profile)
        save_image(overlay_dir / f"{stem}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        depth = np.clip(profile.height_um - profile.z_um, 0.0, None)
        depth = depth - float(np.nanmin(depth))
        pd.DataFrame(
            {
                profile.coordinate_name: profile.coord_um,
                "depth_from_apex_um": depth,
                "height_from_baseline_um": profile.z_um,
            }
        ).to_csv(profile_dir / f"{stem}_profile.csv", index=False)
        cd_result.to_frame().to_csv(cd_dir / f"{stem}_cd_by_depth.csv", index=False)
        if taper_result is not None:
            taper_frame = taper_result.to_frame()
            taper_frame.to_csv(taper_dir / f"{stem}_taper_by_offset.csv", index=False)
            taper_raw_frames.append(taper_frame.assign(image_name=name))
        plot_single_profile_cd_depth(profile, cd_result, plot_dir / f"{stem}_profile_cd_depth.png", taper_result=taper_result)
        raw_data = _graph_raw_data_frame(name, profile, cd_result)
        profile_raw_frames.append(raw_data[raw_data["graph"] == "top_boundary_profile"])
        cd_raw_frames.append(raw_data[raw_data["graph"] == "cd_by_depth"])

        summary_rows.append(
            {
                "image": name,
                "status": "OK",
                "error": "",
                "CD_um": profile.cd_um,
                "H_um": profile.height_um,
                "requested_max_depth_um": cd_result.requested_max_depth_um,
                "effective_max_depth_um": cd_result.effective_max_depth_um,
                "depth_step_um": cd_result.depth_step_um,
                "depth_count": int(cd_result.depth_um.size),
                "taper_step_um": taper_result.step_um if taper_result is not None else np.nan,
                "left_taper_angle_mean_deg": _mean_taper_angle(taper_result, "left"),
                "right_taper_angle_mean_deg": _mean_taper_angle(taper_result, "right"),
                "roi_x": roi[0] if roi else np.nan,
                "roi_y": roi[1] if roi else np.nan,
                "roi_w": roi[2] if roi else np.nan,
                "roi_h": roi[3] if roi else np.nan,
                "pixel_size_um": float(params["pixel_size_um"]),
                "threshold_sensitivity": float(params["threshold_sensitivity"]),
                "smoothing_strength": float(params["smoothing_strength"]),
                "morph_strength": float(params["morph_strength"]),
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "batch_result_summary.csv", index=False)
    if profile_raw_frames:
        pd.concat(profile_raw_frames, ignore_index=True).to_csv(raw_data_dir / "top_boundary_profile_raw_data.csv", index=False)
    if cd_raw_frames:
        pd.concat(cd_raw_frames, ignore_index=True).to_csv(raw_data_dir / "cd_by_depth_raw_data.csv", index=False)
    if taper_raw_frames:
        pd.concat(taper_raw_frames, ignore_index=True).to_csv(raw_data_dir / "taper_by_offset_raw_data.csv", index=False)
    plot_single_batch_cd_depth(results, output_dir / "batch_profile_cd_depth.png")
    return output_dir
