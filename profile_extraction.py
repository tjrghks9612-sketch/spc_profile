from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from image_processing import (
    DetectionError,
    ROI,
    clamp_roi,
    crop_roi,
    detect_dark_structure,
    extract_contour,
)


@dataclass
class ProfileResult:
    axis_name: str
    coordinate_name: str
    roi: ROI
    threshold_value: float
    component_mask: np.ndarray
    contour: np.ndarray
    top_x_px_roi: np.ndarray
    top_y_px_roi: np.ndarray
    baseline_y_px_roi: float
    left_x_px_roi: int
    right_x_px_roi: int
    center_x_px_roi: float
    cd_um: float
    height_um: float
    coord_um: np.ndarray
    z_um: np.ndarray
    normalized_r: np.ndarray
    normalized_f: np.ndarray


def estimate_baseline(component_mask: np.ndarray, contour: np.ndarray | None = None) -> float:
    mask = component_mask > 0
    if not np.any(mask):
        raise DetectionError("baseline 추정을 위한 mask가 비어 있습니다.")

    ys, xs = np.where(mask)
    left = int(xs.min())
    right = int(xs.max())
    width = right - left + 1
    if width <= 2:
        raise DetectionError("baseline 추정에 필요한 구조물 폭이 부족합니다.")

    bottom_by_col = []
    col_xs = []
    for x in range(left, right + 1):
        col = np.where(mask[:, x])[0]
        if col.size:
            bottom_by_col.append(int(col.max()))
            col_xs.append(x)
    if len(bottom_by_col) < 4:
        raise DetectionError("baseline 후보점이 부족합니다.")

    bottom_by_col = np.asarray(bottom_by_col, dtype=np.float32)
    col_xs = np.asarray(col_xs, dtype=np.int32)
    flank_width = max(3, int(round(width * 0.16)))
    flank_mask = (col_xs <= left + flank_width) | (col_xs >= right - flank_width)
    candidates = bottom_by_col[flank_mask]

    if candidates.size >= 4:
        baseline = float(np.median(candidates))
    else:
        baseline = float(np.percentile(bottom_by_col, 90))

    if contour is not None and contour.size > 0:
        contour_y = contour.reshape(-1, 2)[:, 1].astype(np.float32)
        lower_contour = float(np.percentile(contour_y, 90))
        baseline = float(np.median([baseline, lower_contour]))

    h = component_mask.shape[0]
    baseline = float(np.clip(round(baseline), 0, h - 1))
    top_min = float(ys.min())
    if baseline <= top_min:
        raise DetectionError("baseline이 top boundary보다 위에 있어 height를 계산할 수 없습니다.")
    return baseline


def _smooth_1d(values: np.ndarray, strength: float) -> np.ndarray:
    level = int(max(0, min(10, round(float(strength)))))
    if level <= 0 or values.size < 5:
        return values.astype(np.float32)
    window = min(values.size if values.size % 2 == 1 else values.size - 1, 2 * level + 1)
    if window < 3:
        return values.astype(np.float32)
    pad = window // 2
    padded = np.pad(values.astype(np.float32), pad, mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")


def extract_top_boundary_profile(component_mask: np.ndarray, smoothing_strength: float = 0) -> tuple[np.ndarray, np.ndarray]:
    mask = component_mask > 0
    xs = np.where(mask.any(axis=0))[0]
    if xs.size < 8:
        raise DetectionError("top boundary profile point 수가 부족합니다.")

    top_y = np.empty(xs.shape, dtype=np.float32)
    for idx, x in enumerate(xs):
        top_y[idx] = float(np.where(mask[:, x])[0].min())

    top_y = _smooth_1d(top_y, smoothing_strength)
    return xs.astype(np.int32), top_y.astype(np.float32)


def measure_cd_height(
    top_x_px_roi: np.ndarray,
    top_y_px_roi: np.ndarray,
    baseline_y_px_roi: float,
    pixel_size_um: float,
) -> tuple[int, int, float, float, float, np.ndarray, np.ndarray]:
    if pixel_size_um <= 0:
        raise DetectionError("pixel_size_um은 0보다 커야 합니다.")

    valid = np.isfinite(top_y_px_roi) & (baseline_y_px_roi > top_y_px_roi)
    if np.count_nonzero(valid) < 8:
        raise DetectionError("height profile로 사용할 수 있는 point 수가 부족합니다.")

    xs = top_x_px_roi[valid]
    ys = top_y_px_roi[valid]
    left = int(xs.min())
    right = int(xs.max())
    cd_px = right - left + 1
    if cd_px <= 1:
        raise DetectionError("CD 값이 0 이하입니다.")

    center = (left + right) / 2.0
    coord_um = (xs.astype(np.float32) - center) * float(pixel_size_um)
    z_um = (float(baseline_y_px_roi) - ys.astype(np.float32)) * float(pixel_size_um)
    z_um = np.clip(z_um, 0, None)

    cd_um = float(cd_px * pixel_size_um)
    height_um = float(np.max(z_um))
    if cd_um <= 0:
        raise DetectionError("CD 값이 0 이하입니다.")
    if height_um <= 0:
        raise DetectionError("Height 값이 0 이하입니다.")

    return left, right, center, cd_um, height_um, coord_um.astype(np.float32), z_um.astype(np.float32)


def normalize_profile(coord_um: np.ndarray, z_um: np.ndarray, cd_um: float, height_um: float) -> tuple[np.ndarray, np.ndarray]:
    if cd_um <= 0 or height_um <= 0:
        raise DetectionError("profile 정규화에 필요한 CD 또는 Height가 유효하지 않습니다.")

    half_cd = cd_um * 0.5
    r = np.abs(coord_um.astype(np.float64)) / half_cd
    f = z_um.astype(np.float64) / float(height_um)

    valid = np.isfinite(r) & np.isfinite(f) & (r <= 1.05)
    r = np.clip(r[valid], 0.0, 1.0)
    f = np.clip(f[valid], 0.0, 1.5)
    if r.size < 8:
        raise DetectionError("정규화 profile point 수가 부족합니다.")

    rounded = np.round(r, 5)
    unique_r = np.unique(rounded)
    avg_r = []
    avg_f = []
    for rv in unique_r:
        mask = rounded == rv
        avg_r.append(float(np.mean(r[mask])))
        avg_f.append(float(np.mean(f[mask])))

    r_out = np.asarray(avg_r, dtype=np.float64)
    f_out = np.asarray(avg_f, dtype=np.float64)
    order = np.argsort(r_out)
    r_out = r_out[order]
    f_out = f_out[order]

    center_value = 1.0
    if r_out[0] > 1e-6:
        r_out = np.insert(r_out, 0, 0.0)
        f_out = np.insert(f_out, 0, center_value)
    else:
        f_out[0] = center_value

    edge_value = float(np.clip(np.percentile(f_out[r_out > 0.88], 20), 0.0, 0.08)) if np.any(r_out > 0.88) else 0.0
    if r_out[-1] < 1.0 - 1e-6:
        r_out = np.append(r_out, 1.0)
        f_out = np.append(f_out, edge_value)
    else:
        f_out[-1] = min(float(f_out[-1]), edge_value)

    f_out = np.clip(f_out, 0.0, 1.2)
    return r_out.astype(np.float32), f_out.astype(np.float32)


def analyze_section(
    gray_image: np.ndarray,
    roi: ROI,
    pixel_size_um: float,
    threshold_sensitivity: float,
    smoothing_strength: float,
    morph_strength: float,
    axis_name: str,
    coordinate_name: str,
) -> ProfileResult:
    roi = clamp_roi(roi, gray_image.shape)
    crop = crop_roi(gray_image, roi)
    detection = detect_dark_structure(
        crop,
        threshold_sensitivity=threshold_sensitivity,
        smoothing_strength=smoothing_strength,
        morph_strength=morph_strength,
    )
    contour = extract_contour(detection.component_mask)
    baseline = estimate_baseline(detection.component_mask, contour)
    top_x, top_y = extract_top_boundary_profile(detection.component_mask, smoothing_strength=smoothing_strength)
    left, right, center, cd_um, height_um, coord_um, z_um = measure_cd_height(top_x, top_y, baseline, pixel_size_um)
    normalized_r, normalized_f = normalize_profile(coord_um, z_um, cd_um, height_um)

    return ProfileResult(
        axis_name=axis_name,
        coordinate_name=coordinate_name,
        roi=roi,
        threshold_value=detection.threshold_value,
        component_mask=detection.component_mask,
        contour=contour,
        top_x_px_roi=top_x,
        top_y_px_roi=top_y,
        baseline_y_px_roi=baseline,
        left_x_px_roi=left,
        right_x_px_roi=right,
        center_x_px_roi=center,
        cd_um=cd_um,
        height_um=height_um,
        coord_um=coord_um,
        z_um=z_um,
        normalized_r=normalized_r,
        normalized_f=normalized_f,
    )
