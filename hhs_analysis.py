from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_HHS_SETTINGS = {
    "light_smooth_sigma": 0.015,
    "baseline_smooth_sigma": 0.12,
    "center_width": 0.35,
}


def _validate_settings(settings: dict[str, Any] | None) -> dict[str, float]:
    merged = dict(DEFAULT_HHS_SETTINGS)
    if settings:
        merged.update(settings)

    light = float(merged["light_smooth_sigma"])
    baseline = float(merged["baseline_smooth_sigma"])
    center_width = float(merged["center_width"])
    if light <= 0:
        raise ValueError("light_smooth_sigma는 0보다 커야 합니다.")
    if baseline <= 0:
        raise ValueError("baseline_smooth_sigma는 0보다 커야 합니다.")
    if center_width <= 0 or center_width > 1:
        raise ValueError("center_width는 0보다 크고 1 이하여야 합니다.")
    return {
        "light_smooth_sigma": light,
        "baseline_smooth_sigma": baseline,
        "center_width": center_width,
    }


def _gaussian_kernel(sigma_samples: float) -> np.ndarray:
    radius = max(1, int(np.ceil(float(sigma_samples) * 4.0)))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / max(float(sigma_samples), 1e-9)) ** 2)
    total = float(np.sum(kernel))
    if total <= 0:
        return np.asarray([1.0], dtype=np.float64)
    return kernel / total


def _gaussian_smooth(y: np.ndarray, x: np.ndarray, sigma_x: float) -> np.ndarray:
    diffs = np.diff(x.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return y.astype(np.float64).copy()
    dx = float(np.median(diffs))
    sigma_samples = max(float(sigma_x) / max(dx, 1e-12), 1e-9)
    kernel = _gaussian_kernel(sigma_samples)
    pad = kernel.size // 2
    padded = np.pad(y.astype(np.float64), pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _unavailable_result(
    reason: str,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    settings: dict[str, float],
    x_center_used: float,
    profile_width_used: float,
) -> dict[str, Any]:
    nan_array = np.full_like(y_norm, np.nan, dtype=np.float64)
    return {
        "status": "UNAVAILABLE",
        "reason": reason,
        "hhs": float("nan"),
        "bump_area": float("nan"),
        "total_area": float("nan"),
        "light_smooth_sigma": settings["light_smooth_sigma"],
        "baseline_smooth_sigma": settings["baseline_smooth_sigma"],
        "center_width": settings["center_width"],
        "x_center_used": float(x_center_used),
        "profile_width_used": float(profile_width_used),
        "x_norm": x_norm.astype(np.float64),
        "y_norm": y_norm.astype(np.float64),
        "y_smooth": nan_array,
        "baseline": nan_array,
        "residual": nan_array,
        "positive_residual": np.zeros_like(y_norm, dtype=np.float64),
        "center_mask": np.zeros_like(y_norm, dtype=bool),
    }


def compute_hill_on_hill_score(
    x: np.ndarray,
    y: np.ndarray,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hhs_settings = _validate_settings(settings)
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 10:
        raise ValueError("HHS 계산에는 유효한 profile point가 10개 이상 필요합니다.")

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    unique_x, unique_idx = np.unique(x_arr, return_index=True)
    x_arr = unique_x
    y_arr = y_arr[unique_idx]
    if x_arr.size < 10:
        raise ValueError("HHS 계산에는 서로 다른 x 좌표가 10개 이상 필요합니다.")

    y_min = float(np.nanmin(y_arr))
    y_max = float(np.nanmax(y_arr))
    if y_max <= y_min:
        x_mid = (x_arr - float(np.mean([x_arr[0], x_arr[-1]]))) / max(float(x_arr[-1] - x_arr[0]) * 0.5, 1e-12)
        return _unavailable_result("y 값이 모두 같아 HHS를 계산할 수 없습니다.", x_mid, np.zeros_like(x_mid), hhs_settings, 0.0, 0.0)

    y_nonnegative = np.clip(y_arr - y_min, 0.0, None)
    y_scale = float(np.nanmax(y_nonnegative))
    if y_scale <= 0:
        x_mid = (x_arr - float(np.mean([x_arr[0], x_arr[-1]]))) / max(float(x_arr[-1] - x_arr[0]) * 0.5, 1e-12)
        return _unavailable_result("profile 높이 범위가 0입니다.", x_mid, np.zeros_like(x_mid), hhs_settings, 0.0, 0.0)
    y_norm = y_nonnegative / y_scale

    x_span = float(x_arr[-1] - x_arr[0])
    if x_span <= 0:
        raise ValueError("HHS 계산에는 0보다 큰 x 범위가 필요합니다.")
    x_mid_norm = (x_arr - float(np.mean([x_arr[0], x_arr[-1]]))) / (x_span * 0.5)
    preliminary = _gaussian_smooth(y_norm, x_mid_norm, hhs_settings["light_smooth_sigma"])
    center_idx = int(np.nanargmax(preliminary))
    x_center = float(x_arr[center_idx])
    width = float(max(abs(x_arr[0] - x_center), abs(x_arr[-1] - x_center)))
    if width <= 0:
        raise ValueError("HHS 계산에는 0보다 큰 profile 폭이 필요합니다.")
    x_norm = (x_arr - x_center) / width

    y_smooth = _gaussian_smooth(y_norm, x_norm, hhs_settings["light_smooth_sigma"])
    baseline = _gaussian_smooth(y_norm, x_norm, hhs_settings["baseline_smooth_sigma"])
    residual = y_smooth - baseline
    center_mask = np.abs(x_norm) <= hhs_settings["center_width"]
    positive_residual = np.where(center_mask, np.clip(residual, 0.0, None), 0.0)

    bump_area = float(np.trapezoid(positive_residual, x_norm))
    total_area = float(np.trapezoid(np.clip(y_norm, 0.0, None), x_norm))
    hhs = float(bump_area / total_area) if total_area > 0 else float("nan")
    status = "OK" if total_area > 0 else "UNAVAILABLE"
    reason = "" if total_area > 0 else "profile 전체 면적이 0 이하라 HHS를 계산할 수 없습니다."

    return {
        "status": status,
        "reason": reason,
        "hhs": hhs,
        "bump_area": bump_area if total_area > 0 else float("nan"),
        "total_area": total_area,
        "light_smooth_sigma": hhs_settings["light_smooth_sigma"],
        "baseline_smooth_sigma": hhs_settings["baseline_smooth_sigma"],
        "center_width": hhs_settings["center_width"],
        "x_center_used": x_center,
        "profile_width_used": width,
        "x_norm": x_norm.astype(np.float64),
        "y_norm": y_norm.astype(np.float64),
        "y_smooth": y_smooth.astype(np.float64),
        "baseline": baseline.astype(np.float64),
        "residual": residual.astype(np.float64),
        "positive_residual": positive_residual.astype(np.float64),
        "center_mask": center_mask.astype(bool),
    }
