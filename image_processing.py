from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class DetectionError(RuntimeError):
    """Raised when a dark structure cannot be detected reliably."""


@dataclass
class DetectionResult:
    threshold_value: float
    binary_mask: np.ndarray
    component_mask: np.ndarray
    component_area_px: int


ROI = Tuple[int, int, int, int]


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("지원하지 않는 이미지 형식입니다.")

    if gray.dtype == np.uint8:
        return gray

    gray_float = gray.astype(np.float32)
    min_v = float(np.nanmin(gray_float))
    max_v = float(np.nanmax(gray_float))
    if max_v <= min_v:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip((gray_float - min_v) / (max_v - min_v) * 255.0, 0, 255).astype(np.uint8)


def clamp_roi(roi: ROI, image_shape: tuple[int, int]) -> ROI:
    x, y, w, h = [int(round(v)) for v in roi]
    image_h, image_w = image_shape[:2]
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    w = max(1, min(w, image_w - x))
    h = max(1, min(h, image_h - y))
    return x, y, w, h


def crop_roi(gray: np.ndarray, roi: ROI) -> np.ndarray:
    x, y, w, h = clamp_roi(roi, gray.shape)
    return gray[y : y + h, x : x + w].copy()


def _slider_to_int(value: float, low: int = 0, high: int = 10) -> int:
    return int(max(low, min(high, round(float(value)))))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    flood = mask_u8.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return ((mask_u8 | holes) > 0).astype(np.uint8)


def detect_dark_structure(
    crop_gray: np.ndarray,
    threshold_sensitivity: float,
    smoothing_strength: float,
    morph_strength: float,
) -> DetectionResult:
    if crop_gray.size == 0:
        raise DetectionError("ROI crop이 비어 있습니다.")

    gray = to_grayscale(crop_gray)
    smooth_level = _slider_to_int(smoothing_strength)
    if smooth_level > 0:
        kernel = 2 * smooth_level + 1
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), sigmaX=0)
    else:
        blurred = gray

    otsu_value, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    p05, p95 = np.percentile(blurred, [5, 95])
    sensitivity = max(0.0, min(100.0, float(threshold_sensitivity)))
    adjusted = otsu_value + (sensitivity - 50.0) * 1.2
    threshold_value = float(np.clip(adjusted, p05, p95))

    binary_mask = (blurred <= threshold_value).astype(np.uint8)

    morph_level = _slider_to_int(morph_strength)
    if morph_level > 0:
        kernel_size = 2 * morph_level + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    component_mask, component_area = extract_largest_component(binary_mask)
    component_mask = _fill_holes(component_mask)
    component_area = int(np.count_nonzero(component_mask))

    return DetectionResult(
        threshold_value=threshold_value,
        binary_mask=binary_mask.astype(np.uint8),
        component_mask=component_mask.astype(np.uint8),
        component_area_px=component_area,
    )


def extract_largest_component(mask: np.ndarray, min_area_ratio: float = 0.0015) -> tuple[np.ndarray, int]:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if labels_count <= 1:
        raise DetectionError("ROI 내부에서 dark object를 찾지 못했습니다.")

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
    min_area = max(25, int(mask.size * min_area_ratio))
    if largest_area < min_area:
        raise DetectionError(f"dark component 면적이 너무 작습니다: {largest_area}px")

    return (labels == largest_idx).astype(np.uint8), largest_area


def extract_contour(component_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours((component_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise DetectionError("구조물 contour를 추출하지 못했습니다.")
    return max(contours, key=cv2.contourArea)


def generate_synthetic_mound_image(
    width: int = 900,
    height: int = 520,
    cd_px: int = 420,
    mound_height_px: int = 170,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    background = 172 + 24 * yy + rng.normal(0, 6, (height, width)).astype(np.float32)

    scanline = 3.5 * np.sin(np.linspace(0, 42, height, dtype=np.float32))[:, None]
    image = background + scanline

    center_x = width // 2 + int(rng.normal(0, 8))
    baseline_y = int(height * 0.70 + rng.normal(0, 3))
    radius = cd_px / 2.0
    xs = np.arange(width, dtype=np.float32)
    rr = np.abs(xs - center_x) / radius
    mound = rr <= 1.0
    dome_profile = np.zeros(width, dtype=np.float32)
    dome_profile[mound] = mound_height_px * np.power(np.clip(1.0 - rr[mound] ** 2, 0, 1), 0.58)
    dome_profile += rng.normal(0, 2.0, width).astype(np.float32) * mound

    for x in np.where(mound)[0]:
        top_y = int(round(baseline_y - dome_profile[x]))
        top_y = max(8, min(baseline_y - 1, top_y))
        texture = rng.normal(0, 7, baseline_y - top_y + 1).astype(np.float32)
        image[top_y : baseline_y + 1, x] = 62 + 0.11 * (np.arange(top_y, baseline_y + 1) - top_y) + texture

    image = cv2.GaussianBlur(np.clip(image, 0, 255).astype(np.uint8), (3, 3), sigmaX=0)
    return image


def generate_synthetic_pair() -> tuple[np.ndarray, np.ndarray]:
    horizontal = generate_synthetic_mound_image(cd_px=450, mound_height_px=176, seed=41)
    vertical = generate_synthetic_mound_image(cd_px=320, mound_height_px=166, seed=73)
    return horizontal, vertical
