from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CapMetrics:
    cap_mask: np.ndarray
    z_cut_um: float
    cap_curved_surface_area_um2: float
    cap_projected_area_um2: float


def compute_cap_mask(Z: np.ndarray, valid_mask: np.ndarray, cap_depth_um: float) -> tuple[np.ndarray, float]:
    if cap_depth_um <= 0:
        raise ValueError("cap_depth_um은 0보다 커야 합니다.")
    max_depth = float(np.nanmax(Z[valid_mask]))
    z_cut = min(float(cap_depth_um), max_depth)
    cap_mask = (Z <= z_cut) & valid_mask
    return cap_mask, float(z_cut)


def _cell_mask(valid_mask: np.ndarray, cap_mask: np.ndarray | None = None) -> np.ndarray:
    mask = (
        valid_mask[:-1, :-1]
        & valid_mask[1:, :-1]
        & valid_mask[:-1, 1:]
        & valid_mask[1:, 1:]
    )
    if cap_mask is not None:
        mask = mask & (
            cap_mask[:-1, :-1]
            & cap_mask[1:, :-1]
            & cap_mask[:-1, 1:]
            & cap_mask[1:, 1:]
        )
    return mask


def compute_mesh_surface_area(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    valid_mask: np.ndarray,
    cap_mask: np.ndarray,
) -> float:
    cells = _cell_mask(valid_mask, cap_mask)
    if not np.any(cells):
        return 0.0

    p00 = np.stack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]], axis=-1)
    p10 = np.stack([X[1:, :-1], Y[1:, :-1], Z[1:, :-1]], axis=-1)
    p01 = np.stack([X[:-1, 1:], Y[:-1, 1:], Z[:-1, 1:]], axis=-1)
    p11 = np.stack([X[1:, 1:], Y[1:, 1:], Z[1:, 1:]], axis=-1)

    tri1 = 0.5 * np.linalg.norm(np.cross(p10 - p00, p11 - p00), axis=-1)
    tri2 = 0.5 * np.linalg.norm(np.cross(p11 - p00, p01 - p00), axis=-1)
    area = np.where(cells, tri1 + tri2, 0.0)
    return float(np.nansum(area))


def compute_projected_area(X: np.ndarray, Y: np.ndarray, valid_mask: np.ndarray, cap_mask: np.ndarray) -> float:
    cells = _cell_mask(valid_mask, cap_mask)
    if not np.any(cells):
        return 0.0
    dx = np.abs(X[:-1, 1:] - X[:-1, :-1])
    dy = np.abs(Y[1:, :-1] - Y[:-1, :-1])
    cell_area = dx * dy
    return float(np.nansum(np.where(cells, cell_area, 0.0)))


def compute_cap_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    valid_mask: np.ndarray,
    cap_depth_um: float,
) -> CapMetrics:
    cap_mask, z_cut = compute_cap_mask(Z, valid_mask, cap_depth_um)
    curved_area = compute_mesh_surface_area(X, Y, Z, valid_mask, cap_mask)
    projected_area = compute_projected_area(X, Y, valid_mask, cap_mask)
    return CapMetrics(
        cap_mask=cap_mask,
        z_cut_um=z_cut,
        cap_curved_surface_area_um2=curved_area,
        cap_projected_area_um2=projected_area,
    )
