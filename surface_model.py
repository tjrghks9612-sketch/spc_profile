from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from profile_extraction import ProfileResult


@dataclass
class SurfaceGrid:
    X: np.ndarray
    Y: np.ndarray
    Z_height_um: np.ndarray
    Z: np.ndarray
    valid_mask: np.ndarray
    A_um: float
    B_um: float
    H_global_um: float
    H_surface_max_um: float
    sanity_check_x_rmse_um: float
    sanity_check_y_rmse_um: float


def _interp_profile(rho: np.ndarray, r: np.ndarray, f: np.ndarray) -> np.ndarray:
    flat = rho.ravel()
    values = np.interp(flat, r.astype(np.float64), f.astype(np.float64), left=float(f[0]), right=float(f[-1]))
    return values.reshape(rho.shape)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 4:
        return float("nan")
    delta = a[valid] - b[valid]
    return float(np.sqrt(np.mean(delta * delta)))


def _centered_axis(half_width: float, grid_resolution: int) -> np.ndarray:
    axis = np.linspace(-half_width, half_width, grid_resolution, dtype=np.float64)
    axis[int(np.argmin(np.abs(axis)))] = 0.0
    return axis


def _profile_depth_um(profile: ProfileResult) -> np.ndarray:
    depth = profile.height_um - profile.z_um.astype(np.float64)
    return np.clip(depth, 0.0, None)


def _profile_rmse_along_x(surface: "SurfaceGrid", profile: ProfileResult) -> float:
    row = int(np.nanargmin(np.abs(surface.Y[:, 0])))
    depth_surface = surface.Z[row, :]
    depth_profile = np.interp(
        surface.X[row, :],
        profile.coord_um.astype(np.float64),
        _profile_depth_um(profile),
        left=np.nan,
        right=np.nan,
    )
    return _rmse(depth_surface, depth_profile)


def _profile_rmse_along_y(surface: "SurfaceGrid", profile: ProfileResult) -> float:
    col = int(np.nanargmin(np.abs(surface.X[0, :])))
    depth_surface = surface.Z[:, col]
    depth_profile = np.interp(
        surface.Y[:, col],
        profile.coord_um.astype(np.float64),
        _profile_depth_um(profile),
        left=np.nan,
        right=np.nan,
    )
    return _rmse(depth_surface, depth_profile)


def build_profile_based_surface(
    horizontal: ProfileResult,
    vertical: ProfileResult,
    grid_resolution: int = 400,
) -> SurfaceGrid:
    grid_resolution = int(grid_resolution)
    if grid_resolution < 30:
        raise ValueError("grid_resolution은 30 이상이어야 합니다.")

    A = horizontal.cd_um * 0.5
    B = vertical.cd_um * 0.5
    if A <= 0 or B <= 0:
        raise ValueError("CD 값이 유효하지 않아 surface를 생성할 수 없습니다.")

    h_global = (horizontal.height_um + vertical.height_um) * 0.5
    x = _centered_axis(A, grid_resolution)
    y = _centered_axis(B, grid_resolution)
    X, Y = np.meshgrid(x, y)

    U = X / A
    V = Y / B
    rho = np.sqrt(U * U + V * V)
    valid = rho <= 1.0

    f_x = _interp_profile(rho, horizontal.normalized_r, horizontal.normalized_f)
    f_y = _interp_profile(rho, vertical.normalized_r, vertical.normalized_f)

    theta = np.arctan2(np.abs(V), np.abs(U))
    wx = np.cos(theta) ** 2
    wy = np.sin(theta) ** 2
    center = rho < 1e-12
    if np.any(center):
        wx[center] = 0.5
        wy[center] = 0.5

    # Blend actual section heights so the x/y axes preserve the measured profiles.
    height = wx * (horizontal.height_um * f_x) + wy * (vertical.height_um * f_y)
    height = np.clip(height, 0.0, None)
    height[~valid] = np.nan

    depth = wx * (horizontal.height_um * np.clip(1.0 - f_x, 0.0, None))
    depth += wy * (vertical.height_um * np.clip(1.0 - f_y, 0.0, None))
    depth = np.clip(depth, 0.0, None)
    depth[~valid] = np.nan

    h_surface_max = float(np.nanmax(depth))
    surface = SurfaceGrid(
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        Z_height_um=height.astype(np.float32),
        Z=depth.astype(np.float32),
        valid_mask=valid,
        A_um=float(A),
        B_um=float(B),
        H_global_um=float(h_global),
        H_surface_max_um=h_surface_max,
        sanity_check_x_rmse_um=float("nan"),
        sanity_check_y_rmse_um=float("nan"),
    )
    surface.sanity_check_x_rmse_um = _profile_rmse_along_x(surface, horizontal)
    surface.sanity_check_y_rmse_um = _profile_rmse_along_y(surface, vertical)
    return surface
