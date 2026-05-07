from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from matplotlib.figure import Figure

from area_calculation import CapMetrics
from profile_extraction import ProfileResult
from surface_model import SurfaceGrid


BG = "#111827"
PANEL = "#172033"
FG = "#E5E7EB"
MUTED = "#9CA3AF"
ACCENT = "#38BDF8"
ACCENT_2 = "#22C55E"
WARN = "#F59E0B"


def _style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color="#334155", alpha=0.45, linewidth=0.7)


def _save_fig(fig: Figure, output_path: str | Path | None) -> None:
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())


def _apply_surface_3d_aspect(ax, surface: SurfaceGrid) -> None:
    x_span = float(surface.A_um * 2.0)
    y_span = float(surface.B_um * 2.0)
    z_span = float(max(surface.H_surface_max_um, np.nanmax(surface.Z), 1e-9))

    ax.set_xlim(-surface.A_um, surface.A_um)
    ax.set_ylim(-surface.B_um, surface.B_um)
    ax.set_zlim(z_span, 0.0)
    ax.set_box_aspect((x_span, y_span, z_span))
    ax.text2D(
        0.02,
        0.96,
        f"X span {x_span:.3f} um | Y span {y_span:.3f} um",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8,
    )


def plot_detection_overlay(gray: np.ndarray, profile: ProfileResult, output_path: str | Path | None = None) -> np.ndarray:
    if gray.ndim == 2:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

    x0, y0, w, h = profile.roi
    overlay = rgb.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w - 1, y0 + h - 1), (245, 158, 11), 2)

    contour = profile.contour + np.array([[[x0, y0]]], dtype=np.int32)
    cv2.drawContours(overlay, [contour], -1, (56, 189, 248), 2)

    top_points = np.column_stack(
        [
            profile.top_x_px_roi + x0,
            np.round(profile.top_y_px_roi).astype(np.int32) + y0,
        ]
    ).astype(np.int32)
    if len(top_points) >= 2:
        cv2.polylines(overlay, [top_points.reshape(-1, 1, 2)], False, (34, 197, 94), 2)

    baseline_y = int(round(y0 + profile.baseline_y_px_roi))
    left = int(x0 + profile.left_x_px_roi)
    right = int(x0 + profile.right_x_px_roi)
    center = int(round(x0 + profile.center_x_px_roi))
    top_idx = int(np.argmax(profile.z_um))
    top_y = int(round(y0 + profile.top_y_px_roi[top_idx]))

    cv2.line(overlay, (left, baseline_y), (right, baseline_y), (251, 191, 36), 2)
    cv2.line(overlay, (left, baseline_y + 12), (right, baseline_y + 12), (56, 189, 248), 1)
    cv2.line(overlay, (left, baseline_y + 7), (left, baseline_y + 17), (56, 189, 248), 1)
    cv2.line(overlay, (right, baseline_y + 7), (right, baseline_y + 17), (56, 189, 248), 1)
    cv2.line(overlay, (center, baseline_y), (center, top_y), (248, 113, 113), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, f"CD {profile.cd_um:.3f} um", (left, min(overlay.shape[0] - 8, baseline_y + 35)), font, 0.55, (56, 189, 248), 2)
    cv2.putText(overlay, f"H {profile.height_um:.3f} um", (min(center + 8, overlay.shape[1] - 160), max(18, top_y - 8)), font, 0.55, (248, 113, 113), 2)

    if output_path:
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return overlay


def plot_profiles(horizontal: ProfileResult, vertical: ProfileResult, output_path: str | Path | None = None) -> Figure:
    fig = Figure(figsize=(7.2, 4.2), facecolor=BG)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    horizontal_depth = np.clip(horizontal.height_um - horizontal.z_um, 0.0, None)
    vertical_depth = np.clip(vertical.height_um - vertical.z_um, 0.0, None)
    ax.plot(horizontal.coord_um, horizontal_depth, color=ACCENT, linewidth=2.0, label="Horizontal x-depth")
    ax.plot(vertical.coord_um, vertical_depth, color=ACCENT_2, linewidth=2.0, label="Vertical y-depth")
    ax.set_xlabel("Centered coordinate (um)")
    ax.set_ylabel("Depth from apex (um, down +)")
    ax.set_title("Extracted top boundary profiles, apex-referenced")
    ax.invert_yaxis()
    ax.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=FG)
    fig.tight_layout()
    _save_fig(fig, output_path)
    return fig


def plot_single_profile_cd_depth(profile: ProfileResult, cd_result, output_path: str | Path | None = None) -> Figure:
    fig = Figure(figsize=(7.2, 4.2), facecolor=BG)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    depth = np.clip(profile.height_um - profile.z_um, 0.0, None)
    depth = depth - float(np.nanmin(depth))
    ax.plot(profile.coord_um, depth, color=ACCENT, linewidth=2.1, label="Top boundary depth")

    count = len(cd_result.depth_um)
    if count:
        if count <= 28:
            indices = np.arange(count)
        else:
            indices = np.unique(np.linspace(0, count - 1, 28).astype(int))
        for idx in indices:
            d = float(cd_result.depth_um[idx])
            lx = float(cd_result.left_x_um[idx])
            rx = float(cd_result.right_x_um[idx])
            ax.plot([lx, rx], [d, d], color=WARN, alpha=0.42, linewidth=1.0)
        ax.scatter(cd_result.left_x_um, cd_result.depth_um, s=12, color=ACCENT_2, label="CD intersections")
        ax.scatter(cd_result.right_x_um, cd_result.depth_um, s=12, color=ACCENT_2)

    ax.set_xlabel("Centered coordinate (um)")
    ax.set_ylabel("Depth from apex (um, down +)")
    ax.set_title("Single-section CD by depth")
    ax.invert_yaxis()
    ax.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=FG)
    fig.tight_layout()
    _save_fig(fig, output_path)
    return fig


def plot_single_batch_cd_depth(results, output_path: str | Path | None = None) -> Figure:
    fig = Figure(figsize=(8.2, 4.4), facecolor=BG)
    ax_profile = fig.add_subplot(121)
    ax_cd = fig.add_subplot(122)
    _style_axes(ax_profile)
    _style_axes(ax_cd)

    success = [result for result in results if result.profile is not None and result.cd_result is not None]
    colors = [
        "#38BDF8",
        "#22C55E",
        "#F59E0B",
        "#F472B6",
        "#A78BFA",
        "#2DD4BF",
        "#FB7185",
        "#EAB308",
    ]

    for idx, result in enumerate(success[:24]):
        profile = result.profile
        cd_result = result.cd_result
        color = colors[idx % len(colors)]
        label = result.item.name if idx < 8 else None
        depth = np.clip(profile.height_um - profile.z_um, 0.0, None)
        depth = depth - float(np.nanmin(depth))
        ax_profile.plot(profile.coord_um, depth, color=color, linewidth=1.3, alpha=0.82, label=label)
        ax_cd.plot(cd_result.cd_um, cd_result.depth_um, color=color, linewidth=1.4, alpha=0.86, label=label)

    if len(success) > 24:
        ax_profile.text(
            0.02,
            0.96,
            f"showing 24 of {len(success)}",
            transform=ax_profile.transAxes,
            color=MUTED,
            fontsize=8,
            va="top",
        )

    ax_profile.set_xlabel("Centered coordinate (um)")
    ax_profile.set_ylabel("Depth from apex (um)")
    ax_profile.set_title("Top boundary profiles")
    ax_profile.invert_yaxis()

    ax_cd.set_xlabel("CD (um)")
    ax_cd.set_ylabel("Depth from apex (um)")
    ax_cd.set_title("CD by depth")
    ax_cd.invert_yaxis()

    if success:
        ax_profile.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=FG, fontsize=7)
    fig.tight_layout()
    _save_fig(fig, output_path)
    return fig


def plot_3d_surface(surface: SurfaceGrid, output_path: str | Path | None = None) -> Figure:
    fig = Figure(figsize=(7.2, 4.8), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL)
    z = np.ma.masked_invalid(surface.Z)
    ax.plot_surface(surface.X, surface.Y, z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.96)
    ax.set_xlabel("x (um)", color=FG)
    ax.set_ylabel("y (um)", color=FG)
    ax.set_zlabel("Depth z (um, down +)", color=FG)
    ax.set_title("2-section profile-based 3D surface, apex z=0", color=FG)
    ax.tick_params(colors=MUTED, labelsize=8)
    _apply_surface_3d_aspect(ax, surface)
    ax.view_init(elev=28, azim=-135)
    fig.tight_layout()
    _save_fig(fig, output_path)
    return fig


def plot_cap_highlight(surface: SurfaceGrid, cap: CapMetrics, output_path: str | Path | None = None) -> Figure:
    fig = Figure(figsize=(7.2, 4.8), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL)
    base_z = np.ma.masked_invalid(surface.Z)
    cap_z = np.ma.masked_where(~cap.cap_mask, surface.Z)
    ax.plot_surface(surface.X, surface.Y, base_z, color="#1F2937", linewidth=0, alpha=0.45)
    ax.plot_surface(surface.X, surface.Y, cap_z, cmap="plasma", linewidth=0, antialiased=True, alpha=0.98)
    ax.set_xlabel("x (um)", color=FG)
    ax.set_ylabel("y (um)", color=FG)
    ax.set_zlabel("Depth z (um, down +)", color=FG)
    ax.set_title(f"Cap region depth <= {cap.z_cut_um:.3f} um", color=FG)
    ax.tick_params(colors=MUTED, labelsize=8)
    _apply_surface_3d_aspect(ax, surface)
    ax.view_init(elev=30, azim=-135)
    fig.tight_layout()
    _save_fig(fig, output_path)
    return fig


def plot_cap_top_view(surface: SurfaceGrid, cap: CapMetrics, output_path: str | Path | None = None) -> Figure:
    fig = Figure(figsize=(7.2, 4.8), facecolor=BG)
    ax = fig.add_axes([0.22, 0.22, 0.56, 0.66])
    _style_axes(ax)
    data = np.ma.masked_where(~surface.valid_mask, surface.Z)
    extent = [-surface.A_um, surface.A_um, -surface.B_um, surface.B_um]
    im = ax.imshow(
        data,
        extent=extent,
        origin="lower",
        cmap="viridis",
        alpha=0.75,
        interpolation="nearest",
        aspect="equal",
    )
    cap_outline = np.ma.masked_where(~cap.cap_mask, cap.cap_mask.astype(float))
    ax.contour(surface.X, surface.Y, cap_outline, levels=[0.5], colors=[WARN], linewidths=1.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-surface.A_um, surface.A_um)
    ax.set_ylim(-surface.B_um, surface.B_um)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Cap top-view footprint, centered")
    cax = fig.add_axes([0.32, 0.10, 0.36, 0.035])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Depth from apex (um)", color=FG)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    _save_fig(fig, output_path)
    return fig
