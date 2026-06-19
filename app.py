from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from area_calculation import CapMetrics, compute_cap_metrics
from image_processing import DetectionError, clamp_roi, generate_synthetic_mound_image, generate_synthetic_pair, load_image, to_grayscale
from hhs_analysis import DEFAULT_HHS_SETTINGS, compute_hill_on_hill_score
from io_utils import build_summary, create_output_dir, save_outputs
from plotting import (
    plot_3d_surface,
    plot_cap_highlight,
    plot_cap_top_view,
    plot_detection_overlay,
    plot_hhs_batch,
    plot_hhs_profile,
    plot_profiles,
    plot_single_batch_cd_depth,
)
from profile_extraction import ProfileResult, analyze_section
from single_profile import SingleDepthCDResult, SingleTaperResult, compute_cd_by_depth, compute_taper_by_offset, save_single_batch_outputs
from surface_model import SurfaceGrid, build_profile_based_surface


@dataclass
class AnalysisState:
    horizontal_profile: ProfileResult
    vertical_profile: ProfileResult
    surface: SurfaceGrid
    cap: CapMetrics
    params: dict


@dataclass
class SingleImageItem:
    name: str
    image: np.ndarray


@dataclass
class SingleProfileRunResult:
    item: SingleImageItem
    roi: Optional[tuple[int, int, int, int]] = None
    profile: Optional[ProfileResult] = None
    cd_result: Optional[SingleDepthCDResult] = None
    taper_result: Optional[SingleTaperResult] = None
    error: str = ""


@dataclass
class SingleBatchState:
    results: list[SingleProfileRunResult]
    params: dict


@dataclass
class HHSRunResult:
    item: SingleImageItem
    roi: Optional[tuple[int, int, int, int]] = None
    profile: Optional[ProfileResult] = None
    hhs_result: Optional[dict] = None
    error: str = ""
    group_label: str = ""


@dataclass
class HHSBatchState:
    results: list[HHSRunResult]
    params: dict


class ROIImageView(QLabel):
    roi_changed = Signal(tuple)

    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setMinimumSize(320, 190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#0B1120;border:1px solid #273449;border-radius:8px;color:#94A3B8;")
        self.setText(f"{title}\n이미지를 불러오고 ROI를 드래그하세요")
        self._image_rgb: Optional[np.ndarray] = None
        self._roi: Optional[tuple[int, int, int, int]] = None
        self._drag_start: Optional[QPoint] = None
        self._drag_current: Optional[QPoint] = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._scaled_size = (0, 0)

    @property
    def roi(self) -> Optional[tuple[int, int, int, int]]:
        return self._roi

    def set_image(self, image: np.ndarray) -> None:
        if image.ndim == 2:
            self._image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            self._image_rgb = image.copy()
        self._roi = None
        self._drag_start = None
        self._drag_current = None
        self._refresh()

    def set_overlay(self, image_rgb: np.ndarray) -> None:
        self._image_rgb = image_rgb.copy()
        self._refresh()

    def set_roi(self, roi: tuple[int, int, int, int]) -> None:
        if self._image_rgb is not None:
            self._roi = clamp_roi(roi, self._image_rgb.shape)
        else:
            self._roi = roi
        self.roi_changed.emit(self._roi)
        self._refresh()

    def clear_roi(self) -> None:
        self._roi = None
        self._drag_start = None
        self._drag_current = None
        self._refresh()

    def image_shape(self) -> Optional[tuple[int, int]]:
        if self._image_rgb is None:
            return None
        return self._image_rgb.shape[:2]

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._image_rgb is not None:
            pos = self._widget_to_image(event.position().toPoint())
            if pos is not None:
                self._drag_start = QPoint(*pos)
                self._drag_current = QPoint(*pos)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None and self._image_rgb is not None:
            pos = self._widget_to_image(event.position().toPoint())
            if pos is not None:
                self._drag_current = QPoint(*pos)
                self._roi = self._points_to_roi(self._drag_start, self._drag_current)
                self._refresh()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drag_start is not None and self._drag_current is not None:
            self._roi = self._points_to_roi(self._drag_start, self._drag_current)
            self._drag_start = None
            self._drag_current = None
            if self._roi[2] >= 8 and self._roi[3] >= 8:
                self.roi_changed.emit(self._roi)
            self._refresh()

    def _points_to_roi(self, p1: QPoint, p2: QPoint) -> tuple[int, int, int, int]:
        x1, x2 = sorted([p1.x(), p2.x()])
        y1, y2 = sorted([p1.y(), p2.y()])
        return int(x1), int(y1), max(1, int(x2 - x1 + 1)), max(1, int(y2 - y1 + 1))

    def _widget_to_image(self, point: QPoint) -> Optional[tuple[int, int]]:
        if self._image_rgb is None:
            return None
        x = point.x() - self._offset.x()
        y = point.y() - self._offset.y()
        if x < 0 or y < 0 or x >= self._scaled_size[0] or y >= self._scaled_size[1]:
            return None
        image_h, image_w = self._image_rgb.shape[:2]
        ix = int(np.clip(x / self._scale, 0, image_w - 1))
        iy = int(np.clip(y / self._scale, 0, image_h - 1))
        return ix, iy

    def _refresh(self) -> None:
        if self._image_rgb is None:
            return
        image = self._image_rgb.copy()
        if self._roi is not None:
            x, y, w, h = self._roi
            cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (245, 158, 11), 2)

        h, w = image.shape[:2]
        qimage = QImage(image.data, w, h, image.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = (scaled.width(), scaled.height())
        self._scale = scaled.width() / w if w else 1.0
        self._offset = QPoint((self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2)

        canvas = QPixmap(self.size())
        canvas.fill(Qt.transparent)
        painter = QPainter(canvas)
        painter.drawPixmap(self._offset, scaled)
        painter.end()
        self.setPixmap(canvas)


def _configure_spinbox(spinbox: QAbstractSpinBox, step: float) -> None:
    spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    spinbox.setAccelerated(True)
    spinbox.setKeyboardTracking(False)
    if isinstance(spinbox, (QDoubleSpinBox, QSpinBox)):
        spinbox.setSingleStep(step)


def _configure_slider(slider: QSlider, single_step: int = 1, page_step: int = 5) -> None:
    slider.setSingleStep(single_step)
    slider.setPageStep(page_step)


def _make_canvas(figure) -> FigureCanvas:
    canvas = FigureCanvas(figure)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    canvas.updateGeometry()
    return canvas


class ResultPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("Card")
        self.labels: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        title = QLabel("측정 결과")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        items = [
            ("CD_x_um", "가로 CD"),
            ("H_x_um", "가로 Height"),
            ("CD_y_um", "세로 CD"),
            ("H_y_um", "세로 Height"),
            ("H_global_um", "Global Height"),
            ("H_surface_max_um", "Surface max depth"),
            ("cap_depth_um", "Cap depth"),
            ("z_cut_um", "z cut depth"),
            ("cap_curved_surface_area_um2", "Cap curved area"),
            ("cap_projected_area_um2", "Cap projected area"),
            ("sanity_check_x_rmse_um", "Sanity x RMSE"),
            ("sanity_check_y_rmse_um", "Sanity y RMSE"),
        ]
        for key, label in items:
            row = QFrame()
            row.setObjectName("MetricRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 4, 8, 4)
            name = QLabel(label)
            value = QLabel("-")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setObjectName("MetricValue")
            row_layout.addWidget(name)
            row_layout.addWidget(value)
            layout.addWidget(row)
            self.labels[key] = value

        self.message = QLabel("검출 메시지가 여기에 표시됩니다.")
        self.message.setWordWrap(True)
        self.message.setObjectName("MessageBox")
        layout.addWidget(self.message)
        layout.addStretch(1)

    def clear(self) -> None:
        for label in self.labels.values():
            label.setText("-")
        self.message.setText("검출 메시지가 여기에 표시됩니다.")

    def set_summary(self, summary: dict, message: str) -> None:
        for key, label in self.labels.items():
            value = summary.get(key)
            if value is None or not np.isfinite(float(value)):
                label.setText("n/a")
            elif "area" in key:
                label.setText(f"{float(value):.4f} um²")
            elif key == "cap_depth_um" or key.endswith("_um") or "rmse" in key.lower():
                label.setText(f"{float(value):.4f} um")
            else:
                label.setText(f"{float(value):.4f}")
        self.message.setText(message)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FIB-SEM 2-Section Dome Cap Surface Area")
        self.resize(1540, 980)
        self.horizontal_image: Optional[np.ndarray] = None
        self.vertical_image: Optional[np.ndarray] = None
        self.state: Optional[AnalysisState] = None
        self.single_images: list[SingleImageItem] = []
        self.single_rois: list[Optional[tuple[int, int, int, int]]] = []
        self.single_results: list[SingleProfileRunResult] = []
        self.single_current_index = 0
        self.single_state: Optional[SingleBatchState] = None
        self.hhs_images: list[SingleImageItem] = []
        self.hhs_rois: list[Optional[tuple[int, int, int, int]]] = []
        self.hhs_results: list[HHSRunResult] = []
        self.hhs_current_index = 0
        self.hhs_state: Optional[HHSBatchState] = None

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        self.setCentralWidget(root)

        mode_bar = QHBoxLayout()
        mode_bar.setSpacing(8)
        self.two_section_mode_btn = QPushButton("2-Section 3D Cap Area")
        self.single_profile_mode_btn = QPushButton("Single Section CD-Depth")
        self.hhs_mode_btn = QPushButton("HHS")
        self.two_section_mode_btn.setCheckable(True)
        self.single_profile_mode_btn.setCheckable(True)
        self.hhs_mode_btn.setCheckable(True)
        self.two_section_mode_btn.setChecked(True)
        self.two_section_mode_btn.clicked.connect(lambda: self.switch_mode(0))
        self.single_profile_mode_btn.clicked.connect(lambda: self.switch_mode(1))
        self.hhs_mode_btn.clicked.connect(lambda: self.switch_mode(2))
        mode_bar.addWidget(self.two_section_mode_btn)
        mode_bar.addWidget(self.single_profile_mode_btn)
        mode_bar.addWidget(self.hhs_mode_btn)
        mode_bar.addStretch(1)
        root_layout.addLayout(mode_bar)

        self.mode_stack = QStackedWidget()
        root_layout.addWidget(self.mode_stack)
        self.mode_stack.addWidget(self._build_two_section_page())
        self.mode_stack.addWidget(self._build_single_profile_page())
        self.mode_stack.addWidget(self._build_hhs_page())

    def switch_mode(self, index: int) -> None:
        self.mode_stack.setCurrentIndex(index)
        self.two_section_mode_btn.setChecked(index == 0)
        self.single_profile_mode_btn.setChecked(index == 1)
        self.hhs_mode_btn.setChecked(index == 2)
        if index == 0:
            self.setWindowTitle("FIB-SEM 2-Section Dome Cap Surface Area")
        elif index == 1:
            self.setWindowTitle("FIB-SEM Single Section CD-Depth Profile")
        else:
            self.setWindowTitle("FIB-SEM Hill-on-Hill Score")

    def _build_two_section_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        main_splitter = QSplitter(Qt.Vertical)
        page_layout.addWidget(main_splitter)

        top_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(top_splitter)

        top_splitter.addWidget(self._build_left_panel())
        top_splitter.addWidget(self._build_center_panel())
        self.result_panel = ResultPanel()
        top_splitter.addWidget(self.result_panel)
        top_splitter.setSizes([300, 900, 330])

        self.tabs = QTabWidget()
        self.profile_canvas = _make_canvas(plot_profiles_empty())
        self.surface_canvas = _make_canvas(plot_empty_3d("3D surface"))
        self.cap_canvas = _make_canvas(plot_empty_3d("Cap highlighted surface"))
        self.topview_canvas = _make_canvas(plot_empty_2d("Cap top-view footprint"))
        self.tabs.addTab(self.profile_canvas, "Profiles")
        self.tabs.addTab(self.surface_canvas, "3D Surface")
        self.tabs.addTab(self.cap_canvas, "Cap 3D")
        self.tabs.addTab(self.topview_canvas, "Cap Footprint")

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 2)
        toolbar.addStretch(1)
        self.reset_3d_view_btn = QPushButton("3D 뷰 초기화")
        self.reset_3d_view_btn.setObjectName("CompactButton")
        self.reset_3d_view_btn.setEnabled(False)
        self.reset_3d_view_btn.clicked.connect(self.reset_3d_views)
        toolbar.addWidget(self.reset_3d_view_btn)
        plot_layout.addLayout(toolbar)
        plot_layout.addWidget(self.tabs)
        main_splitter.addWidget(plot_panel)
        main_splitter.setSizes([470, 510])
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 5)
        return page

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)

        title = QLabel("입력 및 파라미터")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.load_horizontal_btn = QPushButton("가로 단면 이미지 불러오기")
        self.load_vertical_btn = QPushButton("세로 단면 이미지 불러오기")
        self.synthetic_btn = QPushButton("Synthetic 샘플 생성")
        self.load_horizontal_btn.clicked.connect(self.load_horizontal_image)
        self.load_vertical_btn.clicked.connect(self.load_vertical_image)
        self.synthetic_btn.clicked.connect(self.load_synthetic_images)
        layout.addWidget(self.load_horizontal_btn)
        layout.addWidget(self.load_vertical_btn)
        layout.addWidget(self.synthetic_btn)

        form = QGridLayout()
        form.setVerticalSpacing(12)

        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.000001, 1000.0)
        self.pixel_size_spin.setDecimals(6)
        self.pixel_size_spin.setValue(0.01)
        self.pixel_size_spin.setSuffix(" um/px")
        _configure_spinbox(self.pixel_size_spin, 0.001)

        self.cap_depth_spin = QDoubleSpinBox()
        self.cap_depth_spin.setRange(0.000001, 100000.0)
        self.cap_depth_spin.setDecimals(4)
        self.cap_depth_spin.setValue(1.0)
        self.cap_depth_spin.setSuffix(" um")
        _configure_spinbox(self.cap_depth_spin, 0.1)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        _configure_slider(self.threshold_slider, 1, 5)
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(0, 10)
        self.smoothing_slider.setValue(2)
        _configure_slider(self.smoothing_slider, 1, 2)
        self.morph_slider = QSlider(Qt.Horizontal)
        self.morph_slider.setRange(0, 10)
        self.morph_slider.setValue(2)
        _configure_slider(self.morph_slider, 1, 2)

        self.edge_mode_combo = QComboBox()
        self.edge_mode_combo.addItem("Outer coating edge", "outer")
        self.edge_mode_combo.addItem("Inner gradient edge", "inner_gradient")

        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(30, 1200)
        self.grid_spin.setValue(400)
        _configure_spinbox(self.grid_spin, 10)

        rows = [
            ("pixel_size_um", self.pixel_size_spin),
            ("cap_depth_um", self.cap_depth_spin),
            ("threshold_sensitivity", self.threshold_slider),
            ("smoothing_strength", self.smoothing_slider),
            ("morph_strength", self.morph_slider),
            ("edge_mode", self.edge_mode_combo),
            ("grid_resolution", self.grid_spin),
        ]
        for row, (label, widget) in enumerate(rows):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(widget, row, 1)
        layout.addLayout(form)

        self.analyze_btn = QPushButton("분석 실행")
        self.save_btn = QPushButton("결과 저장")
        self.save_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.save_btn.clicked.connect(self.save_results)
        layout.addWidget(self.analyze_btn)
        layout.addWidget(self.save_btn)
        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        title = QLabel("이미지 및 ROI")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        viewers = QSplitter(Qt.Vertical)
        h_box = QFrame()
        h_layout = QVBoxLayout(h_box)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(QLabel("가로 단면 이미지"))
        self.horizontal_view = ROIImageView("Horizontal section")
        h_layout.addWidget(self.horizontal_view)

        v_box = QFrame()
        v_layout = QVBoxLayout(v_box)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.addWidget(QLabel("세로 단면 이미지"))
        self.vertical_view = ROIImageView("Vertical section")
        v_layout.addWidget(self.vertical_view)

        viewers.addWidget(h_box)
        viewers.addWidget(v_box)
        viewers.setSizes([300, 300])
        layout.addWidget(viewers)
        return panel

    def _build_single_profile_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        main_splitter = QSplitter(Qt.Vertical)
        page_layout.addWidget(main_splitter)

        top_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(top_splitter)
        top_splitter.addWidget(self._build_single_left_panel())
        top_splitter.addWidget(self._build_single_center_panel())
        top_splitter.addWidget(self._build_single_result_panel())
        top_splitter.setSizes([310, 760, 460])

        self.single_profile_canvas = _make_canvas(plot_empty_2d("Single-section CD-depth profile"))
        main_splitter.addWidget(self.single_profile_canvas)
        main_splitter.setSizes([650, 330])
        return page

    def _build_single_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        title = QLabel("단일 단면 입력")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.single_load_btn = QPushButton("이미지 여러장 불러오기")
        self.single_synthetic_btn = QPushButton("Synthetic batch 샘플")
        self.single_load_btn.clicked.connect(self.load_single_image)
        self.single_synthetic_btn.clicked.connect(self.load_single_synthetic_image)
        layout.addWidget(self.single_load_btn)
        layout.addWidget(self.single_synthetic_btn)

        form = QGridLayout()
        form.setVerticalSpacing(12)

        self.single_pixel_size_spin = QDoubleSpinBox()
        self.single_pixel_size_spin.setRange(0.000001, 1000.0)
        self.single_pixel_size_spin.setDecimals(6)
        self.single_pixel_size_spin.setValue(0.01)
        self.single_pixel_size_spin.setSuffix(" um/px")
        _configure_spinbox(self.single_pixel_size_spin, 0.001)

        self.single_max_depth_spin = QDoubleSpinBox()
        self.single_max_depth_spin.setRange(0.000001, 100000.0)
        self.single_max_depth_spin.setDecimals(4)
        self.single_max_depth_spin.setValue(1.0)
        self.single_max_depth_spin.setSuffix(" um")
        _configure_spinbox(self.single_max_depth_spin, 0.1)

        self.single_depth_step_spin = QDoubleSpinBox()
        self.single_depth_step_spin.setRange(0.000001, 100000.0)
        self.single_depth_step_spin.setDecimals(4)
        self.single_depth_step_spin.setValue(0.1)
        self.single_depth_step_spin.setSuffix(" um")
        _configure_spinbox(self.single_depth_step_spin, 0.01)

        self.single_taper_step_spin = QDoubleSpinBox()
        self.single_taper_step_spin.setRange(0.000001, 100000.0)
        self.single_taper_step_spin.setDecimals(4)
        self.single_taper_step_spin.setValue(0.1)
        self.single_taper_step_spin.setSuffix(" um")
        _configure_spinbox(self.single_taper_step_spin, 0.01)

        self.single_threshold_slider = QSlider(Qt.Horizontal)
        self.single_threshold_slider.setRange(0, 100)
        self.single_threshold_slider.setValue(50)
        _configure_slider(self.single_threshold_slider, 1, 5)
        self.single_smoothing_slider = QSlider(Qt.Horizontal)
        self.single_smoothing_slider.setRange(0, 10)
        self.single_smoothing_slider.setValue(2)
        _configure_slider(self.single_smoothing_slider, 1, 2)
        self.single_morph_slider = QSlider(Qt.Horizontal)
        self.single_morph_slider.setRange(0, 10)
        self.single_morph_slider.setValue(2)
        _configure_slider(self.single_morph_slider, 1, 2)

        self.single_edge_mode_combo = QComboBox()
        self.single_edge_mode_combo.addItem("Outer coating edge", "outer")
        self.single_edge_mode_combo.addItem("Inner gradient edge", "inner_gradient")

        rows = [
            ("pixel_size_um", self.single_pixel_size_spin),
            ("max_depth_um", self.single_max_depth_spin),
            ("depth_step_um", self.single_depth_step_spin),
            ("taper_step_um", self.single_taper_step_spin),
            ("threshold_sensitivity", self.single_threshold_slider),
            ("smoothing_strength", self.single_smoothing_slider),
            ("morph_strength", self.single_morph_slider),
            ("edge_mode", self.single_edge_mode_combo),
        ]
        for row, (label, widget) in enumerate(rows):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(widget, row, 1)
        layout.addLayout(form)

        self.single_analyze_btn = QPushButton("프로파일 분석 실행")
        self.single_save_btn = QPushButton("Batch CD-depth 결과 저장")
        self.single_save_btn.setEnabled(False)
        self.single_analyze_btn.clicked.connect(self.run_single_profile_analysis)
        self.single_save_btn.clicked.connect(self.save_single_profile_results)
        layout.addWidget(self.single_analyze_btn)
        layout.addWidget(self.single_save_btn)
        layout.addStretch(1)
        return panel

    def _build_single_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        title = QLabel("공유 ROI 미리보기")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        nav_layout = QHBoxLayout()
        self.single_prev_btn = QPushButton("Previous")
        self.single_next_btn = QPushButton("Next")
        self.single_image_index_label = QLabel("-")
        self.single_image_index_label.setAlignment(Qt.AlignCenter)
        self.single_prev_btn.clicked.connect(self.show_previous_single_image)
        self.single_next_btn.clicked.connect(self.show_next_single_image)
        nav_layout.addWidget(self.single_prev_btn)
        nav_layout.addWidget(self.single_image_index_label, 1)
        nav_layout.addWidget(self.single_next_btn)
        layout.addLayout(nav_layout)

        self.single_view = ROIImageView("Single section")
        self.single_view.roi_changed.connect(self._store_current_single_roi)
        layout.addWidget(self.single_view)
        self._update_single_navigation()
        return panel

    def _build_single_result_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        title = QLabel("CD-depth 결과")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.single_result_labels: dict[str, QLabel] = {}
        items = [
            ("image_count", "Images"),
            ("success_count", "Success"),
            ("fail_count", "Failed"),
            ("depth_step_um", "Depth step"),
            ("taper_step_um", "Taper step"),
            ("roi", "Current ROI"),
        ]
        for key, label in items:
            row = QFrame()
            row.setObjectName("MetricRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 7, 10, 7)
            name = QLabel(label)
            value = QLabel("-")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setObjectName("MetricValue")
            row_layout.addWidget(name)
            row_layout.addWidget(value)
            layout.addWidget(row)
            self.single_result_labels[key] = value

        self.single_table = QTableWidget(0, 7)
        self.single_table.setHorizontalHeaderLabels(["image", "status", "CD_um", "H_um", "depth_count", "left_taper_deg", "right_taper_deg"])
        self.single_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.single_table.verticalHeader().setVisible(False)
        self.single_table.setAlternatingRowColors(True)
        layout.addWidget(self.single_table)

        self.single_message = QLabel("이미지를 여러 장 불러온 뒤 첫 이미지에서 ROI를 드래그하세요.")
        self.single_message.setWordWrap(True)
        self.single_message.setObjectName("MessageBox")
        layout.addWidget(self.single_message)
        return panel

    def _build_hhs_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        main_splitter = QSplitter(Qt.Vertical)
        page_layout.addWidget(main_splitter)

        top_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(top_splitter)
        top_splitter.addWidget(self._build_hhs_left_panel())
        top_splitter.addWidget(self._build_hhs_center_panel())
        top_splitter.addWidget(self._build_hhs_result_panel())
        top_splitter.setSizes([330, 760, 460])

        self.hhs_canvas = _make_canvas(plot_empty_2d("Hill-on-Hill Score"))
        main_splitter.addWidget(self.hhs_canvas)
        main_splitter.setSizes([650, 330])
        return page

    def _build_hhs_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        title = QLabel("HHS 입력")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.hhs_load_btn = QPushButton("이미지 여러장 불러오기")
        self.hhs_synthetic_btn = QPushButton("Synthetic HHS 샘플")
        self.hhs_load_btn.clicked.connect(self.load_hhs_images)
        self.hhs_synthetic_btn.clicked.connect(self.load_hhs_synthetic_images)
        layout.addWidget(self.hhs_load_btn)
        layout.addWidget(self.hhs_synthetic_btn)

        form = QGridLayout()
        form.setVerticalSpacing(12)

        self.hhs_pixel_size_spin = QDoubleSpinBox()
        self.hhs_pixel_size_spin.setRange(0.000001, 1000.0)
        self.hhs_pixel_size_spin.setDecimals(6)
        self.hhs_pixel_size_spin.setValue(0.01)
        self.hhs_pixel_size_spin.setSuffix(" um/px")
        _configure_spinbox(self.hhs_pixel_size_spin, 0.001)

        self.hhs_light_sigma_spin = QDoubleSpinBox()
        self.hhs_light_sigma_spin.setRange(0.000001, 1.0)
        self.hhs_light_sigma_spin.setDecimals(4)
        self.hhs_light_sigma_spin.setValue(DEFAULT_HHS_SETTINGS["light_smooth_sigma"])
        _configure_spinbox(self.hhs_light_sigma_spin, 0.005)

        self.hhs_baseline_sigma_spin = QDoubleSpinBox()
        self.hhs_baseline_sigma_spin.setRange(0.000001, 1.0)
        self.hhs_baseline_sigma_spin.setDecimals(4)
        self.hhs_baseline_sigma_spin.setValue(DEFAULT_HHS_SETTINGS["baseline_smooth_sigma"])
        _configure_spinbox(self.hhs_baseline_sigma_spin, 0.01)

        self.hhs_center_width_spin = QDoubleSpinBox()
        self.hhs_center_width_spin.setRange(0.000001, 1.0)
        self.hhs_center_width_spin.setDecimals(4)
        self.hhs_center_width_spin.setValue(DEFAULT_HHS_SETTINGS["center_width"])
        _configure_spinbox(self.hhs_center_width_spin, 0.01)

        self.hhs_threshold_slider = QSlider(Qt.Horizontal)
        self.hhs_threshold_slider.setRange(0, 100)
        self.hhs_threshold_slider.setValue(50)
        _configure_slider(self.hhs_threshold_slider, 1, 5)
        self.hhs_smoothing_slider = QSlider(Qt.Horizontal)
        self.hhs_smoothing_slider.setRange(0, 10)
        self.hhs_smoothing_slider.setValue(2)
        _configure_slider(self.hhs_smoothing_slider, 1, 2)
        self.hhs_morph_slider = QSlider(Qt.Horizontal)
        self.hhs_morph_slider.setRange(0, 10)
        self.hhs_morph_slider.setValue(2)
        _configure_slider(self.hhs_morph_slider, 1, 2)

        self.hhs_edge_mode_combo = QComboBox()
        self.hhs_edge_mode_combo.addItem("Outer coating edge", "outer")
        self.hhs_edge_mode_combo.addItem("Inner gradient edge", "inner_gradient")

        rows = [
            ("pixel_size_um", self.hhs_pixel_size_spin),
            ("light_smooth_sigma", self.hhs_light_sigma_spin),
            ("baseline_smooth_sigma", self.hhs_baseline_sigma_spin),
            ("center_width", self.hhs_center_width_spin),
            ("threshold_sensitivity", self.hhs_threshold_slider),
            ("smoothing_strength", self.hhs_smoothing_slider),
            ("morph_strength", self.hhs_morph_slider),
            ("edge_mode", self.hhs_edge_mode_combo),
        ]
        for row, (label, widget) in enumerate(rows):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(widget, row, 1)
        layout.addLayout(form)

        self.hhs_analyze_btn = QPushButton("HHS 분석 실행")
        self.hhs_save_btn = QPushButton("HHS 결과 저장")
        self.hhs_save_btn.setEnabled(False)
        self.hhs_analyze_btn.clicked.connect(self.run_hhs_analysis)
        self.hhs_save_btn.clicked.connect(self.save_hhs_results)
        layout.addWidget(self.hhs_analyze_btn)
        layout.addWidget(self.hhs_save_btn)
        layout.addStretch(1)
        return panel

    def _build_hhs_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        title = QLabel("HHS ROI 미리보기")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        nav_layout = QHBoxLayout()
        self.hhs_prev_btn = QPushButton("Previous")
        self.hhs_next_btn = QPushButton("Next")
        self.hhs_image_index_label = QLabel("-")
        self.hhs_image_index_label.setAlignment(Qt.AlignCenter)
        self.hhs_prev_btn.clicked.connect(self.show_previous_hhs_image)
        self.hhs_next_btn.clicked.connect(self.show_next_hhs_image)
        nav_layout.addWidget(self.hhs_prev_btn)
        nav_layout.addWidget(self.hhs_image_index_label, 1)
        nav_layout.addWidget(self.hhs_next_btn)
        layout.addLayout(nav_layout)

        self.hhs_view = ROIImageView("HHS section")
        self.hhs_view.roi_changed.connect(self._store_current_hhs_roi)
        layout.addWidget(self.hhs_view)
        self._update_hhs_navigation()
        return panel

    def _build_hhs_result_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        title = QLabel("HHS 결과")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.hhs_result_labels: dict[str, QLabel] = {}
        items = [
            ("image_count", "Images"),
            ("success_count", "Success"),
            ("fail_count", "Failed"),
            ("hhs", "HHS"),
            ("bump_area", "bump_area"),
            ("total_area", "total_area"),
            ("roi", "Current ROI"),
        ]
        for key, label in items:
            row = QFrame()
            row.setObjectName("MetricRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 7, 10, 7)
            name = QLabel(label)
            value = QLabel("-")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setObjectName("MetricValue")
            row_layout.addWidget(name)
            row_layout.addWidget(value)
            layout.addWidget(row)
            self.hhs_result_labels[key] = value

        self.hhs_table = QTableWidget(0, 5)
        self.hhs_table.setHorizontalHeaderLabels(["sample_id", "status", "hhs", "bump_area", "total_area"])
        self.hhs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hhs_table.verticalHeader().setVisible(False)
        self.hhs_table.setAlternatingRowColors(True)
        layout.addWidget(self.hhs_table)

        self.hhs_message = QLabel("이미지를 여러 장 불러온 뒤 ROI를 드래그하세요.")
        self.hhs_message.setWordWrap(True)
        self.hhs_message.setObjectName("MessageBox")
        layout.addWidget(self.hhs_message)
        return panel

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#0F172A; color:#E5E7EB; font-size:13px; }
            QFrame#Card { background:#111827; border:1px solid #273449; border-radius:8px; }
            QLabel#PanelTitle { font-size:16px; font-weight:700; padding:4px 2px 10px 2px; color:#F8FAFC; }
            QPushButton { background:#1E293B; border:1px solid #334155; border-radius:7px; padding:9px 11px; color:#E5E7EB; }
            QPushButton:hover { background:#273449; border-color:#38BDF8; }
            QPushButton:pressed { background:#0EA5E9; color:#06111F; }
            QPushButton:checked { background:#0EA5E9; color:#06111F; border-color:#7DD3FC; font-weight:700; }
            QPushButton:disabled { color:#64748B; background:#111827; border-color:#1F2937; }
            QPushButton#CompactButton { padding:5px 9px; }
            QComboBox { background:#0B1120; border:1px solid #334155; border-radius:6px; padding:6px 24px 6px 6px; color:#E5E7EB; }
            QComboBox::drop-down { border-left:1px solid #334155; width:22px; background:#172033; }
            QComboBox QAbstractItemView { background:#0B1120; color:#E5E7EB; selection-background-color:#1E293B; border:1px solid #334155; }
            QDoubleSpinBox, QSpinBox { background:#0B1120; border:1px solid #334155; border-radius:6px; padding:6px 24px 6px 6px; color:#E5E7EB; }
            QDoubleSpinBox::up-button, QSpinBox::up-button { subcontrol-origin:border; subcontrol-position:top right; width:20px; border-left:1px solid #334155; border-bottom:1px solid #273449; border-top-right-radius:6px; background:#172033; }
            QDoubleSpinBox::down-button, QSpinBox::down-button { subcontrol-origin:border; subcontrol-position:bottom right; width:20px; border-left:1px solid #334155; border-bottom-right-radius:6px; background:#172033; }
            QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover { background:#273449; }
            QDoubleSpinBox::up-arrow, QSpinBox::up-arrow { width:0; height:0; border-left:4px solid transparent; border-right:4px solid transparent; border-bottom:5px solid #CBD5E1; }
            QDoubleSpinBox::down-arrow, QSpinBox::down-arrow { width:0; height:0; border-left:4px solid transparent; border-right:4px solid transparent; border-top:5px solid #CBD5E1; }
            QSlider::groove:horizontal { height:5px; background:#273449; border-radius:2px; }
            QSlider::handle:horizontal { width:16px; height:16px; margin:-6px 0; border-radius:8px; background:#38BDF8; }
            QTabWidget::pane { border:1px solid #273449; background:#111827; border-radius:8px; }
            QTabBar::tab { background:#172033; color:#CBD5E1; padding:8px 14px; border-top-left-radius:6px; border-top-right-radius:6px; }
            QTabBar::tab:selected { background:#1E293B; color:#F8FAFC; border-bottom:2px solid #38BDF8; }
            QFrame#MetricRow { background:#172033; border:1px solid #243244; border-radius:6px; }
            QLabel#MetricValue { color:#BAE6FD; font-weight:700; }
            QLabel#MessageBox { background:#0B1120; border:1px solid #334155; border-radius:7px; padding:10px; color:#CBD5E1; }
            QTableWidget { background:#0B1120; alternate-background-color:#111827; border:1px solid #334155; border-radius:7px; gridline-color:#273449; color:#E5E7EB; }
            QHeaderView::section { background:#172033; color:#CBD5E1; border:1px solid #273449; padding:6px; }
            """
        )

    def load_horizontal_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "가로 단면 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*.*)")
        if not path:
            return
        try:
            self.horizontal_image = to_grayscale(load_image(path))
            self.horizontal_view.set_image(self.horizontal_image)
            self.state = None
            self.save_btn.setEnabled(False)
            self._set_3d_reset_enabled(False)
        except Exception as exc:
            QMessageBox.warning(self, "이미지 로드 실패", str(exc))

    def load_vertical_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "세로 단면 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*.*)")
        if not path:
            return
        try:
            self.vertical_image = to_grayscale(load_image(path))
            self.vertical_view.set_image(self.vertical_image)
            self.state = None
            self.save_btn.setEnabled(False)
            self._set_3d_reset_enabled(False)
        except Exception as exc:
            QMessageBox.warning(self, "이미지 로드 실패", str(exc))

    def load_synthetic_images(self) -> None:
        self.horizontal_image, self.vertical_image = generate_synthetic_pair()
        self.horizontal_view.set_image(self.horizontal_image)
        self.vertical_view.set_image(self.vertical_image)
        self.horizontal_view.set_roi(_default_synthetic_roi(self.horizontal_image))
        self.vertical_view.set_roi(_default_synthetic_roi(self.vertical_image))
        self.result_panel.message.setText("Synthetic 이미지가 생성되었습니다. ROI는 기본값으로 지정되어 있으며 바로 분석할 수 있습니다.")
        self.state = None
        self.save_btn.setEnabled(False)
        self._set_3d_reset_enabled(False)

    def load_single_image(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "단일 모드 batch 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*.*)")
        if not paths:
            return
        try:
            self.single_images = [
                SingleImageItem(name=Path(path).name, image=to_grayscale(load_image(path)))
                for path in paths
            ]
            self._reset_single_image_state()
            self._show_single_image(0)
            self.single_state = None
            self.single_save_btn.setEnabled(False)
            self._clear_single_results()
            self._populate_single_loaded_table()
            self.single_message.setText(f"{len(self.single_images)}개 이미지를 불러왔습니다. 첫 이미지에서 공유 ROI를 드래그하세요.")
        except Exception as exc:
            QMessageBox.warning(self, "이미지 로드 실패", str(exc))

    def load_single_synthetic_image(self) -> None:
        specs = [(410, 168, 119), (360, 156, 131), (455, 174, 149)]
        self.single_images = [
            SingleImageItem(
                name=f"synthetic_{idx + 1}.png",
                image=generate_synthetic_mound_image(cd_px=cd, mound_height_px=height, seed=seed),
            )
            for idx, (cd, height, seed) in enumerate(specs)
        ]
        self._reset_single_image_state()
        for index, item in enumerate(self.single_images):
            roi = _default_synthetic_roi(item.image)
            self.single_rois[index] = roi
            self.single_results[index].roi = roi
        self._show_single_image(0)
        self.single_state = None
        self.single_save_btn.setEnabled(False)
        self._clear_single_results()
        self._populate_single_loaded_table()
        self.single_message.setText("Synthetic batch 이미지가 생성되었습니다. 첫 이미지 ROI가 전체 이미지에 공유됩니다.")

    def load_hhs_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "HHS 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*.*)")
        if not paths:
            return
        try:
            self.hhs_images = [
                SingleImageItem(name=Path(path).name, image=to_grayscale(load_image(path)))
                for path in paths
            ]
            self._reset_hhs_image_state()
            self._show_hhs_image(0)
            self.hhs_state = None
            self.hhs_save_btn.setEnabled(False)
            self._clear_hhs_results()
            self._populate_hhs_loaded_table()
            self.hhs_message.setText(f"{len(self.hhs_images)}개 이미지를 불러왔습니다. ROI를 드래그하세요.")
        except Exception as exc:
            QMessageBox.warning(self, "이미지 로드 실패", str(exc))

    def load_hhs_synthetic_images(self) -> None:
        specs = [(410, 168, 119), (410, 168, 120), (410, 168, 121)]
        self.hhs_images = []
        for idx, (cd, height, seed) in enumerate(specs):
            image = generate_synthetic_mound_image(cd_px=cd, mound_height_px=height, seed=seed)
            if idx:
                h, w = image.shape[:2]
                xs = np.arange(w, dtype=np.float32)
                center = w * 0.5
                bump = 52.0 * np.exp(-0.5 * ((xs - center) / (w * 0.045)) ** 2)
                for x, delta in enumerate(bump.astype(np.int32)):
                    if delta <= 0:
                        continue
                    col = image[:, x]
                    dark = np.where(col < 115)[0]
                    if dark.size:
                        top = int(dark.min())
                        y0 = max(0, top - delta)
                        image[y0:top, x] = np.minimum(image[y0:top, x], 55)
            self.hhs_images.append(SingleImageItem(name=f"hhs_synthetic_{idx + 1}.png", image=image))
        self._reset_hhs_image_state()
        for index, item in enumerate(self.hhs_images):
            roi = _default_synthetic_roi(item.image)
            self.hhs_rois[index] = roi
            self.hhs_results[index].roi = roi
        self._show_hhs_image(0)
        self.hhs_state = None
        self.hhs_save_btn.setEnabled(False)
        self._clear_hhs_results()
        self._populate_hhs_loaded_table()
        self.hhs_message.setText("Synthetic HHS 이미지가 생성되었습니다. ROI는 기본값입니다.")

    def _params(self) -> dict:
        return {
            "pixel_size_um": self.pixel_size_spin.value(),
            "cap_depth_um": self.cap_depth_spin.value(),
            "threshold_sensitivity": self.threshold_slider.value(),
            "smoothing_strength": self.smoothing_slider.value(),
            "morph_strength": self.morph_slider.value(),
            "edge_mode": self.edge_mode_combo.currentData(),
            "grid_resolution": self.grid_spin.value(),
        }

    def _single_params(self) -> dict:
        return {
            "pixel_size_um": self.single_pixel_size_spin.value(),
            "max_depth_um": self.single_max_depth_spin.value(),
            "depth_step_um": self.single_depth_step_spin.value(),
            "taper_step_um": self.single_taper_step_spin.value(),
            "threshold_sensitivity": self.single_threshold_slider.value(),
            "smoothing_strength": self.single_smoothing_slider.value(),
            "morph_strength": self.single_morph_slider.value(),
            "edge_mode": self.single_edge_mode_combo.currentData(),
        }

    def _hhs_params(self) -> dict:
        return {
            "pixel_size_um": self.hhs_pixel_size_spin.value(),
            "light_smooth_sigma": self.hhs_light_sigma_spin.value(),
            "baseline_smooth_sigma": self.hhs_baseline_sigma_spin.value(),
            "center_width": self.hhs_center_width_spin.value(),
            "threshold_sensitivity": self.hhs_threshold_slider.value(),
            "smoothing_strength": self.hhs_smoothing_slider.value(),
            "morph_strength": self.hhs_morph_slider.value(),
            "edge_mode": self.hhs_edge_mode_combo.currentData(),
        }

    def _reset_single_image_state(self) -> None:
        self.single_current_index = 0
        self.single_rois = [None for _ in self.single_images]
        self.single_results = [SingleProfileRunResult(item=item) for item in self.single_images]
        self._update_single_navigation()

    def _reset_hhs_image_state(self) -> None:
        self.hhs_current_index = 0
        self.hhs_rois = [None for _ in self.hhs_images]
        self.hhs_results = [HHSRunResult(item=item) for item in self.hhs_images]
        self._update_hhs_navigation()

    def _store_current_single_roi(self, roi: tuple[int, int, int, int]) -> None:
        if 0 <= self.single_current_index < len(self.single_rois):
            self.single_rois[self.single_current_index] = roi
            self.single_results[self.single_current_index].roi = roi
            if self.single_state is not None:
                self.single_state.results = self.single_results
            self.single_result_labels["roi"].setText(str(roi))

    def _store_current_hhs_roi(self, roi: tuple[int, int, int, int]) -> None:
        if 0 <= self.hhs_current_index < len(self.hhs_rois):
            self.hhs_rois[self.hhs_current_index] = roi
            self.hhs_results[self.hhs_current_index].roi = roi
            if self.hhs_state is not None:
                self.hhs_state.results = self.hhs_results
            self.hhs_result_labels["roi"].setText(str(roi))

    def _show_single_image(self, index: int) -> None:
        if not (0 <= index < len(self.single_images)):
            self._update_single_navigation()
            return
        self.single_current_index = index
        result = self.single_results[index] if index < len(self.single_results) else None
        if result is not None and result.profile is not None:
            self.single_view.set_overlay(plot_detection_overlay(result.item.image, result.profile))
        else:
            self.single_view.set_image(self.single_images[index].image)
        roi = self.single_rois[index] if index < len(self.single_rois) else None
        if roi is not None:
            self.single_view.set_roi(roi)
        self._update_single_navigation()
        if hasattr(self, "single_table") and index < self.single_table.rowCount():
            self.single_table.selectRow(index)

    def _update_single_navigation(self) -> None:
        count = len(self.single_images)
        has_images = count > 0
        if hasattr(self, "single_prev_btn"):
            self.single_prev_btn.setEnabled(has_images and self.single_current_index > 0)
            self.single_next_btn.setEnabled(has_images and self.single_current_index < count - 1)
            if has_images:
                name = self.single_images[self.single_current_index].name
                self.single_image_index_label.setText(f"{self.single_current_index + 1}/{count} - {name}")
            else:
                self.single_image_index_label.setText("-")

    def _show_hhs_image(self, index: int) -> None:
        if not (0 <= index < len(self.hhs_images)):
            self._update_hhs_navigation()
            return
        self.hhs_current_index = index
        result = self.hhs_results[index] if index < len(self.hhs_results) else None
        if result is not None and result.profile is not None:
            self.hhs_view.set_overlay(plot_detection_overlay(result.item.image, result.profile))
            if result.hhs_result is not None:
                self._replace_hhs_canvas(plot_hhs_profile(result.hhs_result))
        else:
            self.hhs_view.set_image(self.hhs_images[index].image)
        roi = self.hhs_rois[index] if index < len(self.hhs_rois) else None
        if roi is not None:
            self.hhs_view.set_roi(roi)
        self._update_hhs_navigation()
        if hasattr(self, "hhs_table") and index < self.hhs_table.rowCount():
            self.hhs_table.selectRow(index)

    def _update_hhs_navigation(self) -> None:
        count = len(self.hhs_images)
        has_images = count > 0
        if hasattr(self, "hhs_prev_btn"):
            self.hhs_prev_btn.setEnabled(has_images and self.hhs_current_index > 0)
            self.hhs_next_btn.setEnabled(has_images and self.hhs_current_index < count - 1)
            if has_images:
                name = self.hhs_images[self.hhs_current_index].name
                self.hhs_image_index_label.setText(f"{self.hhs_current_index + 1}/{count} - {name}")
            else:
                self.hhs_image_index_label.setText("-")

    def show_previous_single_image(self) -> None:
        self._show_single_image(self.single_current_index - 1)

    def show_next_single_image(self) -> None:
        self._show_single_image(self.single_current_index + 1)

    def show_previous_hhs_image(self) -> None:
        self._show_hhs_image(self.hhs_current_index - 1)

    def show_next_hhs_image(self) -> None:
        self._show_hhs_image(self.hhs_current_index + 1)

    def run_analysis(self) -> None:
        self.result_panel.clear()
        if self.horizontal_image is None or self.vertical_image is None:
            self._fail("가로/세로 단면 이미지를 모두 불러와야 합니다.")
            return
        if self.horizontal_view.roi is None or self.vertical_view.roi is None:
            self._fail("각 이미지에서 ROI를 먼저 지정해야 합니다.")
            return

        params = self._params()
        if params["pixel_size_um"] <= 0:
            self._fail("pixel_size_um은 0보다 커야 합니다.")
            return
        if params["cap_depth_um"] <= 0:
            self._fail("cap_depth_um은 0보다 커야 합니다.")
            return

        try:
            horizontal_profile = analyze_section(
                self.horizontal_image,
                self.horizontal_view.roi,
                pixel_size_um=params["pixel_size_um"],
                threshold_sensitivity=params["threshold_sensitivity"],
                smoothing_strength=params["smoothing_strength"],
                morph_strength=params["morph_strength"],
                axis_name="horizontal",
                coordinate_name="x_um",
                edge_mode=params["edge_mode"],
            )
            vertical_profile = analyze_section(
                self.vertical_image,
                self.vertical_view.roi,
                pixel_size_um=params["pixel_size_um"],
                threshold_sensitivity=params["threshold_sensitivity"],
                smoothing_strength=params["smoothing_strength"],
                morph_strength=params["morph_strength"],
                axis_name="vertical",
                coordinate_name="y_um",
                edge_mode=params["edge_mode"],
            )
            surface = build_profile_based_surface(horizontal_profile, vertical_profile, params["grid_resolution"])
            nan_ratio = float(np.count_nonzero(~np.isfinite(surface.Z)) / surface.Z.size)
            cap = compute_cap_metrics(surface.X, surface.Y, surface.Z, surface.valid_mask, params["cap_depth_um"])
        except (DetectionError, ValueError) as exc:
            self._fail(str(exc))
            return
        except Exception as exc:
            self._fail(f"분석 중 예기치 못한 오류가 발생했습니다: {exc}")
            return

        self.state = AnalysisState(horizontal_profile, vertical_profile, surface, cap, params)
        self.save_btn.setEnabled(True)
        self._set_3d_reset_enabled(True)

        self.horizontal_view.set_overlay(plot_detection_overlay(self.horizontal_image, horizontal_profile))
        self.vertical_view.set_overlay(plot_detection_overlay(self.vertical_image, vertical_profile))
        self._update_plots()

        summary = build_summary(horizontal_profile, vertical_profile, surface, cap, params)
        warnings = []
        rmse_limit = max(0.05, 0.08 * surface.H_surface_max_um)
        if surface.sanity_check_x_rmse_um > rmse_limit:
            warnings.append(f"x-section RMSE가 큽니다 ({surface.sanity_check_x_rmse_um:.4f} um).")
        if surface.sanity_check_y_rmse_um > rmse_limit:
            warnings.append(f"y-section RMSE가 큽니다 ({surface.sanity_check_y_rmse_um:.4f} um).")
        if nan_ratio > 0.45:
            warnings.append(f"surface grid NaN 비율이 높습니다 ({nan_ratio:.1%}).")
        if cap.cap_curved_surface_area_um2 <= 0:
            warnings.append("cap_depth가 너무 작거나 grid_resolution이 낮아 cap cell이 선택되지 않았습니다.")
        if not warnings:
            warnings.append("분석 완료. apex z=0, 아래 방향 positive depth 좌표로 cap area를 계산했습니다.")
        self.result_panel.set_summary(summary, "\n".join(warnings))

    def run_single_profile_analysis(self) -> None:
        self._clear_single_results()
        if not self.single_images:
            self._single_fail("단일 모드 이미지를 먼저 불러와야 합니다.")
            return
        if self.single_view.roi is None:
            self._single_fail("첫 이미지에서 공유 ROI를 먼저 드래그해야 합니다.")
            return

        params = self._single_params()
        if params["pixel_size_um"] <= 0:
            self._single_fail("pixel_size_um은 0보다 커야 합니다.")
            return
        if params["max_depth_um"] <= 0:
            self._single_fail("max_depth_um은 0보다 커야 합니다.")
            return
        if params["depth_step_um"] <= 0:
            self._single_fail("depth_step_um은 0보다 커야 합니다.")
            return
        if params["taper_step_um"] <= 0:
            self._single_fail("taper_step_um은 0보다 커야 합니다.")
            return

        shared_roi = self.single_view.roi
        results: list[SingleProfileRunResult] = []
        for item in self.single_images:
            try:
                profile = analyze_section(
                    item.image,
                    shared_roi,
                    pixel_size_um=params["pixel_size_um"],
                    threshold_sensitivity=params["threshold_sensitivity"],
                    smoothing_strength=params["smoothing_strength"],
                    morph_strength=params["morph_strength"],
                    axis_name="single",
                    coordinate_name="x_um",
                    edge_mode=params["edge_mode"],
                )
                cd_result = compute_cd_by_depth(profile, params["max_depth_um"], params["depth_step_um"])
                taper_result = compute_taper_by_offset(profile, params["taper_step_um"])
                results.append(SingleProfileRunResult(item=item, profile=profile, cd_result=cd_result, taper_result=taper_result))
            except (DetectionError, ValueError) as exc:
                results.append(SingleProfileRunResult(item=item, error=str(exc)))
            except Exception as exc:
                results.append(SingleProfileRunResult(item=item, error=f"예기치 못한 오류: {exc}"))

        success = [result for result in results if result.profile is not None and result.cd_result is not None]
        if not success:
            self.single_state = SingleBatchState(results=results, roi=shared_roi, params=params)
            self._update_single_results()
            self._single_fail("모든 이미지 분석에 실패했습니다. ROI와 threshold 조건을 확인하세요.")
            return

        self.single_state = SingleBatchState(results=results, roi=shared_roi, params=params)
        self.single_save_btn.setEnabled(True)
        first_success = success[0]
        self.single_view.set_overlay(plot_detection_overlay(first_success.item.image, first_success.profile))
        self._update_single_results()
        self._replace_single_profile_canvas(plot_single_batch_cd_depth(results))

        fail_count = len(results) - len(success)
        messages = [f"Batch 분석 완료. 총 {len(results)}개 중 {len(success)}개 성공, {fail_count}개 실패."]
        clipped_count = sum(
            1
            for result in success
            if result.cd_result.effective_max_depth_um < result.cd_result.requested_max_depth_um
        )
        if clipped_count:
            messages.append(f"{clipped_count}개 이미지는 요청 depth보다 profile height가 작아 유효 height까지만 계산했습니다.")
        self.single_message.setText("\n".join(messages))

    def run_single_profile_analysis(self) -> None:
        if not self.single_images:
            self._single_fail("단일 모드 이미지를 먼저 불러와야 합니다.")
            return
        if not (0 <= self.single_current_index < len(self.single_images)):
            self._single_fail("No current image is selected.")
            return
        current_roi = self.single_view.roi
        if current_roi is None:
            self._single_fail("Set ROI for the current image before analysis.")
            return

        params = self._single_params()
        if params["pixel_size_um"] <= 0:
            self._single_fail("pixel_size_um은 0보다 커야 합니다.")
            return
        if params["max_depth_um"] <= 0:
            self._single_fail("max_depth_um은 0보다 커야 합니다.")
            return
        if params["depth_step_um"] <= 0:
            self._single_fail("depth_step_um은 0보다 커야 합니다.")
            return
        if params["taper_step_um"] <= 0:
            self._single_fail("taper_step_um은 0보다 커야 합니다.")
            return

        item = self.single_images[self.single_current_index]
        self._store_current_single_roi(current_roi)
        try:
            profile = analyze_section(
                item.image,
                current_roi,
                pixel_size_um=params["pixel_size_um"],
                threshold_sensitivity=params["threshold_sensitivity"],
                smoothing_strength=params["smoothing_strength"],
                morph_strength=params["morph_strength"],
                axis_name="single",
                coordinate_name="x_um",
                edge_mode=params["edge_mode"],
            )
            cd_result = compute_cd_by_depth(profile, params["max_depth_um"], params["depth_step_um"])
            taper_result = compute_taper_by_offset(profile, params["taper_step_um"])
            self.single_results[self.single_current_index] = SingleProfileRunResult(
                item=item,
                roi=current_roi,
                profile=profile,
                cd_result=cd_result,
                taper_result=taper_result,
            )
            self.single_view.set_overlay(plot_detection_overlay(item.image, profile))
            self.single_view.set_roi(current_roi)
        except (DetectionError, ValueError) as exc:
            self.single_results[self.single_current_index] = SingleProfileRunResult(item=item, roi=current_roi, error=str(exc))
        except Exception as exc:
            self.single_results[self.single_current_index] = SingleProfileRunResult(item=item, roi=current_roi, error=f"Unexpected error: {exc}")

        success = [result for result in self.single_results if result.profile is not None and result.cd_result is not None]
        self.single_state = SingleBatchState(results=self.single_results, params=params)
        self.single_save_btn.setEnabled(bool(success))
        self._update_single_results()
        self._replace_single_profile_canvas(plot_single_batch_cd_depth(self.single_results))
        self._update_single_navigation()

        analyzed = [result for result in self.single_results if result.profile is not None or result.error]
        fail_count = len([result for result in analyzed if result.profile is None])
        messages = [f"Analysis updated. {len(success)} success, {fail_count} failed, {len(self.single_images) - len(analyzed)} not analyzed."]
        clipped_count = sum(
            1
            for result in success
            if result.cd_result.effective_max_depth_um < result.cd_result.requested_max_depth_um
        )
        if clipped_count:
            messages.append(f"{clipped_count} images were clipped to their effective profile height.")
        self.single_message.setText("\n".join(messages))

    def run_hhs_analysis(self) -> None:
        if not self.hhs_images:
            self._hhs_fail("HHS 이미지를 먼저 불러와야 합니다.")
            return
        current_roi = self.hhs_view.roi
        if current_roi is None:
            self._hhs_fail("ROI를 먼저 드래그해야 합니다.")
            return

        params = self._hhs_params()
        try:
            compute_hill_on_hill_score(
                np.linspace(-1.0, 1.0, 20),
                np.linspace(0.0, 1.0, 20),
                params,
            )
        except ValueError as exc:
            self._hhs_fail(str(exc))
            return

        results: list[HHSRunResult] = []
        for item in self.hhs_images:
            try:
                profile = analyze_section(
                    item.image,
                    current_roi,
                    pixel_size_um=params["pixel_size_um"],
                    threshold_sensitivity=params["threshold_sensitivity"],
                    smoothing_strength=params["smoothing_strength"],
                    morph_strength=params["morph_strength"],
                    axis_name="hhs",
                    coordinate_name="x_um",
                    edge_mode=params["edge_mode"],
                )
                hhs_result = compute_hill_on_hill_score(profile.coord_um, profile.z_um, params)
                error = hhs_result["reason"] if hhs_result["status"] != "OK" else ""
                results.append(HHSRunResult(item=item, roi=current_roi, profile=profile, hhs_result=hhs_result, error=error))
            except (DetectionError, ValueError) as exc:
                results.append(HHSRunResult(item=item, roi=current_roi, error=str(exc)))
            except Exception as exc:
                results.append(HHSRunResult(item=item, roi=current_roi, error=f"Unexpected error: {exc}"))

        self.hhs_results = results
        self.hhs_rois = [current_roi for _ in self.hhs_images]
        self.hhs_state = HHSBatchState(results=results, params=params)
        success = [result for result in results if result.hhs_result is not None and result.hhs_result["status"] == "OK"]
        self.hhs_save_btn.setEnabled(bool(results))
        self._update_hhs_results()
        if success:
            first_success = success[0]
            self.hhs_view.set_overlay(plot_detection_overlay(first_success.item.image, first_success.profile))
            self._replace_hhs_canvas(plot_hhs_profile(first_success.hhs_result))
        else:
            self._replace_hhs_canvas(plot_hhs_batch(results))
        fail_count = len([result for result in results if result.hhs_result is None or result.hhs_result["status"] != "OK"])
        self.hhs_message.setText(f"HHS 분석 완료. 총 {len(results)}개 중 {len(success)}개 계산 가능, {fail_count}개 계산 불가.")

    def _clear_single_results(self) -> None:
        for label in self.single_result_labels.values():
            label.setText("-")
        self.single_table.setRowCount(0)
        self.single_message.setText("이미지를 여러 장 불러온 뒤 첫 이미지에서 ROI를 드래그하세요.")

    def _clear_hhs_results(self) -> None:
        for label in self.hhs_result_labels.values():
            label.setText("-")
        self.hhs_table.setRowCount(0)
        self.hhs_message.setText("이미지를 여러 장 불러온 뒤 ROI를 드래그하세요.")

    def _update_single_results(self) -> None:
        if self.single_state is None:
            return
        results = self.single_state.results
        success = [result for result in results if result.profile is not None and result.cd_result is not None]
        failed = [result for result in results if result.profile is None and bool(result.error)]
        self.single_result_labels["image_count"].setText(str(len(results)))
        self.single_result_labels["success_count"].setText(str(len(success)))
        self.single_result_labels["fail_count"].setText(str(len(failed)))
        self.single_result_labels["depth_step_um"].setText(f"{self.single_state.params['depth_step_um']:.4f} um")
        self.single_result_labels["taper_step_um"].setText(f"{self.single_state.params['taper_step_um']:.4f} um")
        current_roi = self.single_rois[self.single_current_index] if self.single_rois else None
        self.single_result_labels["roi"].setText(str(current_roi) if current_roi else "-")

        self.single_table.setRowCount(len(results))
        for row_idx, result in enumerate(results):
            if result.profile is not None and result.cd_result is not None:
                values = [
                    result.item.name,
                    "OK",
                    f"{result.profile.cd_um:.5f}",
                    f"{result.profile.height_um:.5f}",
                    str(result.cd_result.depth_um.size),
                    _format_taper_mean(result.taper_result, "left"),
                    _format_taper_mean(result.taper_result, "right"),
                ]
            else:
                values = [result.item.name, result.error or "NOT_ANALYZED", "-", "-", "-", "-", "-"]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col_idx >= 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.single_table.setItem(row_idx, col_idx, item)

    def _populate_single_loaded_table(self) -> None:
        results = self.single_results or [SingleProfileRunResult(item=item) for item in self.single_images]
        self.single_table.setRowCount(len(results))
        for row_idx, result in enumerate(results):
            values = [result.item.name, "LOADED", "-", "-", "-"]
            for col_idx, value in enumerate(values):
                self.single_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self.single_result_labels["image_count"].setText(str(len(self.single_images)))
        self.single_result_labels["success_count"].setText("-")
        self.single_result_labels["fail_count"].setText("-")
        self.single_result_labels["depth_step_um"].setText(f"{self.single_depth_step_spin.value():.4f} um")
        self.single_result_labels["taper_step_um"].setText(f"{self.single_taper_step_spin.value():.4f} um")
        current_roi = self.single_rois[self.single_current_index] if self.single_rois else None
        self.single_result_labels["roi"].setText(str(current_roi) if current_roi else "-")
        self._update_single_navigation()

    def _update_hhs_results(self) -> None:
        if self.hhs_state is None:
            return
        results = self.hhs_state.results
        success = [result for result in results if result.hhs_result is not None and result.hhs_result["status"] == "OK"]
        failed = [result for result in results if result not in success]
        self.hhs_result_labels["image_count"].setText(str(len(results)))
        self.hhs_result_labels["success_count"].setText(str(len(success)))
        self.hhs_result_labels["fail_count"].setText(str(len(failed)))
        current_roi = self.hhs_rois[self.hhs_current_index] if self.hhs_rois else None
        self.hhs_result_labels["roi"].setText(str(current_roi) if current_roi else "-")
        if success:
            first = success[0].hhs_result
            self.hhs_result_labels["hhs"].setText(_format_float(first["hhs"]))
            self.hhs_result_labels["bump_area"].setText(_format_float(first["bump_area"]))
            self.hhs_result_labels["total_area"].setText(_format_float(first["total_area"]))
        else:
            self.hhs_result_labels["hhs"].setText("-")
            self.hhs_result_labels["bump_area"].setText("-")
            self.hhs_result_labels["total_area"].setText("-")

        self.hhs_table.setRowCount(len(results))
        for row_idx, result in enumerate(results):
            hhs_result = result.hhs_result
            if hhs_result is not None:
                values = [
                    result.item.name,
                    hhs_result["status"],
                    _format_float(hhs_result["hhs"]),
                    _format_float(hhs_result["bump_area"]),
                    _format_float(hhs_result["total_area"]),
                ]
            else:
                values = [result.item.name, result.error or "FAILED", "-", "-", "-"]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col_idx >= 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.hhs_table.setItem(row_idx, col_idx, item)

    def _populate_hhs_loaded_table(self) -> None:
        results = self.hhs_results or [HHSRunResult(item=item) for item in self.hhs_images]
        self.hhs_table.setRowCount(len(results))
        for row_idx, result in enumerate(results):
            values = [result.item.name, "LOADED", "-", "-", "-"]
            for col_idx, value in enumerate(values):
                self.hhs_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self.hhs_result_labels["image_count"].setText(str(len(self.hhs_images)))
        self.hhs_result_labels["success_count"].setText("-")
        self.hhs_result_labels["fail_count"].setText("-")
        self.hhs_result_labels["hhs"].setText("-")
        self.hhs_result_labels["bump_area"].setText("-")
        self.hhs_result_labels["total_area"].setText("-")
        current_roi = self.hhs_rois[self.hhs_current_index] if self.hhs_rois else None
        self.hhs_result_labels["roi"].setText(str(current_roi) if current_roi else "-")
        self._update_hhs_navigation()

    def _replace_single_profile_canvas(self, figure) -> None:
        new_canvas = _make_canvas(figure)
        parent_splitter = self.single_profile_canvas.parent()
        if isinstance(parent_splitter, QSplitter):
            index = parent_splitter.indexOf(self.single_profile_canvas)
            self.single_profile_canvas.setParent(None)
            parent_splitter.insertWidget(index, new_canvas)
            parent_splitter.setSizes([650, 330])
        self.single_profile_canvas = new_canvas

    def _replace_hhs_canvas(self, figure) -> None:
        new_canvas = _make_canvas(figure)
        parent_splitter = self.hhs_canvas.parent()
        if isinstance(parent_splitter, QSplitter):
            index = parent_splitter.indexOf(self.hhs_canvas)
            self.hhs_canvas.setParent(None)
            parent_splitter.insertWidget(index, new_canvas)
            parent_splitter.setSizes([650, 330])
        self.hhs_canvas = new_canvas

    def _update_plots(self) -> None:
        if self.state is None:
            return
        current = self.tabs.currentIndex()
        self._replace_plot_tab(0, "profile_canvas", plot_profiles(self.state.horizontal_profile, self.state.vertical_profile), "Profiles")
        self._replace_plot_tab(1, "surface_canvas", plot_3d_surface(self.state.surface), "3D Surface")
        self._replace_plot_tab(2, "cap_canvas", plot_cap_highlight(self.state.surface, self.state.cap), "Cap 3D")
        self._replace_plot_tab(3, "topview_canvas", plot_cap_top_view(self.state.surface, self.state.cap), "Cap Footprint")
        self.tabs.setCurrentIndex(current)

    def _set_3d_reset_enabled(self, enabled: bool) -> None:
        if hasattr(self, "reset_3d_view_btn"):
            self.reset_3d_view_btn.setEnabled(enabled)

    def reset_3d_views(self) -> None:
        if self.state is None:
            return
        current = self.tabs.currentIndex()
        self._replace_plot_tab(1, "surface_canvas", plot_3d_surface(self.state.surface), "3D Surface")
        self._replace_plot_tab(2, "cap_canvas", plot_cap_highlight(self.state.surface, self.state.cap), "Cap 3D")
        self.tabs.setCurrentIndex(current)

    def _replace_plot_tab(self, index: int, attr_name: str, figure, title: str) -> None:
        old_canvas = getattr(self, attr_name)
        new_canvas = _make_canvas(figure)
        self.tabs.removeTab(index)
        old_canvas.setParent(None)
        self.tabs.insertTab(index, new_canvas, title)
        setattr(self, attr_name, new_canvas)

    def save_results(self) -> None:
        if self.state is None or self.horizontal_image is None or self.vertical_image is None:
            self._fail("저장할 분석 결과가 없습니다.")
            return
        try:
            output_dir = create_output_dir(Path.cwd())
            save_outputs(
                output_dir,
                self.horizontal_image,
                self.vertical_image,
                self.state.horizontal_profile,
                self.state.vertical_profile,
                self.state.surface,
                self.state.cap,
                self.state.params,
            )
            self.result_panel.message.setText(f"결과 저장 완료:\n{output_dir}")
        except Exception as exc:
            self._fail(f"결과 저장 실패: {exc}")

    def save_single_profile_results(self) -> None:
        if self.single_state is None:
            self._single_fail("저장할 batch 프로파일 분석 결과가 없습니다.")
            return
        try:
            output_dir = create_output_dir(Path.cwd())
            save_single_batch_outputs(
                output_dir,
                self.single_state.results,
                self.single_state.params,
            )
            self.single_message.setText(f"Batch 프로파일 결과 저장 완료:\n{output_dir}")
        except Exception as exc:
            self._single_fail(f"Batch 프로파일 결과 저장 실패: {exc}")

    def save_hhs_results(self) -> None:
        if self.hhs_state is None:
            self._hhs_fail("저장할 HHS 분석 결과가 없습니다.")
            return
        try:
            output_dir = create_output_dir(Path.cwd())
            hhs_profile_dir = output_dir / "hhs_profiles"
            hhs_overlay_dir = output_dir / "hhs_overlays"
            hhs_plot_dir = output_dir / "hhs_plots"
            for directory in [hhs_profile_dir, hhs_overlay_dir, hhs_plot_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            rows = []
            for index, result in enumerate(self.hhs_state.results):
                sample_id = result.item.name
                hhs_result = result.hhs_result
                if result.profile is not None:
                    stem = Path(sample_id).stem or f"sample_{index + 1}"
                    profile_depth = np.clip(result.profile.z_um, 0.0, None)
                    pd.DataFrame(
                        {
                            result.profile.coordinate_name: result.profile.coord_um,
                            "height_from_baseline_um": profile_depth,
                            "edge_mode": result.profile.edge_mode,
                        }
                    ).to_csv(hhs_profile_dir / f"{stem}_hhs_profile.csv", index=False)
                    plot_detection_overlay(result.item.image, result.profile, hhs_overlay_dir / f"{stem}_hhs_overlay.png")
                    if hhs_result is not None:
                        plot_hhs_profile(hhs_result, hhs_plot_dir / f"{stem}_hhs_profile.png")
                rows.append(_hhs_summary_row(sample_id, result, self.hhs_state.params))

            pd.DataFrame(rows).to_csv(output_dir / "hhs_result_summary.csv", index=False)
            plot_hhs_batch(self.hhs_state.results, output_dir / "hhs_batch_distribution.png")
            self.hhs_message.setText(f"HHS 결과 저장 완료:\n{output_dir}")
        except Exception as exc:
            self._hhs_fail(f"HHS 결과 저장 실패: {exc}")

    def _fail(self, message: str) -> None:
        self.result_panel.message.setText(message)
        self.save_btn.setEnabled(False)

    def _single_fail(self, message: str) -> None:
        self.single_message.setText(message)
        self.single_save_btn.setEnabled(False)

    def _hhs_fail(self, message: str) -> None:
        self.hhs_message.setText(message)
        self.hhs_save_btn.setEnabled(False)


def _default_synthetic_roi(image: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    roi_w = int(w * 0.66)
    roi_h = int(h * 0.62)
    x = (w - roi_w) // 2
    y = int(h * 0.17)
    return x, y, roi_w, roi_h


def _format_taper_mean(taper_result: SingleTaperResult | None, side: str) -> str:
    if taper_result is None:
        return "-"
    mask = taper_result.side == side
    if not np.any(mask):
        return "-"
    return f"{float(np.nanmean(taper_result.taper_angle_deg[mask])):.3f}"


def _format_float(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "NaN"
    return f"{number:.6f}"


def _hhs_summary_row(sample_id: str, result: HHSRunResult, params: dict) -> dict:
    hhs_result = result.hhs_result or {}
    profile = result.profile
    roi = result.roi
    return {
        "sample_id": sample_id,
        "status": hhs_result.get("status", "FAILED"),
        "error": result.error or hhs_result.get("reason", ""),
        "hhs": hhs_result.get("hhs", np.nan),
        "bump_area": hhs_result.get("bump_area", np.nan),
        "total_area": hhs_result.get("total_area", np.nan),
        "light_smooth_sigma": params["light_smooth_sigma"],
        "baseline_smooth_sigma": params["baseline_smooth_sigma"],
        "center_width": params["center_width"],
        "x_center_used": hhs_result.get("x_center_used", np.nan),
        "profile_width_used": hhs_result.get("profile_width_used", np.nan),
        "CD_um": profile.cd_um if profile is not None else np.nan,
        "H_um": profile.height_um if profile is not None else np.nan,
        "pixel_size_um": params["pixel_size_um"],
        "threshold_sensitivity": params["threshold_sensitivity"],
        "smoothing_strength": params["smoothing_strength"],
        "morph_strength": params["morph_strength"],
        "edge_mode": params.get("edge_mode", profile.edge_mode if profile is not None else ""),
        "roi_x": roi[0] if roi else np.nan,
        "roi_y": roi[1] if roi else np.nan,
        "roi_w": roi[2] if roi else np.nan,
        "roi_h": roi[3] if roi else np.nan,
    }


def plot_profiles_empty():
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.2, 4.2), facecolor="#111827")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#172033")
    ax.set_title("Profiles", color="#E5E7EB")
    ax.set_xlabel("coordinate (um)", color="#E5E7EB")
    ax.set_ylabel("depth z (um, down +)", color="#E5E7EB")
    ax.tick_params(colors="#9CA3AF")
    fig.tight_layout()
    return fig


def plot_empty_2d(title: str):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.2, 4.2), facecolor="#111827")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#172033")
    ax.set_title(title, color="#E5E7EB")
    ax.tick_params(colors="#9CA3AF")
    fig.tight_layout()
    return fig


def plot_empty_3d(title: str):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.2, 4.2), facecolor="#111827")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#172033")
    ax.set_title(title, color="#E5E7EB")
    ax.set_zlabel("depth z (um, down +)", color="#E5E7EB")
    ax.invert_zaxis()
    ax.tick_params(colors="#9CA3AF")
    fig.tight_layout()
    return fig


def run() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
