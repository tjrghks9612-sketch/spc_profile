from __future__ import annotations

import unittest

import numpy as np

from hhs_analysis import compute_hill_on_hill_score


def _single_hill(x: np.ndarray) -> np.ndarray:
    y = 1.0 - x * x
    return np.clip(y, 0.0, None)


def _hill_on_hill(x: np.ndarray, height: float) -> np.ndarray:
    return _single_hill(x) + height * np.exp(-0.5 * (x / 0.09) ** 2)


class HHSTests(unittest.TestCase):
    def test_single_hill_hhs_is_low(self) -> None:
        x = np.linspace(-1.0, 1.0, 401)
        result = compute_hill_on_hill_score(x, _single_hill(x))
        self.assertEqual(result["status"], "OK")
        self.assertLess(result["hhs"], 0.04)

    def test_hill_on_hill_hhs_is_higher_than_single_hill(self) -> None:
        x = np.linspace(-1.0, 1.0, 401)
        single = compute_hill_on_hill_score(x, _single_hill(x))
        double = compute_hill_on_hill_score(x, _hill_on_hill(x, 0.35))
        self.assertGreater(double["hhs"], single["hhs"] * 2.0)

    def test_hhs_increases_with_center_bump_height(self) -> None:
        x = np.linspace(-1.0, 1.0, 401)
        low = compute_hill_on_hill_score(x, _hill_on_hill(x, 0.18))
        high = compute_hill_on_hill_score(x, _hill_on_hill(x, 0.45))
        self.assertGreater(high["hhs"], low["hhs"])

    def test_nan_values_are_ignored(self) -> None:
        x = np.linspace(-1.0, 1.0, 401)
        y = _hill_on_hill(x, 0.35)
        y[40] = np.nan
        y[130] = np.inf
        result = compute_hill_on_hill_score(x, y)
        self.assertEqual(result["status"], "OK")
        self.assertTrue(np.isfinite(result["hhs"]))

    def test_constant_y_is_unavailable(self) -> None:
        x = np.linspace(-1.0, 1.0, 80)
        result = compute_hill_on_hill_score(x, np.ones_like(x))
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertTrue(np.isnan(result["hhs"]))

    def test_existing_cd_depth_mode_still_computes(self) -> None:
        try:
            from profile_extraction import ProfileResult
            from single_profile import compute_cd_by_depth
        except ImportError as exc:
            self.skipTest(f"OpenCV/PySide dependencies unavailable: {exc}")

        coord = np.linspace(-1.0, 1.0, 101).astype(np.float32)
        height = 1.0
        z = (height - np.abs(coord)).astype(np.float32)
        profile = ProfileResult(
            axis_name="single",
            coordinate_name="x_um",
            roi=(0, 0, 101, 80),
            threshold_value=0.0,
            component_mask=np.ones((80, 101), dtype=np.uint8),
            contour=np.asarray([[[0, 0]], [[100, 0]], [[100, 79]], [[0, 79]]], dtype=np.int32),
            top_x_px_roi=np.arange(101, dtype=np.int32),
            top_y_px_roi=np.zeros(101, dtype=np.float32),
            baseline_y_px_roi=79.0,
            left_x_px_roi=0,
            right_x_px_roi=100,
            center_x_px_roi=50.0,
            cd_um=2.0,
            height_um=height,
            coord_um=coord,
            z_um=z,
            normalized_r=np.abs(coord).astype(np.float32),
            normalized_f=z.astype(np.float32),
        )
        result = compute_cd_by_depth(profile, max_depth_um=0.8, depth_step_um=0.2)
        self.assertGreater(result.depth_um.size, 1)
        self.assertTrue(np.all(np.isfinite(result.cd_um)))


if __name__ == "__main__":
    unittest.main()
