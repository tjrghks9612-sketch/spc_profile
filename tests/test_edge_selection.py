from __future__ import annotations

import unittest

import numpy as np


class EdgeSelectionTests(unittest.TestCase):
    def test_inner_gradient_edge_selects_lower_boundary_than_outer(self) -> None:
        try:
            from profile_extraction import extract_boundary_profile
        except ImportError as exc:
            self.skipTest(f"OpenCV dependencies unavailable: {exc}")

        gray = np.full((40, 30), 180, dtype=np.uint8)
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[6:32, 4:26] = 1
        gray[6:14, 4:26] = 70
        gray[14:32, 4:26] = 125

        outer_x, outer_y = extract_boundary_profile(gray, mask, edge_mode="outer")
        inner_x, inner_y = extract_boundary_profile(gray, mask, edge_mode="inner_gradient")

        self.assertEqual(outer_x.size, inner_x.size)
        self.assertGreater(float(np.nanmedian(inner_y)), float(np.nanmedian(outer_y)) + 4.0)


if __name__ == "__main__":
    unittest.main()
