# ---------------- Begin_Citation [4] ---------------- # noqa: E266
"""Unit tests for penpal.font_trajectory.FontTrajectory.

These tests focus on the basic TTF-outline path generation that is actually
used in the project. Hershey fonts and skeletonization are NOT exercised here,
to keep dependencies and runtime small.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from penpal.font_trajectory import FontTrajectory


class TestFontTrajectoryTTF(unittest.TestCase):
    """Tests for TTF-based trajectories in FontTrajectory."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load a known TTF font once for all tests."""
        # When this file lives under penpal/test/, parents[1] is penpal/.
        fonts_dir = Path(__file__).resolve().parents[1] / "fonts"
        cls.roboto_path = fonts_dir / "Roboto-Regular.ttf"

        if not cls.roboto_path.is_file():
            # If the font is missing, skip all tests in this class
            raise unittest.SkipTest(
                f"Roboto-Regular.ttf not found at {cls.roboto_path}"
            )

    def _make_font(self, font_size_mm: float = 10.0) -> FontTrajectory:
        """Helper: create a FontTrajectory and register Roboto-Regular."""
        ft = FontTrajectory()
        ft.add_font(self.roboto_path)
        # Sanity check: config looks reasonable
        self.assertGreater(ft.c.target_step_mm, 0.0)
        self.assertGreater(font_size_mm, 0.0)
        return ft

    # ------------------------------------------------------------------
    # Basic geometry / shape tests
    # ------------------------------------------------------------------

    def test_single_character_generates_nonempty_trajectory(self) -> None:
        """A simple character should produce a non-empty (N,3) trajectory."""
        ft = self._make_font(font_size_mm=10.0)

        chars = ft.write_text(
            text="H",
            font_name="Roboto-Regular",
            font_size_mm=10.0,
            pen_thickness_mm=1.0,
            const_speed=False,  # use raw per-path sampling here
        )

        # We expect exactly one Character object for text "H".
        self.assertEqual(len(chars), 1)

        ch = chars[0]
        self.assertEqual(ch.char, "H")
        self.assertIsNotNone(ch.trajectory)

        traj = np.asarray(ch.trajectory, dtype=float)
        self.assertEqual(traj.ndim, 2)
        self.assertEqual(traj.shape[1], 3)
        self.assertGreater(traj.shape[0], 0)

        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]

        # Some pen-down samples must exist.
        self.assertTrue(np.any(z > 0.0))

        # Bounding box should be non-degenerate and not absurdly large.
        width = float(x.max() - x.min())
        height = float(y.max() - y.min())

        self.assertGreater(width, 0.1)
        self.assertGreater(height, 0.1)

        # If unitsPerEm handling were wrong, this would blow up by ~1000×.
        font_size_mm = 10.0
        self.assertLess(width, font_size_mm * 10.0)
        self.assertLess(height, font_size_mm * 10.0)

    # ------------------------------------------------------------------
    # Layout tests (multi-line, newlines)
    # ------------------------------------------------------------------

    def test_newline_moves_text_down_one_line(self) -> None:
        """A newline should move the next line down by roughly line_height."""
        ft = self._make_font(font_size_mm=15.0)
        font_size = 15.0

        chars = ft.write_text(
            text="A\nB",
            font_name="Roboto-Regular",
            font_size_mm=font_size,
            pen_thickness_mm=1.0,
            const_speed=False,
        )

        # "A" and "B" become two Character objects in order.
        self.assertEqual(len(chars), 2)

        traj_A = np.asarray(chars[0].trajectory, dtype=float)
        traj_B = np.asarray(chars[1].trajectory, dtype=float)

        mean_y_A = float(traj_A[:, 1].mean())
        mean_y_B = float(traj_B[:, 1].mean())

        # In the implementation, cursor_y starts at 0 and each newline does:
        #   cursor_y -= line_height
        # so the second line should have a clearly smaller (more negative) y.
        self.assertLess(
            mean_y_B,
            mean_y_A - 0.5 * font_size,  # quite a loose bound, but robust
            msg=f"Expected second line below first: mean_y_A={mean_y_A}, mean_y_B={mean_y_B}",
        )

    # ------------------------------------------------------------------
    # Constant-speed flattening tests
    # ------------------------------------------------------------------

    def test_flattened_path_has_approximately_constant_step(self) -> None:
        """build_flat_path_constant_speed should produce near-uniform step size."""
        ft = self._make_font(font_size_mm=12.0)

        # Use two characters so there is at least one pen-up jump between them.
        chars = ft.write_text(
            text="AB",
            font_name="Roboto-Regular",
            font_size_mm=12.0,
            pen_thickness_mm=1.0,
            const_speed=False,  # keep original per-character sampling
        )

        flat = ft.build_flat_path_constant_speed(chars, step_mm=1.0)

        self.assertEqual(flat.ndim, 2)
        self.assertEqual(flat.shape[1], 3)
        self.assertGreater(flat.shape[0], 2)

        # There should be both pen-down and pen-up samples.
        zs = flat[:, 2]
        self.assertTrue(np.any(zs > 0.0))
        self.assertTrue(np.any(zs <= 0.0))

        # Compute step lengths in xy-plane, ignoring zero-length steps.
        diffs = np.diff(flat[:, 0:2], axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        dists = dists[dists > 1e-6]

        self.assertGreater(dists.size, 0)

        mean_step = float(dists.mean())
        max_step = float(dists.max())
        min_step = float(dists.min())

        # Step size should be on the same order as requested step_mm=1.0.
        self.assertGreater(mean_step, 0.1)
        self.assertLess(mean_step, 2.0)

        # Spread should not be huge; constant-speed reparameterization should
        # keep steps roughly uniform.
        self.assertLess(
            max_step - min_step,
            mean_step,
            msg=(
                f"Step sizes too uneven: "
                f"min={min_step:.3f}, max={max_step:.3f}, mean={mean_step:.3f}"
            ),
        )

    # ------------------------------------------------------------------
    # Error handling tests
    # ------------------------------------------------------------------

    def test_unknown_font_raises_value_error(self) -> None:
        """Requesting a font that was never added should raise ValueError."""
        ft = FontTrajectory()

        with self.assertRaises(ValueError):
            _ = ft.write_text(
                text="Hi",
                font_name="NonexistentFontName",
                font_size_mm=10.0,
                pen_thickness_mm=1.0,
                const_speed=False,
            )

    def test_empty_text_returns_empty_char_list(self) -> None:
        """Empty text should yield an empty character list."""
        ft = self._make_font(font_size_mm=10.0)

        chars = ft.write_text(
            text="",
            font_name="Roboto-Regular",
            font_size_mm=10.0,
            pen_thickness_mm=1.0,
            const_speed=False,
        )

        self.assertEqual(chars, [])


if __name__ == "__main__":
    unittest.main()

# ---------------- End_Citation [4] ---------------- # noqa: E266