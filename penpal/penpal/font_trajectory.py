"""Generation of 2D trajectories (plus pressure) given font + text."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

from penpal.write_planner import WritePlanner, Character


class PathCollectorPen(BasePen):
    """Pen that converts glyph outlines (lines + curves) into polylines."""

    def __init__(self, glyph_set, steps_per_curve: int = 20) -> None:
        super().__init__(glyph_set)
        self.steps = steps_per_curve
        # List of contours; each contour is a list of (x, y) points
        self.paths: List[List[Tuple[float, float]]] = []
        # Points for the current contour
        self._current_path: List[Tuple[float, float]] = []

    def _moveTo(self, p0) -> None:
        """Start a new contour."""
        if self._current_path:
            self.paths.append(self._current_path)
        self._current_path = [p0]

    def _lineTo(self, p1) -> None:
        """Straight segment: just add the end point."""
        self._current_path.append(p1)

    def _curveToOne(self, p1, p2, p3) -> None:
        """Cubic Bezier: current point -> p1 -> p2 -> p3."""
        p0 = self._getCurrentPoint()
        for i in range(1, self.steps + 1):
            t = i / self.steps
            x = (
                (1 - t) ** 3 * p0[0]
                + 3 * (1 - t) ** 2 * t * p1[0]
                + 3 * (1 - t) * t ** 2 * p2[0]
                + t ** 3 * p3[0]
            )
            y = (
                (1 - t) ** 3 * p0[1]
                + 3 * (1 - t) ** 2 * t * p1[1]
                + 3 * (1 - t) * t ** 2 * p2[1]
                + t ** 3 * p3[1]
            )
            self._current_path.append((x, y))

    def _qCurveToOne(self, p1, p2) -> None:
        """Quadratic Bezier: current point -> p1 -> p2."""
        p0 = self._getCurrentPoint()
        for i in range(1, self.steps + 1):
            t = i / self.steps
            x = (
                (1 - t) ** 2 * p0[0]
                + 2 * (1 - t) * t * p1[0]
                + t ** 2 * p2[0]
            )
            y = (
                (1 - t) ** 2 * p0[1]
                + 2 * (1 - t) * t * p1[1]
                + t ** 2 * p2[1]
            )
            self._current_path.append((x, y))

    def _closePath(self) -> None:
        """Close current contour."""
        if self._current_path:
            self.paths.append(self._current_path)
            self._current_path = []

    def _endPath(self) -> None:
        """End open contour."""
        if self._current_path:
            self.paths.append(self._current_path)
            self._current_path = []


class FontTrajectory:
    """Generates 2D trajectories + pressure given font & text."""

    @dataclass
    class Config:
        """Configuration for the object."""

        # Number of samples per Bezier curve segment
        steps_per_curve: int = 20
        # Target spatial step between trajectory points, in mm
        target_step_mm: float = 1.0
        # Line spacing factor relative to font_size_mm
        line_spacing_factor: float = 1.2
        # Character advance factor (approx width) relative to font_size_mm
        char_advance_factor: float = 0.6
        # Space width factor relative to font_size_mm
        space_advance_factor: float = 0.5
        # Default drawing pressure in [0, 1]
        default_pressure: float = 1.0

    def __init__(
        self, writer: WritePlanner, cfg: Config | None = None
    ) -> None:
        """Initialize the object."""
        self._writer = writer
        self.c = cfg if cfg is not None else self.Config()
        # Loaded fonts: key is font_name, value is TTFont object
        self._fonts: Dict[str, TTFont] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_font(self, otf_path: pathlib.Path) -> None:
        """
        Register a new font from an OTF/TTF file so it can be used by this class.

        The font will be stored under a name derived from the file stem,
        e.g. "DejaVuSans" for "DejaVuSans.ttf".
        """
        font = TTFont(str(otf_path))
        font_name = otf_path.stem
        self._fonts[font_name] = font

    def write_text(
        self,
        text: str,
        font_name: str,
        font_size_mm: float,
        pen_thickness_mm: float,  # currently unused, reserved for pressure mapping
    ) -> None:
        """Generate trajectories for a string of text and send them to WritePlanner.

        Args:
            text (str): Text string to generate.
            font_name (str): Name of the font to use. Must have been added
                using add_font().
            font_size_mm (float): Approximate height of tallest glyphs in mm.
            pen_thickness_mm (float): Thickness of the pen we're using to draw.
                (Not yet used; can be mapped to pressure in future.)
        """
        if font_name not in self._fonts:
            raise ValueError(
                f"Font '{font_name}' has not been added via add_font()."
            )
        font = self._fonts[font_name]

        characters: list[Character] = []

        # Simple left-to-right, top-to-bottom layout in a "virtual board" frame.
        cursor_x = 0.0  # in mm
        cursor_y = 0.0  # in mm

        line_height = self.c.line_spacing_factor * font_size_mm
        char_advance = self.c.char_advance_factor * font_size_mm
        space_advance = self.c.space_advance_factor * font_size_mm

        for ch in text:
            # Newline handling
            if ch == "\n":
                cursor_x = 0.0
                cursor_y -= line_height
                continue

            # Space handling
            if ch == " ":
                cursor_x += space_advance
                continue

            # 1) Extract glyph outline paths in normalized units (roughly [0,1]).
            paths_norm = self._glyph_to_paths(
                font, ch, steps_per_curve=self.c.steps_per_curve
            )

            if not paths_norm:
                # Skip characters without outlines (e.g., unsupported codepoints).
                cursor_x += char_advance
                continue

            # 2) Scale to physical size in mm.
            scale = font_size_mm
            paths_mm: list[list[tuple[float, float]]] = []
            for path in paths_norm:
                scaled = [(x * scale, y * scale) for (x, y) in path]
                paths_mm.append(scaled)
            
                # Compute approximate glyph width in normalized units
            all_x = [p[0] for path in paths_norm for p in path]
            glyph_width_norm = max(all_x) - min(all_x)
            # Convert to mm
            glyph_width_mm = glyph_width_norm * font_size_mm

            # 3) Shift to current cursor position (simple layout).
            shifted_paths: list[list[tuple[float, float]]] = []
            for path in paths_mm:
                shifted = [
                    (x + cursor_x, y + cursor_y)
                    for (x, y) in path
                ]
                shifted_paths.append(shifted)

            # 4) Convert these paths into a Character (Nx3 trajectory).
            char_obj = self._paths_to_character(
                char=ch,
                paths_mm=shifted_paths,
                target_step_mm=self.c.target_step_mm,
                pressure=self.c.default_pressure,
            )
            characters.append(char_obj)

            # 5) Advance cursor to the right for the next character.
            #    Use actual glyph width plus a small extra spacing.
            extra_spacing_mm = 0.1 * font_size_mm  # you can tune this
            cursor_x += glyph_width_mm + extra_spacing_mm

        # 6) Hand the characters to WritePlanner, which will handle
        #    virtual-board -> real-robot mapping and execution.
        if characters:
            self._writer.write_characters(characters)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _glyph_to_paths(
        self,
        font: TTFont,
        char: str,
        steps_per_curve: int = 20,
    ) -> list[list[tuple[float, float]]]:
        """Return a list of polylines for one character.

        Each polyline is a list of (x, y) in normalized font units
        (i.e., divided by unitsPerEm).
        """
        glyph_set = font.getGlyphSet()

        cmap = font.getBestCmap()
        codepoint = ord(char)
        if codepoint not in cmap:
            # Character not supported by this font.
            return []

        glyph_name = cmap[codepoint]
        glyph = glyph_set[glyph_name]

        pen = PathCollectorPen(glyph_set, steps_per_curve=steps_per_curve)
        glyph.draw(pen)

        units_per_em = font["head"].unitsPerEm
        normalized_paths: list[list[tuple[float, float]]] = []
        for path in pen.paths:
            norm_path = [(x / units_per_em, y / units_per_em) for (x, y) in path]
            if len(norm_path) >= 2:
                normalized_paths.append(norm_path)

        return normalized_paths

    def _resample_path(
        self,
        path: list[tuple[float, float]],
        target_step_mm: float,
    ) -> list[tuple[float, float]]:
        """Resample a closed path so that distances between successive points
        are roughly <= target_step_mm.

        Path is assumed to be in physical units (mm).
        The path is treated as closed: last point connects back to the first.
        """
        if len(path) < 2:
            return path

        # Make an explicit closed version: ... p[n-1], p[0]
        closed = list(path) + [path[0]]

        new_path: list[tuple[float, float]] = [closed[0]]

        for i in range(len(closed) - 1):
            x0, y0 = closed[i]
            x1, y1 = closed[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            seg_len = float(np.hypot(dx, dy))
            if seg_len == 0.0:
                continue

            n_sub = max(1, int(seg_len / target_step_mm))

            for j in range(1, n_sub + 1):
                t = j / n_sub
                x = x0 + t * dx
                y = y0 + t * dy
                new_path.append((x, y))

        return new_path

    def _paths_to_character(
        self,
        char: str,
        paths_mm: list[list[tuple[float, float]]],
        target_step_mm: float,
        pressure: float,
    ) -> Character:
        """Convert a list of 2D paths (in mm) into a Character with Nx3 trajectory.

        For each path, we:
          - Move pen (z=0) to the first point.
          - Press pen down (z=pressure).
          - Follow the closed path with z=pressure (approximately constant speed).
        """
        points: list[list[float]] = []

        for path in paths_mm:
            if len(path) == 0:
                continue

            # Resample each path (treated as closed) for approximate constant speed.
            resampled = self._resample_path(path, target_step_mm=target_step_mm)
            if len(resampled) == 0:
                continue

            x0, y0 = resampled[0]

            # Pen up move to the start of this path.
            points.append([x0, y0, 0.0])

            # Pen down at start.
            points.append([x0, y0, pressure])

            # Draw remaining points with constant pressure.
            for (x, y) in resampled[1:]:
                points.append([x, y, pressure])

            # Optional: at the end we could lift the pen again, but it is
            # not strictly necessary here. If desired:
            # x_end, y_end = resampled[-1]
            # points.append([x_end, y_end, 0.0])

        traj = np.array(points, dtype=float)  # shape (N, 3)
        return Character(char=char, trajectory=traj)
