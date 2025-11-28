"""Generation of 2D trajectories (plus pressure) given font + text."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

from penpal.write_planner import WritePlanner, Character


class PathCollectorPen(BasePen):
    """Pen that converts glyph outlines (lines + curves) into polylines."""

    def __init__(self, glyph_set, steps_per_curve: int = 20) -> None:
        super().__init__(glyph_set)
        self.steps = steps_per_curve
        # List of contours; each contour is a list of (x, y) points.
        self.paths: List[List[Tuple[float, float]]] = []
        # Points for the current contour.
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
    """Generates 2D trajectories + pressure given font & text.
    
    Supports two font types:
    - TTF/OTF fonts via add_font()
    - Hershey single-stroke fonts via add_hershey_font() (recommended for plotters)
    """

    # Available Hershey font names
    HERSHEY_FONT_NAMES = [
        'futural', 'futuram', 'scripts', 'scriptc', 'cursive',
        'rowmans', 'rowmand', 'rowmant', 'timesr', 'timesi', 'timesib',
        'timesg', 'timesrb', 'gothiceng', 'gothicger', 'gothicita',
        'gothgbt', 'gothgrt', 'gothitt', 'greek', 'greekc', 'greeks',
        'cyrillic', 'cyrilc_1', 'japanese', 'markers', 'mathlow',
        'mathupp', 'meteorology', 'music', 'symbolic', 'astrology',
    ]

    @dataclass
    class Config:
        """Configuration for the object."""

        # Number of samples per Bezier curve segment.
        steps_per_curve: int = 20
        # Target spatial step between trajectory points, in mm.
        target_step_mm: float = 1.0
        # Line spacing factor relative to font_size_mm.
        line_spacing_factor: float = 1.08
        # Character advance factor (approx width) relative to font_size_mm.
        char_advance_factor: float = 0.6
        # Space width factor relative to font_size_mm.
        space_advance_factor: float = 0.5
        # Default drawing pressure in [0, 1].
        default_pressure: float = 1.0
        # Whether to convert glyph outline to a single-stroke skeleton (TTF only).
        use_skeleton: bool = False
        # Rasterized image size used for skeletonization.
        skeleton_img_size: int = 256

    def __init__(
        self, writer: WritePlanner, cfg: Config | None = None
    ) -> None:
        """Initialize the object."""
        self._writer = writer
        self.c = cfg if cfg is not None else self.Config()
        # Loaded fonts: key is font_name, value is TTFont object.
        self._fonts: Dict[str, TTFont] = {}
        # Loaded Hershey fonts: key is font_name, value is HersheyFonts object.
        self._hershey_fonts: Dict[str, object] = {}

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

    def add_hershey_font(self, font_name: str) -> None:
        """
        Register a Hershey font (single-stroke font ideal for plotters).

        Recommended fonts:
        - 'futural': Future Light (simple sans-serif) - BEST for legibility
        - 'scripts': Script Simplex (handwriting-like)
        - 'rowmans': Roman Simplex (serif)
        """
        try:
            from HersheyFonts import HersheyFonts
        except ImportError:
            raise RuntimeError(
                "Hershey fonts require the Hershey-Fonts package. "
                "Install with: pip install Hershey-Fonts"
            )

        hf = HersheyFonts()
        available = hf.default_font_names
        if font_name not in available:
            raise ValueError(
                f"Unknown Hershey font '{font_name}'. "
                f"Available fonts: {available}"
            )

        hf.load_default_font(font_name)
        self._hershey_fonts[font_name] = hf


    def write_text(
        self,
        text: str,
        font_name: str,
        font_size_mm: float,
        pen_thickness_mm: float,
    ) -> None:
        """Generate trajectories for a string of text and send them to WritePlanner.

        This preserves the original public API: it creates a list of Character
        objects and calls self._writer.write_characters(characters).
        """
        characters = self._text_to_characters(
            text=text,
            font_name=font_name,
            font_size_mm=font_size_mm,
            pen_thickness_mm=pen_thickness_mm,
        )

        if characters:
            self._writer.write_characters(characters)

    def write_text_flat(
        self,
        text: str,
        font_name: str,
        font_size_mm: float,
        pen_thickness_mm: float,
        step_mm: float | None = None,
    ) -> np.ndarray:
        """Generate a single continuous trajectory for the whole text.

        The returned array is shape (N, 3) in the virtual-board frame:
        columns [x_mm, y_mm, z_pressure], where z = 0 means pen up and z > 0
        means pen down. This method does NOT call WritePlanner.
        """
        characters = self._text_to_characters(
            text=text,
            font_name=font_name,
            font_size_mm=font_size_mm,
            pen_thickness_mm=pen_thickness_mm,
        )
        if not characters:
            return np.zeros((0, 3), dtype=float)

        step = step_mm if step_mm is not None else self.c.target_step_mm
        return self.build_flat_path_constant_speed(characters, step_mm=step)

    # ------------------------------------------------------------------
    # Internal helpers: text -> per-character trajectories
    # ------------------------------------------------------------------

    def _text_to_characters(
        self,
        text: str,
        font_name: str,
        font_size_mm: float,
        pen_thickness_mm: float,  # currently unused, reserved for pressure mapping
    ) -> list[Character]:
        """Convert a text string into a list of Character objects.

        This is the core text-to-glyph-to-trajectory pipeline shared by
        write_text() and write_text_flat().
        """
        # Check if it's a Hershey font
        if font_name in self._hershey_fonts:
            return self._text_to_characters_hershey(
                text=text,
                font_name=font_name,
                font_size_mm=font_size_mm,
            )

        # Otherwise, use TTF font
        if font_name not in self._fonts:
            raise ValueError(
                f"Font '{font_name}' has not been added via add_font() or add_hershey_font()."
            )

        return self._text_to_characters_ttf(
            text=text,
            font_name=font_name,
            font_size_mm=font_size_mm,
        )

    def _text_to_characters_hershey(
        self,
        text: str,
        font_name: str,
        font_size_mm: float,
    ) -> list[Character]:
        """Convert text to Character objects using a Hershey font."""
        hf = self._hershey_fonts[font_name]

        # Normalize rendering so that typical glyph height ~= font_size_mm.
        # After this call, coordinates returned by strokes_for_text() are in
        # the same scale (we treat them directly as millimeters in the board frame).
        hf.normalize_rendering(font_size_mm)

        # Optional: make sure there is no extra internal spacing from the font,
        # because we handle spacing ourselves via current_x.
        if "spacing" in hf.render_options:
            hf.render_options["spacing"] = 1.0

        characters: list[Character] = []

        # Multi-line layout (top-to-bottom)
        lines = text.split("\n")
        line_height = self.c.line_spacing_factor * font_size_mm

        for line_idx, line in enumerate(lines):
            if not line:
                continue

            # Each new line is shifted downward by line_height
            y_offset = -line_idx * line_height
            current_x = 0.0

            for ch in line:
                # Space character: advance cursor only.
                if ch == " ":
                    current_x += self.c.space_advance_factor * font_size_mm
                    continue

                # Get strokes for this single character.
                try:
                    strokes = list(hf.strokes_for_text(ch))
                except Exception:
                    # Unsupported character: just advance a default width.
                    current_x += self.c.char_advance_factor * font_size_mm
                    continue

                if not strokes:
                    current_x += self.c.char_advance_factor * font_size_mm
                    continue

                paths_mm: list[list[tuple[float, float]]] = []
                min_x = float("inf")
                max_x = float("-inf")

                # Convert each stroke to a polyline path
                for stroke in strokes:
                    pts = list(stroke)
                    if len(pts) < 2:
                        continue

                    path: list[tuple[float, float]] = []
                    for (x, y) in pts:
                        # Coordinates from HersheyFonts are already scaled
                        # by normalize_rendering(), so we treat them as mm.
                        path.append((float(x), float(y)))
                        if x < min_x:
                            min_x = x
                        if x > max_x:
                            max_x = x

                    if len(path) >= 2:
                        paths_mm.append(path)

                if not paths_mm or not math.isfinite(min_x) or not math.isfinite(max_x):
                    current_x += self.c.char_advance_factor * font_size_mm
                    continue

                # Estimate character width from bounding box.
                char_width = max_x - min_x
                if char_width <= 0.0:
                    char_width = 0.5 * font_size_mm

                # Shift all paths to the current cursor position.
                shifted_paths: list[list[tuple[float, float]]] = []
                for path in paths_mm:
                    shifted = [
                        (x - min_x + current_x, y + y_offset)
                        for (x, y) in path
                    ]
                    shifted_paths.append(shifted)

                # Build Character. Hershey strokes are open paths (no closing).
                char_obj = self._paths_to_character(
                    char=ch,
                    paths_mm=shifted_paths,
                    target_step_mm=self.c.target_step_mm,
                    pressure=self.c.default_pressure,
                    closed_paths=False,
                )
                characters.append(char_obj)

                # Advance cursor for the next character.
                extra_spacing = 0.1 * font_size_mm
                current_x += char_width + extra_spacing

        return characters


    def _text_to_characters_ttf(
        self,
        text: str,
        font_name: str,
        font_size_mm: float,
    ) -> list[Character]:
        """Convert text to Character objects using a TTF font."""
        font = self._fonts[font_name]

        characters: list[Character] = []

        # Simple left-to-right, top-to-bottom layout in a "virtual board" frame.
        cursor_x = 0.0  # in mm
        cursor_y = 0.0  # in mm

        line_height = self.c.line_spacing_factor * font_size_mm
        char_advance = self.c.char_advance_factor * font_size_mm
        space_advance = self.c.space_advance_factor * font_size_mm

        for ch in text:
            # Newline handling.
            if ch == "\n":
                cursor_x = 0.0
                cursor_y -= line_height
                continue

            # Space handling.
            if ch == " ":
                cursor_x += space_advance
                continue

            # 1) Extract glyph paths in normalized units (roughly [0,1]).
            if self.c.use_skeleton:
                paths_norm = self._glyph_to_paths_single_stroke(
                    font, ch, steps_per_curve=self.c.steps_per_curve
                )
            else:
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

            # 3) Compute approximate glyph width in normalized units.
            all_x = [p[0] for path in paths_norm for p in path]
            glyph_width_norm = max(all_x) - min(all_x)
            glyph_width_mm = glyph_width_norm * font_size_mm

            # 4) Shift to current cursor position (simple layout).
            shifted_paths: list[list[tuple[float, float]]] = []
            for path in paths_mm:
                shifted = [(x + cursor_x, y + cursor_y) for (x, y) in path]
                shifted_paths.append(shifted)

            # 5) Convert these paths into a Character (Nx3 trajectory).
            char_obj = self._paths_to_character(
                char=ch,
                paths_mm=shifted_paths,
                target_step_mm=self.c.target_step_mm,
                pressure=self.c.default_pressure,
                closed_paths=not self.c.use_skeleton,  # outline: closed, skeleton: open
            )
            characters.append(char_obj)

            # 6) Advance cursor to the right for the next character.
            #    Use actual glyph width plus a small extra spacing.
            extra_spacing_mm = 0.1 * font_size_mm  # tunable
            cursor_x += glyph_width_mm + extra_spacing_mm

        return characters

    # ------------------------------------------------------------------
    # Internal helpers: glyph outlines
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

    def _glyph_to_paths_single_stroke(
        self,
        font: TTFont,
        char: str,
        steps_per_curve: int = 20,
    ) -> list[list[tuple[float, float]]]:
        """Approximate single-stroke (skeleton) paths for a glyph.

        Strategy:
          - Get outline polylines in normalized coords.
          - For 'O'/'o'/'0', always use a single outer outline loop.
          - Only for a small whitelist of simple characters do we:
              * rasterize + skeletonize,
              * decompose skeleton into strokes,
              * map strokes back to normalized coords.
          - For all other characters, just return the outline paths.
        """

        # ---------- 0) Outline ----------
        outline_paths = self._glyph_to_paths(
            font, char, steps_per_curve=steps_per_curve
        )
        if not outline_paths:
            return []

        # ---------- 0a) Special-case donut glyphs ----------
        # Always draw 'O'/'o'/'0' as a single outer loop (no skeleton).
        if char in ("O", "o", "0"):
            def bbox_area(path: list[tuple[float, float]]) -> float:
                xs = [p[0] for p in path]
                ys = [p[1] for p in path]
                return (max(xs) - min(xs)) * (max(ys) - min(ys))

            outer = max(outline_paths, key=bbox_area)
            return [outer]

        # ---------- 0b) Character whitelist for skeleton ----------
        # Only these letters will use skeleton; everything else uses outline.
        SIMPLE_SKELETON_CHARS = set(
            "HIJKLMNTUVWXYZ"      # uppercase with simple strokes
            "cijlmnruvwxyz"       # lowercase that usually skeletonize nicely
        )
        # You can add or remove characters here based on visual results.

        if char not in SIMPLE_SKELETON_CHARS:
            # Do not attempt skeletonization for more complex glyphs.
            return outline_paths

        # ---------- 1) Bounding box (normalized coords) ----------
        all_x = [p[0] for path in outline_paths for p in path]
        all_y = [p[1] for path in outline_paths for p in path]
        min_x0, max_x0 = min(all_x), max(all_x)
        min_y0, max_y0 = min(all_y), max(all_y)

        width = max_x0 - min_x0
        height = max_y0 - min_y0
        if width <= 0.0 or height <= 0.0:
            return outline_paths

        pad = 0.05 * max(width, height)
        min_x = min_x0 - pad
        max_x = max_x0 + pad
        min_y = min_y0 - pad
        max_y = max_y0 + pad

        size = self.c.skeleton_img_size

        # Normalized coords -> pixel coords (leave 1-pixel padding).
        scale_x = (size - 2) / (max_x - min_x)
        scale_y = (size - 2) / (max_y - min_y)

        # ---------- 2) Rasterize outline into a filled mask ----------
        try:
            from PIL import Image, ImageDraw
            from skimage.morphology import skeletonize
            from skimage.measure import label
        except ImportError as e:
            raise RuntimeError(
                "Skeleton mode requires Pillow and scikit-image. "
                "Install them with 'pip install pillow scikit-image'."
            ) from e

        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)

        for path in outline_paths:
            if len(path) < 3:
                continue
            poly: list[tuple[float, float]] = []
            for (x, y) in path:
                px = (x - min_x) * scale_x + 1.0
                py = (max_y - y) * scale_y + 1.0  # flip y
                poly.append((px, py))
            if poly[0] != poly[-1]:
                poly.append(poly[0])
            draw.polygon(poly, outline=255, fill=255)

        mask = np.array(img, dtype=bool)
        if not mask.any():
            return outline_paths

        # ---------- 3) Skeletonize ----------
        skel = skeletonize(mask)

        # ---------- 4) Label components and build strokes ----------
        labels = label(skel, connectivity=2)
        n_labels = labels.max()
        if n_labels == 0:
            return outline_paths

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        skeleton_paths_pix: list[list[tuple[int, int]]] = []

        for lab in range(1, n_labels + 1):
            comp = labels == lab
            ys, xs = np.nonzero(comp)
            if len(xs) == 0:
                continue

            pixels = list(zip(ys, xs))
            pix_set = set(pixels)

            def pixel_neighbors(p: tuple[int, int]):
                r, c = p
                for dr, dc in neighbors:
                    q = (r + dr, c + dc)
                    if q in pix_set:
                        yield q

            # Degree of each pixel in skeleton graph.
            degrees: dict[tuple[int, int], int] = {
                p: sum(1 for _ in pixel_neighbors(p)) for p in pixels
            }

            # Nodes: pixels with degree != 2.
            nodes = [p for p, deg in degrees.items() if deg != 2]

            visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

            def add_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
                if a <= b:
                    visited_edges.add((a, b))
                else:
                    visited_edges.add((b, a))

            def edge_visited(a: tuple[int, int], b: tuple[int, int]) -> bool:
                return (a, b) in visited_edges or (b, a) in visited_edges

            # Case 1 – graph with nodes (endpoints / junctions).
            if nodes:
                for u in nodes:
                    for v in pixel_neighbors(u):
                        if edge_visited(u, v):
                            continue

                        path_pix: list[tuple[int, int]] = [u, v]
                        add_edge(u, v)
                        prev = u
                        cur = v

                        while True:
                            deg_cur = degrees.get(cur, 0)
                            if deg_cur != 2:
                                # Reached another node; stop stroke here.
                                break

                            nbrs = [w for w in pixel_neighbors(cur) if w != prev]
                            if not nbrs:
                                break
                            w = nbrs[0]
                            if edge_visited(cur, w):
                                break

                            path_pix.append(w)
                            add_edge(cur, w)
                            prev, cur = cur, w

                        if len(path_pix) >= 2:
                            skeleton_paths_pix.append(path_pix)

            else:
                # Case 2: pure loop (all degrees == 2), no endpoints/junctions.
                start = pixels[0]
                path_pix: list[tuple[int, int]] = [start]
                prev = None
                cur = start

                for _ in range(len(pixels) + 5):
                    nbrs = list(pixel_neighbors(cur))
                    if not nbrs:
                        break
                    if prev is None:
                        nxt = nbrs[0]
                    else:
                        candidates = [w for w in nbrs if w != prev] or nbrs
                        nxt = candidates[0]
                    if nxt == start and prev is not None:
                        path_pix.append(nxt)
                        break
                    if nxt == prev:
                        break
                    path_pix.append(nxt)
                    prev, cur = cur, nxt

                if len(path_pix) >= 2:
                    skeleton_paths_pix.append(path_pix)

        # ---------- 5) Convert pixel paths back to normalized coords ----------
        def pix_path_to_norm(path_pix: list[tuple[int, int]]):
            pts_norm: list[tuple[float, float]] = []
            for (r, c) in path_pix:
                x_norm = (c - 1.0) / scale_x + min_x
                y_norm = max_y - (r - 1.0) / scale_y
                pts_norm.append((x_norm, y_norm))
            return pts_norm

        skeleton_paths: list[list[tuple[float, float]]] = [
            pix_path_to_norm(path_pix)
            for path_pix in skeleton_paths_pix
            if len(path_pix) >= 2
        ]

        if not skeleton_paths:
            # Skeletonization failed → fall back to outline.
            return outline_paths

        # ---------- 6) Normal case: use skeleton paths ----------
        return skeleton_paths

    # ------------------------------------------------------------------
    # Internal helpers: per-character path resampling
    # ------------------------------------------------------------------

    def _resample_path(
        self,
        path: list[tuple[float, float]],
        target_step_mm: float,
        closed: bool = True,
    ) -> list[tuple[float, float]]:
        """Resample a path so that distances between successive points
        are roughly <= target_step_mm.

        Path is assumed to be in physical units (mm).
        If closed=True, last point connects back to the first.
        """
        if len(path) < 2:
            return path

        if closed:
            pts = list(path) + [path[0]]
        else:
            pts = list(path)

        new_path: list[tuple[float, float]] = [pts[0]]

        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
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
        closed_paths: bool = True,
    ) -> Character:
        """Convert a list of 2D paths (in mm) into a Character with Nx3 trajectory.

        For each path, we:
          - Move pen (z=0) to the first point.
          - Press pen down (z=pressure).
          - Follow the path with z=pressure (approximately constant speed).
        """
        points: list[list[float]] = []

        for path in paths_mm:
            if len(path) == 0:
                continue

            # Resample each path for approximate constant speed.
            resampled = self._resample_path(
                path, target_step_mm=target_step_mm, closed=closed_paths
            )
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

            # Optionally we could lift the pen at the end here.

        traj = np.array(points, dtype=float)  # shape (N, 3)
        return Character(char=char, trajectory=traj)

    # ------------------------------------------------------------------
    # Internal helper: build one continuous, constant-speed path
    # ------------------------------------------------------------------

    def build_flat_path_constant_speed(
        self,
        characters: list[Character],
        step_mm: float | None = None,
    ) -> np.ndarray:
        """Flatten a list of Character trajectories into a single path.

        The returned array has shape (N, 3) with columns [x_mm, y_mm, z],
        where z = 0 means pen up and z > 0 means pen down. The spacing in
        the xy-plane is approximately constant.

        Characters are written in the given order. Between the end of
        character i and the beginning of character i+1 we insert a straight
        pen-up segment.
        """
        if step_mm is None:
            step_mm = self.c.target_step_mm

        seg_starts: list[np.ndarray] = []
        seg_ends: list[np.ndarray] = []
        seg_down: list[bool] = []

        prev_end_xy: np.ndarray | None = None

        for ch in characters:
            traj = ch.trajectory
            if traj is None:
                continue
            traj = np.asarray(traj, dtype=float)
            if traj.size == 0:
                continue

            # Ensure shape (N, 3).
            if traj.ndim != 2 or traj.shape[1] not in (2, 3):
                raise ValueError("Character.trajectory must be (N,2) or (N,3).")

            if traj.shape[1] == 2:
                # If no z column, assume the whole trajectory is pen down.
                z_col = np.ones((traj.shape[0], 1), dtype=float)
                traj = np.hstack([traj, z_col])

            # If all z are 0, also interpret as pen-down strokes.
            if np.allclose(traj[:, 2], 0.0):
                traj[:, 2] = 1.0

            # Pen-up link from previous character end to this character start.
            start_xy = traj[0, :2]
            if prev_end_xy is not None:
                link_vec = start_xy - prev_end_xy
                if np.linalg.norm(link_vec) > 1e-6:
                    seg_starts.append(prev_end_xy.copy())
                    seg_ends.append(start_xy.copy())
                    seg_down.append(False)  # pen up during this jump

            # Segments inside this character.
            for i in range(1, traj.shape[0]):
                p0 = traj[i - 1, :2]
                p1 = traj[i, :2]
                if np.allclose(p0, p1):
                    continue
                seg_starts.append(p0)
                seg_ends.append(p1)
                pen_down = (traj[i - 1, 2] > 0.0) and (traj[i, 2] > 0.0)
                seg_down.append(pen_down)

            prev_end_xy = traj[-1, :2]

        if not seg_starts:
            return np.zeros((0, 3), dtype=float)

        seg_starts_arr = np.vstack(seg_starts)
        seg_ends_arr = np.vstack(seg_ends)
        seg_down_arr = np.asarray(seg_down, dtype=bool)

        # Segment lengths and cumulative arc-length.
        diffs = seg_ends_arr - seg_starts_arr
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])

        # Remove zero-length segments just in case.
        mask_nonzero = seg_lens > 1e-9
        seg_starts_arr = seg_starts_arr[mask_nonzero]
        seg_ends_arr = seg_ends_arr[mask_nonzero]
        seg_down_arr = seg_down_arr[mask_nonzero]
        seg_lens = seg_lens[mask_nonzero]

        if seg_starts_arr.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)

        cumlen = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total_len = float(cumlen[-1])
        if total_len <= 0.0:
            p = np.hstack(
                [
                    seg_starts_arr[0],
                    [1.0 if seg_down_arr[0] else 0.0],
                ]
            )
            return p[None, :]

        n_steps = max(1, int(total_len / step_mm))
        s_values = np.linspace(0.0, total_len, n_steps + 1)

        new_pts: list[list[float]] = []
        idx = 0
        for s in s_values:
            # Find segment such that cumlen[idx] <= s <= cumlen[idx+1].
            while idx < len(seg_lens) - 1 and s > cumlen[idx + 1]:
                idx += 1
            seg_len = seg_lens[idx]
            if seg_len <= 0.0:
                t = 0.0
            else:
                t = (s - cumlen[idx]) / seg_len

            p0 = seg_starts_arr[idx]
            p1 = seg_ends_arr[idx]
            xy = p0 + t * (p1 - p0)
            z = 1.0 if seg_down_arr[idx] else 0.0
            new_pts.append([xy[0], xy[1], z])

        return np.asarray(new_pts, dtype=float)