"""Generation of 2D trajectories (plus pressure) given font + text."""

import pathlib
from dataclasses import dataclass
from penpal.penpal.write_planner import WritePlanner


class FontTrajectory:
    """Generates 2D trajectories + pressure given font & text."""

    @dataclass
    class Config:
        """Configuration for the object."""

        pass

    def __init__(self, writer: WritePlanner, cfg: Config | None = None) -> None:
        """Initialize the object."""
        # self._fonts: dict[str, TTFont] = dict()
        self._writer = writer
        self.c = cfg if cfg is not None else self.Config()

    def add_font(self, otf_path: pathlib.Path) -> None:
        """
        Register a new font from an OTF file so it can be used by this class.

        Args:
            otf_path (pathlib.Path): _description_
        """
        # load the font from the file
        # and add it to self._fonts
        pass

    def write_text(
        self, text: str, font_name: str, font_size_mm: float, pen_thickness_mm: float
    ) -> None:
        """Generate trajectories for a string of text.

        Args:
            text (str): text string to generate.
            font_name (str): name of the font to use. must have been added
                using add_font()
            font_size_mm (float): Height of the tallest glyphs in mm
            pen_thickness_mm (float): thickness of the pen we're using to draw.
        """
        # get what font to use from
        # self._fonts[font_name]

        # then call self._writer.write_characters()
        pass
