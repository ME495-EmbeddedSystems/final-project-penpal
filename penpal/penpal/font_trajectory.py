"""
Generation of 2D trajectories (plus pressure) given font + text.
"""

import pathlib
from dataclasses import dataclass
from penpal.penpal.write_planner import WritePlanner


class FontTrajectory:
    """
    Generates 2D trajectories + pressure given font & text.
    """

    @dataclass
    class Config:
        pass

    def __init__(self, writer: WritePlanner, cfg: Config) -> None:
        # self._fonts: dict[str, TTFont] = dict()
        self._writer = writer
        pass

    def add_font(self, otf_path: pathlib.Path) -> None:
        # load the font from the file
        # and add it to self._fonts
        pass

    def write_text(
        self, text: str, font_name: str, font_size_mm: float, pen_thickness_mm: float
    ) -> None:
        # get what font to use from
        # self._fonts[font_name]

        # then call self._writer.write_characters()
        pass
