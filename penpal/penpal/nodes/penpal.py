"""
PenPal node.

TODO better docs.
"""

from typing import Any, List
from rclpy import Context, Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, qos_profile_rosout_default

import board_detector, font_trajectory, grab_planner, pen_detector, write_planner
from control import moveit_control, hybrid_control, pp_control


class PenPal(Node):
    """
    PenPal node.

    TODO better docs.
    """

    def __init__(self) -> None:
        """Initialize the node."""
        super().__init__('PenPal')

        # todo get this from a parameter
        p_control_type = 'moveit'

        match p_control_type:
            case 'moveit':
                ctl = moveit_control.MoveItPPControl(self)
            case 'hybrid':
                ctl = hybrid_control.HybridPPControl(self)

        pen = pen_detector.PenDetector(self)
        board = board_detector.BoardDetector(self)
        writer = write_planner.WritePlanner(self, ctl, board)
        self.fonts = font_trajectory.FontTrajectory(writer)
        self.grabber = grab_planner.GrabPlanner(self, ctl, pen)
