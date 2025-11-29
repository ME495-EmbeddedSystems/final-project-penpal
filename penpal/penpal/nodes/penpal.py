"""
PenPal node.

TODO better docs.
"""

from typing import Any, List
from rclpy import Context, Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, qos_profile_rosout_default

import font_trajectory
import grab_planner
import pen_detector
import write_planner
from control import moveit_control, position_control


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
                ctl = position_control.PositionPPControl(self)

        pen = pen_detector.PenDetector(self)
        writer = write_planner.WritePlanner(self, ctl)
        self.fonts = font_trajectory.FontTrajectory(writer)
        self.grabber = grab_planner.GrabPlanner(self, ctl, pen)
