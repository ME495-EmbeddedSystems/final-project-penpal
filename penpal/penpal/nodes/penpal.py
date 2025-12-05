"""
PenPal node.

TODO better docs.
"""

import asyncio
from typing import Any, List
from rclpy import Context, Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, qos_profile_rosout_default
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle

from penpal_interfaces.action import WriteMessage

from penpal import font_trajectory
from penpal import grab_planner
from penpal import pen_detector
from penpal import write_planner
from penpal.control import moveit_control, position_control, pp_control


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
            case _:
                raise NotImplementedError

        self._pen = pen_detector.PenDetector(self)
        self._writer = write_planner.WritePlanner(self, ctl)
        self._fonts = font_trajectory.FontTrajectory()
        self._grabber = grab_planner.GrabPlanner(self, ctl, self._pen)
        self._loop = asyncio.get_event_loop()

        self._asrv_write_message = ActionServer(
            self, WriteMessage, 'write_message', self._cb_execute_writemessage
        )

    def _cb_execute_writemessage(
        self, goal_handle: ServerGoalHandle
    ) -> WriteMessage.Result:
        req: WriteMessage.Goal = goal_handle.request
        self.get_logger().info('Writing message to board: ' + req.text)

        chars = self._fonts.write_text(
            req.text, req.font_name, req.font_size_mm, req.pen_thickness_mm
        )
        write_task = self._loop.create_task(
            self._writer.write_characters(
                chars, self._fonts.c.line_spacing_factor
            )
        )
        try:
            unwritten_chars = self._loop.run_until_complete(write_task)

            goal_handle.succeed()
            res = WriteMessage.Result()
            res.unwritten_characters = ''.join(
                [c.char for c in unwritten_chars]
            )
            return res
        except Exception as err:
            self.get_logger().error(f'Error during async tasks {err}')
            goal_handle.abort()
            res = WriteMessage.Result()
            res.unwritten_characters = req.text
            return res
