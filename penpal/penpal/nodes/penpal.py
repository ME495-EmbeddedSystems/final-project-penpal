"""
PenPal node.

TODO better docs.
"""

import asyncio
from pathlib import Path
from typing import Any, List, Literal

from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, qos_profile_rosout_default
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from ament_index_python.packages import get_package_share_directory

from penpal_interfaces.action import WriteMessage

from penpal import font_trajectory
from penpal import grab_planner
from penpal import write_planner
from penpal.control import moveit_control, position_control


class PenPal(Node):
    """
    PenPal node.

    TODO better docs.
    """

    def __init__(self) -> None:
        """Initialize the node."""
        super().__init__('PenPal')

        # todo get this from a parameter
        self.declare_parameter(
            'write_control_type',
            'mock',
            ParameterDescriptor(
                description='Type of controller to use for writing.'
            ),
        )
        write_control_type = (
            self.get_parameter('write_control_type')
            .get_parameter_value()
            .string_value
        )

        match write_control_type:
            case 'moveit':
                ctl = moveit_control.MoveItPPControl(self)
            case 'hybrid':
                ctl = position_control.PositionPPControl(self)
            case 'mock':
                from penpal.integration_tests.int_test_write_planner import (
                    MockController,
                )

                ctl = MockController(self)
            case _:
                raise NotImplementedError

        self._writer = write_planner.WritePlanner(self, ctl)
        self._fonts = font_trajectory.FontTrajectory()
        self._grabber = grab_planner.GrabPlanner(self, ctl)
        self._loop = asyncio.get_event_loop()
        self._package_share = Path(get_package_share_directory('penpal'))

        self._asrv_write_message = ActionServer(
            self, WriteMessage, 'write_message', self._cb_execute_writemessage
        )

        self._load_fonts(self._package_share / 'fonts')

    def _load_fonts(self, fonts_dir: Path) -> None:
        """Load fonts from a directory."""
        self.get_logger().info(f'Loading fonts from {fonts_dir}...')
        for p in fonts_dir.iterdir():
            if p.is_file() and p.suffix.lower() in ['.ttf', '.otf']:
                self._fonts.add_font(p)
                self.get_logger().info(f'-- Added {p}.')

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
            self.get_logger().error(
                f'{type(err).__name__} during async tasks: {err}'
            )
            goal_handle.abort()
            res = WriteMessage.Result()
            res.unwritten_characters = req.text
            return res


def main():
    """Node entry point."""
    import rclpy

    rclpy.init()
    node = PenPal()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
