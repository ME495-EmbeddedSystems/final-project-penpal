"""
PenPal node.

TODO better docs.
"""

import asyncio
from pathlib import Path
import traceback
from typing import Any, List, Literal
from threading import Lock

from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, qos_profile_rosout_default
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.action.server import ServerGoalHandle
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from penpal_interfaces.action import WriteMessage

from penpal import font_trajectory
from penpal import grab_planner
from penpal import write_planner
from penpal.control import moveit_control, position_control
from penpal import ppstate


class PenPal(Node):
    """
    Controls a 7DoF FER arm to write conversational responses to a whiteboard.

    Publishers
    ---------
    move_group: motion & planning commands to MoveIt

    Actions
    -------
    WriteMessage: write a message to the whiteboard.
        Can only handle one of these requests at a time, and will reject
        them if awake.

    Services
    --------
    Wake: Start the live conversation feature. When the whiteboard is
        clearly visible, read what's written on it and get a response to it
        using the chatbot node, then write the response to the board.
        Then wait for the board to be taken away & repeat the process.
    Sleep: Stop the live conversation feature, making the node available for
        explicit WriteMessage requests.

    Parameters
    ----------
    write_control_type: type of controller to use for writing

    Subscribers
    ----------
    whiteboard_pose: 6dof pose of the whiteboard
    whiteboard_outline: markers for the whiteboard boundary

    Clients
    -------
    read_and_answer_board (QwenOCRNode): used to get chatbot responses to images.

    """

    def __init__(self) -> None:
        """Initialize the node."""
        super().__init__('PenPal')

        self.declare_parameter(
            'write_control_type',
            'mock',
            ParameterDescriptor(
                description='Type of controller to use for writing. [moveit, hybrid, mock]'  # noqa: E501
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
        self._cbgroup = MutuallyExclusiveCallbackGroup()

        state_lock = Lock()
        self._fsm = ppstate.PPFSM(
            state_lock, self.get_logger().get_child('FSM')
        )

        self._asrv_write_message = ActionServer(
            self,
            WriteMessage,
            'write_message',
            self._cb_execute_writemessage,
            callback_group=self._cbgroup,
        )

        self._load_fonts(self._package_share / 'fonts')

        self.get_logger().info('PenPal node started.')

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

            self.get_logger().info('Finished writing message!')
            cstr = ''.join([c.char for c in unwritten_chars])

            if len(unwritten_chars) > 0:
                self.get_logger().warning(
                    f'Unable to write the end of the message: {cstr}'
                )

            goal_handle.succeed()
            res = WriteMessage.Result()
            res.unwritten_characters = cstr
            return res
        except Exception as err:
            tb = traceback.format_exc()
            self.get_logger().error(
                f'{type(err).__name__} during async tasks: {err}\n\nTraceback:\n{tb}'
            )
            goal_handle.abort()
            res = WriteMessage.Result()
            res.unwritten_characters = req.text
            return res


def main():
    """Node entry point."""
    import rclpy

    rclpy.init()
    penpal = PenPal()
    # need >=2 threads so we can still receive subscription
    # callbacks while in synchronous writing code.
    ex = MultiThreadedExecutor(num_threads=2)
    try:
        ex.add_node(penpal)
        ex.spin()
    finally:
        penpal.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
