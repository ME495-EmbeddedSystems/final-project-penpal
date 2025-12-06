"""
PenPal node.

TODO better docs.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Any, List, Literal
from threading import Lock

import numpy as np
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, qos_profile_rosout_default
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.action.server import ServerGoalHandle
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from example_interfaces.srv import Trigger
from rclpy.timer import TimerInfo

from penpal_interfaces.action import WriteMessage
from penpal_interfaces.msg import BoardInfo as BoardInfoMsg

from penpal import font_trajectory
from penpal import grab_planner
from penpal import write_planner
from penpal.control import moveit_control, position_control
from penpal import ppstate


class PenPal(Node):
    """
    Controls a 7DoF FER arm to write conversational responses to a whiteboard.

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

    Publishers
    ---------
    move_group: motion & planning commands to MoveIt

    Subscribers
    ----------
    whiteboard_pose: 6dof pose of the whiteboard
    whiteboard_outline: markers for the whiteboard boundary

    Clients
    -------
    read_and_answer_board (QwenOCRNode): used to get chatbot responses to images.

    """

    @dataclass
    class Config:
        """Node configuration."""

        board_visibility_thresh_s: float = 1.0
        board_visibility_tags_thresh: int = 2
        write_control_type: str = 'mock'
        timer_freq_hz: float = 20.0

    def __init__(self) -> None:
        """Initialize the node."""
        super().__init__('PenPal')
        self.c = self.Config()

        self.declare_parameter(
            'write_control_type',
            self.c.write_control_type,
            ParameterDescriptor(
                description='Type of controller to use for writing. [moveit, hybrid, mock]'  # noqa: E501
            ),
        )
        self.c.write_control_type = (
            self.get_parameter('write_control_type')
            .get_parameter_value()
            .string_value
        )
        self.declare_parameter(
            'board_visibility_thresh_s',
            self.c.board_visibility_thresh_s,
            ParameterDescriptor(
                description='Duration of continuous board pose readings after '
                'which the board is declared to be visible'
            ),
        )
        self.c.board_visibility_thresh_s = (
            self.get_parameter('board_visibility_thresh_s')
            .get_parameter_value()
            .double_value
        )
        self.declare_parameter(
            'board_visibility_tags_thresh',
            self.c.board_visibility_tags_thresh,
            ParameterDescriptor(
                description='Number of apriltags that must be detected'
                'in order to qualify as visible '
            ),
        )
        self.c.board_visibility_tags_thresh = (
            self.get_parameter('board_visibility_tags_thresh')
            .get_parameter_value()
            .integer_value
        )

        match self.c.write_control_type:
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
        self._cbgroup_arm_users = MutuallyExclusiveCallbackGroup()
        """Callback group for anything that synchronously uses the arm."""

        state_lock = Lock()
        self._fsm = ppstate.ConvoFSM(
            state_lock, self.get_logger().get_child('FSM')
        )
        self._load_fonts(self._package_share / 'fonts')

        self._asrv_write_message = ActionServer(
            self,
            WriteMessage,
            'write_message',
            self._cb_execute_writemessage,
            callback_group=self._cbgroup_arm_users,
        )
        self._srv_wake = self.create_service(Trigger, 'wake', self._cb_wake)
        self._srv_sleep = self.create_service(Trigger, 'sleep', self._cb_sleep)
        self._tick = self.create_timer(
            1.0 / self.c.timer_freq_hz, self._cb_tick
        )
        self._c_ocr = self.create_client(Trigger, 'read_and_answer_board')
        self._sub_wbinfo = self.create_subscription(
            BoardInfoMsg, 'whiteboard_info', self._cb_wbinfo, 10
        )

        self.get_logger().info('PenPal node started.')

    def _cb_wbinfo(self, msg: BoardInfoMsg) -> None:
        """Handle BoardInfo callback."""
        # translate BoardInfoMsg to writeplanner's BoardInfo struct
        msg_pos = msg.pose.pose.position
        msg_ori = msg.pose.pose.orientation
        tl_pos = np.array([msg_pos.x, msg_pos.y, msg_pos.z])
        ori = R.from_quat([msg_ori.x, msg_ori.y, msg_ori.z, msg_ori.w])
        wa = np.array(
            [msg.writeable_area[0], msg.writeable_area[1]],
            [msg.writeable_area[2], msg.writeable_area[3]],
        )
        board = write_planner.BoardInfo(
            pos=tl_pos,
            ori=ori,
            width_m=msg.width_m,
            height_m=msg.height_m,
            writeable_area=wa,
        )
        self._writer.set_board_info(board)

    def _cb_tick(self, info: TimerInfo) -> None:
        """Handle periodic tasks. Timer callback."""
        s = self._fsm.get_state()
        match s:
            case ppstate.S.ASLEEP:
                # do nothing.
                pass
            case ppstate.S.ASLEEP_IN_USE:
                # do nothing.
                pass
            case ppstate.S.READY_TO_READ:
                # TODO check if the board's been visible for more
                # than the visibility threshold
                pass
            case ppstate.S.READING:
                pass
            case ppstate.S.READY_TO_WRITE:
                pass
            case ppstate.S.WRITING:
                pass
            case ppstate.S.WRITE_COMPLETE:
                pass
            case _:
                raise NotImplementedError(f'Unrecognized state {self._s}')

    def _load_fonts(self, fonts_dir: Path) -> None:
        """Load fonts from a directory."""
        self.get_logger().info(f'Loading fonts from {fonts_dir}...')
        for p in fonts_dir.iterdir():
            if p.is_file() and p.suffix.lower() in ['.ttf', '.otf']:
                self._fonts.add_font(p)
                self.get_logger().info(f'-- Added {p}.')

    def _cb_wake(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Handle wake service call."""
        if self._fsm.get_state() == ppstate.S.ASLEEP_IN_USE:
            self.get_logger().error('Cannot enter convo mode; arm is in use.')
            response.success = False
            return response
        self._fsm.transition(ppstate.E.WAKE)
        return response

    def _cb_sleep(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Handle sleep service call."""
        self._fsm.transition(ppstate.E.SLEEP)
        # TODO end the loop
        return response

    def _cb_execute_writemessage(
        self, goal_handle: ServerGoalHandle
    ) -> WriteMessage.Result:
        """Handle WriteMessage action execute callback."""
        req: WriteMessage.Goal = goal_handle.request

        self.get_logger().debug(f'WriteMessage called with text: {req.text}')

        # transition must be called first for concurrency
        self._fsm.transition(ppstate.E.WRITEMESSAGE_CALLED)
        if self._fsm.is_awake():
            self.get_logger().error(
                'Call to WriteMessage not permitted while PenPal '
                'is awake (conversational mode active).'
            )
            goal_handle.abort()
            res = WriteMessage.Result()
            res.unwritten_characters = req.text
            return res

        self.get_logger().info('Writing message to board...')

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
                self._fsm.transition(ppstate.E.WRITE_INCOMPLETE)
            else:
                self._fsm.transition(ppstate.E.WRITE_SUCCEEDED)

            goal_handle.succeed()
            res = WriteMessage.Result()
            res.unwritten_characters = cstr
            return res
        except Exception as err:
            tb = traceback.format_exc()
            self.get_logger().error(
                f'{type(err).__name__} during async tasks: {err}\n\nTraceback:\n{tb}'
            )
            self._fsm.transition(ppstate.E.WRITE_FAILED)
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
