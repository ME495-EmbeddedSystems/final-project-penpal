"""
PenPal node.

Authors: Conor
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import traceback
from threading import Lock
import json

import concurrent.futures
import numpy as np
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
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
from penpal import freespace_planner
from penpal import write_planner
from penpal.control import (
    moveit_control,
    impedance_control,
    moveit_control_freespace,
)
from penpal import ppstate
from penpal.utils import LockedString


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
    convo_font_name: font to use in writing
    board_visibility_thresh_s
    board_visibility_tags_thresh

    Publishers
    ---------
    move_group: motion & planning commands to MoveIt

    Subscribers
    ----------
    board_info: pose & metadata about the whiteboard

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
        timer_freq_hz: float = 10.0

        convo_font_size_mm: float = 30.0
        convo_pen_thickness_mm: float = 2.0
        convo_font_name: str = 'Roboto-Regular'

        workspace_dimensions_m: np.ndarray = field(
            default_factory=lambda: np.array([1.0, 1.0])
        )
        """
        Dimensions of the robot's cylindrical workspace in meters.

        This is represented as a cylinder centered at the origin
        of the base frame, and resting on the base xy plane in z
        (since we can't reach under the table)

        This is only used to heuristically evaluate whether the board
        is reachable; a final determination of actual reachability is
        left to MoveIt/franka's IK solvers. So this should be a
        conservative definition.

        format: (radius_m, height_m)
        """

    def __init__(self) -> None:
        """Initialize the node."""
        super().__init__('PenPal')
        self.c = self.Config()

        self.declare_parameter(
            'write_control_type',
            self.c.write_control_type,
            ParameterDescriptor(
                description='Type of controller to use for writing. [moveit, impedance, mock]'  # noqa: E501
            ),
        )
        self.c.write_control_type = (
            self.get_parameter('write_control_type')
            .get_parameter_value()
            .string_value
        )

        self.declare_parameter(
            'convo_font_name',
            self.c.convo_font_name,
            ParameterDescriptor(
                description='Font used in conversational mode.'
            ),
        )
        self.c.convo_font_name = (
            self.get_parameter('convo_font_name')
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
            case 'impedance':
                ctl = impedance_control.ImpedancePPControl(self)
            case 'mock':
                from penpal.integration_tests.int_test_write_planner import (
                    MockController,
                )

                ctl = MockController(self)
            case _:
                raise NotImplementedError

        self._write_planner = write_planner.WritePlanner(self, ctl)
        self._fonts = font_trajectory.FontTrajectory()

        # grab planner will always use a moveit controller instance
        # it _cannot_ be used at the same time as the other one, and this
        # isn't hard-enforced in the controller code, so we must take care
        # to enforce this in our logic here. We use the FSM formalism for this.
        grab_ctl = moveit_control_freespace.FreeSpaceMoveItPPControl(self)
        self._fplanner = freespace_planner.FreespacePlanner(self, grab_ctl)

        # thread pool executor for long running arm tasks
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
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
        self._srv_sleep = self.create_service(
            Trigger, 'grab_pen', self._cb_grab_pen
        )
        self._tick = self.create_timer(
            1.0 / self.c.timer_freq_hz, self._cb_tick
        )
        self._c_ocr = self.create_client(Trigger, 'read_and_answer_board')
        self._sub_wbinfo = self.create_subscription(
            BoardInfoMsg, 'board_info', self._cb_wbinfo, 10
        )

        # bookkeeping vars
        self._board_sequence_no = 0
        self._board_sequence_start_t = None
        self._board_last_reading_t = None
        self._prev_state = self._fsm.get_state()
        self._text_to_write = LockedString()

        self.get_logger().info('PenPal node started.')

    def _cb_wbinfo(self, msg: BoardInfoMsg) -> None:
        """Handle BoardInfo callback."""
        # translate BoardInfoMsg's data to the WritePlanner
        msg_pos = msg.pose.pose.position
        msg_ori = msg.pose.pose.orientation
        tl_pos = np.array([msg_pos.x, msg_pos.y, msg_pos.z])
        ori = R.from_quat([msg_ori.x, msg_ori.y, msg_ori.z, msg_ori.w])
        wa = np.array([float(val) for val in msg.writeable_area])
        wa = wa.reshape(2, 2)

        board = write_planner.BoardInfo(
            pos=tl_pos,
            ori=ori,
            width_m=msg.width_m,
            height_m=msg.height_m,
            writeable_area=wa,
        )
        self._write_planner.set_board_info(board)

        # bookkeep variables to help with determination of last valid reading
        now = self.get_clock().now()
        if (
            msg.sequence_number <= self._board_sequence_no
            or self._board_sequence_start_t is None
        ):
            # we've restarted the sequence due to a bad reading.
            # (or somehow the messages arrived out of order--is that
            # possible in ros2? unsure. if it's a big issue i'll deal w/ it)
            self._board_sequence_start_t = now
        else:
            if msg.n_tags < self.c.board_visibility_tags_thresh:
                # this counds as a bad reading cuz the number of tags
                # is too low.
                self._board_sequence_start_t = now
            else:
                # valid reading
                self._board_last_reading_t = now

        self._board_sequence_no = msg.sequence_number

    def board_is_visible(self) -> bool:
        """
        Return true if board is visible.

        Applies the relevant thresholds to evaluate this.
        """
        now_s = self.get_clock().now().nanoseconds / 1.0e9

        first_info_received = (
            self._board_last_reading_t is not None
            and self._board_sequence_start_t is not None
        )

        if first_info_received:
            last_valid_s = self._board_last_reading_t.nanoseconds / 1.0e9  # type: ignore
            seq_start_s = self._board_sequence_start_t.nanoseconds / 1.0e9  # type: ignore
            t_since_valid = now_s - last_valid_s
            t_since_invalid = now_s - seq_start_s
            if (
                t_since_valid <= self.c.board_visibility_thresh_s
                and t_since_invalid > self.c.board_visibility_thresh_s
            ):
                return True

        return False

    def board_is_in_workspace(self) -> bool:
        """Return true if the board is in range of the arm."""
        if self.board_is_visible():
            rthresh = self.c.workspace_dimensions_m[0]
            hthresh = self.c.workspace_dimensions_m[1]
            board = self._write_planner.get_latest_board_info()

            # board's position is already in base frame.
            # evaluate if all 4 corners of the writing area are in reach.
            corners = board.get_writeable_area_corners_world_frame()
            for i in range(corners.shape[0]):
                corner = corners[i]
                r = np.linalg.norm(corner[0:2])
                h = corner[2]
                is_in_workspace = r < rthresh and (h > 0 and h < hthresh)
                if not is_in_workspace:
                    return False
            return True

        return False

    def _run_async_worker_in_thread(self, coroutine_func):
        """Run an async function using a new event loop. Intended for use in worker thread."""
        try:
            result = asyncio.run(coroutine_func)
            return result
        except Exception as err:  # noqa: B902
            tb = traceback.format_exc()
            self.get_logger().error(
                f'{type(err).__name__} while running event loop'
                f': {err}\n\nTraceback:\n{tb}'
            )

    def schedule_in_worker(self, coroutine_func) -> concurrent.futures.Future:
        """Schedule an async function into the worker thread."""
        return self._executor.submit(
            self._run_async_worker_in_thread, coroutine_func
        )

    async def _perform_home(self) -> None:
        """Perform async home."""
        await self._fplanner.ctl.configure()
        await self._fplanner.home_arm()

    def worker_home(self) -> None:
        """Send the robot to the home position in the worker thread."""
        self.schedule_in_worker(self._perform_home())

    async def _perform_write(
        self,
        chars: list[font_trajectory.Character],
    ) -> None:
        """Perform the actual write sequence."""
        await self._fplanner.ctl.configure()
        await self._fplanner.move_to_board(
            self._write_planner.get_latest_board_info(),
            self._write_planner.c.off_board_height_m,
        )

        await self._write_planner.control.configure()
        await self._write_planner.write_characters(
            chars, self._fonts.c.line_spacing_factor
        )

    def worker_write(
        self,
        text: str,
        font_name: str,
        font_size_mm: float,
        pen_thickness_mm: float,
    ) -> list[write_planner.Character]:
        """Use the worker thread to write with the robot."""
        self.get_logger().info(f'Writing message "{text}" to board...')
        self._fsm.transition(ppstate.E.WRITE_STARTED)
        chars = self._fonts.write_text(
            text, font_name, font_size_mm, pen_thickness_mm
        )
        future = self.schedule_in_worker(self._perform_write(chars))

        # block here for now.
        # TODO figure out how to not need to block.
        unwritten_chars: list[write_planner.Character] = future.result()  # type: ignore

        self.get_logger().info(f'Finished writing message "{text}"!')
        cstr = ''.join([c.char for c in unwritten_chars])

        if len(unwritten_chars) > 0:
            self.get_logger().warning(
                f'Unable to write the end of the message: {cstr}'
            )
            self._fsm.transition(ppstate.E.WRITE_INCOMPLETE)
        else:
            self._fsm.transition(ppstate.E.WRITE_SUCCEEDED)

        return unwritten_chars

    async def _perform_trigger_vlm(self) -> None:
        """Actual async function to trigger the vlm and wait for the response."""
        resp: Trigger.Response = await self._c_ocr.call_async(
            Trigger.Request()
        )  # type: ignore
        self._logger.debug(f'Received payload from VLM: {resp.message}')
        payload = json.loads(resp.message)

        self._text_to_write.text = payload['answer']
        self._fsm.transition(ppstate.E.OCR_VLM_TEXT_RECEIVED)

    def worker_trigger_vlm(self) -> None:
        """Use the worker thread to get text to write from the VLM."""
        self.schedule_in_worker(self._perform_trigger_vlm())
        self._fsm.transition(ppstate.E.OCR_VLM_TRIGGERED)

    async def _perform_startup_actions(self) -> None:
        """Perform startup actions in worker thread."""
        await asyncio.sleep(3.0)
        await self._fplanner.ctl.remove_pen()
        await self._fplanner.reset_gripper()
        await self._fplanner.home_arm()

    def worker_startup_actions(self) -> None:
        """Perform startup actions, blocking."""
        future = self.schedule_in_worker(self._perform_startup_actions())
        future.result(20.0)

    def _cb_tick(self, info: TimerInfo) -> None:
        """Handle periodic tasks. Timer callback."""
        # non-state specific logic
        if self.board_is_visible():
            self._fsm.transition(ppstate.E.BOARD_VISIBLE)
        else:
            self._fsm.transition(ppstate.E.BOARD_NOT_VISIBLE)

        if self.board_is_in_workspace():
            self._fsm.transition(ppstate.E.BOARD_IN_WORKSPACE)

        # state-specific logic & on-transition functions
        s = self._fsm.get_state()
        enter = s != self._prev_state
        match s:
            case ppstate.S.STARTUP:
                # go straight to asleep after a reset.
                self.worker_startup_actions()
                self._fsm.transition(ppstate.E.STARTUP_COMPLETE)

            case ppstate.S.ASLEEP:
                # wait to be woken up
                pass
            case ppstate.S.ASLEEP_IN_USE:
                # wait to be woken up
                pass
            case ppstate.S.READY_TO_READ:
                if enter:
                    self.worker_home()
                # nothing to do; we just wait for visibility
                pass
            case ppstate.S.READING:
                if enter:
                    self.worker_trigger_vlm()
            # otherwise we just wait around for the VLM to get back to us
            case ppstate.S.READY_TO_WRITE:
                # nothing to do; wait for board to enter workspace
                pass
            case ppstate.S.WRITING:
                if enter:
                    text = self._text_to_write.text
                    if text is None:
                        raise ValueError(
                            'No text available to write. This should be unreachable!'
                        )
                    self.worker_write(
                        text,
                        self.c.convo_font_name,
                        self.c.convo_font_size_mm,
                        self.c.convo_pen_thickness_mm,
                    )
                # wait around for writing to finish
            case ppstate.S.WRITE_COMPLETE:
                if enter:
                    self.worker_home()
                # wait around for the board to be hidden & arm to be homed
                # again.
                pass
            case _:
                raise NotImplementedError(f'Unrecognized state {self._s}')

        self._prev_state = s

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
            errstr = 'Cannot enter convo mode; arm is in use.'
            self.get_logger().error(errstr)
            response.success = False
            response.message = errstr
            return response
        self._fsm.transition(ppstate.E.WAKE)
        response.success = True
        response.message = (
            'Awake. Display board with writing to begin conversation'
        )
        return response

    async def _perform_grab_pen(self) -> None:
        """Grab the pen; called in the worker thread."""
        await self._fplanner.ctl.configure()
        await self._fplanner.grab_pen()
        self._fsm.transition(ppstate.E.GRAB_PEN_COMPLETE)

    def _cb_grab_pen(
        self, req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        """Handle grab pen service call."""
        self._fsm.transition(ppstate.E.GRAB_PEN_CALLED)
        self.schedule_in_worker(self._perform_grab_pen())
        resp.success = True
        resp.message = 'Retrieving pen.'
        return resp

    def _cb_sleep(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Handle sleep service call."""
        self._fsm.transition(ppstate.E.SLEEP)
        response.success = True
        response.message = 'Sleeping.'
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

        try:
            unwritten_chars = self.worker_write(
                req.text, req.font_name, req.font_size_mm, req.pen_thickness_mm
            )
            cstr = (
                ''.join([c.char for c in unwritten_chars])
                if unwritten_chars is not None
                else None
            )

            goal_handle.succeed()
            res = WriteMessage.Result()
            res.unwritten_characters = cstr
            return res

        except Exception:  # noqa: B902
            # we already make a fuss about this in the worker_write function.
            # we can just swallow this.
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
