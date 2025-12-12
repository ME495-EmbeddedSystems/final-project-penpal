"""Mock node for publishing BoardInfo to PenPal."""

from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import transforms3d.quaternions as tquat
from scipy.spatial.transform import Rotation

from penpal_interfaces.msg import BoardInfo
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from example_interfaces.srv import Trigger
from tf2_ros import StaticTransformBroadcaster


class MockBoardDetector(Node):
    """Detects pose + dimensions of one whiteboard using two AprilTags."""

    def __init__(self) -> None:
        """Initialize board detector."""
        super().__init__('mock_board_detector')

        # board + tag geometry
        self.width: float = self.declare_parameter('board_width_m', 0.8).value
        self.height: float = self.declare_parameter(
            'board_height_m', 0.61
        ).value
        self.tag_size: float = self.declare_parameter('tag_size_m', 0.07).value

        # tag ids at known board corners
        self.tag_tl: int = self.declare_parameter('top_left_id', 0).value
        self.tag_br: int = self.declare_parameter('bottom_right_id', 1).value

        # ---- Frames + calibration tag info ----
        self.base_frame_id: str = self.declare_parameter(
            'base_frame_id', 'base'
        ).value
        self.camera_frame_id: str = self.declare_parameter(
            'camera_frame_id', 'camera_link'
        ).value

        # id of the calibration tag (on the table)
        self.calib_tag_id: int = self.declare_parameter(
            'calib_tag_id', 2
        ).value

        # known pose of calib tag in BASE frame: [x, y, z]
        base_calib_xyz = self.declare_parameter(
            'base_calib_tag_xyz',
            [-0.3, 0.0, 0.0],
        ).value

        # known orientation of calib tag in BASE frame, [qx, qy, qz, qw]
        base_calib_quat = self.declare_parameter(
            'base_calib_tag_quat',
            [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ).value

        # homogeneous transform T_base_calib_tag
        self.T_base_calib = np.eye(4)
        # transforms3d expects (w, x, y, z)
        qw = float(base_calib_quat[3])
        qx = float(base_calib_quat[0])
        qy = float(base_calib_quat[1])
        qz = float(base_calib_quat[2])
        self.T_base_calib[:3, :3] = tquat.quat2mat([qw, qx, qy, qz])
        self.T_base_calib[:3, 3] = np.array(base_calib_xyz, dtype=float)

        # only need to publish BASE -> CAMERA once
        self.camera_calibrated: bool = False
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        # cached board pose/orientation for fallback when only one tag is seen
        self.R_board: Optional[np.ndarray] = None
        self.center_board: Optional[np.ndarray] = None
        self.board_visible: bool = False

        # ------------------ Publishers ------------------
        self.pose_pub = self.create_publisher(
            PoseStamped, 'whiteboard_pose', 10
        )

        self.board_info_pub = self.create_publisher(
            BoardInfo, 'board_info', 10
        )
        self.sequence_number: int = 0

        # --------------- Marker Publishers ---------------
        self.marker_pub = self.create_publisher(
            Marker, 'whiteboard_outline', 10
        )

        # write-space visualization
        self.write_space_pub = self.create_publisher(
            Marker,
            'penpal_write_space',
            10,
        )

        # debug line from BASE -> CALIB_TAG
        self._debug_pub = self.create_publisher(
            Marker,
            'calib_link',
            10,
        )

        self.get_logger().info('BoardDetector running')

        # --------------- MOCK STUFF ---------------
        self.T_base_camera = np.eye(4)
        self.T_base_camera[0:3, 3] = [2, 1, 0]

        pub_freq_hz = 3
        self.create_timer(1 / pub_freq_hz, self.tag_cb)

        self.create_service(
            Trigger, 'toggle_publish_board_info', self._cb_toggle_publish
        )
        self._is_publishing_mock = True

    def _cb_toggle_publish(
        self, req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        """Restart the sequence from 0. mocking helper."""
        self._is_publishing_mock = not self._is_publishing_mock
        self.get_logger().info(
            f'Mock - setting publish to {
                "ON" if self._is_publishing_mock else "OFF"
            }'
        )
        self.sequence_number = 0
        resp.success = True
        return resp

    def _publish_base_camera_tf(self, T_base_camera: np.ndarray) -> None:
        """Publish BASE -> CAMERA transform as a static TF."""
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.base_frame_id
        tf.child_frame_id = self.camera_frame_id

        # translation
        tf.transform.translation.x = float(T_base_camera[0, 3])
        tf.transform.translation.y = float(T_base_camera[1, 3])
        tf.transform.translation.z = float(T_base_camera[2, 3])

        # rotation: convert R to quaternion
        R_bc = T_base_camera[:3, :3]
        qw, qx, qy, qz = tquat.mat2quat(R_bc)
        tf.transform.rotation.w = float(qw)
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)

        self._static_tf_broadcaster.sendTransform(tf)
        self.get_logger().debug(
            f'Published static TF {self.base_frame_id} -> {self.camera_frame_id}'
        )

    def publish_calibration_link(
        self,
        header,
        t_cam_calib: np.ndarray,
    ) -> None:
        """
        Publish a line marker from the robot base to the calibration tag.

        Express in the CAMERA frame so it shows in both 3D and camera views.
        """
        if self.T_base_camera is None:
            return

        # invert to get T_camera_base
        T_cam_base = np.linalg.inv(self.T_base_camera)
        # base origin in camera frame
        p_base_cam = T_cam_base @ np.array([0.0, 0.0, 0.0, 1.0])

        marker = Marker()
        # same frame as the board outline/camera image
        marker.header = header
        marker.ns = 'calibration_link'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.scale.x = 0.01
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # endpoints in CAMERA frame
        p0 = Point()
        p0.x = float(p_base_cam[0])
        p0.y = float(p_base_cam[1])
        p0.z = float(p_base_cam[2])

        p1 = Point()
        p1.x = float(t_cam_calib[0])
        p1.y = float(t_cam_calib[1])
        p1.z = float(t_cam_calib[2])

        marker.points = [p0, p1]

        self.marker_pub.publish(marker)

    # ---------------- Main callback --------------------
    def tag_cb(self) -> None:
        """Use TL + BR tags to compute board pose and outline."""
        self._publish_base_camera_tf(self.T_base_camera)

        # tl_pose = (
        #     self.estimate_tag_pose(dets[self.tag_tl])
        #     if self.tag_tl in dets
        #     else None
        # )
        # br_pose = (
        #     self.estimate_tag_pose(dets[self.tag_br])
        #     if self.tag_br in dets
        #     else None
        # )

        # # --- No tags visible: mark board as not visible and delete marker ---
        # if tl_pose is None and br_pose is None:
        #     # reset sequence number
        #     self.sequence_number = 0

        #     if getattr(self, 'board_visible', False):
        #         self.get_logger().info('Board not visible (no tags).')
        #         self.board_visible = False

        #         m = Marker()
        #         m.header = msg.header
        #         m.ns = 'whiteboard'
        #         m.id = 0
        #         m.action = Marker.DELETE
        #         self.marker_pub.publish(m)

        #     return

        # --- Board geometry in BOARD frame ---
        W: float = self.width
        H: float = self.height
        S: float = self.tag_size
        hw, hh, hs = W / 2.0, H / 2.0, S / 2.0

        # # BOARD frame convention: origin at center, +x right, +y down
        # P_tag_tl_b = np.array([-hw + hs, -hh + hs, 0.0])  # TL tag center
        # P_tag_br_b = np.array([hw - hs, hh - hs, 0.0])  # BR tag center

        # CONOR'S MOCK STUFF
        mock_header = Header()
        mock_header.frame_id = 'base'
        mock_header.stamp = self.get_clock().now().to_msg()

        center = np.array([0.1, -0.4, 0.6])  # board center in base frame
        rot = Rotation.from_euler('xz', (90, 180), degrees=True)
        R = rot.as_matrix()
        n_tags = 2

        # # ---- Publish PoseStamped ----
        # pose = PoseStamped()
        # pose.header = mock_header
        # pose.pose.position.x = float(center[0])
        # pose.pose.position.y = float(center[1])
        # pose.pose.position.z = float(center[2])

        # q = tquat.mat2quat(R)  # (w, x, y, z)
        # pose.pose.orientation.w = float(q[0])
        # pose.pose.orientation.x = float(q[1])
        # pose.pose.orientation.y = float(q[2])
        # pose.pose.orientation.z = float(q[3])

        # self.pose_pub.publish(pose)

        # ---- Publish BoardInfo ----
        if self._is_publishing_mock:
            self.publish_board_info(
                mock_header,
                R,
                center,
                n_tags,
            )

            # ---- Publish outline markers ----
            self.publish_outline(mock_header, R, center)
            self.publish_write_space(mock_header, R, center)

    # ---------------- Publish BoardInfo ----------------
    def publish_board_info(
        self,
        header: Header,
        R: np.ndarray,
        center: np.ndarray,
        n_tags: int,
    ) -> None:
        """
        Publish BoardInfo message.

        pose:
            Pose of the TOP-LEFT corner of the board in the CAMERA frame.
            Orientation = board orientation (R).

        writeable_area:
            Bottom half of the board, in board-plane coordinates with:
            - origin at top left of board
        """
        hw = self.width / 2.0
        hh = self.height / 2.0

        # top-left corner in CAMERA frame
        P_tl_b = np.array([-hw, hh, 0.0])
        top_left_c = R @ P_tl_b + center

        # fill board info message
        msg = BoardInfo()

        # pose of top-left corner
        msg.pose = PoseStamped()
        msg.pose.header = header
        msg.pose.pose.position.x = float(top_left_c[0])
        msg.pose.pose.position.y = float(top_left_c[1])
        msg.pose.pose.position.z = float(top_left_c[2])

        q = tquat.mat2quat(R)  # (w, x, y, z)
        msg.pose.pose.orientation.w = float(q[0])
        msg.pose.pose.orientation.x = float(q[1])
        msg.pose.pose.orientation.y = float(q[2])
        msg.pose.pose.orientation.z = float(q[3])

        # board dimensions
        msg.width_m = float(self.width)
        msg.height_m = float(self.height)

        # writespace in board coordinates:
        # origin at top left of board: +x right, -y down
        # bottom edge: y = -height, midline: y = -hh
        x_tl = 0.0
        y_tl = -hh  # top of writeable strip
        x_br = msg.width_m
        y_br = -msg.height_m  # bottom of board

        msg.writeable_area = [
            float(x_tl),
            float(y_tl),
            float(x_br),
            float(y_br),
        ]

        # tag + sequence info
        msg.n_tags = int(n_tags)

        self.sequence_number += 1
        msg.sequence_number = int(self.sequence_number)

        self.board_info_pub.publish(msg)

    # --------------- Visualization ---------------------
    def publish_outline(
        self,
        header: Header,
        R: np.ndarray,
        center: np.ndarray,
    ) -> None:
        """Draw board rectangle in CAMERA frame using board pose (R, center)."""
        hw, hh = self.width / 2.0, self.height / 2.0

        # board corners in BOARD frame with origin at center
        corners_b = np.array(
            [
                [-hw, -hh, 0.0],  # bottom-left
                [hw, -hh, 0.0],  # bottom-right
                [hw, hh, 0.0],  # top-right
                [-hw, hh, 0.0],  # top-left
            ],
            dtype=float,
        ).T

        # transform to CAMERA frame: corners_c = R * corners_b + center
        corners_c = R @ corners_b + center.reshape(3, 1)

        m = Marker()
        m.header = header
        m.ns = 'whiteboard'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.01
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0

        for i in range(4):
            pt = Point()
            pt.x = float(corners_c[0, i])
            pt.y = float(corners_c[1, i])
            pt.z = float(corners_c[2, i])
            m.points.append(pt)

        m.points.append(m.points[0])

        self.marker_pub.publish(m)

    def publish_write_space(
        self,
        header: Header,
        R: np.ndarray,
        center: np.ndarray,
    ) -> None:
        """
        Draw the penpal write space = bottom half of the board.

        BOARD frame convention matches publish_outline: origin at center,
        width along x, height along y. We take y from -hh (bottom edge)
        up to 0 (midline) as the write-space.
        """
        hw, hh = self.width / 2.0, self.height / 2.0

        # bottom half in BOARD frame
        corners_b = np.array(
            [
                [-hw, -hh, 0.0],  # bottom-left
                [hw, -hh, 0.0],  # bottom-right
                [hw, 0.0, 0.0],  # mid-right
                [-hw, 0.0, 0.0],  # mid-left
            ],
            dtype=float,
        ).T

        # transform to CAMERA frame
        corners_c = R @ corners_b + center.reshape(3, 1)

        m = Marker()
        m.header = header
        m.ns = 'penpal_write_space'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.01
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0

        for i in range(4):
            pt = Point()
            pt.x = float(corners_c[0, i])
            pt.y = float(corners_c[1, i])
            pt.z = float(corners_c[2, i])
            m.points.append(pt)

        m.points.append(m.points[0])

        self.write_space_pub.publish(m)


def main(args=None) -> None:
    """Spin the node."""
    rclpy.init(args=args)
    node = MockBoardDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
