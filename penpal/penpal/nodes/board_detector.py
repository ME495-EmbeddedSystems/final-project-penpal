"""Detect pose + dimensions of a rectangular whiteboard using AprilTags."""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import transforms3d.quaternions as tquat

from penpal_interfaces.msg import BoardInfo
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker


class BoardDetector(Node):
    """Detects pose + dimensions of one whiteboard using two AprilTags."""

    def __init__(self) -> None:
        """Initialize board detector."""
        super().__init__('board_detector')

        self.T_base_camera: Optional[np.ndarray] = None

        marker_qos = QoSProfile(
            depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # board + tag geometry
        self.width: float = self.declare_parameter('board_width_m', 0.8).value
        self.height: float = self.declare_parameter(
            'board_height_m', 0.61
        ).value
        self.tag_size: float = self.declare_parameter('tag_size_m', 0.07).value

        # tag ids at known board corners
        self.tag_tl: int = self.declare_parameter('top_left_id', 0).value
        self.tag_br: int = self.declare_parameter('bottom_right_id', 1).value

        # detection topic
        self.tag_topic: str = self.declare_parameter(
            'tag_topic', '/detections'
        ).value

        # ---- Frames + calibration tag info ----
        # id of the calibration tag (on the table)
        self.calib_tag_id: int = self.declare_parameter(
            'calib_tag_id', 2
        ).value

        # known pose of calib tag in BASE frame: [x, y, z]
        base_calib_xyz = self.declare_parameter(
            'base_calib_tag_xyz',
            [-0.3, 0.0, 0.0],
        ).value

        # homogeneous transform T_base_calib_tag
        self.T_base_calib = np.eye(4)
        # need x,y,z,w
        # R_base_calib = Rotation.from_quat(base_calib_quat)
        R_base_calib = Rotation.from_euler('zyz', (90, 180, -90), degrees=True)

        self.T_base_calib[:3, :3] = R_base_calib.as_matrix()
        self.T_base_calib[:3, 3] = np.array(base_calib_xyz, dtype=float)
        self.get_logger().info(
            f'BASE->CALIB hardcoded transformation matrix: {self.T_base_calib}'
        )

        # only need to publish BASE -> CAMERA once
        self.camera_calibrated: bool = False
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        # camera intrinsics
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None

        # cached board pose/orientation for fallback when only one tag is seen
        self.R_board: Optional[np.ndarray] = None
        self.center_board: Optional[np.ndarray] = None
        self.board_visible: bool = False

        # ---------------- Subscriptions ----------------
        self.caminfo_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.cam_info_cb,
            1,
        )

        self.create_subscription(
            AprilTagDetectionArray,
            self.tag_topic,
            self.tag_cb,
            10,
        )

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
            Marker, 'whiteboard_outline', marker_qos
        )

        # write-space visualization
        self.write_space_pub = self.create_publisher(
            Marker,
            'penpal_write_space',
            marker_qos,
        )

        # debug line from BASE -> CALIB_TAG
        self._debug_pub = self.create_publisher(
            Marker,
            'calib_link',
            marker_qos,
        )

        self.get_logger().info('BoardDetector running')

    # ---------------- Camera intrinsics ----------------
    def cam_info_cb(self, msg: CameraInfo) -> None:
        """Cache camera intrinsics K, D from CameraInfo."""
        if self.K is not None:
            return

        # use for solvePnP
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.D = np.array(msg.d, dtype=float)

        self.get_logger().info('Camera intrinsics received')
        # we only need intrinsics once
        self.destroy_subscription(self.caminfo_sub)

    # --------------- SolvePnP helper -------------------
    def estimate_tag_pose(
        self,
        detection: AprilTagDetection,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate tag pose (R, t) in CAMERA frame from 4 corners using IPPE_SQUARE.

        Returns
        -------
        (R, t): if successful
            R: 3x3 rotation matrix (TAG frame -> CAMERA frame)
            t: shape (3,) translation vector (tag origin in CAMERA frame)
        None:
            if intrinsics missing or PnP fails

        """
        if self.K is None:
            return None

        if len(detection.corners) != 4:
            self.get_logger().warn(
                f'Tag id={detection.id} has {len(detection.corners)} corners, expected 4'
            )
            return None

        # pixel corners -> shape (4, 2)
        uv = np.array([[c.x, c.y] for c in detection.corners], dtype=float)

        # TAG frame 3D corners (square in z = 0 plane, centered at origin)
        s: float = self.tag_size / 2.0
        XYZ = np.array(
            [
                [-s, s, 0],
                [s, s, 0],
                [s, -s, 0],
                [-s, -s, 0],
            ],
            dtype=float,
        )

        success, rvec, tvec = cv2.solvePnP(
            XYZ,  # real-world coordinates of tag corners
            uv,  # pixel coordinates of corners
            self.K,
            self.D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not success:
            self.get_logger().warn(
                f'solvePnP failed for tag id={detection.id}'
            )
            return None

        # rotation matrix from TAG frame -> CAMERA frame
        R, _ = cv2.Rodrigues(rvec)
        # translation vector from tag origin -> CAMERA frame
        t = tvec.reshape(3)

        return R, t

    def _publish_base_camera_tf(self, T_base_camera: np.ndarray) -> None:
        """Publish BASE -> CAMERA transform as a static TF."""
        base_frame_id = 'base'
        # camera_frame_id = 'camera_link'
        camera_frame_id = 'camera_link'
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = base_frame_id
        tf.child_frame_id = camera_frame_id

        # translation
        tf.transform.translation.x = float(T_base_camera[0, 3])
        tf.transform.translation.y = float(T_base_camera[1, 3])
        tf.transform.translation.z = float(T_base_camera[2, 3])

        # rotation: convert R to quaternion
        R_bc = Rotation.from_matrix(T_base_camera[:3, :3])
        qx, qy, qz, qw = R_bc.as_quat(True)
        tf.transform.rotation.w = float(qw)
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)

        self._static_tf_broadcaster.sendTransform(tf)
        self.get_logger().info(
            f'Published static TF {base_frame_id} -> {camera_frame_id}: '
            f'\n\n{T_base_camera}'
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
    def tag_cb(self, msg: AprilTagDetectionArray) -> None:
        """Use TL + BR tags to compute board pose and outline."""
        if self.K is None:
            return

        # map id -> detection
        dets: Dict[int, AprilTagDetection] = {d.id: d for d in msg.detections}

        # ---------- Camera calibration ----------
        if not self.camera_calibrated and self.calib_tag_id in dets:
            calib = self.estimate_tag_pose(dets[self.calib_tag_id])
            if calib is not None:
                R_cam_calib, t_cam_calib = calib

                # build T_camera_calib_tag as 4x4 homogeneous
                T_camera_to_calib = np.eye(4)
                T_camera_to_calib[:3, :3] = R_cam_calib
                T_camera_to_calib[:3, 3] = t_cam_calib

                # T_base_camera = T_base_calib_tag * (T_camera_calib_tag)^-1
                T_base_to_camera = self.T_base_calib @ np.linalg.inv(
                    T_camera_to_calib
                )

                self.T_base_camera = T_base_to_camera
                self._publish_base_camera_tf(T_base_to_camera)
                self.camera_calibrated = True
                self.get_logger().info(
                    f'Camera calibrated using tag id={self.calib_tag_id}'
                )

                # draw a line in rviz from BASE -> CALIB_TAG
                self.publish_calibration_link(
                    msg.header,
                    t_cam_calib,
                )

        tl_pose = (
            self.estimate_tag_pose(dets[self.tag_tl])
            if self.tag_tl in dets
            else None
        )
        br_pose = (
            self.estimate_tag_pose(dets[self.tag_br])
            if self.tag_br in dets
            else None
        )

        # --- No tags visible: mark board as not visible and delete marker ---
        if tl_pose is None and br_pose is None:
            # reset sequence number
            self.sequence_number = 0

            if not self.board_visible:
                self.get_logger().debug('Board not visible (no tags).')
                self.board_visible = False

                m = Marker()
                m.header = msg.header
                m.ns = 'whiteboard'
                m.id = 0
                m.action = Marker.DELETE
                self.marker_pub.publish(m)

            return

        # --- Board geometry in BOARD frame ---
        W: float = self.width
        H: float = self.height
        S: float = self.tag_size
        hw, hh, hs = W / 2.0, H / 2.0, S / 2.0

        # BOARD frame convention: origin at center, +x right, +y down
        P_tag_tl_b = np.array([-hw + hs, -hh + hs, 0.0])  # TL tag center
        P_tag_br_b = np.array([hw - hs, hh - hs, 0.0])  # BR tag center

        # --- Choose R and center depending on which tags are in view ---
        if tl_pose is not None and br_pose is not None:
            # both tags visible
            R_tl, t_tl = tl_pose
            R_br, t_br = br_pose

            # center inferred from each tag separately
            center_from_tl = t_tl - R_tl @ P_tag_tl_b
            center_from_br = t_br - R_br @ P_tag_br_b
            center = 0.5 * (center_from_tl + center_from_br)

            # take average rotation via SVD on R_tl + R_br
            R_sum = R_tl + R_br
            U, _, Vt = np.linalg.svd(R_sum)
            R = U @ Vt

            # count number of visible tags
            n_tags = 2

            if self.board_visible:
                self.get_logger().debug('Board visible (both TL + BR).')
            self.board_visible = True

        elif tl_pose is not None:
            # only top-left tag visible: lock TL to top-left of board
            R_tl, t_tl = tl_pose
            R = R_tl
            center = t_tl - R @ P_tag_tl_b

            # count number of visible tags
            n_tags = 1

            if self.board_visible:
                self.get_logger().debug('Board visible (TL only).')
            self.board_visible = True

        else:
            # only bottom-right tag visible: lock BR to bottom-right of board
            R_br, t_br = br_pose
            R = R_br
            center = t_br - R @ P_tag_br_b

            # count number of visible tags
            n_tags = 1

            if self.board_visible:
                self.get_logger().debug('Board visible (BR only).')
            self.board_visible = True

        # cache for potential later use/debugging
        self.R_board = R
        self.center_board = center

        # ---- Publish PoseStamped ----
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = float(center[0])
        pose.pose.position.y = float(center[1])
        pose.pose.position.z = float(center[2])

        q = tquat.mat2quat(R)  # (w, x, y, z)
        pose.pose.orientation.w = float(q[0])
        pose.pose.orientation.x = float(q[1])
        pose.pose.orientation.y = float(q[2])
        pose.pose.orientation.z = float(q[3])

        self.pose_pub.publish(pose)

        # ---- Publish BoardInfo ----
        self.publish_board_info(
            msg.header,
            R,
            center,
            n_tags,
        )

        # ---- Publish outline markers ----
        self.publish_outline(msg.header, R, center)
        self.publish_write_space(msg.header, R, center)

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
            - origin at top left of the board
        """
        # can't publish anything unless we know where the base frame is
        if self.T_base_camera is None:
            return

        hw = self.width / 2.0
        hh = self.height / 2.0

        # top-left corner in BASE frame
        P_tl_b = np.array([-hw, hh, 0.0])
        top_left_c_camera = R @ P_tl_b + center
        homog_tl = np.array([*top_left_c_camera, 1])
        top_left_c = (self.T_base_camera @ homog_tl)[0:3]

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

        # hacky fix - flip R over in the plane of the board's z axis
        R = R @ Rotation.from_euler('z', 180, degrees=True).as_matrix()

        # bottom half in BOARD frame
        corners_b = np.array(
            [
                [-hw, -hh, 0.0],  # bottom-left
                [hw, -hh, 0.0],  # bottom-right
                [hw, 0.0, 0.0],  # top-right
                [-hw, 0.0, 0.0],  # top-left
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
    node = BoardDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
