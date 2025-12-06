"""PenDetector node.

Detects a dedicated AprilTag attached to the pen holder, applies a fixed
offset from the tag center to a desired grasp point on the pen, and publishes
the resulting poses (plus an optional RViz marker).

Inputs
------
- Subscribes to:
    * /camera/camera/color/camera_info  (sensor_msgs/CameraInfo)
        Used once to get camera intrinsics for solvePnP.
    * /detections  (apriltag_msgs/AprilTagDetectionArray)
        AprilTag detections from apriltag_ros; looks for tag_id.

Parameters (main ones)
----------------------
- tag_id (int, default=3)
    AprilTag ID of the pen holder tag.
- tag_size_m (float, default=0.032)
    Physical side length of the black AprilTag square (meters).
- pen_offset_xyz (list[float], default=[0, 0, 0])
    Fixed offset from tag center to the grasp point, in the tag frame (meters).
- pen_offset_quat (list[float], default=[0, 0, 0, 1])
    Orientation of the grasp frame relative to the tag frame (xyzw).
- publish_marker (bool, default=False)
    If true, publishes a debug marker at the grasp point for RViz.

Outputs
-------
- Publishes:
    * tag_pen_pose    (geometry_msgs/PoseStamped)
        Pose of the tag center in camera_frame_id.
    * pen_grasp_pose  (geometry_msgs/PoseStamped)
        Pose of the grasp point:
          - in base_frame_id if base->camera is calibrated,
          - otherwise in camera_frame_id.
    * pen_grasp_marker (visualization_msgs/Marker, optional)
        Sphere marker at the grasp point in camera_frame_id, for visual check
        when publish_marker=true.

TF
--
- Static transform: tag_frame_id -> pen_grasp (grasp_frame_id)
    Encodes the fixed pen_offset_xyz / pen_offset_quat.
- Optional static transform: base_frame_id -> camera_frame_id
    Only published when calibrate_camera=true and a calibration tag is used.
"""


from __future__ import annotations

from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import transforms3d.quaternions as tquat

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from visualization_msgs.msg import Marker
from apriltag_msgs.msg import AprilTagDetectionArray, AprilTagDetection
from tf2_ros import StaticTransformBroadcaster


def _build_T(xyz: List[float], quat_xyzw: List[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform from xyz + quaternion (xyzw)."""
    x, y, z = [float(v) for v in xyz]
    qx, qy, qz, qw = [float(v) for v in quat_xyzw]

    # transforms3d expects quaternion as (w, x, y, z)
    R = tquat.quat2mat([qw, qx, qy, qz])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z], dtype=float)
    return T


def _mat_to_quat_xyzw(R: np.ndarray) -> List[float]:
    """Convert rotation matrix to quaternion in xyzw order."""
    qw, qx, qy, qz = tquat.mat2quat(R)  # returns (w, x, y, z)
    return [float(qx), float(qy), float(qz), float(qw)]


class PenDetector(Node):
    """Detects a single pen AprilTag and publishes an offset grasp pose + marker."""

    def __init__(self) -> None:
        """Initialize the PenDetector node and set up parameters, topics, and TF."""
        super().__init__('pen_detector')

        # ---------------- Parameters ----------------
        # ID and frame of the pen tag
        self.tag_id: int = self.declare_parameter('tag_id', 3).value
        self.tag_frame_id: str = self.declare_parameter(
            'tag_frame_id', 'tag_pen'
        ).value

        # Physical size of the detected tag (black square side length, in meters)
        self.tag_size: float = self.declare_parameter('tag_size_m', 0.032).value

        # AprilTag detection topic (from apriltag_ros)
        self.tag_topic: str = self.declare_parameter('tag_topic', '/detections').value

        # Frames for robot and camera
        self.base_frame_id: str = self.declare_parameter(
            'base_frame_id', 'base'
        ).value
        self.camera_frame_id: str = self.declare_parameter(
            'camera_frame_id', 'camera_color_optical_frame'
        ).value

        # Optional camera calibration using a known tag pose in base frame.
        # Keep default False if another node (e.g., BoardDetector) already
        # publishes base->camera.
        self.calibrate_camera: bool = self.declare_parameter(
            'calibrate_camera', False
        ).value

        self.calib_tag_id: int = self.declare_parameter(
            'calib_tag_id', 2
        ).value
        self.calib_tag_size: float = self.declare_parameter(
            'calib_tag_size_m', 0.07
        ).value

        # Known pose of calibration tag in base frame
        base_calib_xyz = self.declare_parameter(
            'base_calib_tag_xyz',
            [0.30, 0.0, 0.0],
        ).value

        base_calib_quat = self.declare_parameter(
            'base_calib_tag_quat',
            [1.0, 0.0, 0.0, 0.0],
        ).value

        # ---------------- Fixed pen offset (measured by user) ----------------
        # Offset from tag center to desired grasp point, expressed in tag frame.
        self.pen_offset_xyz: List[float] = self.declare_parameter(
            'pen_offset_xyz',
            [0.0, 0.0, 0.0],
        ).value

        # Orientation of the grasp frame relative to tag frame (xyzw).
        self.pen_offset_quat: List[float] = self.declare_parameter(
            'pen_offset_quat',
            [0.0, 0.0, 0.0, 1.0],
        ).value

        # Name of the grasp frame, used for static TF tag -> grasp.
        self.grasp_frame_id: str = self.declare_parameter(
            'grasp_frame_id', 'pen_grasp'
        ).value

        # Marker settings (for debugging in RViz)
        self.marker_scale: float = self.declare_parameter(
            'marker_scale', 0.02
        ).value
        # Debug switch for publishing marker
        self.publish_marker: bool = self.declare_parameter(
            'publish_marker', False
        ).value

        # ---------------- Calibration transforms ----------------
        # T_base_calib: known pose of calibration tag in base frame
        self.T_base_calib = _build_T(base_calib_xyz, base_calib_quat)

        # Will be filled after calibration if enabled
        self.T_base_camera: Optional[np.ndarray] = None
        self.camera_calibrated: bool = False
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Publish a static TF: tag_pen -> pen_grasp using measured offset
        self._publish_tag_to_grasp_static_tf()

        # ---------------- Camera intrinsics ----------------
        self.K: Optional[np.ndarray] = None  # 3x3 intrinsic matrix
        self.D: Optional[np.ndarray] = None  # distortion coefficients

        # ---------------- Subscriptions ----------------
        # Camera intrinsics (used for solvePnP)
        self.caminfo_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.cam_info_cb,
            1,
        )

        # AprilTag detections
        self.create_subscription(
            AprilTagDetectionArray,
            self.tag_topic,
            self.tag_cb,
            10,
        )

        # ---------------- Publishers ----------------
        # Pose of the tag center
        self.tag_pose_pub = self.create_publisher(
            PoseStamped,
            'tag_pen_pose',
            10,
        )

        # Pose of the grasp point
        self.grasp_pose_pub = self.create_publisher(
            PoseStamped,
            'pen_grasp_pose',
            10,
        )

        # Optional debug marker for RViz alignment checks
        self.marker_pub = None
        if self.publish_marker:
            self.marker_pub = self.create_publisher(
                Marker,
                'pen_grasp_marker',
                10,
            )

        self.get_logger().info(
            "PenDetector running "
            f"(tag_id={self.tag_id}, tag_size_m={self.tag_size}, "
            f"offset_xyz={self.pen_offset_xyz})"
        )

    # ---------------- Static TF: tag -> grasp ----------------
    def _publish_tag_to_grasp_static_tf(self) -> None:
        """Publish a static transform tag_frame_id -> grasp_frame_id."""
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.tag_frame_id
        tf.child_frame_id = self.grasp_frame_id

        tf.transform.translation.x = float(self.pen_offset_xyz[0])
        tf.transform.translation.y = float(self.pen_offset_xyz[1])
        tf.transform.translation.z = float(self.pen_offset_xyz[2])

        qx, qy, qz, qw = [float(v) for v in self.pen_offset_quat]
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw

        self._static_tf_broadcaster.sendTransform(tf)

    # ---------------- Camera intrinsics ----------------
    def cam_info_cb(self, msg: CameraInfo) -> None:
        """Cache camera intrinsics K, D from CameraInfo (first message only)."""
        if self.K is not None:
            return

        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.D = np.array(msg.d, dtype=float)

        self.get_logger().info("Camera intrinsics received")
        # Only need intrinsics once
        self.destroy_subscription(self.caminfo_sub)

    # --------------- SolvePnP helper -------------------
    def estimate_tag_pose(
        self,
        detection: AprilTagDetection,
        tag_size_override: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate tag pose (R, t) in camera frame from 4 corners using IPPE_SQUARE.

        Returns
        -------
        (R, t): if successful
            R: 3x3 rotation matrix (tag frame -> camera frame)
            t: shape (3,) translation vector (tag origin in camera frame)
        None:
            if intrinsics missing or PnP fails
        """
        if self.K is None:
            return None

        if len(detection.corners) != 4:
            self.get_logger().warn(
                f"Tag id={detection.id} has {len(detection.corners)} corners, expected 4"
            )
            return None

        # Pixel coordinates of the four detected corners (u, v)
        uv = np.array([[c.x, c.y] for c in detection.corners], dtype=float)

        # Use override size if given (e.g., for calibration tag), else node parameter
        tag_size = float(tag_size_override) if tag_size_override is not None else float(self.tag_size)
        s: float = tag_size / 2.0

        # Square corners in tag frame, z=0, centered at origin
        XYZ = np.array(
            [
                [-s,  s, 0],
                [ s,  s, 0],
                [ s, -s, 0],
                [-s, -s, 0],
            ],
            dtype=float,
        )

        # Solve PnP using IPPE for planar square tags
        success, rvec, tvec = cv2.solvePnP(
            XYZ,
            uv,
            self.K,
            self.D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not success:
            self.get_logger().warn(
                f"solvePnP failed for tag id={detection.id}"
            )
            return None

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

        return R, t

    # ---------------- Base->Camera static TF (optional) ----------------
    def _publish_base_camera_tf(self, T_base_camera: np.ndarray) -> None:
        """Publish base -> camera transform as a static TF."""
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.base_frame_id
        tf.child_frame_id = self.camera_frame_id

        tf.transform.translation.x = float(T_base_camera[0, 3])
        tf.transform.translation.y = float(T_base_camera[1, 3])
        tf.transform.translation.z = float(T_base_camera[2, 3])

        R_bc = T_base_camera[:3, :3]
        qx, qy, qz, qw = _mat_to_quat_xyzw(R_bc)
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw

        self._static_tf_broadcaster.sendTransform(tf)
        self.get_logger().info(
            f"Published static TF {self.base_frame_id} -> {self.camera_frame_id}"
        )

    # ---------------- Main callback --------------------
    def tag_cb(self, msg: AprilTagDetectionArray) -> None:
        """Main callback: detect pen tag, compute grasp pose, and (optionally) publish marker."""
        if self.K is None:
            # Cannot do pose estimation without intrinsics
            return

        # Build map: id -> detection
        dets: Dict[int, AprilTagDetection] = {d.id: d for d in msg.detections}

        # Optional camera calibration using a known tag in base frame
        if (
            self.calibrate_camera
            and not self.camera_calibrated
            and self.calib_tag_id in dets
        ):
            calib = self.estimate_tag_pose(
                dets[self.calib_tag_id],
                tag_size_override=self.calib_tag_size,
            )
            if calib is not None:
                R_cam_calib, t_cam_calib = calib

                T_camera_calib = np.eye(4)
                T_camera_calib[:3, :3] = R_cam_calib
                T_camera_calib[:3, 3] = t_cam_calib

                # T_base_camera = T_base_calib * (T_camera_calib)^-1
                T_base_camera = self.T_base_calib @ np.linalg.inv(T_camera_calib)

                self.T_base_camera = T_base_camera
                self._publish_base_camera_tf(T_base_camera)
                self.camera_calibrated = True

        # Pen tag not visible: nothing to do
        if self.tag_id not in dets:
            return

        est = self.estimate_tag_pose(dets[self.tag_id])
        if est is None:
            return

        R_cam_tag, t_cam_tag = est

        # Build T_camera_tag
        T_camera_tag = np.eye(4)
        T_camera_tag[:3, :3] = R_cam_tag
        T_camera_tag[:3, 3] = t_cam_tag

        # Build fixed T_tag_grasp from measured offset
        T_tag_grasp = _build_T(self.pen_offset_xyz, self.pen_offset_quat)

        # Compute grasp pose in camera frame
        T_camera_grasp = T_camera_tag @ T_tag_grasp

        # ---------------- Publish tag pose ----------------
        tag_pose = PoseStamped()
        tag_pose.header = msg.header
        tag_pose.header.frame_id = self.camera_frame_id

        tag_pose.pose.position.x = float(t_cam_tag[0])
        tag_pose.pose.position.y = float(t_cam_tag[1])
        tag_pose.pose.position.z = float(t_cam_tag[2])

        qx, qy, qz, qw = _mat_to_quat_xyzw(R_cam_tag)
        tag_pose.pose.orientation.x = qx
        tag_pose.pose.orientation.y = qy
        tag_pose.pose.orientation.z = qz
        tag_pose.pose.orientation.w = qw

        self.tag_pose_pub.publish(tag_pose)

        # ---------------- Publish grasp pose ----------------
        # Prefer base frame if base->camera is known, otherwise publish in camera frame.
        if self.T_base_camera is not None:
            # Transform grasp pose into base frame
            T_base_grasp = self.T_base_camera @ T_camera_grasp
            R_out = T_base_grasp[:3, :3]
            t_out = T_base_grasp[:3, 3]
            out_frame = self.base_frame_id
        else:
            # Only know camera frame; publish in camera frame
            R_out = T_camera_grasp[:3, :3]
            t_out = T_camera_grasp[:3, 3]
            out_frame = self.camera_frame_id

        grasp_pose = PoseStamped()
        grasp_pose.header = msg.header
        grasp_pose.header.frame_id = out_frame

        grasp_pose.pose.position.x = float(t_out[0])
        grasp_pose.pose.position.y = float(t_out[1])
        grasp_pose.pose.position.z = float(t_out[2])

        qx, qy, qz, qw = _mat_to_quat_xyzw(R_out)
        grasp_pose.pose.orientation.x = qx
        grasp_pose.pose.orientation.y = qy
        grasp_pose.pose.orientation.z = qz
        grasp_pose.pose.orientation.w = qw

        self.grasp_pose_pub.publish(grasp_pose)

        # ---------------- Publish marker in CAMERA frame for visual alignment ----------------
        if self.publish_marker and self.marker_pub is not None:
            self._publish_grasp_marker(msg.header, T_camera_grasp)

    # ---------------- Marker ----------------
    def _publish_grasp_marker(self, header, T_camera_grasp: np.ndarray) -> None:
        """Publish a simple sphere marker at the grasp point in the camera frame."""
        if self.marker_pub is None:
            return

        m = Marker()
        m.header = header
        m.header.frame_id = self.camera_frame_id

        m.ns = 'pen_grasp'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        # Uniform sphere size
        m.scale.x = float(self.marker_scale)
        m.scale.y = float(self.marker_scale)
        m.scale.z = float(self.marker_scale)

        # Bright, visible color
        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 1.0
        m.color.a = 1.0

        # Position at grasp point in camera frame
        m.pose.position.x = float(T_camera_grasp[0, 3])
        m.pose.position.y = float(T_camera_grasp[1, 3])
        m.pose.position.z = float(T_camera_grasp[2, 3])

        # Orientation from grasp frame rotation
        R = T_camera_grasp[:3, :3]
        qx, qy, qz, qw = _mat_to_quat_xyzw(R)
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw

        self.marker_pub.publish(m)


def main(args=None) -> None:
    """Spin the PenDetector node."""
    rclpy.init(args=args)
    node = PenDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
