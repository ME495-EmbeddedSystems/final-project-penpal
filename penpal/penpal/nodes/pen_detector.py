"""PenDetector node.

Detects a dedicated AprilTag (default id=3) attached to the pen holder.
Reuses BoardDetector utilities for:
  - caching camera intrinsics from CameraInfo
  - estimating tag pose from corners via IPPE_SQUARE solvePnP
  - publishing optional static base->camera TF

This node adds the pen-specific logic:
  - applies a fixed measured transform from tag center -> desired grasp point
  - publishes tag and grasp poses
  - optionally publishes a debug marker for RViz alignment

Subscribes
----------
- /camera/camera/color/camera_info (sensor_msgs/CameraInfo)
- /detections (apriltag_msgs/AprilTagDetectionArray)

Publishes
---------
- tag_pen_pose (geometry_msgs/PoseStamped)
- pen_grasp_pose (geometry_msgs/PoseStamped)
- pen_grasp_marker (visualization_msgs/Marker) if publish_marker=true

Static TF
---------
- tag_frame_id -> grasp_frame_id (fixed offset)
- base_frame_id -> camera_frame_id if calibrate_camera=true
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, List

import numpy as np
import transforms3d.quaternions as tquat

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from apriltag_msgs.msg import AprilTagDetectionArray, AprilTagDetection
from tf2_ros import StaticTransformBroadcaster

# Reuse these implementations instead of duplicating PnP / intrinsics / TF code.
from penpal.board_detector import BoardDetector


def _build_T(xyz: List[float], quat_xyzw: List[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform from xyz + quaternion (xyzw)."""
    x, y, z = [float(v) for v in xyz]
    qx, qy, qz, qw = [float(v) for v in quat_xyzw]

    # transforms3d expects (w, x, y, z)
    R = tquat.quat2mat([qw, qx, qy, qz])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z], dtype=float)
    return T


def _mat_to_quat_xyzw(R: np.ndarray) -> List[float]:
    """Convert rotation matrix to quaternion in xyzw order."""
    qw, qx, qy, qz = tquat.mat2quat(R)
    return [float(qx), float(qy), float(qz), float(qw)]


class PenDetector(Node):
    """Detect a single pen tag and publish a fixed-offset grasp pose."""

    def __init__(self) -> None:
        super().__init__('pen_detector')

        # ---------------- Parameters ----------------
        self.tag_id: int = self.declare_parameter('tag_id', 3).value
        self.tag_frame_id: str = self.declare_parameter(
            'tag_frame_id', 'tag_pen'
        ).value

        # Black square side length (meters)
        self.tag_size: float = self.declare_parameter('tag_size_m', 0.032).value

        self.tag_topic: str = self.declare_parameter(
            'tag_topic', '/detections'
        ).value

        self.base_frame_id: str = self.declare_parameter(
            'base_frame_id', 'base'
        ).value
        self.camera_frame_id: str = self.declare_parameter(
            'camera_frame_id', 'camera_color_optical_frame'
        ).value

        # Optional camera calibration using a known tag in base frame.
        # Keep False if another node already publishes base->camera.
        self.calibrate_camera: bool = self.declare_parameter(
            'calibrate_camera', False
        ).value

        self.calib_tag_id: int = self.declare_parameter(
            'calib_tag_id', 2
        ).value
        self.calib_tag_size: float = self.declare_parameter(
            'calib_tag_size_m', 0.07
        ).value

        base_calib_xyz = self.declare_parameter(
            'base_calib_tag_xyz',
            [0.30, 0.0, 0.0],
        ).value

        base_calib_quat = self.declare_parameter(
            'base_calib_tag_quat',
            [1.0, 0.0, 0.0, 0.0],
        ).value

        # Fixed offset from tag center -> desired grasp point (tag frame)
        self.pen_offset_xyz: List[float] = self.declare_parameter(
            'pen_offset_xyz',
            [0.0, 0.0, 0.0],
        ).value

        self.pen_offset_quat: List[float] = self.declare_parameter(
            'pen_offset_quat',
            [0.0, 0.0, 0.0, 1.0],
        ).value

        self.grasp_frame_id: str = self.declare_parameter(
            'grasp_frame_id', 'pen_grasp'
        ).value

        # Marker debug
        self.publish_marker: bool = self.declare_parameter(
            'publish_marker', False
        ).value
        self.marker_scale: float = self.declare_parameter(
            'marker_scale', 0.02
        ).value

        # ---------------- Shared-state required by borrowed methods ----------------
        # BoardDetector methods assume these fields exist on self.
        self.T_base_camera: Optional[np.ndarray] = None
        self.camera_calibrated: bool = False

        # Homogeneous transform for known base->calib tag pose
        self.T_base_calib = _build_T(base_calib_xyz, base_calib_quat)

        # Intrinsics cached by BoardDetector.cam_info_cb
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None

        # Static TF broadcaster reused for tag->grasp and optional base->camera
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Publish static tag->grasp TF once
        self._publish_tag_to_grasp_static_tf()

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

        # ---------------- Publishers ----------------
        self.tag_pose_pub = self.create_publisher(
            PoseStamped,
            'tag_pen_pose',
            10,
        )

        self.grasp_pose_pub = self.create_publisher(
            PoseStamped,
            'pen_grasp_pose',
            10,
        )

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
            f"offset_xyz={self.pen_offset_xyz}, publish_marker={self.publish_marker})"
        )

    # ---------------- Reused methods from BoardDetector ----------------
    def cam_info_cb(self, msg: CameraInfo) -> None:
        """Reuse BoardDetector intrinsics caching logic."""
        BoardDetector.cam_info_cb(self, msg)

    def _publish_base_camera_tf(self, T_base_camera: np.ndarray) -> None:
        """Reuse BoardDetector base->camera static TF publisher."""
        BoardDetector._publish_base_camera_tf(self, T_base_camera)

    def _estimate_tag_pose(
        self,
        detection: AprilTagDetection,
        tag_size_override: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Reuse BoardDetector.solvePnP implementation.

        BoardDetector.estimate_tag_pose uses self.tag_size.
        We temporarily override self.tag_size when needed.
        """
        if tag_size_override is None:
            return BoardDetector.estimate_tag_pose(self, detection)

        old = float(self.tag_size)
        self.tag_size = float(tag_size_override)
        try:
            return BoardDetector.estimate_tag_pose(self, detection)
        finally:
            self.tag_size = old

    # ---------------- Static TF: tag -> grasp ----------------
    def _publish_tag_to_grasp_static_tf(self) -> None:
        """Publish static tag_frame_id -> grasp_frame_id using measured offset."""
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

    # ---------------- Main callback --------------------
    def tag_cb(self, msg: AprilTagDetectionArray) -> None:
        """Detect pen tag, compute grasp pose, publish poses and optional marker."""
        if self.K is None:
            return

        dets: Dict[int, AprilTagDetection] = {d.id: d for d in msg.detections}

        # Optional base->camera calibration using a known tag pose in base frame
        if (
            self.calibrate_camera
            and not self.camera_calibrated
            and self.calib_tag_id in dets
        ):
            calib = self._estimate_tag_pose(
                dets[self.calib_tag_id],
                tag_size_override=self.calib_tag_size,
            )
            if calib is not None:
                R_cam_calib, t_cam_calib = calib

                T_camera_calib = np.eye(4)
                T_camera_calib[:3, :3] = R_cam_calib
                T_camera_calib[:3, 3] = t_cam_calib

                # T_base_camera = T_base_calib * inv(T_camera_calib)
                T_base_camera = self.T_base_calib @ np.linalg.inv(T_camera_calib)

                self.T_base_camera = T_base_camera
                self._publish_base_camera_tf(T_base_camera)
                self.camera_calibrated = True

        # Pen tag not visible
        if self.tag_id not in dets:
            return

        est = self._estimate_tag_pose(dets[self.tag_id])
        if est is None:
            return

        R_cam_tag, t_cam_tag = est

        # T_camera_tag
        T_camera_tag = np.eye(4)
        T_camera_tag[:3, :3] = R_cam_tag
        T_camera_tag[:3, 3] = t_cam_tag

        # Fixed T_tag_grasp
        T_tag_grasp = _build_T(self.pen_offset_xyz, self.pen_offset_quat)

        # Grasp in camera frame
        T_camera_grasp = T_camera_tag @ T_tag_grasp

        # ---------------- Publish tag pose (camera frame) ----------------
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
        if self.T_base_camera is not None:
            T_base_grasp = self.T_base_camera @ T_camera_grasp
            R_out = T_base_grasp[:3, :3]
            t_out = T_base_grasp[:3, 3]
            out_frame = self.base_frame_id
        else:
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

        # ---------------- Optional marker (camera frame) ----------------
        if self.publish_marker and self.marker_pub is not None:
            self._publish_grasp_marker(msg.header, T_camera_grasp)

    # ---------------- Marker ----------------
    def _publish_grasp_marker(self, header, T_camera_grasp: np.ndarray) -> None:
        """Publish a sphere marker at the grasp point in the camera frame."""
        m = Marker()
        m.header = header
        m.header.frame_id = self.camera_frame_id

        m.ns = 'pen_grasp'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.scale.x = float(self.marker_scale)
        m.scale.y = float(self.marker_scale)
        m.scale.z = float(self.marker_scale)

        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 1.0
        m.color.a = 1.0

        m.pose.position.x = float(T_camera_grasp[0, 3])
        m.pose.position.y = float(T_camera_grasp[1, 3])
        m.pose.position.z = float(T_camera_grasp[2, 3])

        qx, qy, qz, qw = _mat_to_quat_xyzw(T_camera_grasp[:3, :3])
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw

        self.marker_pub.publish(m)


def main(args=None) -> None:
    """Spin the node."""
    rclpy.init(args=args)
    node = PenDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
