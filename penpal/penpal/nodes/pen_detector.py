"""Detect pose of a single AprilTag used for pen localization."""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import transforms3d.quaternions as tquat

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from apriltag_msgs.msg import AprilTagDetectionArray, AprilTagDetection
from tf2_ros import StaticTransformBroadcaster


class PenDetector(Node):
    """Detects pose of a single pen tag (default id=3)."""

    def __init__(self) -> None:
        super().__init__('pen_detector')

        # ---------------- Parameters ----------------
        self.tag_id: int = self.declare_parameter('tag_id', 3).value
        self.tag_frame_id: str = self.declare_parameter(
            'tag_frame_id', 'tag_pen'
        ).value

        # Physical size of the pen tag (black square side length)
        self.tag_size: float = self.declare_parameter('tag_size_m', 0.032).value

        # Detection topic
        self.tag_topic: str = self.declare_parameter('tag_topic', '/detections').value

        # Frames
        self.base_frame_id: str = self.declare_parameter(
            'base_frame_id', 'base'
        ).value
        self.camera_frame_id: str = self.declare_parameter(
            'camera_frame_id', 'camera_color_optical_frame'
        ).value

        # Optional camera calibration using a known tag pose in base frame
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

        # ---------------- Calibration transforms ----------------
        self.T_base_calib = np.eye(4)
        qw = float(base_calib_quat[3])
        qx = float(base_calib_quat[0])
        qy = float(base_calib_quat[1])
        qz = float(base_calib_quat[2])
        self.T_base_calib[:3, :3] = tquat.quat2mat([qw, qx, qy, qz])
        self.T_base_calib[:3, 3] = np.array(base_calib_xyz, dtype=float)

        self.T_base_camera: Optional[np.ndarray] = None
        self.camera_calibrated: bool = False
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ---------------- Camera intrinsics ----------------
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None

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
        self.pose_pub = self.create_publisher(
            PoseStamped,
            'tag_pen_pose',
            10,
        )

        self.get_logger().info(
            f"PenDetector running (tag_id={self.tag_id}, tag_size_m={self.tag_size})"
        )

    # ---------------- Camera intrinsics ----------------
    def cam_info_cb(self, msg: CameraInfo) -> None:
        """Cache camera intrinsics K, D from CameraInfo."""
        if self.K is not None:
            return

        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.D = np.array(msg.d, dtype=float)

        self.get_logger().info("Camera intrinsics received")
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

        uv = np.array([[c.x, c.y] for c in detection.corners], dtype=float)

        tag_size = float(tag_size_override) if tag_size_override is not None else float(self.tag_size)
        s: float = tag_size / 2.0

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
        qw, qx, qy, qz = tquat.mat2quat(R_bc)
        tf.transform.rotation.w = float(qw)
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)

        self._static_tf_broadcaster.sendTransform(tf)
        self.get_logger().info(
            f"Published static TF {self.base_frame_id} -> {self.camera_frame_id}"
        )

    # ---------------- Main callback --------------------
    def tag_cb(self, msg: AprilTagDetectionArray) -> None:
        """Detect the pen tag and publish its pose."""
        if self.K is None:
            return

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

                T_base_camera = self.T_base_calib @ np.linalg.inv(T_camera_calib)

                self.T_base_camera = T_base_camera
                self._publish_base_camera_tf(T_base_camera)
                self.camera_calibrated = True

        # Pen tag not visible
        if self.tag_id not in dets:
            return

        est = self.estimate_tag_pose(dets[self.tag_id])
        if est is None:
            return

        R_cam_pen, t_cam_pen = est

        # Build T_camera_pen
        T_camera_pen = np.eye(4)
        T_camera_pen[:3, :3] = R_cam_pen
        T_camera_pen[:3, 3] = t_cam_pen

        # Decide output frame
        if self.T_base_camera is not None:
            T_out = self.T_base_camera @ T_camera_pen
            frame_id = self.base_frame_id
            R_out = T_out[:3, :3]
            t_out = T_out[:3, 3]
        else:
            # Fall back to camera frame
            frame_id = msg.header.frame_id if msg.header.frame_id else self.camera_frame_id
            R_out = R_cam_pen
            t_out = t_cam_pen

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = frame_id

        pose.pose.position.x = float(t_out[0])
        pose.pose.position.y = float(t_out[1])
        pose.pose.position.z = float(t_out[2])

        qw, qx, qy, qz = tquat.mat2quat(R_out)
        pose.pose.orientation.w = float(qw)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)

        self.pose_pub.publish(pose)


def main(args=None) -> None:
    """Spin the node."""
    rclpy.init(args=args)
    node = PenDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
