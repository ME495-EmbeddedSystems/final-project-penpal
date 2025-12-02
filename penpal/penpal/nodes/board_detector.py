"""Detect pose + dimensions of a rectangular whiteboard using AprilTags."""

import numpy as np
import cv2
import transforms3d.quaternions as tquat

from typing import Optional, Tuple, Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from apriltag_msgs.msg import AprilTagDetectionArray, AprilTagDetection


class BoardDetector(Node):
    """Detects pose + dimensions of one whiteboard using two AprilTags."""

    def __init__(self):
        """Initialize board detector."""
        super().__init__("board_detector")

        # board + tag geometry
        self.width: float = self.declare_parameter("board_width_m", 0.8).value
        self.height: float = self.declare_parameter("board_height_m", 0.61).value
        self.tag_size: float = self.declare_parameter("tag_size_m", 0.07).value

        # tag IDs at known board corners
        self.tag_tl: int = self.declare_parameter("top_left_id", 0).value
        self.tag_br: int = self.declare_parameter("bottom_right_id", 1).value

        # detection topic
        self.tag_topic: str = self.declare_parameter("tag_topic", "/detections").value

        # camera intrinsics
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None

        # ---------------- Subscriptions ----------------
        self.caminfo_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
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
            PoseStamped,
            "whiteboard_pose",
            10
        )

        self.marker_pub = self.create_publisher(
            Marker,
            "whiteboard_outline",
            10
        )

        self.get_logger().info("BoardDetector running")

    # ---------------- Camera intrinsics ----------------
    def cam_info_cb(self, msg: CameraInfo) -> None:
        """Cache camera intrinsics K, D from CameraInfo."""
        if self.K is not None:
            return
        # use for solvePnP
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.D = np.array(msg.d, dtype=float)

        self.get_logger().info("Camera intrinsics received")
        # we only need intrinsics once
        self.destroy_subscription(self.caminfo_sub)

    # --------------- SolvePnP helper -------------------
    def estimate_tag_pose(
        self,
        detection: AprilTagDetection
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

        # pixel corners -> shape (4,2)
        uv = np.array([[c.x, c.y] for c in detection.corners], dtype=float)

        # tag-frame 3D corners (square in z=0 plane, centered at origin)
        s: float = self.tag_size / 2.0
        XYZ = np.array([
            [-s, s, 0],
            [s, s, 0],
            [s, -s, 0],
            [-s, -s, 0],
        ], dtype=float)

        success, rvec, tvec = cv2.solvePnP(
            # real-world coodinates of corner
            XYZ,
            # pixel coordinates of corners
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

        # rotation matrix from tag frame -> camera frame
        R, _ = cv2.Rodrigues(rvec)
        # translation vector from tag origin -> camera frame
        t = tvec.reshape(3)

        return R, t

    # ---------------- Main callback --------------------
    def tag_cb(self, msg: AprilTagDetectionArray) -> None:
        """Use TL + BR tags to compute board pose and outline."""
        if self.K is None:
            return

        # map id -> detection
        dets: Dict[int, AprilTagDetection] = {d.id: d for d in msg.detections}
        if self.tag_tl not in dets or self.tag_br not in dets:
            # need both corner tags for this model
            return

        tl = self.estimate_tag_pose(dets[self.tag_tl])
        br = self.estimate_tag_pose(dets[self.tag_br])

        if tl is None or br is None:
            return

        R_tl, t_tl = tl
        R_br, t_br = br

        # ---- Board orientation ----
        # assume tags are aligned with board so board axes = TL tag axes
        # R columns = x, y, z axes of board in camera frame
        R = R_tl

        # board center from tag centers
        W: float = self.width
        H: float = self.height
        S: float = self.tag_size
        hw, hh, hs = W / 2.0, H / 2.0, S / 2.0

        # tag centers expressed in board frame
        # TL tag is inset by half a tag along +x and half a tag along -y
        P_tag_tl_b = np.array([-hw + hs, hh - hs, 0.0])
        # BR tag is inset by half a tag along -x and half a tag along +y
        P_tag_br_b = np.array([hw - hs, -hh + hs, 0.0])

        # solve for board center from each tag:
        # t_tag ≈ R * P_tag_b + center
        center0 = t_tl - R @ P_tag_tl_b
        center1 = t_br - R @ P_tag_br_b

        # average value
        center = 0.5 * (center0 + center1)

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

        # ---- Publish outline marker ----
        self.publish_outline(msg.header, R, center)

    # --------------- Visualization ---------------------
    def publish_outline(
        self,
        header: PoseStamped.header,
        R: np.ndarray,
        center: np.ndarray,
    ) -> None:
        """Draw board rectangle in camera frame using board pose (R, center)."""
        hw, hh = self.width / 2.0, self.height / 2.0

        # board corners in board frame
        corners_b = np.array(
            [
                [-hw, -hh, 0.0],
                [hw, -hh, 0.0],
                [hw, hh, 0.0],
                [-hw, hh, 0.0],
            ], dtype=float,).T

        # transform to camera frame
        corners_c = R @ corners_b + center.reshape(3, 1)

        m = Marker()
        m.header = header
        m.ns = "whiteboard"
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


def main(args=None):
    """Spin the node."""
    rclpy.init(args=args)
    node = BoardDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
