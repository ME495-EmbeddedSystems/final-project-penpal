"""ROS 2 node wrapper for QwenOCREngine."""

from typing import Optional
import json

import cv2
import numpy as np
from cv_bridge import CvBridge
from example_interfaces.srv import Trigger
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage

from penpal.ocr_engine import QwenOCREngine, BoardQAResult


class QwenOCRNode(Node):
    """ROS 2 node using Qwen-based OCR + QA as a Trigger service."""

    def __init__(self) -> None:
        """Initialize the OCR node."""
        super().__init__("ocr_node")

        image_topic: str = self.declare_parameter(
            "image_topic",
            "/board/image_rectified",
        ).value
        service_name: str = self.declare_parameter(
            "service_name",
            "read_and_answer_board",
        ).value

        self._engine = QwenOCREngine()
        self._bridge = CvBridge()
        self._last_board_rgb: Optional[np.ndarray] = None

        # ------------------ Subscribers ------------------
        self._image_sub = self.create_subscription(
            RosImage,
            image_topic,
            self._image_cb,
            10,
        )

        # ------------------- Services --------------------
        # trigger OCR + answer using latest cached image
        self._srv = self.create_service(
            Trigger,
            service_name,
            self._handle_read_and_answer,
        )

        self.get_logger().info(
            f"OCRNode started.\n"
            f"image_topic: '{image_topic}'\n"
            f"service: '{service_name}'"
        )

    def _image_cb(self, msg: RosImage) -> None:
        """
        Cache the latest board image from the subscribed topic.

        Args:
        ----
        msg:
            Incoming rectified board image (sensor_msgs/Image).

        """
        try:
            # ROS Image -> OpenCV BGR
            cv_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # BGR -> RGB for OCR model
            rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
            self._last_board_rgb = rgb
        except Exception as exc:
            self.get_logger().error(f"Failed to convert board image: {exc}")

    def _handle_read_and_answer(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """
        Handle `read_and_answer_board` Trigger service calls.

        Returns
        -------
        Trigger.Response:
            success:
                True if OCR+QA ran successfully.
            message:
                JSON string with question, answer, and raw outputs.

        """
        if self._last_board_rgb is None:
            response.success = False
            response.message = "No board image received yet."
            return response

        try:
            qa: BoardQAResult = self._engine.read_and_answer_board(
                self._last_board_rgb
            )

            payload = {
                "question": qa.question,
                "answer": qa.answer,
                "ocr_text": qa.ocr.text,
                "ocr_lines": qa.ocr.lines,
                "ocr_raw": qa.ocr.raw_output,
                "answer_raw": qa.raw_answer_output,
            }

            response.success = True
            response.message = json.dumps(payload)
        except Exception as exc:
            self.get_logger().error(f"OCR/QA failed: {exc}")
            response.success = False
            response.message = f"OCR/QA failed: {exc}"

        return response


def main(args: Optional[list[str]] = None) -> None:
    """Entry point for `ros2 run penpal ocr_node`."""
    import rclpy

    rclpy.init(args=args)
    node = QwenOCRNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
