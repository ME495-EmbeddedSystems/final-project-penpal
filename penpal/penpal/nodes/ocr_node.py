"""
ROS 2 node wrapper for GeminiOCREngine.

Wraps a vision-language model (VLM) OCR engine as a ROS 2 Trigger service.
When triggered, the node reads the latest available camera image, transcribes the
whiteboard text, and produces a short answer suitable for writing back onto the board.

Service
-------
- read_and_answer_board (example_interfaces/srv/Trigger)
    Request: empty
    Response:
      - success: bool
      - message: JSON string payload with fields like:
          {
            "question": "string",
            "answer": "string (concise)",
            "ocr_text": "string",
            "ocr_lines": ["..."],
            "ocr_raw": "string",
            "answer_raw": "string"
          }

Topics
------
- image_topic (sensor_msgs/msg/Image)
    Camera RGB image stream used for OCR/QA.

Parameters
----------
- image_topic (string):
    Topic name to subscribe to for RGB images.
- service_name (string):
    Trigger service ('read_and_answer_board').
- model configuration parameters (strings / floats):
    Model id/name, device, temperature, and any prompt controls as implemented
    by the OCR engine.

"""

import json
from typing import Optional

import cv2
from cv_bridge import CvBridge
from example_interfaces.srv import Trigger
import numpy as np
from penpal.ocr_engine import BoardQAResult, GeminiOCREngine
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage


class GeminiOCRNode(Node):
    """ROS 2 node using Gemini-based OCR + QA as a Trigger service."""

    def __init__(self) -> None:
        """Initialize the OCR node."""
        super().__init__('ocr_node')

        image_topic: str = self.declare_parameter(
            'image_topic',
            '/camera/camera/color/image_raw',
        ).value

        service_name: str = self.declare_parameter(
            'service_name',
            'read_and_answer_board',
        ).value

        api_key = self.declare_parameter('gemini_api_key', '').value
        self._engine = GeminiOCREngine(api_key=api_key)

        self._bridge = CvBridge()
        self._last_board_rgb: Optional[np.ndarray] = None

        self._image_sub = self.create_subscription(
            RosImage,
            image_topic,
            self._image_cb,
            10,
        )

        self._srv = self.create_service(
            Trigger,
            service_name,
            self._handle_read_and_answer,
        )

        self.get_logger().info(
            f'GeminiOCRNode started.\n'
            f'image_topic: {image_topic}\n'
            f'service: {service_name}'
        )

    def _image_cb(self, msg: RosImage) -> None:
        """Cache the latest board image."""
        try:
            cv_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
            self._last_board_rgb = rgb
        except Exception as exc:
            self.get_logger().error(f'Failed to convert board image: {exc}')

    def _handle_read_and_answer(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Handle service calls."""
        if self._last_board_rgb is None:
            response.success = False
            response.message = 'No board image received yet.'
            return response

        self.get_logger().info('Processing board request with Gemini...')

        try:
            qa: BoardQAResult = self._engine.read_and_answer_board(
                self._last_board_rgb
            )

            payload = {
                'question': qa.question,
                'answer': qa.answer,
                'ocr_text': qa.ocr.text,
                'ocr_lines': qa.ocr.lines,
                'ocr_raw': qa.ocr.raw_output,
                'answer_raw': qa.raw_answer_output,
            }

            # validation
            if not qa.question and 'Error' in qa.answer:
                response.success = False
                response.message = qa.answer
            else:
                response.success = True
                response.message = json.dumps(payload)

        except Exception as exc:
            self.get_logger().error(f'OCR/QA failed: {exc}')
            response.success = False
            response.message = f'OCR/QA failed: {exc}'

        return response


def main(args: Optional[list[str]] = None) -> None:
    """Node entry point."""
    import rclpy
    rclpy.init(args=args)
    node = GeminiOCRNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
