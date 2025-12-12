"""ROS 2 node wrapper for QwenOCREngine."""

import json
from typing import Optional

from example_interfaces.srv import Trigger
from rclpy.node import Node


class QwenOCRNode(Node):
    """ROS 2 node using Qwen-based OCR + QA as a Trigger service."""

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

        self._srv = self.create_service(
            Trigger,
            service_name,
            self._handle_read_and_answer,
        )

        self.get_logger().info(
            f'OCRNode started.\n'
            f'image_topic: {image_topic}\n'
            f'service: {service_name}'
        )

    def _handle_read_and_answer(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Return static response."""
        payload = {
            'question': 'What is the capital of Florida?',
            'answer': 'Talahassee',
            'ocr_text': 'test',
            'ocr_lines': 'test',
            'ocr_raw': 'test',
            'answer_raw': 'test',
        }

        response.success = True
        response.message = json.dumps(payload)
        return response


def main(args: Optional[list[str]] = None) -> None:
    """Node entry point."""
    import rclpy

    rclpy.init(args=args)
    node = QwenOCRNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
