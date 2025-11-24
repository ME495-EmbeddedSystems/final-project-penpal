"""
Integration test node for developing PP control.
"""

import asyncio
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from penpal.control.position_control import PositionPPControl


async def integration_test(node: Node, ctl: PositionPPControl) -> None:
    """Test move plan functions."""
    logger = node.get_logger()
    try:
        logger.info('helloworld from integration test')
        pass
    finally:
        node.get_logger().info('Integration test finished.')


# ---------------- End_Citation [2] ----------------


def main():
    """Run main."""
    rclpy.init()
    node = rclpy.create_node('test_position_controller_node')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    ctl = PositionPPControl(node)

    try:
        asyncio.run(integration_test(node, ctl))
    finally:
        executor.shutdown()
        executor_thread.join()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
