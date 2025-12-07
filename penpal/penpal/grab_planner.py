"""Grabs the pen."""

from penpal.control.moveit_control import MoveItPPControl
from rclpy.node import Node


class GrabPlanner:
    """Compute trajectories to write on the real board."""

    def __init__(self, node: Node, controller: MoveItPPControl) -> None:
        """Initialize the object."""
        self.control = controller
        self._node = node
        self._logger = node.get_logger().get_child('GrabPlanner')

    async def grab_pen(self) -> None:
        """Grab the pen (must be visible to camera)."""
        pass

    async def home_arm(self) -> None:
        """Send the arm to the home position."""
        self._logger.info("Homing the arm to 'ready' position...")
        await self.control.plan_to_named_config(
            'ready', execute_immediately=True
        )
