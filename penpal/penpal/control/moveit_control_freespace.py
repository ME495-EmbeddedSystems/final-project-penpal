"""MoveIt-based controller for moving through free space (not writing)."""

from penpal.control.moveit_control import MoveItPPControl
from penpal.integration_tests.int_test_ppcontrol import (
    SetFullCollisionBehavior)


class FreeSpaceMoveItPPControl(MoveItPPControl):
    """MoveIt-based controller for moving through free space (not writing)."""

    async def configure(self) -> None:
        """One-time robot configuration."""
        # for moving through free space, we don't need anything special.
        free_space_req = SetFullCollisionBehavior.Request()
        free_space_req.upper_torque_thresholds_nominal = [
            60.0,
            60.0,
            60.0,
            60.0,
            50.0,
            50.0,
            50.0,
        ]
        free_space_req.upper_force_thresholds_nominal = [
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
        ]
        free_space_req.lower_torque_thresholds_nominal = [
            50.0,
            50.0,
            50.0,
            50.0,
            40.0,
            40.0,
            40.0,
        ]
        free_space_req.lower_force_thresholds_nominal = [
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
        ]
        # await self.set_collision_thresholds(free_space_req)
