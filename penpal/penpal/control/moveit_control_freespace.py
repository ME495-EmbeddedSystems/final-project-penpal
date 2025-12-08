"""MoveIt-based controller for moving through free space (not writing)."""

from penpal.control.moveit_control import MoveItPPControl


class FreeSpaceMoveItPPControl(MoveItPPControl):
    """MoveIt-based controller for moving through free space (not writing)."""

    async def configure(self) -> None:
        """One-time robot configuration."""
        # for moving through free space, we currently don't need anything special.
        # TODO - maybe reset to original defaults?
        pass
