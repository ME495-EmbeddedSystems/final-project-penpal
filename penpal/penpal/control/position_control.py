"""Direct position control using individual joint controllers."""

from rclpy.node import Node

from penpal.control.pp_control import PPControlBase, Trajectory


class PositionPPControl(PPControlBase):
    """Base class for control of the pen tip."""

    def __init__(
        self, node: Node, cfg: PPControlBase.Config | None = None
    ) -> None:
        """Initialize the object."""
        super().__init__(node, cfg)

    def execute_trajectory(
        self, traj: Trajectory, target_ee_velocity_m_s: float
    ) -> None:
        """
        Move the EE through a trajectory.

        Args:
            traj (Trajectory): path to send the EE through space
            target_ee_velocity_m_s (float): target average velocity for the trajectory
            execution

        """
        pass

    def grip(self, offset_m: float, grip_force_N: float | None = None) -> None:
        """
        Open or close the gripper to the desired offset, then applies a force.

        Args:
            offset_m: Offset (meters) of each finger from the EE frame.
            grip_force_N: Force to apply once gripped (i.e. to the marker when closed).
            If None, don't control the force.

        """
        # not intended to be used in this control scheme. use moveit controller
        raise NotImplementedError
