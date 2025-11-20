"""Impedance Control."""
from penpal.control.pp_control import PPControlBase, PPControlError, Trajectory
from rclpy import Node
from geometry_msgs.msg import PoseStamped


class CartesianImpedanceControl():
    """Implement Impedance Control from Franka Library."""

    def __init__(self):
        """Initialize the object."""
        super().__init__()
        self.topic = '/cartesian_impedance_example_controller/equilibrium_pose'
        self.pub = self.node.create_publisher(PoseStamped, self.topic, 10)

    def execute_trajectory(
        self, traj: Trajectory, target_ee_velocity_m_s: float
    ) -> None:
        """
        Move the EE through a trajectory.

        Args:
            traj (Trajectory): path to send the EE through space
            target_ee_velocity_m_s (float): target average velocity for 
            the trajectory execution.
        """
        for i in range(len(traj)):
            point = traj[i]
            msg = PoseStamped()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = self.c.world_frame

            # Position
            msg.pose.position.x = point[0]
            msg.pose.position.y = point[1]
            msg.pose.position.z = point[2]

            # Orientation
            msg.pose.orientation.x = point[3]
            msg.pose.orientation.y = point[4]
            msg.pose.orientation.z = point[5]
            msg.pose.orientation.w = point[6]

            self.target_pub.publish(msg)

    def grip(self, offset_m: float, grip_force_N: float | None = None) -> None:
        """
        Open or close the gripper to the desired offset, then applies a force.

        Args:
            offset_m: Offset (meters) of each finger from the EE frame.
            grip_force_N: Force to apply once gripped (i.e. to the marker when closed).
            If None, don't control the force.
        """
        pass
