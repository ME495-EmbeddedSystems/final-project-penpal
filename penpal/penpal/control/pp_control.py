"""Base class - executes trajectories in EE space."""

import abc
from dataclasses import dataclass

import numpy as np

from geometry_msgs.msg import Point
from rclpy.node import Node
from visualization_msgs.msg import Marker


class PPControlError(Exception):
    """Base exception for pp_control module."""

    pass


@dataclass
class Trajectory:
    """Represents a trajectory of the EE through space + wrench to apply."""

    label: str
    """human-readable label for debugging purposes."""
    data: np.ndarray
    """
    list of points to move the EE through. Nx12 array for N trajectory waypoints.
    Each waypoint provides [pose, force scalar in the orientation direction],
    pose being x,y,z and orientation qx, qy, qz, qw as quaternion
    like so:
    [x, y, z, qx, qy, qz, qw , f]
    """

    def __post_init__(self) -> None:
        """Post-initialization input checking."""
        if self.data.shape[1] != 8:
            raise PPControlError(
                f'Incorrect shape {self.data.shape} for Trajectory.data'
            )


class PPControlBase(abc.ABC):
    """Base class for control of the pen tip."""

    @dataclass
    class Config:
        """Configuration for PPControl."""

        # todo make these have correct values
        ee_frame: str = 'fer_hand_tcp'
        """Frame ID in tf tree to be considered as the end-effector."""
        world_frame: str = 'base'
        """Frame ID in tf tree for the world frame (probably robot {base})."""

    def __init__(self, node: Node, cfg: Config | None = None) -> None:
        """Initialize the object."""
        # TODO figure out what else we need here
        self._node = node
        self.c = cfg if cfg is not None else self.Config()
        self._logger = node.get_logger().get_child(self.__class__.__name__)

        # publish trajectory markers in rviz
        self._marker_pub = None
        self._next_marker_id: int = 0

    @abc.abstractmethod
    async def _execute_trajectory(
        self,
        traj: Trajectory,
        target_ee_velocity_m_s: float,
        publish_markers: bool = False,
    ) -> None:
        """
        Move the EE through a trajectory.

        Args:
            traj (Trajectory): path to send the EE through space
            target_ee_velocity_m_s (float): target average velocity for the trajectory
            execution

        """
        pass

    async def execute_trajectory(
        self,
        traj: Trajectory,
        target_ee_velocity_m_s: float,
        publish_markers: bool = False,
    ) -> None:
        """
        Move the EE through a trajectory.

        Args:
            traj (Trajectory): path to send the EE through space
            target_ee_velocity_m_s (float): target average velocity for the trajectory
            execution

        """
        self._logger.info(f"Executing trajectory '{traj.label}'")
        if publish_markers:
            self._logger.info('Publishing debug markers...')
            await self.publish_marker(traj)
        return await self._execute_trajectory(
            traj, target_ee_velocity_m_s, publish_markers
        )

    @abc.abstractmethod
    async def grip(
        self, offset_m: float, grip_force_N: float | None = None
    ) -> None:
        """
        Open or close the gripper to the desired offset, then applies a force.

        Args:
            offset_m: Offset (meters) of each finger from the EE frame.
            grip_force_N: Force to apply once gripped (i.e. to the marker when closed).
            If None, don't control the force.

        """
        pass

    async def publish_marker(
        self,
        traj: Trajectory,
    ) -> None:
        """Publish trajectory as a line marker to rviz."""
        if self._marker_pub is None:
            self._marker_pub = self._node.create_publisher(
                Marker,
                'pp_trajectories',
                10,
            )
        if traj.label == 'space':
            self._logger.debug(
                f"Skipping visualization for hover trajectory '{traj.label}'"
            )
            return

        marker = Marker()
        marker.header.frame_id = self.c.world_frame
        marker.header.stamp = self._node.get_clock().now().to_msg()
        marker.ns = 'pp_trajectories'
        marker.id = self._next_marker_id
        self._next_marker_id += 1

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.005
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # add XYZ points
        for wp in traj.data:
            pt = Point()
            pt.x = float(wp[0])
            pt.y = float(wp[1])
            pt.z = float(wp[2])
            marker.points.append(pt)

        self._marker_pub.publish(marker)
        self._logger.debug(
            f"Published marker id={marker.id} with {len(marker.points)} points "
            f"on /pp_trajectories"
        )
        pass

    async def configure(self) -> None:
        """Configure control parameters, TCP frame, collision behavior, etc."""
        # TODO
        # raise NotImplementedError
