"""Integration test launchfile for motion controller."""

from launch import LaunchDescription
from launch.actions import (
    RegisterEventHandler,
    Shutdown,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """ROS2 launch description generator."""
    return LaunchDescription(
        [
            Node(
                package='penpal',
                executable='penpal',
                arguments=[
                    '--ros-args',
                    '--log-level',
                    'penpal:=DEBUG',
                ],
                parameters=[{'write_control_type': 'mock'}],
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments=[
                    # TODO fill this in correctly.
                    # this is if we do end up just hardcoding pen position.
                    '--x',
                    '0.5',
                    '--y',
                    '0',
                    '--z',
                    '0',
                    '--roll',
                    '0',
                    '--pitch',
                    '0',
                    '--yaw',
                    '0',
                    '--frame-id',
                    'world',
                    '--child-frame-id',
                    'pen',
                ],
            ),
        ]
    )
