"""Integration test launchfile for motion controller."""

from launch import LaunchDescription
from launch.actions import (
    RegisterEventHandler,
    Shutdown,
    IncludeLaunchDescription,
    DeclareLaunchArgument,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    PathJoinSubstitution,
    EqualsSubstitution,
    LaunchConfiguration,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """ROS2 launch description generator."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'vision',
                default_value='true',
                description='If true, launch with vision nodes running.',
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [
                                get_package_share_directory('penpal'),
                                'launch',
                                'penpal_vision.launch.py',
                            ]
                        )
                    ]
                ),
                launch_arguments={
                    'run_rviz': 'false',
                }.items(),
                condition=IfCondition(
                    EqualsSubstitution(LaunchConfiguration('vision'), 'true')
                ),
            ),
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
                    'base',  # robot base frame
                    '--child-frame-id',
                    'pen',
                ],
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=[
                    '-d',
                    PathJoinSubstitution(
                        [
                            FindPackageShare('penpal'),
                            'config',
                            'robot_view.rviz',
                        ]
                    ),
                ],
            ),
        ]
    )
