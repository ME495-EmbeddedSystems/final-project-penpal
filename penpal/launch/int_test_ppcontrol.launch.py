"""Integration test launchfile for motion controller."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """ROS2 launch description generator."""
    moveit_config = MoveItConfigsBuilder(
        'fer', package_name='franka_fer_moveit_config'
    ).to_moveit_configs()

    return LaunchDescription(
        [
            Node(
                package='moveit_ros_move_group',
                executable='move_group',
                parameters=[
                    moveit_config.to_dict(),
                    {'use_sim_time': True},
                    {'fake_execution_type': 'simple'},
                    {'allow_trajectory_execution': True},
                    {'moveit_manage_controllers': False},
                    {
                        'moveit_controller_manager': 'moveit_simple_controller'
                        '_manager/MoveItSimpleControllerManager'
                    },
                ],
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                output='log',
                arguments=[
                    '-d',
                    PathJoinSubstitution(
                        [
                            FindPackageShare('franka_fer_moveit_config'),
                            'config',
                            'moveit.rviz',
                        ]
                    ),
                ],
                parameters=[
                    moveit_config.planning_pipelines,
                    moveit_config.robot_description_kinematics,
                ],
                remappings=[('/move_action', '/viz/move_action')],
            ),
            Node(
                package='penpal',
                executable='int_test_ppcontrol',
            ),
        ]
    )
