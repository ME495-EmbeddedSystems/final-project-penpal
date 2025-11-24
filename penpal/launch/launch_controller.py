"""Launch file for PenPal Controller."""
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Launch cartesian pose controller.

    Run ros2 control list_controllers
    to see if cartesian pose controller is an option.
    """
    penpal_pkg = FindPackageShare('penpal')
    controller_config_path = PathJoinSubstitution(
        [penpal_pkg, 'config', 'controller.yaml']
    )

    load_params_action = ExecuteProcess(
        cmd=['ros2', 'param', 'load',
             '/controller_manager',
             controller_config_path]
    )

    delayed_spawner = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['cartesian_pose_example_controller',
                           '--inactive'],
            )
        ]
    )

    return LaunchDescription([
        load_params_action,
        delayed_spawner,
    ])
