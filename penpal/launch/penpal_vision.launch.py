"""Vision launch file."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import (
    PathJoinSubstitution,
    EqualsSubstitution,
    LaunchConfiguration,
)
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate ROS Launch description."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'run_rviz',
                default_value='true',
                description='If true, launch with the internal rviz config.',
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [
                                get_package_share_directory(
                                    'realsense2_camera'
                                ),
                                'launch',
                                'rs_launch.py',
                            ]
                        )
                    ]
                ),
                launch_arguments={
                    'enable_color': 'true',
                    'enable_depth': 'true',
                    'align_depth.enable': 'true',
                    'rgb_camera.profile': '640x480x30',
                    'depth_module.profile': '640x480x30',
                }.items(),
            ),
            Node(
                package='apriltag_ros',
                executable='apriltag_node',
                name='apriltag',
                output='screen',
                parameters=[
                    PathJoinSubstitution(
                        [
                            get_package_share_directory('penpal'),
                            'config',
                            'tag.yaml',
                        ]
                    )
                ],
                remappings=[
                    ('image_rect', '/camera/camera/color/image_raw'),
                    ('camera_info', '/camera/camera/color/camera_info'),
                ],
            ),
            Node(
                package='penpal',
                executable='board_detector',
                name='board_detector',
                parameters=[
                    {
                        'top_left_id': 0,
                        'bottom_right_id': 1,
                        'board_width_m': 0.8,
                        'board_height_m': 0.61,
                        'tag_topic': '/detections',
                        'board_frame_id': 'whiteboard',
                        # calibration settings:
                        'calib_tag_id': 2,
                        'base_frame_id': 'base',
                        'camera_frame_id': 'camera_color_optical_frame',
                        'base_calib_tag_xyz': [0.30, 0.0, 0.0],
                        'base_calib_tag_quat': [1.0, 0.0, 0.0, 0.0],
                    }
                ],
                output='screen',
            ),
            Node(
                package='penpal',
                executable='pen_detector',
                name='pen_detector',
                output='screen',
                parameters=[
                    {
                        'tag_id': 3,
                        'tag_frame_id': 'tag_pen',
                        'tag_size_m': 0.032,
                        'tag_topic': '/detections',

                        # Measured offset from tag center to desired grasp point
                        #'pen_offset_xyz': [-0.042, 0.003, -0.018], # for OLD pen support
                        'pen_offset_xyz': [0.042, 0.003, -0.018], # for NEW pen support
                        #'pen_offset_quat': [0.0, 0.0, 0.0, 1.0], # for OLD pen support
                        'pen_offset_quat': [0.0, 1.0, 0.0, 0.0], # for NEW pen support

                        # Marker visual size (meters)
                        'publish_marker': True,
                        'marker_scale': 0.02,

                        # Keep False if another node already handles base->camera calibration.
                        'calibrate_camera': False,
                        'base_frame_id': 'base',
                        'camera_frame_id': 'camera_color_optical_frame',
                    }
                ],
            ),
            Node(
                package='penpal',
                executable='ocr_node',
                name='ocr_node',
                output='screen',
                parameters=[
                    {
                        'image_topic': '/camera/camera/color/image_raw',
                        'service_name': 'read_and_answer_board',
                    }
                ],
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                condition=IfCondition(
                    EqualsSubstitution(LaunchConfiguration('run_rviz'), 'true')
                ),
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
