"""Vision launch file."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node as RosNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate ROS Launch description."""
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        get_package_share_directory('realsense2_camera'),
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
    )

    apriltag_config = PathJoinSubstitution(
        [
            get_package_share_directory('penpal'),
            'config',
            'tag.yaml',
        ]
    )

    apriltag_node = RosNode(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag',
        output='screen',
        parameters=[apriltag_config],
        remappings=[
            ('image_rect', '/camera/camera/color/image_raw'),
            ('camera_info', '/camera/camera/color/camera_info'),
        ],
    )

    board_detector_node = RosNode(
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
    )

    ocr_node = RosNode(
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
    )

    rviz_node = RosNode(
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
    )

    return LaunchDescription(
        [
            realsense_launch,
            apriltag_node,
            board_detector_node,
            ocr_node,
            rviz_node,
        ]
    )
