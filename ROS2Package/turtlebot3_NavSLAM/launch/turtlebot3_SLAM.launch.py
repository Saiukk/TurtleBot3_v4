import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, get_package_share_path
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.descriptions import ParameterValue  # Need master or Galactic branch for this feature
import xacro

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam_params_file = LaunchConfiguration('slam_params_file')
    rviz_config_file = LaunchConfiguration('rviz_config')

    path_to_urdf = get_package_share_path('turtlebot3_description') / 'urdf' / 'turtlebot3_burger.urdf'

    declare_use_sim_time_argument = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation/Gazebo clock')

    declare_slam_params_file_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(get_package_share_directory("turtlebot3_NavSLAM"),
                                   'config', 'SLAM_param.yaml'),
        description='Full path to the ROS2 parameters file to use for the slam_toolbox node')

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(get_package_share_directory("turtlebot3_NavSLAM"), 'rviz', 'SLAM.rviz'),
        description='Full path to the RVIZ config file to use')

    # start_rviz_cmd = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     output='screen',
    #     name='rviz2',
    #     arguments=['-d', rviz_config_file],
    #     )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro ', str(path_to_urdf)]), value_type=str
            )
        }]
    )

    start_async_slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        #output='screen',
        parameters=[
          slam_params_file,
          {'use_sim_time': use_sim_time}
        ],
    )



    ld = LaunchDescription()

    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_use_sim_time_argument)
    ld.add_action(declare_slam_params_file_cmd)
    
    ld.add_action(robot_state_publisher_node)
    ld.add_action(start_async_slam_toolbox_node)
    #d.add_action(start_rviz_cmd)

    return ld