import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory,get_package_share_path
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import launch_ros
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.descriptions import ParameterValue  # Need master or Galactic branch for this feature
import xacro


def generate_launch_description():
    package_name = 'turtlebot3_visualizer'
    
    package_dir = get_package_share_directory(package_name)
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', os.path.join(package_dir, 'rviz', 'SLAM.rviz')],
    )


    ld = LaunchDescription()

    ld.add_action(rviz_node)


    return ld