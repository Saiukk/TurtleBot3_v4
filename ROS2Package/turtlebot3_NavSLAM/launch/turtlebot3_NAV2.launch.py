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
    package_name = 'turtlebot3_NavSLAM'
    
    package_dir = get_package_share_directory(package_name)
    path_to_urdf = get_package_share_path('turtlebot3_description') / 'urdf' / 'turtlebot3_burger.urdf'

    nav2_params_file = LaunchConfiguration('nav2_params_file')
    bt_params_file = LaunchConfiguration('bt_params_file')
    namespace = LaunchConfiguration('namespace')
    map_yaml_file = LaunchConfiguration('map_yaml_file')
    autostart = LaunchConfiguration('autostart')
    use_sim_time = LaunchConfiguration('use_sim_time')

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_use_sim_time_argument = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation/Gazebo clock')

    declare_map_yaml_file = DeclareLaunchArgument(
        'map_yaml_file',
        default_value=os.path.join(package_dir,'map', '<map_name.yaml>'),
        description='Full path to map file to load')

    declare_nav2_params_file_cmd = DeclareLaunchArgument(
        'nav2_params_file',
        default_value=os.path.join(package_dir,'config', 'NAV2_param.yaml'),
        description='Full path to the ROS2 parameters file to use for the navigation node')

    declare_bt_params_file = DeclareLaunchArgument(
        'bt_params_file',
        default_value=os.path.join(
                get_package_share_directory('nav2_bt_navigator'),
                'behavior_trees', 'navigate_w_replanning_and_recovery.xml'),
        description='Full path to the ROS2 parameters file to use for the navigation node')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the nav2 stack')



    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     output='screen',
    #     arguments=['-d', os.path.join(package_dir, 'rviz', 'NAV2.rviz')],
    # )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro ', str(path_to_urdf)]), value_type=str
            )
        }]
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(package_dir, 'launch', 'turtlebot3_AMCL.launch.py')),
        launch_arguments={'namespace': namespace,
                            'map': map_yaml_file,
                            'use_sim_time': use_sim_time,
                            'autostart': autostart,
                            'params_file': nav2_params_file,
                            'robot_state_cmd': 'False'}.items()
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'namespace': namespace,
            'use_sim_time': 'false',
            'autostart': autostart,
            'params_file': nav2_params_file,
            'use_lifecycle_mgr': 'false',
            'map_subscribe_transient_local': 'true',
            'default_bt_xml_filename': bt_params_file,
        }.items()
    )


    
    ld = LaunchDescription()

    ld.add_action(stdout_linebuf_envvar)

    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_argument)
    ld.add_action(declare_map_yaml_file)
    ld.add_action(declare_nav2_params_file_cmd)
    ld.add_action(declare_bt_params_file)
    ld.add_action(declare_autostart_cmd)
    #ld.add_action(rviz_node)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(localization_launch)
    ld.add_action(navigation_launch)

    return ld