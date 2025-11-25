from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Path to YAML config file
    params_file = PathJoinSubstitution([
        FindPackageShare('mav_formation_control'),
        'config',
        'params.yaml'
    ])

    # Launch arguments (for overriding YAML params if needed)
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=params_file,
        description='Path to YAML parameters file'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 for visualization'
    )

    # Virtual Leader Trajectory Node
    virtual_leader_node = Node(
        package='mav_formation_control',
        executable='virtual_leader_trajectory_node',
        name='virtual_leader_trajectory_node',
        parameters=[LaunchConfiguration('params_file')],
        output='screen'
    )

    # Formation Controller Node
    formation_controller_node = Node(
        package='mav_formation_control',
        executable='formation_controller_node',
        name='formation_controller_node',
        parameters=[LaunchConfiguration('params_file')],
        output='screen'
    )

    # Follower Controller Node
    follower_controller_node = Node(
        package='mav_formation_control',
        executable='follower_controller_node',
        name='follower_controller_node',
        parameters=[LaunchConfiguration('params_file')],
        output='screen'
    )

    # RViz2 Node (conditional)
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('mav_formation_control'),
        'config',
        'rviz_config.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    return LaunchDescription([
        params_file_arg,
        use_rviz_arg,
        virtual_leader_node,
        formation_controller_node,
        follower_controller_node,
        rviz_node
    ])

