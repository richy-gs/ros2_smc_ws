"""
Crazyswarm2 Integration Launch File

Launches the formation-containment controller with Crazyswarm2.
Requires Crazyswarm2 to be running separately.

Usage:
1. First launch Crazyswarm2:
   ros2 launch crazyflie launch.py
   
2. Then launch this controller:
   ros2 launch formation_containment_control crazyswarm2.launch.py

This launches:
1. Virtual leader node (generates trajectory)
2. Formation containment controller
3. Crazyswarm2 bridge (translates commands)
4. Visualization node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('formation_containment_control')
    
    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'formation_params.yaml'
    ])
    
    rviz_config = PathJoinSubstitution([
        pkg_share, 'config', 'rviz_config.rviz'
    ])
    
    # Arguments
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2'
    )
    
    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='velocity',
        description='Control mode: velocity, position, full_state'
    )
    
    trajectory_type_arg = DeclareLaunchArgument(
        'trajectory_type',
        default_value='hover',
        description='Trajectory type (start with hover for safety)'
    )
    
    # Nodes
    virtual_leader_node = Node(
        package='formation_containment_control',
        executable='virtual_leader_node.py',
        name='virtual_leader_node',
        output='screen',
        parameters=[config_file, {
            'trajectory_type': LaunchConfiguration('trajectory_type'),
        }]
    )
    
    formation_controller_node = Node(
        package='formation_containment_control',
        executable='formation_containment_node.py',
        name='formation_containment_node',
        output='screen',
        parameters=[config_file]
    )
    
    crazyswarm_bridge_node = Node(
        package='formation_containment_control',
        executable='crazyswarm_bridge_node.py',
        name='crazyswarm_bridge_node',
        output='screen',
        parameters=[config_file, {
            'control_mode': LaunchConfiguration('control_mode'),
        }]
    )
    
    visualization_node = Node(
        package='formation_containment_control',
        executable='visualization_node.py',
        name='visualization_node',
        output='screen',
        parameters=[config_file]
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )
    
    # Delay controller start to allow Crazyswarm2 to initialize
    delayed_controller = TimerAction(
        period=2.0,
        actions=[formation_controller_node]
    )
    
    return LaunchDescription([
        use_rviz_arg,
        control_mode_arg,
        trajectory_type_arg,
        
        virtual_leader_node,
        crazyswarm_bridge_node,
        delayed_controller,
        visualization_node,
        rviz_node,
    ])

