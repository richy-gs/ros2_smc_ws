"""
Simulation Launch File

Launches the complete formation-containment control system in simulation mode.
No Crazyswarm2 or hardware required.

This launches:
1. Simulation node (simulates drone dynamics)
2. Virtual leader node (generates trajectory)
3. Formation containment controller
4. Visualization node
5. RViz2 (optional)

All parameters are loaded from config/formation_params.yaml.
To override parameters, modify the YAML file or use command line:
    ros2 launch formation_containment_control simulation.launch.py use_rviz:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package share directory
    pkg_share = FindPackageShare('formation_containment_control')
    
    # Config file path - ALL parameters come from this YAML file
    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'formation_params.yaml'
    ])
    
    # RViz config
    rviz_config = PathJoinSubstitution([
        pkg_share, 'config', 'rviz_config.rviz'
    ])
    
    # Launch arguments (only for launch-level options, not ROS parameters)
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 for visualization'
    )
    
    # Nodes - all parameters loaded from YAML file only
    # Do NOT override with hardcoded defaults, let YAML be the single source of truth
    simulation_node = Node(
        package='formation_containment_control',
        executable='simulation_node.py',
        name='simulation_node',
        output='screen',
        parameters=[config_file]
    )
    
    virtual_leader_node = Node(
        package='formation_containment_control',
        executable='virtual_leader_node.py',
        name='virtual_leader_node',
        output='screen',
        parameters=[config_file]
    )
    
    formation_controller_node = Node(
        package='formation_containment_control',
        executable='formation_containment_node.py',
        name='formation_containment_node',
        output='screen',
        parameters=[config_file]
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
    
    return LaunchDescription([
        # Arguments
        use_rviz_arg,
        
        # Nodes
        simulation_node,
        virtual_leader_node,
        formation_controller_node,
        visualization_node,
        rviz_node,
    ])

