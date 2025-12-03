"""
Manual Offsets Formation Launch File

Launches the formation-containment control system using manually defined
leader offsets from a YAML file.

This demonstrates how to:
1. Load custom leader offsets from config/leader_offsets.yaml
2. Configure the formation controller to use these offsets
3. Run a complete simulation with the custom formation

Usage:
    ros2 launch formation_containment_control manual_offsets.launch.py

    # With custom offsets file:
    ros2 launch formation_containment_control manual_offsets.launch.py \
        offsets_file:=/path/to/custom_offsets.yaml

    # Without RViz:
    ros2 launch formation_containment_control manual_offsets.launch.py use_rviz:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration, 
    PathJoinSubstitution,
    PythonExpression
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package share directory
    pkg_share = FindPackageShare('formation_containment_control')
    
    # Config file path - parameters for manual offsets example
    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'manual_offsets_params.yaml'
    ])
    
    # Default offsets file path (YAML format)
    default_offsets_file = PathJoinSubstitution([
        pkg_share, 'config', 'leader_offsets.yaml'
    ])
    
    # RViz config
    rviz_config = PathJoinSubstitution([
        pkg_share, 'config', 'rviz_config.rviz'
    ])
    
    # =========================================================================
    # Launch Arguments
    # =========================================================================
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 for visualization'
    )
    
    offsets_file_arg = DeclareLaunchArgument(
        'offsets_file',
        default_value=default_offsets_file,
        description='Path to the leader offsets file (YAML format)'
    )
    
    trajectory_type_arg = DeclareLaunchArgument(
        'trajectory_type',
        default_value='lemniscate',
        description='Virtual leader trajectory: circle, lemniscate, hover, square'
    )
    
    # n_leaders_arg = DeclareLaunchArgument(
    #     'n_leaders',
    #     default_value='5',
    #     description='Number of leader drones (should match offsets file)'
    # )
    
    # n_followers_arg = DeclareLaunchArgument(
    #     'n_followers',
    #     default_value='1',
    #     description='Number of follower drones'
    # )
    
    # =========================================================================
    # Nodes
    # =========================================================================
    
    # Simulation node - simulates drone dynamics
    simulation_node = Node(
        package='formation_containment_control',
        executable='simulation_node.py',
        name='simulation_node',
        output='screen',
        parameters=[
            config_file,
            # {
            #     # 'n_leaders': LaunchConfiguration('n_leaders'),
            #     # 'n_followers': LaunchConfiguration('n_followers'),
            # }
        ]
    )
    
    # Virtual leader node - generates trajectory
    # Note: trajectory_type is now controlled by the YAML config file
    # Override via launch: ros2 launch ... trajectory_type:=circle
    virtual_leader_node = Node(
        package='formation_containment_control',
        executable='virtual_leader_node.py',
        name='virtual_leader_node',
        output='screen',
        parameters=[
            config_file,
            # {
            #     'trajectory_type': LaunchConfiguration('trajectory_type'),
            # }
        ]
    )
    
    # Formation controller node - with custom offsets file
    formation_controller_node = Node(
        package='formation_containment_control',
        executable='formation_containment_node.py',
        name='formation_containment_node',
        output='screen',
        parameters=[
            config_file,
            {
                # 'formation_type': 'custom',
                'offsets_file': LaunchConfiguration('offsets_file'),
                # 'n_leaders': LaunchConfiguration('n_leaders'),
                # 'n_followers': LaunchConfiguration('n_followers'),
            }
        ]
    )
    
    # Visualization node
    visualization_node = Node(
        package='formation_containment_control',
        executable='visualization_node.py',
        name='visualization_node',
        output='screen',
        parameters=[config_file]
    )
    
    # RViz2 node
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
        offsets_file_arg,
        trajectory_type_arg,
        # n_leaders_arg,
        # n_followers_arg,
        
        # Nodes
        simulation_node,
        virtual_leader_node,
        formation_controller_node,
        visualization_node,
        rviz_node,
    ])



