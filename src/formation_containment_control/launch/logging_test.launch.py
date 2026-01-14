#!/usr/bin/env python3
"""
Logging Test Launch File

Launch file for testing the formation control data logging system.
Enables logging and runs the formation control with virtual leader.

Usage:
    ros2 launch formation_containment_control logging_test.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    """Generate launch description with logging enabled."""
    
    pkg_share = FindPackageShare('formation_containment_control')
    
    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'sim_manual_offsets_params.yaml'
    ])

    default_offsets_file = PathJoinSubstitution([
        pkg_share, 'config', 'leader_offsets.yaml'
    ])

    rviz_config = PathJoinSubstitution([
        pkg_share, 'config', 'rviz_config.rviz'
    ])
    
    # =========================================================================
    # Launch Arguments
    # =========================================================================
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2'
    )

    offsets_file_arg = DeclareLaunchArgument(
        'offsets_file',
        default_value=default_offsets_file,
        description='Path to the leader offsets file (YAML format)'
    )

    trajectory_type_arg = DeclareLaunchArgument(
        'trajectory_type',
        default_value='hover',
        description='Virtual leader trajectory: circle, lemniscate, hover, square'
    )
 
    log_rate_arg = DeclareLaunchArgument(
        'log_rate',
        default_value='5.0',
        description='Data logging rate in Hz'
    )
    
    log_directory_arg = DeclareLaunchArgument(
        'log_directory',
        default_value='/home/roli_005/robotarium/ros2_smc_ws/src/formation_containment_control/scripts/test_logs',
        description='Directory for log files'
    )
        
    # =========================================================================
    # Nodes
    # =========================================================================

    # Virtual leader generates the reference trajectory
    virtual_leader_node = Node(
        package='formation_containment_control',
        executable='virtual_leader_node.py',
        name='virtual_leader_node',
        output='screen',
        parameters=[
            config_file,
            {
                'trajectory_type': LaunchConfiguration('trajectory_type'),
            }
        ]
    )
    
    # Formation controller computes control inputs with logging enabled
    formation_controller_node = Node(
        package='formation_containment_control',
        executable='formation_containment_node.py',
        name='formation_containment_node',
        output='screen',
        parameters=[
            config_file,
            {
                'offsets_file': LaunchConfiguration('offsets_file'),
                # LOGGING ENABLED
                'enable_logging': False,
                'log_rate': LaunchConfiguration('log_rate'),
                'log_directory': LaunchConfiguration('log_directory'),
            }
        ]
    )
    
    # Crazyswarm bridge sends GoTo commands (with rate limiting)
    crazyswarm_bridge_node = Node(
        package='formation_containment_control',
        executable='crazyswarm_bridge_node.py',
        name='crazyswarm_bridge_node',
        output='screen',
        parameters=[
            config_file,
            # {
            #     'goto_rate': LaunchConfiguration('goto_rate'),
            #     'position_threshold': LaunchConfiguration('position_threshold'),
            # }
        ]
    )
    
    # Visualization node (subscribes to odometry, publishes markers)
    visualization_node = Node(
        package='formation_containment_control',
        executable='visualization_node.py',
        name='visualization_node',
        output='screen',
        parameters=[config_file]
    )
    
    # RViz visualization
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
        log_rate_arg,
        log_directory_arg,
        
        # Nodes
        virtual_leader_node,
        crazyswarm_bridge_node,
        formation_controller_node,
        visualization_node,
        rviz_node,
    ])

