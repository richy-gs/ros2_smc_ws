"""
Crazyswarm2 Integration Launch File

Launches the formation-containment controller with Crazyswarm2.
Uses only the GoTo service for high-level position control.

Usage:
1. First launch Crazyswarm2:
   ros2 launch crazyflie launch.py

2. Then launch this controller:
   ros2 launch formation_containment_control crazyswarm2.launch.py

3. Start the formation (takeoff + enable control):
   ros2 service call /formation/start std_srvs/srv/Trigger

4. Stop the formation (disable control + land):
   ros2 service call /formation/stop std_srvs/srv/Trigger

State Machine:
    IDLE → TAKING_OFF → HOVERING → ACTIVE → LANDING → LANDED

This launches:
1. Virtual leader node (generates trajectory)
2. Formation containment controller (computes target positions)
3. Crazyswarm2 bridge (sends GoTo commands with rate limiting)
4. RViz visualization (optional)
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
        pkg_share, 'config', 'manual_offsets_params.yaml'
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
    # These are ROS 2 "launch arguments", implemented using DeclareLaunchArgument from the launch.actions module.
    # They act as user-settable options at launch time, allowing customization of the node behavior without modifying code.
    # Each tool below declares a new launch argument for the launch file:
    #
    # - trajectory_type_arg: Allows the user to specify the type of trajectory (e.g., lemniscate, circle, hover, square)
    # - goto_rate_arg: Lets the user set how often GoTo commands are sent to the drones (in Hz)
    # - position_threshold_arg: Sets the minimum distance a drone's position must change before sending a new GoTo command
    #
    # These tools make the code more flexible and parameterizable.

    trajectory_type_arg = DeclareLaunchArgument(
        'trajectory_type',
        default_value='lemniscate',
        description='Virtual leader trajectory: circle, lemniscate, hover, square'
    )

    # goto_rate_arg = DeclareLaunchArgument(
    #     'goto_rate',
    #     default_value='10.0',
    #     description='Rate limit for GoTo commands (Hz)'
    # )

    # position_threshold_arg = DeclareLaunchArgument(
    #     'position_threshold',
    #     default_value='0.05',
    #     description='Minimum position change to send GoTo (meters)'
    # )

    # =========================================================================
    # Nodes
    # =========================================================================

    # Virtual leader generates the reference trajectory
    virtual_leader_node = Node(
        package='formation_containment_control',
        executable='virtual_leader_node.py',
        name='virtual_leader_node',
        output='screen',
        parameters=[config_file]
    )
    
    # Formation controller computes control inputs (starts disabled, waits for bridge)
    formation_controller_node = Node(
        package='formation_containment_control',
        executable='formation_containment_node.py',
        name='formation_containment_node',
        output='screen',
        parameters=[
            config_file,
            {
                'offsets_file': LaunchConfiguration('offsets_file'),
                # 'auto_enable': False,  # Wait for bridge to enable via /formation/enable
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
    
    # Delay formation controller to allow other nodes to initialize
    delayed_formation_controller = TimerAction(
        period=2.0,
        actions=[formation_controller_node]
    )
    
    return LaunchDescription([
        # Arguments
        use_rviz_arg,
        offsets_file_arg,
        trajectory_type_arg,
        # goto_rate_arg,
        # position_threshold_arg,
        
        # Nodes
        virtual_leader_node,
        crazyswarm_bridge_node,
        delayed_formation_controller,
        visualization_node,
        rviz_node,
    ])
