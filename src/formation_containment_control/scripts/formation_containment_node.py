#!/usr/bin/env python3
"""
Formation-Containment Control Node

Main ROS2 node that implements the formation-containment control strategy
from the paper "Collision-Free Formation-Containment Control Based on
Adaptive Sliding Mode Strategy for a Quadrotor Fleet Under Disturbances"

This node:
1. Subscribes to virtual leader trajectory
2. Computes control inputs for all leaders and followers
3. Publishes velocity/position commands to Crazyswarm2
4. Publishes visualization markers
5. Monitors formation and containment status

Compatible with Crazyswarm2 simulation and real Crazyflie hardware.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from typing import Dict, List, Optional

# ROS2 message types
from geometry_msgs.msg import PoseStamped, Twist, Point, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, ColorRGBA, Bool, Float64MultiArray, Empty
from visualization_msgs.msg import Marker, MarkerArray

# Crazyswarm2 full state message
try:
    from crazyflie_interfaces.msg import FullState
    CRAZYSWARM2_AVAILABLE = True
except ImportError:
    CRAZYSWARM2_AVAILABLE = False
    FullState = None

# Formation control modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formation_containment_control.controllers.formation_controller import (
    FormationController, FormationConfig
)
# from formation_containment_control.core.dynamics import QuadrotorState
from formation_containment_control.utils.math_utils import (
    quaternion_to_euler, euler_to_quaternion
)

# Data logging
try:
    from data_logger_utils import DataLogger
    DATA_LOGGER_AVAILABLE = True
except ImportError:
    DATA_LOGGER_AVAILABLE = False
    DataLogger = None


class FormationContainmentNode(Node):
    """D
    Main formation-containment control node.
    
    Implements the two-layer control strategy:
    - Layer 1: Leaders track virtual leader trajectory with formation offsets
    - Layer 2: Followers track positions inside leader convex hull
    
    Topics:
    -------
    Subscriptions:
        /virtual_leader/pose: Virtual leader position and orientation
        /cf<id>/pose: Individual drone state feedback
        
    Publications:
        /cf<id>/cmd_vel_legacy: Velocity commands for each drone
        /cf<id>/cmd_position: Position commands for each drone
        /cf<id>/cmd_full_state: Full state commands for each drone
        /formation/status: Formation status information
        /formation/markers: Visualization markers
        /formation/convex_hull: Convex hull visualization
    """
    
    def __init__(self):
        super().__init__('formation_containment_node')
        
        # Declare parameters
        self._declare_parameters()
        
        # Get parameters
        self._get_parameters()
        
        # Initialize formation controller
        self._init_controller()
        
        # Initialize ROS2 interface
        self._init_ros_interface()
        
        # State variables
        self.virtual_leader_received = False
        self.agent_states_received = {i: False for i in range(self.n_total)}
        # Auto-enable for simulation (True), wait for bridge enable for Crazyswarm2 (False)
        self.controller_enabled = self.auto_enable
        
        # Data logging start time (initialized when logger is created)
        self.log_start_time = None
        
        self.get_logger().info(
            f"Formation Containment Node initialized with "
            f"{self.n_followers} followers and {self.n_leaders} leaders"
        )
    
    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        # Agent configuration
        self.declare_parameter('n_followers', 4)
        self.declare_parameter('n_leaders', 4)
        self.declare_parameter('topology', 'paper')
        
        # Formation configuration
        self.declare_parameter('formation_type', 'square')
        self.declare_parameter('formation_scale', 1.0)
        self.declare_parameter('formation_height', 1.0)
        
        # Custom offsets file path (used when formation_type is "custom")
        self.declare_parameter('offsets_file', '')
        
        # Control parameters (from paper)
        self.declare_parameter('lambda_gain', 3.0)
        self.declare_parameter('alpha', 4.0)
        self.declare_parameter('beta', 0.125)
        self.declare_parameter('safety_distance', 0.3)
        
        # Control rate
        self.declare_parameter('control_rate', 50.0)  # Hz
        self.declare_parameter('dt', 0.02)  # Control timestep
        
        # Control mode: "position", "velocity", "full_state"
        self.declare_parameter('control_mode', 'position')
        
        # Collision avoidance
        self.declare_parameter('use_collision_avoidance', True)
        
        # Velocity limit
        self.declare_parameter('max_velocity', 0.0)  # 0 = no limit
        
        # Drone naming
        self.declare_parameter('drone_prefix', 'cf')
        self.declare_parameter('follower_ids', [1, 2, 3, 4])
        self.declare_parameter('leader_ids', [5, 6, 7, 8])
        
        # Frame
        self.declare_parameter('world_frame', 'world')
        
        # Auto-enable: True for simulation, False when using Crazyswarm2 bridge
        self.declare_parameter('auto_enable', True)
        
        # Data logging parameters
        self.declare_parameter('enable_logging', False)
        self.declare_parameter('log_rate', 10.0)
        self.declare_parameter('log_directory', 'logs')
    
    def _get_parameters(self):
        """Get parameters from ROS2 parameter server."""
        self.n_followers = self.get_parameter('n_followers').value
        self.n_leaders = self.get_parameter('n_leaders').value
        self.n_total = self.n_followers + self.n_leaders
        self.topology = self.get_parameter('topology').value
        
        self.formation_type = self.get_parameter('formation_type').value
        self.formation_scale = self.get_parameter('formation_scale').value
        self.formation_height = self.get_parameter('formation_height').value
        self.offsets_file = self.get_parameter('offsets_file').value
        
        self.lambda_gain = self.get_parameter('lambda_gain').value
        self.alpha = self.get_parameter('alpha').value
        self.beta = self.get_parameter('beta').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.use_collision_avoidance = self.get_parameter('use_collision_avoidance').value
        self.max_velocity = self.get_parameter('max_velocity').value
        
        # Control rate
        self.control_rate = self.get_parameter('control_rate').value
        self.dt = self.get_parameter('dt').value
        
        # Control mode
        self.control_mode = self.get_parameter('control_mode').value
        if self.control_mode not in ['position', 'velocity', 'full_state']:
            self.get_logger().warn(
                f"Invalid control_mode '{self.control_mode}'. "
                f"Using 'position' instead."
            )
            self.control_mode = 'position'
        
        # Drone naming
        self.drone_prefix = self.get_parameter('drone_prefix').value
        self.follower_ids = self.get_parameter('follower_ids').value
        self.leader_ids = self.get_parameter('leader_ids').value
        
        self.world_frame = self.get_parameter('world_frame').value
        
        # Auto-enable for simulation mode (no bridge)
        self.auto_enable = self.get_parameter('auto_enable').value
        
        # Data logging parameters
        self.enable_logging = self.get_parameter('enable_logging').value
        self.log_rate = self.get_parameter('log_rate').value
        self.log_directory = self.get_parameter('log_directory').value
    
    def _init_controller(self):
        """Initialize the formation controller."""
        config = FormationConfig(
            n_followers=self.n_followers,
            n_leaders=self.n_leaders,
            topology=self.topology,
            formation_type=self.formation_type,
            formation_scale=self.formation_scale,
            formation_height=self.formation_height,
            offsets_file=self.offsets_file,
            lambda_gain=self.lambda_gain,
            alpha=self.alpha,
            beta=self.beta,
            safety_distance=self.safety_distance,
            dt=self.dt,
            use_collision_avoidance=self.use_collision_avoidance,
            max_velocity=self.max_velocity
        )
        
        self.formation_controller = FormationController(config)
        
        # Log offsets file usage (only when formation_type is "custom")
        if self.formation_type.lower() == "custom" and self.offsets_file:
            self.get_logger().info(f"Using custom offsets from: {self.offsets_file}")
        
        # State storage
        self.leader_states = np.zeros((self.n_leaders, 4))
        self.leader_velocities = np.zeros((self.n_leaders, 4))
        self.follower_states = np.zeros((self.n_followers, 4))
        self.follower_velocities = np.zeros((self.n_followers, 4))
        
        self.virtual_leader_state = np.array([0.0, 0.0, self.formation_height, 0.0])
        self.virtual_leader_velocity = np.zeros(4)
        
        # Storage for control outputs (for logging)
        self.last_leader_controls = np.zeros((self.n_leaders, 4))
        self.last_follower_controls = np.zeros((self.n_followers, 4))
    
    def _init_ros_interface(self):
        """Initialize ROS2 publishers and subscribers."""
        # QoS profiles
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Virtual leader subscriber
        self.vl_sub = self.create_subscription(
            PoseStamped,
            '/virtual_leader/pose',
            self._virtual_leader_callback,
            qos_reliable
        )
        
        # Enable/disable from mission controller
        # This has to be considered like a service for the future
        self.enable_sub = self.create_subscription(
            Bool,
            '/formation/enable',
            self._enable_callback,
            qos_reliable
        )
        
        # Drone state subscribers and command publishers
        self.odom_subs: Dict[int, any] = {}
        self.cmd_pubs: Dict[int, any] = {}
        self.pose_pubs: Dict[int, any] = {}  # For position commands
        self.full_state_pubs: Dict[str, any] = {}  # For full state commands
        
        # Followers
        for i, drone_id in enumerate(self.follower_ids[:self.n_followers]):
            # State feedback (odometry topic - better than pose for velocity)
            self.odom_subs[f'follower_{i}'] = self.create_subscription(
                Odometry,
                f'/{self.drone_prefix}{drone_id}/odom',
                lambda msg, idx=i: self._follower_odom_callback(msg, idx),
                qos_best_effort
            )
            
            # Velocity command
            self.cmd_pubs[f'follower_{i}'] = self.create_publisher(
                Twist,
                f'/{self.drone_prefix}{drone_id}/cmd_vel_legacy',
                qos_reliable
            )

            # Position command (for Crazyswarm2 goTo)
            self.pose_pubs[f'follower_{i}'] = self.create_publisher(
                PoseStamped,
                f'/{self.drone_prefix}{drone_id}/cmd_position',
                qos_reliable
            )
            
            # Full state command (for Crazyswarm2 full state control)
            if CRAZYSWARM2_AVAILABLE:
                self.full_state_pubs[f'follower_{i}'] = self.create_publisher(
                    FullState,
                    f'/{self.drone_prefix}{drone_id}/cmd_full_state',
                    qos_reliable
                )
        
        # Leaders
        for i, drone_id in enumerate(self.leader_ids[:self.n_leaders]):
            # State feedback (odometry topic - better than pose for velocity)
            self.odom_subs[f'leader_{i}'] = self.create_subscription(
                Odometry,
                f'/{self.drone_prefix}{drone_id}/odom',
                lambda msg, idx=i: self._leader_odom_callback(msg, idx),
                qos_best_effort
            )
            
            self.cmd_pubs[f'leader_{i}'] = self.create_publisher(
                Twist,
                f'/{self.drone_prefix}{drone_id}/cmd_vel_legacy',
                qos_reliable
            )
            
            self.pose_pubs[f'leader_{i}'] = self.create_publisher(
                PoseStamped,
                f'/{self.drone_prefix}{drone_id}/cmd_position',
                qos_reliable
            )
            
            # Full state command (for Crazyswarm2 full state control)
            if CRAZYSWARM2_AVAILABLE:
                self.full_state_pubs[f'leader_{i}'] = self.create_publisher(
                    FullState,
                    f'/{self.drone_prefix}{drone_id}/cmd_full_state',
                    qos_reliable
                )
        
        # Status and visualization publishers
        # self.status_pub = self.create_publisher(
        #     Float64MultiArray,
        #     '/formation/status',
        #     qos_reliable
        # )
        
        # self.marker_pub = self.create_publisher(
        #     MarkerArray,
        #     '/formation/markers',
        #     qos_reliable
        # )
        
        # self.hull_marker_pub = self.create_publisher(
        #     Marker,
        #     '/formation/convex_hull',
        #     qos_reliable
        # )
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self._control_callback
        )
        
        # # Visualization timer (slower rate)
        # self.viz_timer = self.create_timer(
        #     0.1,  # 10 Hz
        #     self._visualization_callback
        # )
        
        # Data logging initialization
        self.data_logger = None
        if self.enable_logging:
            if DATA_LOGGER_AVAILABLE:
                self._init_data_logger()
                # Verify initialization succeeded before creating timer
                if self.data_logger is None or self.log_start_time is None:
                    self.get_logger().error(
                        f"Data logger initialization failed! "
                        f"data_logger={self.data_logger is not None}, "
                        f"log_start_time={self.log_start_time}"
                    )
                    self.enable_logging = False
                else:
                    # Create logging timer
                    self.logging_timer = self.create_timer(
                        1.0 / self.log_rate,
                        self._logging_callback
                    )
                    self.get_logger().info(f"Data logging enabled at {self.log_rate} Hz. log_start_time={self.log_start_time}")
            else:
                self.get_logger().warn(
                    "Data logging requested but data_logger_utils not available. "
                    "Logging disabled."
                )
                self.enable_logging = False
    
    def _init_data_logger(self):
        """Initialize data logger and write metadata."""
        try:
            self.data_logger = DataLogger(self.log_directory)
            
            # Store start time for relative timestamp calculation
            start_time_ns = self.get_clock().now().seconds_nanoseconds()
            self.log_start_time = start_time_ns[0] + start_time_ns[1] * 1e-9
            
            self.get_logger().info(f"Data logger initialized. Start time: {self.log_start_time}")
            
            # Write metadata (inside try block to ensure log_start_time stays set)
            metadata = {
                'n_followers': self.n_followers,
                'n_leaders': self.n_leaders,
                'n_total': self.n_total,
                'topology': self.topology,
                'formation_type': self.formation_type,
                'formation_scale': self.formation_scale,
                'formation_height': self.formation_height,
                'control_rate': self.control_rate,
                'log_rate': self.log_rate,
                'dt': self.dt,
                'control_mode': self.control_mode,
                'lambda_gain': self.lambda_gain,
                'alpha': self.alpha,
                'beta': self.beta,
                'safety_distance': self.safety_distance,
                'use_collision_avoidance': self.use_collision_avoidance,
                'max_velocity': self.max_velocity,
                'drone_prefix': self.drone_prefix,
                'follower_ids': self.follower_ids,
                'leader_ids': self.leader_ids,
                'world_frame': self.world_frame
            }
            
            self.data_logger.write_metadata(metadata)
            
            # Verify log_start_time is still set after metadata write
            if self.log_start_time is None:
                self.get_logger().error("log_start_time became None after metadata write!")
            else:
                self.get_logger().info(f"Data logger fully initialized. log_start_time verified: {self.log_start_time}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to initialize data logger: {e}", exc_info=True)
            self.data_logger = None
            self.log_start_time = None
            raise
    
    def _enable_callback(self, msg: Bool):
        """Handle enable/disable from mission controller."""
        self.controller_enabled = msg.data
        status = "ENABLED" if msg.data else "DISABLED"
        self.get_logger().info(f"Formation controller {status}")
    
    def _virtual_leader_callback(self, msg: PoseStamped):
        """Handle virtual leader pose updates."""
        # Extract position
        pos = msg.pose.position
        
        # Extract yaw from quaternion
        q = msg.pose.orientation
        _, _, yaw = quaternion_to_euler(np.array([q.x, q.y, q.z, q.w]))
        
        # Compute velocity (numerical differentiation)
        new_state = np.array([pos.x, pos.y, pos.z, yaw])          # TODO: Change the differentiator
        if self.virtual_leader_received:
            self.virtual_leader_velocity = (new_state - self.virtual_leader_state) / self.dt
        
        self.virtual_leader_state = new_state
        self.virtual_leader_received = True
        
        # Update formation controller
        self.formation_controller.set_virtual_leader_state(
            self.virtual_leader_state,
            self.virtual_leader_velocity
        )
    
    def _follower_odom_callback(self, msg: Odometry, idx: int):
        """
        Handle follower odometry updates.
        
        Odometry provides both position and velocity directly, which is more
        accurate than numerical differentiation from pose messages.
        """
        # Extract position from pose
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        
        # Extract yaw from quaternion
        _, _, yaw = quaternion_to_euler(np.array([q.x, q.y, q.z, q.w]))
        
        # Store state [x, y, z, yaw]
        self.follower_states[idx] = np.array([pos.x, pos.y, pos.z, yaw])
        
        # Extract velocity directly from odometry (much more accurate!)
        vel = msg.twist.twist.linear
        vyaw = msg.twist.twist.angular.z
        
        # Store velocity [vx, vy, vz, vyaw]
        self.follower_velocities[idx] = np.array([vel.x, vel.y, vel.z, vyaw])
        
        self.agent_states_received[idx] = True
    
    def _leader_odom_callback(self, msg: Odometry, idx: int):
        """
        Handle leader odometry updates.
        
        Odometry provides both position and velocity directly, which is more
        accurate than numerical differentiation from pose messages.
        """
        # Extract position from pose
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        
        # Extract yaw from quaternion
        _, _, yaw = quaternion_to_euler(np.array([q.x, q.y, q.z, q.w]))
        
        # Store state [x, y, z, yaw]
        self.leader_states[idx] = np.array([pos.x, pos.y, pos.z, yaw])
        
        # Extract velocity directly from odometry (much more accurate!)
        vel = msg.twist.twist.linear
        vyaw = msg.twist.twist.angular.z
        
        # Store velocity [vx, vy, vz, vyaw]
        self.leader_velocities[idx] = np.array([vel.x, vel.y, vel.z, vyaw])
        
        leader_idx = self.n_followers + idx
        self.agent_states_received[leader_idx] = True
    
    def _control_callback(self):
        """Main control loop callback."""
        # TODO: Use velocities instead of use Accelerations

        if not self.virtual_leader_received:
            return
        
        if not self.controller_enabled:
            return  # Skip control when disabled
        
        # Get target positions for GoTo commands (open-loop, no odometry needed)
        leader_targets = self.formation_controller.get_leader_target_positions()
        
        # Calculate follower targets using leader targets (open-loop)
        # This uses the desired positions of leaders instead of their actual odometry
        follower_targets = self._compute_follower_targets_open_loop(leader_targets)
        
        # Update formation controller with current states (if available)
        # For open-loop, we can use targets as states or skip this
        # This is still called for status checking, but targets are calculated independently
        self.formation_controller.update_agent_states(
            self.leader_states,
            self.leader_velocities,
            self.follower_states,
            self.follower_velocities
        )
        
        # Compute control inputs (accelerations) - required for full_state mode
        leader_controls, follower_controls = self.formation_controller.compute_all_controls()
        
        # Store for logging
        self.last_leader_controls = leader_controls
        self.last_follower_controls = follower_controls
        
        # Publish leader commands based on control mode
        for i, drone_id in enumerate(self.leader_ids[:self.n_leaders]):
            if self.control_mode == 'full_state':
                # Full state control: use accelerations directly
                self._publish_full_state(f'leader_{i}', leader_controls[i])
            elif self.control_mode == 'velocity':
                # Velocity control: integrate accelerations
                self._publish_velocity(f'leader_{i}', leader_controls[i])
            else:  # position mode (default)
                # Position control: use target positions
                if i < len(leader_targets):
                    self._publish_position(f'leader_{i}', leader_targets[i])
        
        # Publish follower commands based on control mode
        for i, drone_id in enumerate(self.follower_ids[:self.n_followers]):
            if self.control_mode == 'full_state':
                # Full state control: use accelerations directly
                self._publish_full_state(f'follower_{i}', follower_controls[i])
            elif self.control_mode == 'velocity':
                # Velocity control: integrate accelerations
                self._publish_velocity(f'follower_{i}', follower_controls[i])
            else:  # position mode (default)
                # Position control: use target positions
                if i < len(follower_targets):
                    self._publish_position(f'follower_{i}', follower_targets[i])
        
        # Publish status
        # self._publish_status()
    
    # Velocity control
    def _publish_velocity(self, agent_key: str, control: np.ndarray):
        """Publish velocity control command for an agent."""
        cmd = Twist()
        
        # The control is [ax, ay, az, Omega] in body frame
        # Convert to velocity command (simplified - assumes small dt)
        # TODO: change the differentatior
        cmd.linear.x = float(control[0] * self.dt)
        cmd.linear.y = float(control[1] * self.dt)
        cmd.linear.z = float(control[2] * self.dt)
        cmd.angular.z = float(control[3] * self.dt)
        
        if agent_key in self.cmd_pubs:
            self.cmd_pubs[agent_key].publish(cmd)
    
    def _compute_follower_targets_open_loop(self, leader_targets: np.ndarray) -> np.ndarray:
        """
        Compute follower target positions using leader targets (open-loop).
        
        This method calculates follower targets based on the desired positions
        of leaders (not their actual odometry), enabling open-loop control.
        
        Uses the containment weights from the interaction network:
        follower_target = Σ (weights[i,j] × leader_target[j])
        
        Args:
            leader_targets: Array of leader target positions, shape (n_leaders, 4)
            
        Returns:
            Array of follower target positions, shape (n_followers, 4)
        """
        # Get containment weights from the network
        containment_weights = self.formation_controller.network.laplacian.containment_weights
        
        # Calculate follower targets: weights @ leader_targets
        # This line uses the @ operator, which in Python (and numpy) performs matrix multiplication.
        # Here, `containment_weights` is a 2D numpy array of shape (n_followers, n_leaders),
        # and `leader_targets` is an array of shape (n_leaders, 4) containing the desired target
        # positions for each leader.
        # The operation performs a matrix multiplication so that each follower's target position
        # is computed as a weighted sum of the leader targets, according to the containment weights.
        # Effectively: follower_targets[i] = sum_j containment_weights[i, j] * leader_targets[j]
        follower_targets = containment_weights @ leader_targets
        
        return follower_targets
    
    # Position control
    def _publish_position(self, agent_key: str, target_pos: np.ndarray):
        """Publish position command for GoTo service (Crazyswarm2)."""
        pose = PoseStamped()
        pose.header.frame_id = self.world_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = float(target_pos[0])
        pose.pose.position.y = float(target_pos[1])
        pose.pose.position.z = float(target_pos[2]) if len(target_pos) > 2 else self.formation_height
        
        # Set orientation (yaw from target if available)
        yaw = float(target_pos[3]) if len(target_pos) > 3 else 0.0
        q = euler_to_quaternion(0.0, 0.0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        
        if agent_key in self.pose_pubs:
            self.pose_pubs[agent_key].publish(pose)
    
    # Full state control
    def _publish_full_state(self, agent_key: str, control: np.ndarray):
        """
        Publish full state command for Crazyswarm2 using odometry data.
        
        This function constructs a FullState message using:
        - Position & Velocity: Directly from odometry (most accurate)
        - Acceleration: From controller output [a_x, a_y, a_z]
        - Orientation: From odometry yaw
        - Angular rates: [0, 0, Ω] where Ω is from controller output
        
        The controller outputs accelerations [a_x, a_y, a_z, Ω] where:
        - [a_x, a_y, a_z] are linear accelerations in world frame
        - Ω is yaw rate
        
        IMPORTANT: Full state control REQUIRES accurate position and velocity
        from odometry. This function uses the odometry data that was received
        via the _follower_odom_callback and _leader_odom_callback methods.
        
        Args:
            agent_key: Key identifying the agent ('leader_0', 'follower_1', etc.)
            control: Control output from controller [a_x, a_y, a_z, Ω]
        """
        if not CRAZYSWARM2_AVAILABLE or agent_key not in self.full_state_pubs:
            return
        
        # Determine agent index and get current state from odometry
        has_state_feedback = False
        if agent_key.startswith('leader_'):
            idx = int(agent_key.split('_')[1])
            leader_idx = self.n_followers + idx
            has_state_feedback = self.agent_states_received.get(leader_idx, False)
            current_state = self.leader_states[idx]
            current_velocity = self.leader_velocities[idx]
        elif agent_key.startswith('follower_'):
            idx = int(agent_key.split('_')[1])
            has_state_feedback = self.agent_states_received.get(idx, False)
            current_state = self.follower_states[idx]
            current_velocity = self.follower_velocities[idx]
        else:
            self.get_logger().warn(f"Unknown agent key: {agent_key}")
            return
        
        # Warn if odometry data is not available (but continue anyway)
        if not has_state_feedback:
            self.get_logger().warn(
                f"No odometry data received for {agent_key}. "
                f"Full state control requires position/velocity from /{self.drone_prefix}*/odom topic. "
                f"Using zero-initialized state (will cause tracking errors)."
            )
        
        # Create FullState message
        full_state = FullState()
        full_state.header.frame_id = self.world_frame
        full_state.header.stamp = self.get_clock().now().to_msg()
        
        # Position from odometry [x, y, z]
        # This comes from msg.pose.pose.position in the odometry callback
        full_state.pose.position.x = float(current_state[0])
        full_state.pose.position.y = float(current_state[1])
        full_state.pose.position.z = float(current_state[2])
        
        # Velocity from odometry [vx, vy, vz]
        # This comes directly from msg.twist.twist.linear in the odometry callback
        # Much more accurate than numerical differentiation!
        # Clamp linear velocities to [-20, 20] m/s
        full_state.twist.linear.x = float(np.clip(current_velocity[0], -2.0, 2.0))
        full_state.twist.linear.y = float(np.clip(current_velocity[1], -2.0, 2.0))
        full_state.twist.linear.z = float(np.clip(current_velocity[2], -2.0, 2.0))
        
        # Acceleration from controller output [a_x, a_y, a_z]
        # This is the PRIMARY control input from your SGASMC controller
        # Clamp linear accelerations to [-20, 20] m/s²
        full_state.acc.x = float(np.clip(control[0], -2.0, 2.0))
        full_state.acc.y = float(np.clip(control[1], -2.0, 2.0))
        full_state.acc.z = float(np.clip(control[2], -2.0, 2.0))
        
        # Orientation (quaternion) from odometry yaw
        # Yaw comes from msg.pose.pose.orientation in the odometry callback
        yaw = float(current_state[3])
        q = euler_to_quaternion(0.0, 0.0, yaw)
        full_state.pose.orientation.w = q[3]
        full_state.pose.orientation.x = q[0]
        full_state.pose.orientation.y = q[1]
        full_state.pose.orientation.z = q[2]
        
        # Angular rates: [roll_rate, pitch_rate, yaw_rate]
        # Roll and pitch rates are 0 (not controlled by formation controller)
        # Yaw rate comes from controller output Ω
        full_state.twist.angular.x = 0.0  # roll_rate (not controlled)
        full_state.twist.angular.y = 0.0  # pitch_rate (not controlled)
        # full_state.twist.angular.z = float(control[3])  # yaw_rate (Ω from controller)
        # Integrate the angular acceleration (control[3]) to update yaw velocity and assign to yaw_rate
        # Use the yaw rate from odometry as the base value instead of the last calculated value
        # now_ns = self.get_clock().now().seconds_nanoseconds()
        # now = now_ns[0] + now_ns[1] * 1e-9

        # if not hasattr(self, "last_control_time"):
        #     self.last_control_time = now

        # dt = now - self.last_control_time
        # # Prevent large jumps on first call or poor timer
        # if dt < 0 or dt > 2.0:
        #     dt = self.dt if hasattr(self, "dt") else 0.02

        # Integrate: ω_new = ω_odom + alpha * dt
        # Use yaw rate from odometry (current_velocity[3]) instead of last calculated value
        yaw_rate_odom = float(current_velocity[3])  # Yaw rate from odometry
        yaw_rate = yaw_rate_odom + float(control[3]) * self.dt
        # Clamp angular velocity (yaw rate) to [-20, 20] rad/s
        full_state.twist.angular.z = float(np.clip(yaw_rate, -2.0, 2.0))

        # Store time for next integration
        # self.last_control_time = now
        # Publish full state command
        self.full_state_pubs[agent_key].publish(full_state)
    
    def _logging_callback(self):
        """Data logging callback - collects and saves system state."""
        # Debug: Log that callback was invoked (only first few times to avoid spam)
        if not hasattr(self, '_logging_callback_count'):
            self._logging_callback_count = 0
        self._logging_callback_count += 1
        if self._logging_callback_count <= 3:
            self.get_logger().info(f"Logging callback invoked (count: {self._logging_callback_count})")
            # Debug: Check the actual state
            self.get_logger().info(f"  data_logger: {self.data_logger is not None}, log_start_time: {self.log_start_time}, hasattr log_start_time: {hasattr(self, 'log_start_time')}")
        
        if not self.data_logger:
            self.get_logger().warn("Logging callback called but data_logger is None")
            return
        
        if self.log_start_time is None:
            self.get_logger().warn(f"Logging callback called but log_start_time is None. data_logger exists: {self.data_logger is not None}, hasattr: {hasattr(self, 'log_start_time')}")
            # Try to re-initialize if data_logger exists but log_start_time is None
            if self.data_logger is not None:
                self.get_logger().warn("Attempting to fix log_start_time...")
                start_time_ns = self.get_clock().now().seconds_nanoseconds()
                self.log_start_time = start_time_ns[0] + start_time_ns[1] * 1e-9
                self.get_logger().info(f"Re-initialized log_start_time to: {self.log_start_time}")
                return  # Return this time, next call should work
            return
        
        # Get current timestamp (relative time since logging started)
        current_time_ns = self.get_clock().now().seconds_nanoseconds()
        current_time = current_time_ns[0] + current_time_ns[1] * 1e-9
        timestamp = current_time - self.log_start_time
        
        # Ensure timestamp is non-negative (shouldn't happen, but safety check)
        if timestamp < 0:
            self.get_logger().warn(f"Negative timestamp detected: {timestamp}. Skipping log entry.")
            return
        
        # Log follower data
        try:
            for i in range(self.n_followers):
                agent_id = i
                agent_type = 'follower'
                
                # Log state
                self.data_logger.log_state(timestamp, agent_id, agent_type, self.follower_states[i])
                
                # Log velocity
                self.data_logger.log_velocity(timestamp, agent_id, agent_type, self.follower_velocities[i])
                
                # Log acceleration (control input)
                if i < len(self.last_follower_controls):
                    self.data_logger.log_acceleration(timestamp, agent_id, agent_type, self.last_follower_controls[i])
                
                # Log adaptive gain K_c
                try:
                    controller_state = self.formation_controller.follower_controllers[i].controller.get_state()
                    K_c = controller_state['K_c']
                    self.data_logger.log_adaptive_gain(timestamp, agent_id, agent_type, K_c)
                except (IndexError, KeyError, AttributeError):
                    # Silently skip if controller state not available
                    pass
        except Exception as e:
            self.get_logger().error(f"Error logging follower data: {e}")
        
        # Log leader data
        try:
            for i in range(self.n_leaders):
                agent_id = i
                agent_type = 'leader'
                
                # Log state
                self.data_logger.log_state(timestamp, agent_id, agent_type, self.leader_states[i])
                
                # Log velocity
                self.data_logger.log_velocity(timestamp, agent_id, agent_type, self.leader_velocities[i])
                
                # Log acceleration (control input)
                if i < len(self.last_leader_controls):
                    self.data_logger.log_acceleration(timestamp, agent_id, agent_type, self.last_leader_controls[i])
                
                # Log adaptive gain K_c
                try:
                    controller_state = self.formation_controller.leader_controllers[i].controller.get_state()
                    K_c = controller_state['K_c']
                    self.data_logger.log_adaptive_gain(timestamp, agent_id, agent_type, K_c)
                except (IndexError, KeyError, AttributeError):
                    # Silently skip if controller state not available
                    pass
        except Exception as e:
            self.get_logger().error(f"Error logging leader data: {e}")
        
        # Log formation errors (less frequently to avoid overhead)
        # Only compute every 10th logging cycle
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        
        self._log_counter += 1
        if self._log_counter >= 10:
            self._log_counter = 0
            
            try:
                status = self.formation_controller.check_formation_status()
                
                # Log leader errors
                if 'leader_errors' in status:
                    for i, error in enumerate(status['leader_errors']):
                        self.data_logger.log_error(timestamp, i, 'leader', error)
                
                # Log follower errors
                if 'follower_errors' in status:
                    for i, error in enumerate(status['follower_errors']):
                        self.data_logger.log_error(timestamp, i, 'follower', error)
            except Exception as e:
                self.get_logger().warn(f"Error computing formation status for logging: {e}")
        
        # Flush to disk periodically (every 10 samples to ensure data is saved)
        if hasattr(self, '_flush_counter'):
            self._flush_counter += 1
            if self._flush_counter >= 10:
                self.data_logger.flush()
                self._flush_counter = 0
        else:
            self._flush_counter = 0
    
    # def _publish_status(self):
    #     """Publish formation status."""
    #     status = self.formation_controller.check_formation_status()
        
    #     msg = Float64MultiArray()
    #     msg.data = [
    #         float(status['formation_achieved']),
    #         float(status['containment_achieved']),
    #         float(status['collision_free']),
    #         status['max_leader_error'],
    #         status['max_follower_error'],
    #         status['min_inter_agent_distance'],
    #         status['convex_hull_volume']
    #     ]
        
    #     self.status_pub.publish(msg)
    
    # def _visualization_callback(self):
    #     """Publish visualization markers."""
    #     marker_array = MarkerArray()
    #     timestamp = self.get_clock().now().to_msg()
        
    #     # Virtual leader marker (green sphere)
    #     vl_marker = Marker()
    #     vl_marker.header.frame_id = self.world_frame
    #     vl_marker.header.stamp = timestamp
    #     vl_marker.ns = "virtual_leader"
    #     vl_marker.id = 0
    #     vl_marker.type = Marker.SPHERE
    #     vl_marker.action = Marker.ADD
    #     vl_marker.pose.position.x = self.virtual_leader_state[0]
    #     vl_marker.pose.position.y = self.virtual_leader_state[1]
    #     vl_marker.pose.position.z = self.virtual_leader_state[2]
    #     vl_marker.scale.x = 0.3
    #     vl_marker.scale.y = 0.3
    #     vl_marker.scale.z = 0.3
    #     vl_marker.color.r = 0.0
    #     vl_marker.color.g = 1.0
    #     vl_marker.color.b = 0.0
    #     vl_marker.color.a = 1.0
    #     marker_array.markers.append(vl_marker)
        
    #     # Leader markers (colored spheres)
    #     leader_colors = [
    #         (1.0, 0.0, 0.0),  # Red
    #         (0.0, 0.0, 1.0),  # Blue
    #         (1.0, 1.0, 0.0),  # Yellow
    #         (1.0, 0.0, 1.0),  # Magenta
    #     ]
        
    #     for i in range(self.n_leaders):
    #         marker = Marker()
    #         marker.header.frame_id = self.world_frame
    #         marker.header.stamp = timestamp
    #         marker.ns = "leaders"
    #         marker.id = i
    #         marker.type = Marker.SPHERE
    #         marker.action = Marker.ADD
    #         marker.pose.position.x = self.leader_states[i, 0]
    #         marker.pose.position.y = self.leader_states[i, 1]
    #         marker.pose.position.z = self.leader_states[i, 2]
    #         marker.scale.x = 0.25
    #         marker.scale.y = 0.25
    #         marker.scale.z = 0.25
    #         color = leader_colors[i % len(leader_colors)]
    #         marker.color.r = color[0]
    #         marker.color.g = color[1]
    #         marker.color.b = color[2]
    #         marker.color.a = 1.0
    #         marker_array.markers.append(marker)
        
    #     # Follower markers (transparent spheres)
    #     for i in range(self.n_followers):
    #         marker = Marker()
    #         marker.header.frame_id = self.world_frame
    #         marker.header.stamp = timestamp
    #         marker.ns = "followers"
    #         marker.id = i
    #         marker.type = Marker.SPHERE
    #         marker.action = Marker.ADD
    #         marker.pose.position.x = self.follower_states[i, 0]
    #         marker.pose.position.y = self.follower_states[i, 1]
    #         marker.pose.position.z = self.follower_states[i, 2]
    #         marker.scale.x = 0.2
    #         marker.scale.y = 0.2
    #         marker.scale.z = 0.2
    #         marker.color.r = 0.5
    #         marker.color.g = 0.5
    #         marker.color.b = 0.5
    #         marker.color.a = 0.7
    #         marker_array.markers.append(marker)
        
    #     self.marker_pub.publish(marker_array)
        
    #     # Convex hull visualization
    #     self._publish_convex_hull(timestamp)
    
    # def _publish_convex_hull(self, timestamp):
    #     """Publish convex hull marker."""
    #     hull_data = self.formation_controller.convex_hull.get_visualization_data()
        
    #     if len(hull_data['vertices']) < 3:
    #         return
        
    #     # Line strip for hull edges
    #     hull_marker = Marker()
    #     hull_marker.header.frame_id = self.world_frame
    #     hull_marker.header.stamp = timestamp
    #     hull_marker.ns = "convex_hull"
    #     hull_marker.id = 0
    #     hull_marker.type = Marker.LINE_STRIP
    #     hull_marker.action = Marker.ADD
    #     hull_marker.scale.x = 0.1  # Line width
    #     hull_marker.color.r = 0.0
    #     hull_marker.color.g = 1.0
    #     hull_marker.color.b = 1.0
    #     hull_marker.color.a = 0.5
        
    #     # Add hull vertices as a closed polygon
    #     vertices = hull_data['vertices']
    #     for vertex in vertices:
    #         p = Point()
    #         p.x = float(vertex[0])
    #         p.y = float(vertex[1])
    #         p.z = float(vertex[2]) if len(vertex) > 2 else self.formation_height
    #         hull_marker.points.append(p)
        
    #     # Close the polygon
    #     if len(vertices) > 0:
    #         p = Point()
    #         p.x = float(vertices[0][0])
    #         p.y = float(vertices[0][1])
    #         p.z = float(vertices[0][2]) if len(vertices[0]) > 2 else self.formation_height
    #         hull_marker.points.append(p)
        
    #     self.hull_marker_pub.publish(hull_marker)


def main(args=None):
    rclpy.init(args=args)
    node = FormationContainmentNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Flush and close data logger before shutdown
        if hasattr(node, 'data_logger') and node.data_logger is not None:
            node.get_logger().info("Flushing and closing data logger...")
            node.data_logger.flush()
            node.data_logger.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

