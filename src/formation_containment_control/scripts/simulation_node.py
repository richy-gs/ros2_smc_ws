#!/usr/bin/env python3
"""
Simulation Node for Formation-Containment Control

Provides a standalone simulation environment for testing the control algorithms
without requiring the full Crazyswarm2 simulation stack.

Features:
- Simulates quadrotor dynamics (reduced tracking model)
- Applies control inputs and integrates states
- Optional turbulence (Von Kármán model)
- Publishes simulated drone states as odometry

This is useful for:
- Algorithm development and debugging
- Unit testing
- Demonstration without Gazebo/CFLib
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from typing import Dict, List

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formation_containment_control.core.dynamics import (
    ReducedTrackingModel, QuadrotorState, VonKarmanTurbulence
)
from formation_containment_control.utils.math_utils import euler_to_quaternion


class SimulatedDrone:
    """
    Simulated drone with dynamics integration.
    """
    
    def __init__(self, drone_id: int, 
                 initial_position: np.ndarray = None,
                 use_turbulence: bool = False):
        """
        Initialize simulated drone.
        
        Args:
            drone_id: Drone identifier
            initial_position: Initial [x, y, z] position
            use_turbulence: Enable Von Kármán turbulence
        """
        self.drone_id = drone_id
        self.dynamics = ReducedTrackingModel()
        
        # State initialization
        if initial_position is None:
            initial_position = np.array([0.0, 0.0, 0.0])
        
        self.state = QuadrotorState(
            x=initial_position[0],
            y=initial_position[1],
            z=initial_position[2]
        )
        
        # Control input storage
        self.control_input = np.zeros(4)
        
        # Turbulence
        self.use_turbulence = use_turbulence
        if use_turbulence:
            self.turbulence = VonKarmanTurbulence(intensity=0.3)
        else:
            self.turbulence = None
    
    def update(self, dt: float):
        """
        Update drone state with current control input.
        
        Args:
            dt: Time step
        """
        # Get perturbation
        delta = None
        if self.turbulence is not None:
            delta = self.turbulence.generate_disturbance()
        
        # Integrate dynamics
        self.state = self.dynamics.integrate_rk4(
            self.state, self.control_input, dt, delta
        )
    
    def set_control(self, control: np.ndarray):
        """Set control input [ax, ay, az, omega]."""
        self.control_input = control.copy()
    
    def get_state_array(self) -> np.ndarray:
        """Get state as [x, y, z, yaw]."""
        return self.state.to_reduced_state()
    
    def get_velocity_array(self) -> np.ndarray:
        """Get velocity as [vx, vy, vz, yaw_rate]."""
        return self.state.to_reduced_velocity()


class SimulationNode(Node):
    """
    ROS2 node for simulating the drone fleet.
    """
    
    def __init__(self):
        super().__init__('simulation_node')
        
        # Parameters
        self.declare_parameter('n_followers', 4)
        self.declare_parameter('n_leaders', 4)
        self.declare_parameter('simulation_rate', 100.0)  # Hz
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('use_turbulence', False)
        self.declare_parameter('turbulence_start_time', 45.0)  # Start turbulence at t=45s
        self.declare_parameter('drone_prefix', 'cf')
        self.declare_parameter('follower_ids', [1, 2, 3, 4])
        self.declare_parameter('leader_ids', [5, 6, 7, 8])
        
        # Formation initial positions from paper (Section 4)
        self.declare_parameter('initial_follower_positions', 
                              [-0.7, 1.0, 0.0,   # F1
                               -2.0, 0.0, 0.0,   # F2
                                0.7, -1.0, 0.0,  # F3
                                1.0, 1.0, 0.0])  # F4
        self.declare_parameter('initial_leader_positions',
                              [1.0, 0.0, 0.0,    # L1
                               -1.0, 0.0, 0.0,   # L2
                                0.0, 1.0, 0.0,   # L3
                                0.0, -1.0, 0.0]) # L4
        
        self.n_followers = self.get_parameter('n_followers').value
        self.n_leaders = self.get_parameter('n_leaders').value
        self.sim_rate = self.get_parameter('simulation_rate').value
        self.frame = self.get_parameter('world_frame').value
        self.use_turbulence = self.get_parameter('use_turbulence').value
        self.turbulence_start = self.get_parameter('turbulence_start_time').value
        self.drone_prefix = self.get_parameter('drone_prefix').value
        self.follower_ids = self.get_parameter('follower_ids').value
        self.leader_ids = self.get_parameter('leader_ids').value
        
        # Get initial positions
        init_follower_pos = self.get_parameter('initial_follower_positions').value
        init_leader_pos = self.get_parameter('initial_leader_positions').value
        
        # Create simulated drones
        self.followers: List[SimulatedDrone] = []
        self.leaders: List[SimulatedDrone] = []
        
        # Initialize followers
        for i in range(self.n_followers):
            pos = np.array(init_follower_pos[i*3:(i+1)*3]) if len(init_follower_pos) >= (i+1)*3 else np.zeros(3)
            drone = SimulatedDrone(
                drone_id=self.follower_ids[i] if i < len(self.follower_ids) else i,
                initial_position=pos
            )
            self.followers.append(drone)
        
        # Initialize leaders
        for i in range(self.n_leaders):
            pos = np.array(init_leader_pos[i*3:(i+1)*3]) if len(init_leader_pos) >= (i+1)*3 else np.zeros(3)
            drone = SimulatedDrone(
                drone_id=self.leader_ids[i] if i < len(self.leader_ids) else self.n_followers + i,
                initial_position=pos
            )
            self.leaders.append(drone)
        
        # Publishers and subscribers
        self._setup_ros_interface()
        
        # Simulation timing
        self.dt = 1.0 / self.sim_rate
        self.start_time = self.get_clock().now()
        
        # Timer
        self.sim_timer = self.create_timer(self.dt, self._simulation_step)
        
        self.get_logger().info(
            f"Simulation Node started: {self.n_followers} followers, {self.n_leaders} leaders"
        )
    
    def _setup_ros_interface(self):
        """Setup ROS2 publishers and subscribers."""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Odometry publishers (simulated sensor output)
        self.odom_pubs: Dict[str, any] = {}
        
        for i, drone_id in enumerate(self.follower_ids[:self.n_followers]):
            topic = f'/{self.drone_prefix}{drone_id}/odom'
            self.odom_pubs[f'follower_{i}'] = self.create_publisher(Odometry, topic, qos)
        
        for i, drone_id in enumerate(self.leader_ids[:self.n_leaders]):
            topic = f'/{self.drone_prefix}{drone_id}/odom'
            self.odom_pubs[f'leader_{i}'] = self.create_publisher(Odometry, topic, qos)
        
        # Control input subscribers
        self.cmd_subs: Dict[str, any] = {}
        
        for i, drone_id in enumerate(self.follower_ids[:self.n_followers]):
            topic = f'/{self.drone_prefix}{drone_id}/cmd_vel'
            self.cmd_subs[f'follower_{i}'] = self.create_subscription(
                Twist, topic,
                lambda msg, idx=i: self._follower_cmd_callback(msg, idx),
                10
            )
        
        for i, drone_id in enumerate(self.leader_ids[:self.n_leaders]):
            topic = f'/{self.drone_prefix}{drone_id}/cmd_vel'
            self.cmd_subs[f'leader_{i}'] = self.create_subscription(
                Twist, topic,
                lambda msg, idx=i: self._leader_cmd_callback(msg, idx),
                10
            )
    
    def _follower_cmd_callback(self, msg: Twist, idx: int):
        """Handle follower control commands."""
        if idx < len(self.followers):
            # Convert velocity command to acceleration (simple model)
            control = np.array([
                msg.linear.x / self.dt,
                msg.linear.y / self.dt,
                msg.linear.z / self.dt,
                msg.angular.z / self.dt
            ])
            # Limit accelerations
            control = np.clip(control, -5.0, 5.0)
            self.followers[idx].set_control(control)
    
    def _leader_cmd_callback(self, msg: Twist, idx: int):
        """Handle leader control commands."""
        if idx < len(self.leaders):
            control = np.array([
                msg.linear.x / self.dt,
                msg.linear.y / self.dt,
                msg.linear.z / self.dt,
                msg.angular.z / self.dt
            ])
            control = np.clip(control, -5.0, 5.0)
            self.leaders[idx].set_control(control)
    
    def _simulation_step(self):
        """Execute one simulation step."""
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9
        
        # Enable turbulence after specified time
        if self.use_turbulence and t >= self.turbulence_start:
            for drone in self.followers + self.leaders:
                if drone.turbulence is None:
                    drone.turbulence = VonKarmanTurbulence(intensity=0.3)
                    drone.use_turbulence = True
        
        # Update all drone states
        for drone in self.followers:
            drone.update(self.dt)
        
        for drone in self.leaders:
            drone.update(self.dt)
        
        # Publish odometry
        timestamp = now.to_msg()
        
        for i, drone in enumerate(self.followers):
            self._publish_odometry(f'follower_{i}', drone, timestamp)
        
        for i, drone in enumerate(self.leaders):
            self._publish_odometry(f'leader_{i}', drone, timestamp)
    
    def _publish_odometry(self, key: str, drone: SimulatedDrone, timestamp):
        """Publish odometry for a drone."""
        state = drone.get_state_array()
        velocity = drone.get_velocity_array()
        
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = self.frame
        odom.child_frame_id = f'{self.drone_prefix}{drone.drone_id}'
        
        # Position
        odom.pose.pose.position.x = state[0]
        odom.pose.pose.position.y = state[1]
        odom.pose.pose.position.z = state[2]
        
        # Orientation
        q = euler_to_quaternion(0, 0, state[3])
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        # Velocity
        odom.twist.twist.linear.x = velocity[0]
        odom.twist.twist.linear.y = velocity[1]
        odom.twist.twist.linear.z = velocity[2]
        odom.twist.twist.angular.z = velocity[3]
        
        if key in self.odom_pubs:
            self.odom_pubs[key].publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

