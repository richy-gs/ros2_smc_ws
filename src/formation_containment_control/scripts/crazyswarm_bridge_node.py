#!/usr/bin/env python3
"""
Crazyswarm2 Bridge Node

Bridges the formation-containment controller with Crazyswarm2.
Translates control commands to Crazyswarm2 API calls.

Crazyswarm2 Control Modes:
1. cmdVelocityWorld - Velocity in world frame
2. cmdFullState - Position, velocity, acceleration
3. goTo - High-level position waypoint

This bridge supports multiple control modes and handles the
interface between our controller and the Crazyflie firmware.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
from typing import Dict, Optional

# ROS2 messages
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Bool

# Try to import Crazyswarm2 interfaces
try:
    from crazyflie_interfaces.msg import FullState, Hover, Position
    from crazyflie_interfaces.srv import GoTo, Takeoff, Land, NotifySetpointsStop
    CRAZYSWARM2_AVAILABLE = True
except ImportError:
    CRAZYSWARM2_AVAILABLE = False
    print("Warning: crazyflie_interfaces not found. Running in simulation mode.")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formation_containment_control.utils.math_utils import (
    quaternion_to_euler, euler_to_quaternion
)


class CrazyswarmBridge(Node):
    """
    Bridge between formation controller and Crazyswarm2.
    
    Control Modes:
    - velocity: Send velocity commands (cmdVelocityWorld)
    - position: Send position setpoints (cmdPosition)
    - full_state: Send complete state (cmdFullState)
    """
    
    def __init__(self):
        super().__init__('crazyswarm_bridge_node')
        
        # Parameters
        self.declare_parameter('n_drones', 8)
        self.declare_parameter('drone_prefix', 'cf')
        self.declare_parameter('drone_ids', [1, 2, 3, 4, 5, 6, 7, 8])
        self.declare_parameter('control_mode', 'velocity')  # velocity, position, full_state
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('use_takeoff_service', True)
        self.declare_parameter('default_height', 1.0)
        
        self.n_drones = self.get_parameter('n_drones').value
        self.prefix = self.get_parameter('drone_prefix').value
        self.drone_ids = self.get_parameter('drone_ids').value
        self.control_mode = self.get_parameter('control_mode').value
        self.frame = self.get_parameter('world_frame').value
        self.use_takeoff = self.get_parameter('use_takeoff_service').value
        self.default_height = self.get_parameter('default_height').value
        
        # State tracking
        self.drone_states: Dict[int, np.ndarray] = {}
        self.is_flying: Dict[int, bool] = {}
        
        # Initialize interface
        self._setup_interface()
        
        self.get_logger().info(
            f"Crazyswarm Bridge initialized for {self.n_drones} drones, "
            f"control mode: {self.control_mode}"
        )
    
    def _setup_interface(self):
        """Setup ROS2 interface."""
        # Subscribers for control commands from our controller
        self.cmd_subs: Dict[int, any] = {}
        
        for drone_id in self.drone_ids[:self.n_drones]:
            # Subscribe to our controller's output
            self.cmd_subs[drone_id] = self.create_subscription(
                Twist,
                f'/{self.prefix}{drone_id}/cmd_vel',
                lambda msg, did=drone_id: self._cmd_callback(msg, did),
                10
            )
            
            # Initialize state
            self.drone_states[drone_id] = np.zeros(6)  # [x, y, z, vx, vy, vz]
            self.is_flying[drone_id] = False
        
        # Publishers to Crazyswarm2
        self.cf_vel_pubs: Dict[int, any] = {}
        self.cf_pos_pubs: Dict[int, any] = {}
        
        if CRAZYSWARM2_AVAILABLE:
            for drone_id in self.drone_ids[:self.n_drones]:
                # Velocity command publisher
                self.cf_vel_pubs[drone_id] = self.create_publisher(
                    Twist,
                    f'/{self.prefix}{drone_id}/cmd_vel_legacy',
                    10
                )
                
                # Position command (hover) publisher
                if self.control_mode == 'position':
                    self.cf_pos_pubs[drone_id] = self.create_publisher(
                        Hover,
                        f'/{self.prefix}{drone_id}/cmd_hover',
                        10
                    )
        
        # State feedback subscribers (from Crazyswarm2)
        self.odom_subs: Dict[int, any] = {}
        for drone_id in self.drone_ids[:self.n_drones]:
            self.odom_subs[drone_id] = self.create_subscription(
                Odometry,
                f'/{self.prefix}{drone_id}/odom',
                lambda msg, did=drone_id: self._odom_callback(msg, did),
                10
            )
        
        # Takeoff/Land services
        if CRAZYSWARM2_AVAILABLE and self.use_takeoff:
            self.takeoff_clients: Dict[int, any] = {}
            self.land_clients: Dict[int, any] = {}
            
            for drone_id in self.drone_ids[:self.n_drones]:
                self.takeoff_clients[drone_id] = self.create_client(
                    Takeoff,
                    f'/{self.prefix}{drone_id}/takeoff'
                )
                self.land_clients[drone_id] = self.create_client(
                    Land,
                    f'/{self.prefix}{drone_id}/land'
                )
        
        # Takeoff/Land command topics (from user)
        self.takeoff_sub = self.create_subscription(
            Empty,
            '/formation/takeoff',
            self._takeoff_callback,
            10
        )
        
        self.land_sub = self.create_subscription(
            Empty,
            '/formation/land',
            self._land_callback,
            10
        )
    
    def _cmd_callback(self, msg: Twist, drone_id: int):
        """
        Handle velocity command from formation controller.
        
        The controller outputs accelerations, which we've converted to
        velocity deltas. Here we forward to Crazyswarm2.
        """
        if not self.is_flying.get(drone_id, False):
            return
        
        if self.control_mode == 'velocity':
            self._send_velocity_command(drone_id, msg)
        elif self.control_mode == 'position':
            self._send_position_command(drone_id, msg)
    
    def _send_velocity_command(self, drone_id: int, msg: Twist):
        """Send velocity command to Crazyswarm2."""
        if drone_id in self.cf_vel_pubs:
            self.cf_vel_pubs[drone_id].publish(msg)
    
    def _send_position_command(self, drone_id: int, msg: Twist):
        """
        Convert velocity to position increment and send position command.
        """
        if not CRAZYSWARM2_AVAILABLE or drone_id not in self.cf_pos_pubs:
            return
        
        # Get current state
        current = self.drone_states.get(drone_id, np.zeros(6))
        
        # Simple integration for position
        dt = 0.02  # Assume 50Hz
        new_z = current[2] + msg.linear.z * dt
        
        hover = Hover()
        hover.vx = msg.linear.x
        hover.vy = msg.linear.y
        hover.yaw_rate = msg.angular.z
        hover.z_distance = max(0.1, new_z)  # Keep positive height
        
        self.cf_pos_pubs[drone_id].publish(hover)
    
    def _odom_callback(self, msg: Odometry, drone_id: int):
        """Handle odometry feedback from Crazyswarm2."""
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        
        self.drone_states[drone_id] = np.array([
            pos.x, pos.y, pos.z,
            vel.x, vel.y, vel.z
        ])
    
    def _takeoff_callback(self, msg: Empty):
        """Handle takeoff command."""
        self.get_logger().info("Initiating takeoff for all drones")
        
        if CRAZYSWARM2_AVAILABLE and self.use_takeoff:
            for drone_id in self.drone_ids[:self.n_drones]:
                self._takeoff_drone(drone_id)
        else:
            # Simulation mode - just set flying flag
            for drone_id in self.drone_ids[:self.n_drones]:
                self.is_flying[drone_id] = True
    
    def _land_callback(self, msg: Empty):
        """Handle land command."""
        self.get_logger().info("Initiating landing for all drones")
        
        if CRAZYSWARM2_AVAILABLE and self.use_takeoff:
            for drone_id in self.drone_ids[:self.n_drones]:
                self._land_drone(drone_id)
        else:
            for drone_id in self.drone_ids[:self.n_drones]:
                self.is_flying[drone_id] = False
    
    def _takeoff_drone(self, drone_id: int):
        """Takeoff a single drone using Crazyswarm2 service."""
        if drone_id not in self.takeoff_clients:
            return
        
        client = self.takeoff_clients[drone_id]
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Takeoff service not available for cf{drone_id}")
            return
        
        request = Takeoff.Request()
        request.height = self.default_height
        request.duration = rclpy.duration.Duration(seconds=3.0).to_msg()
        
        future = client.call_async(request)
        future.add_done_callback(
            lambda f, did=drone_id: self._takeoff_done(f, did)
        )
    
    def _takeoff_done(self, future, drone_id: int):
        """Callback when takeoff completes."""
        try:
            future.result()
            self.is_flying[drone_id] = True
            self.get_logger().info(f"cf{drone_id} takeoff complete")
        except Exception as e:
            self.get_logger().error(f"Takeoff failed for cf{drone_id}: {e}")
    
    def _land_drone(self, drone_id: int):
        """Land a single drone using Crazyswarm2 service."""
        if drone_id not in self.land_clients:
            return
        
        client = self.land_clients[drone_id]
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Land service not available for cf{drone_id}")
            return
        
        request = Land.Request()
        request.height = 0.05
        request.duration = rclpy.duration.Duration(seconds=3.0).to_msg()
        
        future = client.call_async(request)
        future.add_done_callback(
            lambda f, did=drone_id: self._land_done(f, did)
        )
    
    def _land_done(self, future, drone_id: int):
        """Callback when landing completes."""
        try:
            future.result()
            self.is_flying[drone_id] = False
            self.get_logger().info(f"cf{drone_id} landed")
        except Exception as e:
            self.get_logger().error(f"Landing failed for cf{drone_id}: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CrazyswarmBridge()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

