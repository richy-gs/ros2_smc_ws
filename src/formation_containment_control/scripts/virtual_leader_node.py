#!/usr/bin/env python3
"""
Virtual Leader Trajectory Node

Generates the reference trajectory for the virtual leader.
The virtual leader defines the overall motion of the formation.

Trajectory types supported:
- 3D lemniscate (infinity/figure-8) - as used in paper simulation
- Circle
- Line/waypoints
- Hover (stationary)
- Custom (user-defined function)

The virtual leader state χ_0 = [x_0, ẋ_0, y_0, ẏ_0, z_0, ż_0]^T
is used as reference for leader formation (Equation 7).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from typing import Callable, Optional

from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formation_containment_control.utils.math_utils import euler_to_quaternion


class TrajectoryGenerator:
    """
    Generates different trajectory types for the virtual leader.
    """
    
    @staticmethod
    def infinity_3d(t: float, scale: float = 2.0, 
                    height: float = 1.0, period: float = 30.0) -> tuple:
        """
        3D infinity (lemniscate) trajectory - as used in paper (Figure 5).
        
        Parametric equations:
        x(t) = scale * sin(ωt)
        y(t) = scale * sin(ωt) * cos(ωt)  (or scale * sin(2ωt)/2)
        z(t) = height + amplitude * sin(ωt/2)
        
        Args:
            t: Time in seconds
            scale: Size of the trajectory
            height: Base height
            period: Time for one complete cycle
            
        Returns:
            Tuple of (position, velocity, yaw)
        """
        omega = 2 * np.pi / period
        
        # Position
        x = scale * np.sin(omega * t)
        y = scale * np.sin(omega * t) * np.cos(omega * t)
        z = height + 0.3 * np.sin(omega * t / 2)  # Small vertical variation
        
        # Velocity (analytical derivatives)
        x_dot = scale * omega * np.cos(omega * t)
        y_dot = scale * omega * (np.cos(omega * t)**2 - np.sin(omega * t)**2)
        z_dot = 0.15 * omega * np.cos(omega * t / 2)
        
        # Yaw follows velocity direction
        yaw = np.arctan2(y_dot, x_dot)
        yaw_dot = 0.0  # Simplified
        
        position = np.array([x, y, z, yaw])
        velocity = np.array([x_dot, y_dot, z_dot, yaw_dot])
        
        return position, velocity
    
    @staticmethod
    def circle(t: float, radius: float = 2.0,
               height: float = 1.0, period: float = 20.0,
               transition_time: float = 5.0) -> tuple:
        """
        Circular trajectory with smooth start from origin.
        
        Uses a spiral transition from (0, 0, height) to the full circle,
        ensuring no abrupt jump after takeoff.
        
        Args:
            t: Time in seconds
            radius: Circle radius
            height: Height
            period: Time for one circle
            transition_time: Time to transition from origin to full circle
            
        Returns:
            Tuple of (position, velocity, yaw)
        """
        omega = 2 * np.pi / period
        
        # Smooth radius ramp from 0 to full radius (spiral out)
        if t < transition_time:
            # Smooth ramp using cosine for continuous velocity
            ramp = 0.5 * (1.0 - np.cos(np.pi * t / transition_time))
            ramp_dot = 0.5 * np.pi / transition_time * np.sin(np.pi * t / transition_time)
        else:
            ramp = 1.0
            ramp_dot = 0.0
        
        effective_radius = radius * ramp
        
        # Position (spiral that becomes circle)
        x = effective_radius * np.sin(omega * t)  # sin so x(0)=0
        y = effective_radius * (1.0 - np.cos(omega * t))  # 1-cos so y(0)=0
        z = height
        
        # Velocity (product rule: d/dt[r(t)*f(t)] = r'(t)*f(t) + r(t)*f'(t))
        x_dot = (radius * ramp_dot * np.sin(omega * t) + 
                 effective_radius * omega * np.cos(omega * t))
        y_dot = (radius * ramp_dot * (1.0 - np.cos(omega * t)) + 
                 effective_radius * omega * np.sin(omega * t))
        z_dot = 0.0
        
        # Yaw follows velocity direction
        if np.hypot(x_dot, y_dot) > 1e-6:
            yaw = np.arctan2(y_dot, x_dot)
        else:
            yaw = 0.0
        yaw_dot = omega
        
        position = np.array([x, y, z, yaw])
        velocity = np.array([x_dot, y_dot, z_dot, yaw_dot])
        
        return position, velocity
    
    @staticmethod
    def hover(center: np.ndarray = None, height: float = 1.0) -> tuple:
        """
        Stationary hover at a point.
        
        Args:
            center: Center position [x, y] or [x, y, z]
            height: Height if not specified in center
            
        Returns:
            Tuple of (position, velocity)
        """
        if center is None:
            center = np.array([0.0, 0.0, height, 0.0])
        elif len(center) == 2:
            center = np.array([center[0], center[1], height, 0.0])
        elif len(center) == 3:
            center = np.array([center[0], center[1], center[2], 0.0])
        
        position = center
        velocity = np.zeros(4)
        
        return position, velocity
    
    @staticmethod
    def square_waypoints(t: float, size: float = 2.0,
                        height: float = 1.0, speed: float = 0.5) -> tuple:
        """
        Square waypoint trajectory with smooth start from origin.
        
        Includes initial segment from (0, 0, height) to first waypoint,
        ensuring no abrupt jump after takeoff.
        
        Args:
            t: Time
            size: Square size (diamond shape centered at origin)
            height: Height
            speed: Movement speed
            
        Returns:
            Tuple of (position, velocity)
        """
        # Waypoints - include origin as starting point
        waypoints = np.array([
            [0, 0, height],       # Start from origin (matches hover position)
            [size, 0, height],
            [0, size, height],
            [-size, 0, height],
            [0, -size, height],
        ])
        
        # Calculate segment lengths (origin to first waypoint, then between vertices)
        n_waypoints = len(waypoints)
        segment_lengths = []
        for i in range(n_waypoints):
            start = waypoints[i]
            end = waypoints[(i + 1) % n_waypoints]
            segment_lengths.append(np.linalg.norm(end - start))
        
        total_length = sum(segment_lengths)
        
        # Find current position along path
        distance = (speed * t) % total_length
        
        # Find which segment we're on
        cumulative = 0.0
        segment = 0
        for i, seg_len in enumerate(segment_lengths):
            if cumulative + seg_len > distance:
                segment = i
                break
            cumulative += seg_len
        
        # Progress within current segment
        segment_progress = (distance - cumulative) / segment_lengths[segment]
        
        # Interpolate position
        start = waypoints[segment]
        end = waypoints[(segment + 1) % n_waypoints]
        
        position = start + segment_progress * (end - start)
        
        # Velocity direction
        direction = (end - start) / segment_lengths[segment]
        velocity_3d = direction * speed
        
        yaw = np.arctan2(direction[1], direction[0])
        
        pos = np.array([position[0], position[1], position[2], yaw])
        vel = np.array([velocity_3d[0], velocity_3d[1], velocity_3d[2], 0.0])
        
        return pos, vel


class VirtualLeaderNode(Node):
    """
    ROS2 node for virtual leader trajectory generation.
    """
    
    def __init__(self):
        super().__init__('virtual_leader_node')
        
        # Parameters
        self.declare_parameter('trajectory_type', 'lemniscate')
        self.declare_parameter('trajectory_scale', 2.0)
        self.declare_parameter('trajectory_height', 1.0)
        self.declare_parameter('trajectory_period', 60.0)
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('start_delay', 5.0)  # Wait before starting motion
        
        self.trajectory_type = self.get_parameter('trajectory_type').value
        self.scale = self.get_parameter('trajectory_scale').value
        self.height = self.get_parameter('trajectory_height').value
        self.period = self.get_parameter('trajectory_period').value
        self.rate = self.get_parameter('publish_rate').value
        self.frame = self.get_parameter('world_frame').value
        self.start_delay = self.get_parameter('start_delay').value
        
        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/virtual_leader/pose',
            10
        )
        
        self.velocity_pub = self.create_publisher(
            Twist,
            '/virtual_leader/velocity',
            10
        )
        
        self.marker_pub = self.create_publisher(
            Marker,
            '/virtual_leader/marker',
            10
        )
        
        # Trajectory timer
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.rate, self._timer_callback)
        
        self.get_logger().info(
            f"Virtual Leader Node started with trajectory: {self.trajectory_type}"
        )
    
    def _timer_callback(self):
        """Generate and publish trajectory."""
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9
        
        # Apply start delay (hover at initial position)
        if t < self.start_delay:
            position, velocity = TrajectoryGenerator.hover(
                np.array([0.0, 0.0, self.height])
            )
        else:
            # Adjust time for delay
            t_traj = t - self.start_delay
            
            # Generate trajectory based on type
            if self.trajectory_type == 'lemniscate':
                position, velocity = TrajectoryGenerator.infinity_3d(
                    t_traj, self.scale, self.height, self.period
                )
            elif self.trajectory_type == 'circle':
                position, velocity = TrajectoryGenerator.circle(
                    t_traj, self.scale, self.height, self.period
                )
            elif self.trajectory_type == 'hover':
                position, velocity = TrajectoryGenerator.hover(
                    np.array([0.0, 0.0, self.height])
                )
            elif self.trajectory_type == 'square':
                position, velocity = TrajectoryGenerator.square_waypoints(
                    t_traj, self.scale, self.height
                )
            else:
                position, velocity = TrajectoryGenerator.hover()
        
        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now.to_msg()
        pose_msg.header.frame_id = self.frame
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        
        q = euler_to_quaternion(0, 0, position[3])
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        
        self.pose_pub.publish(pose_msg)
        
        # Publish velocity
        vel_msg = Twist()
        vel_msg.linear.x = velocity[0]
        vel_msg.linear.y = velocity[1]
        vel_msg.linear.z = velocity[2]
        vel_msg.angular.z = velocity[3]
        
        self.velocity_pub.publish(vel_msg)
        
        # Publish marker
        marker = Marker()
        marker.header.stamp = now.to_msg()
        marker.header.frame_id = self.frame
        marker.ns = "virtual_leader"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose_msg.pose
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = VirtualLeaderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

