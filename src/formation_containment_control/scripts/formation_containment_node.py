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
from std_msgs.msg import Header, ColorRGBA, Bool, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry

# Formation control modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formation_containment_control.controllers.formation_controller import (
    FormationController, FormationConfig
)
from formation_containment_control.core.dynamics import QuadrotorState
from formation_containment_control.utils.math_utils import (
    quaternion_to_euler, euler_to_quaternion
)


class FormationContainmentNode(Node):
    """
    Main formation-containment control node.
    
    Implements the two-layer control strategy:
    - Layer 1: Leaders track virtual leader trajectory with formation offsets
    - Layer 2: Followers track positions inside leader convex hull
    
    Topics:
    -------
    Subscriptions:
        /virtual_leader/pose: Virtual leader position and orientation
        /cf<id>/odom or /cf<id>/pose: Individual drone state feedback
        
    Publications:
        /cf<id>/cmd_vel: Velocity commands for each drone
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
        
        # Control parameters (from paper)
        self.declare_parameter('lambda_gain', 3.0)
        self.declare_parameter('alpha', 4.0)
        self.declare_parameter('beta', 0.125)
        self.declare_parameter('safety_distance', 0.3)
        
        # Control rate
        self.declare_parameter('control_rate', 50.0)  # Hz
        self.declare_parameter('dt', 0.02)  # Control timestep
        
        # Collision avoidance
        self.declare_parameter('use_collision_avoidance', True)
        
        # Drone naming
        self.declare_parameter('drone_prefix', 'cf')
        self.declare_parameter('follower_ids', [1, 2, 3, 4])
        self.declare_parameter('leader_ids', [5, 6, 7, 8])
        
        # Frame
        self.declare_parameter('world_frame', 'world')
    
    def _get_parameters(self):
        """Get parameters from ROS2 parameter server."""
        self.n_followers = self.get_parameter('n_followers').value
        self.n_leaders = self.get_parameter('n_leaders').value
        self.n_total = self.n_followers + self.n_leaders
        self.topology = self.get_parameter('topology').value
        
        self.formation_type = self.get_parameter('formation_type').value
        self.formation_scale = self.get_parameter('formation_scale').value
        self.formation_height = self.get_parameter('formation_height').value
        
        self.lambda_gain = self.get_parameter('lambda_gain').value
        self.alpha = self.get_parameter('alpha').value
        self.beta = self.get_parameter('beta').value
        self.safety_distance = self.get_parameter('safety_distance').value
        
        self.control_rate = self.get_parameter('control_rate').value
        self.dt = self.get_parameter('dt').value
        self.use_collision_avoidance = self.get_parameter('use_collision_avoidance').value
        
        self.drone_prefix = self.get_parameter('drone_prefix').value
        self.follower_ids = self.get_parameter('follower_ids').value
        self.leader_ids = self.get_parameter('leader_ids').value
        
        self.world_frame = self.get_parameter('world_frame').value
    
    def _init_controller(self):
        """Initialize the formation controller."""
        config = FormationConfig(
            n_followers=self.n_followers,
            n_leaders=self.n_leaders,
            topology=self.topology,
            formation_type=self.formation_type,
            formation_scale=self.formation_scale,
            formation_height=self.formation_height,
            lambda_gain=self.lambda_gain,
            alpha=self.alpha,
            beta=self.beta,
            safety_distance=self.safety_distance,
            dt=self.dt,
            use_collision_avoidance=self.use_collision_avoidance
        )
        
        self.formation_controller = FormationController(config)
        
        # State storage
        self.leader_states = np.zeros((self.n_leaders, 4))
        self.leader_velocities = np.zeros((self.n_leaders, 4))
        self.follower_states = np.zeros((self.n_followers, 4))
        self.follower_velocities = np.zeros((self.n_followers, 4))
        
        self.virtual_leader_state = np.array([0.0, 0.0, self.formation_height, 0.0])
        self.virtual_leader_velocity = np.zeros(4)
    
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
        
        # Drone state subscribers and command publishers
        self.odom_subs: Dict[int, any] = {}
        self.cmd_pubs: Dict[int, any] = {}
        self.pose_pubs: Dict[int, any] = {}  # For position commands
        
        # Followers
        for i, drone_id in enumerate(self.follower_ids[:self.n_followers]):
            # State feedback (try both odom and pose topics)
            self.odom_subs[f'follower_{i}'] = self.create_subscription(
                Odometry,
                f'/{self.drone_prefix}{drone_id}/odom',
                lambda msg, idx=i: self._follower_odom_callback(msg, idx),
                qos_best_effort
            )
            
            # Velocity command
            self.cmd_pubs[f'follower_{i}'] = self.create_publisher(
                Twist,
                f'/{self.drone_prefix}{drone_id}/cmd_vel',
                qos_reliable
            )
            
            # Position command (for Crazyswarm2 goTo)
            self.pose_pubs[f'follower_{i}'] = self.create_publisher(
                PoseStamped,
                f'/{self.drone_prefix}{drone_id}/cmd_position',
                qos_reliable
            )
        
        # Leaders
        for i, drone_id in enumerate(self.leader_ids[:self.n_leaders]):
            self.odom_subs[f'leader_{i}'] = self.create_subscription(
                Odometry,
                f'/{self.drone_prefix}{drone_id}/odom',
                lambda msg, idx=i: self._leader_odom_callback(msg, idx),
                qos_best_effort
            )
            
            self.cmd_pubs[f'leader_{i}'] = self.create_publisher(
                Twist,
                f'/{self.drone_prefix}{drone_id}/cmd_vel',
                qos_reliable
            )
            
            self.pose_pubs[f'leader_{i}'] = self.create_publisher(
                PoseStamped,
                f'/{self.drone_prefix}{drone_id}/cmd_position',
                qos_reliable
            )
        
        # Status and visualization publishers
        self.status_pub = self.create_publisher(
            Float64MultiArray,
            '/formation/status',
            qos_reliable
        )
        
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/formation/markers',
            qos_reliable
        )
        
        self.hull_marker_pub = self.create_publisher(
            Marker,
            '/formation/convex_hull',
            qos_reliable
        )
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self._control_callback
        )
        
        # Visualization timer (slower rate)
        self.viz_timer = self.create_timer(
            0.1,  # 10 Hz
            self._visualization_callback
        )
    
    def _virtual_leader_callback(self, msg: PoseStamped):
        """Handle virtual leader pose updates."""
        # Extract position
        pos = msg.pose.position
        
        # Extract yaw from quaternion
        q = msg.pose.orientation
        _, _, yaw = quaternion_to_euler(np.array([q.x, q.y, q.z, q.w]))
        
        # Compute velocity (numerical differentiation)
        new_state = np.array([pos.x, pos.y, pos.z, yaw])
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
        """Handle follower odometry updates."""
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        vel = msg.twist.twist.linear
        ang_vel = msg.twist.twist.angular
        
        _, _, yaw = quaternion_to_euler(np.array([q.x, q.y, q.z, q.w]))
        
        self.follower_states[idx] = [pos.x, pos.y, pos.z, yaw]
        self.follower_velocities[idx] = [vel.x, vel.y, vel.z, ang_vel.z]
        
        self.agent_states_received[idx] = True
    
    def _leader_odom_callback(self, msg: Odometry, idx: int):
        """Handle leader odometry updates."""
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        vel = msg.twist.twist.linear
        ang_vel = msg.twist.twist.angular
        
        _, _, yaw = quaternion_to_euler(np.array([q.x, q.y, q.z, q.w]))
        
        self.leader_states[idx] = [pos.x, pos.y, pos.z, yaw]
        self.leader_velocities[idx] = [vel.x, vel.y, vel.z, ang_vel.z]
        
        self.agent_states_received[self.n_followers + idx] = True
    
    def _control_callback(self):
        """Main control loop callback."""
        if not self.virtual_leader_received:
            return
        
        # Update formation controller with current states
        self.formation_controller.update_agent_states(
            self.leader_states,
            self.leader_velocities,
            self.follower_states,
            self.follower_velocities
        )
        
        # Compute control inputs
        leader_controls, follower_controls = self.formation_controller.compute_all_controls()
        
        # Publish leader commands
        for i, drone_id in enumerate(self.leader_ids[:self.n_leaders]):
            self._publish_control(f'leader_{i}', leader_controls[i])
        
        # Publish follower commands
        for i, drone_id in enumerate(self.follower_ids[:self.n_followers]):
            self._publish_control(f'follower_{i}', follower_controls[i])
        
        # Publish status
        self._publish_status()
    
    def _publish_control(self, agent_key: str, control: np.ndarray):
        """Publish control command for an agent."""
        # Convert acceleration control to velocity command (simple integration)
        # For Crazyswarm2, we often use velocity or position commands
        
        cmd = Twist()
        
        # The control is [ax, ay, az, Omega] in body frame
        # Convert to velocity command (simplified - assumes small dt)
        cmd.linear.x = float(control[0] * self.dt)
        cmd.linear.y = float(control[1] * self.dt)
        cmd.linear.z = float(control[2] * self.dt)
        cmd.angular.z = float(control[3] * self.dt)
        
        if agent_key in self.cmd_pubs:
            self.cmd_pubs[agent_key].publish(cmd)
    
    def _publish_status(self):
        """Publish formation status."""
        status = self.formation_controller.check_formation_status()
        
        msg = Float64MultiArray()
        msg.data = [
            float(status['formation_achieved']),
            float(status['containment_achieved']),
            float(status['collision_free']),
            status['max_leader_error'],
            status['max_follower_error'],
            status['min_inter_agent_distance'],
            status['convex_hull_volume']
        ]
        
        self.status_pub.publish(msg)
    
    def _visualization_callback(self):
        """Publish visualization markers."""
        marker_array = MarkerArray()
        timestamp = self.get_clock().now().to_msg()
        
        # Virtual leader marker (green sphere)
        vl_marker = Marker()
        vl_marker.header.frame_id = self.world_frame
        vl_marker.header.stamp = timestamp
        vl_marker.ns = "virtual_leader"
        vl_marker.id = 0
        vl_marker.type = Marker.SPHERE
        vl_marker.action = Marker.ADD
        vl_marker.pose.position.x = self.virtual_leader_state[0]
        vl_marker.pose.position.y = self.virtual_leader_state[1]
        vl_marker.pose.position.z = self.virtual_leader_state[2]
        vl_marker.scale.x = 0.3
        vl_marker.scale.y = 0.3
        vl_marker.scale.z = 0.3
        vl_marker.color.r = 0.0
        vl_marker.color.g = 1.0
        vl_marker.color.b = 0.0
        vl_marker.color.a = 1.0
        marker_array.markers.append(vl_marker)
        
        # Leader markers (colored spheres)
        leader_colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
        ]
        
        for i in range(self.n_leaders):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = timestamp
            marker.ns = "leaders"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = self.leader_states[i, 0]
            marker.pose.position.y = self.leader_states[i, 1]
            marker.pose.position.z = self.leader_states[i, 2]
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            color = leader_colors[i % len(leader_colors)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        
        # Follower markers (transparent spheres)
        for i in range(self.n_followers):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = timestamp
            marker.ns = "followers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = self.follower_states[i, 0]
            marker.pose.position.y = self.follower_states[i, 1]
            marker.pose.position.z = self.follower_states[i, 2]
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 0.7
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
        
        # Convex hull visualization
        self._publish_convex_hull(timestamp)
    
    def _publish_convex_hull(self, timestamp):
        """Publish convex hull marker."""
        hull_data = self.formation_controller.convex_hull.get_visualization_data()
        
        if len(hull_data['vertices']) < 3:
            return
        
        # Line strip for hull edges
        hull_marker = Marker()
        hull_marker.header.frame_id = self.world_frame
        hull_marker.header.stamp = timestamp
        hull_marker.ns = "convex_hull"
        hull_marker.id = 0
        hull_marker.type = Marker.LINE_STRIP
        hull_marker.action = Marker.ADD
        hull_marker.scale.x = 0.02  # Line width
        hull_marker.color.r = 0.0
        hull_marker.color.g = 1.0
        hull_marker.color.b = 1.0
        hull_marker.color.a = 0.5
        
        # Add hull vertices as a closed polygon
        vertices = hull_data['vertices']
        for vertex in vertices:
            p = Point()
            p.x = float(vertex[0])
            p.y = float(vertex[1])
            p.z = float(vertex[2]) if len(vertex) > 2 else self.formation_height
            hull_marker.points.append(p)
        
        # Close the polygon
        if len(vertices) > 0:
            p = Point()
            p.x = float(vertices[0][0])
            p.y = float(vertices[0][1])
            p.z = float(vertices[0][2]) if len(vertices[0]) > 2 else self.formation_height
            hull_marker.points.append(p)
        
        self.hull_marker_pub.publish(hull_marker)


def main(args=None):
    rclpy.init(args=args)
    node = FormationContainmentNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

