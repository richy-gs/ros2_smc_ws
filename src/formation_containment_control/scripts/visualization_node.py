#!/usr/bin/env python3
"""
Visualization Node for Formation-Containment Control

Provides comprehensive visualization in RViz2:
- Agent positions (spheres)
- Agent trajectories (lines)
- Convex hull (mesh/lines)
- Formation connections (lines)
- Error vectors
- Controller state information

Colors follow the convention from the paper:
- Green: Virtual leader
- Red, Blue, Yellow, Magenta: Leaders
- Gray (transparent): Followers
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from typing import Dict, List, Deque
from collections import deque

from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float64MultiArray
from nav_msgs.msg import Odometry

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formation_containment_control.utils.math_utils import quaternion_to_euler


class VisualizationNode(Node):
    """
    ROS2 node for formation visualization in RViz2.
    """
    
    def __init__(self):
        super().__init__('visualization_node')
        
        # Parameters
        self.declare_parameter('n_followers', 4)
        self.declare_parameter('n_leaders', 4)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('drone_prefix', 'cf')
        self.declare_parameter('follower_ids', [1, 2, 3, 4])
        self.declare_parameter('leader_ids', [5, 6, 7, 8])
        self.declare_parameter('trajectory_length', 200)  # Points to keep
        self.declare_parameter('update_rate', 10.0)
        
        self.n_followers = self.get_parameter('n_followers').value
        self.n_leaders = self.get_parameter('n_leaders').value
        self.frame = self.get_parameter('world_frame').value
        self.prefix = self.get_parameter('drone_prefix').value
        self.follower_ids = self.get_parameter('follower_ids').value
        self.leader_ids = self.get_parameter('leader_ids').value
        self.traj_length = self.get_parameter('trajectory_length').value
        self.update_rate = self.get_parameter('update_rate').value
        
        # Colors
        self.vl_color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
        self.leader_colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # Yellow
            ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),  # Magenta
        ]
        self.follower_color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.7)  # Gray
        self.hull_color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3)  # Cyan
        
        # State storage
        self.vl_position = np.zeros(3)
        self.leader_positions = np.zeros((self.n_leaders, 3))
        self.follower_positions = np.zeros((self.n_followers, 3))
        
        # Trajectory history
        self.vl_trajectory: Deque[np.ndarray] = deque(maxlen=self.traj_length)
        self.leader_trajectories: List[Deque[np.ndarray]] = [
            deque(maxlen=self.traj_length) for _ in range(self.n_leaders)
        ]
        self.follower_trajectories: List[Deque[np.ndarray]] = [
            deque(maxlen=self.traj_length) for _ in range(self.n_followers)
        ]
        
        # Setup ROS interface
        self._setup_interface()
        
        self.get_logger().info("Visualization Node initialized")
    
    def _setup_interface(self):
        """Setup ROS2 publishers and subscribers."""
        # QoS profile for odometry - must match simulation_node publisher
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Marker publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, '/formation/visualization', 10
        )
        
        self.traj_pub = self.create_publisher(
            MarkerArray, '/formation/trajectories', 10
        )
        
        # Virtual leader subscriber
        self.vl_sub = self.create_subscription(
            PoseStamped, '/virtual_leader/pose',
            self._vl_callback, 10
        )
        
        # Agent state subscribers (use BEST_EFFORT to match simulation_node publisher)
        for i, did in enumerate(self.follower_ids[:self.n_followers]):
            self.create_subscription(
                Odometry, f'/{self.prefix}{did}/odom',
                lambda msg, idx=i: self._follower_callback(msg, idx),
                qos_best_effort
            )
        
        for i, did in enumerate(self.leader_ids[:self.n_leaders]):
            self.create_subscription(
                Odometry, f'/{self.prefix}{did}/odom',
                lambda msg, idx=i: self._leader_callback(msg, idx),
                qos_best_effort
            )
        
        # Visualization timer
        self.timer = self.create_timer(
            1.0 / self.update_rate, self._publish_visualization
        )
    
    def _vl_callback(self, msg: PoseStamped):
        """Handle virtual leader pose."""
        pos = msg.pose.position
        self.vl_position = np.array([pos.x, pos.y, pos.z])
        self.vl_trajectory.append(self.vl_position.copy())
    
    def _follower_callback(self, msg: Odometry, idx: int):
        """Handle follower odometry."""
        pos = msg.pose.pose.position
        self.follower_positions[idx] = np.array([pos.x, pos.y, pos.z])
        self.follower_trajectories[idx].append(
            self.follower_positions[idx].copy()
        )
    
    def _leader_callback(self, msg: Odometry, idx: int):
        """Handle leader odometry."""
        pos = msg.pose.pose.position
        self.leader_positions[idx] = np.array([pos.x, pos.y, pos.z])
        self.leader_trajectories[idx].append(
            self.leader_positions[idx].copy()
        )
    
    def _publish_visualization(self):
        """Publish all visualization markers."""
        timestamp = self.get_clock().now().to_msg()
        
        # Main markers
        marker_array = MarkerArray()
        marker_id = 0
        
        # Virtual leader sphere
        vl_marker = self._create_sphere_marker(
            marker_id, "virtual_leader",
            self.vl_position, 0.3, self.vl_color, timestamp
        )
        marker_array.markers.append(vl_marker)
        marker_id += 1
        
        # Leader spheres
        for i in range(self.n_leaders):
            color = self.leader_colors[i % len(self.leader_colors)]
            marker = self._create_sphere_marker(
                marker_id, "leaders",
                self.leader_positions[i], 0.25, color, timestamp
            )
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Follower spheres
        for i in range(self.n_followers):
            marker = self._create_sphere_marker(
                marker_id, "followers",
                self.follower_positions[i], 0.2, self.follower_color, timestamp
            )
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Convex hull
        if self.n_leaders >= 3:
            hull_marker = self._create_hull_marker(
                marker_id, timestamp
            )
            marker_array.markers.append(hull_marker)
            marker_id += 1
        
        # Formation lines (VL to leaders)
        for i in range(self.n_leaders):
            color = self.leader_colors[i % len(self.leader_colors)]
            line = self._create_line_marker(
                marker_id, "formation_lines",
                self.vl_position, self.leader_positions[i],
                color, 0.02, timestamp
            )
            marker_array.markers.append(line)
            marker_id += 1
        
        self.marker_pub.publish(marker_array)
        
        # Trajectory markers
        traj_array = MarkerArray()
        marker_id = 0
        
        # Virtual leader trajectory
        if len(self.vl_trajectory) > 1:
            traj = self._create_trajectory_marker(
                marker_id, "vl_trajectory",
                list(self.vl_trajectory), self.vl_color, 0.01, timestamp
            )
            traj_array.markers.append(traj)
            marker_id += 1
        
        # Leader trajectories
        for i in range(self.n_leaders):
            if len(self.leader_trajectories[i]) > 1:
                color = self.leader_colors[i % len(self.leader_colors)]
                traj = self._create_trajectory_marker(
                    marker_id, f"leader_{i}_trajectory",
                    list(self.leader_trajectories[i]), color, 0.008, timestamp
                )
                traj_array.markers.append(traj)
                marker_id += 1
        
        # Follower trajectories
        for i in range(self.n_followers):
            if len(self.follower_trajectories[i]) > 1:
                color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.5)
                traj = self._create_trajectory_marker(
                    marker_id, f"follower_{i}_trajectory",
                    list(self.follower_trajectories[i]), color, 0.005, timestamp
                )
                traj_array.markers.append(traj)
                marker_id += 1
        
        self.traj_pub.publish(traj_array)
    
    def _create_sphere_marker(self, mid: int, ns: str, pos: np.ndarray,
                              scale: float, color: ColorRGBA, timestamp) -> Marker:
        """Create a sphere marker."""
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.header.stamp = timestamp
        marker.ns = ns
        marker.id = mid
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color = color
        return marker
    
    def _create_line_marker(self, mid: int, ns: str, 
                            start: np.ndarray, end: np.ndarray,
                            color: ColorRGBA, width: float, timestamp) -> Marker:
        """Create a line marker between two points."""
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.header.stamp = timestamp
        marker.ns = ns
        marker.id = mid
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color = color
        
        p1 = Point(x=start[0], y=start[1], z=start[2])
        p2 = Point(x=end[0], y=end[1], z=end[2])
        marker.points = [p1, p2]
        
        return marker
    
    def _create_trajectory_marker(self, mid: int, ns: str,
                                   points: List[np.ndarray],
                                   color: ColorRGBA, width: float,
                                   timestamp) -> Marker:
        """Create a trajectory line marker."""
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.header.stamp = timestamp
        marker.ns = ns
        marker.id = mid
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color = color
        
        for p in points:
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
        
        return marker
    
    def _create_hull_marker(self, mid: int, timestamp) -> Marker:
        """Create convex hull visualization."""
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.header.stamp = timestamp
        marker.ns = "convex_hull"
        marker.id = mid
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color = self.hull_color
        
        # Simple polygon connecting leaders
        for i in range(self.n_leaders):
            p = self.leader_positions[i]
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
        
        # Close the polygon
        if self.n_leaders > 0:
            p = self.leader_positions[0]
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
        
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

