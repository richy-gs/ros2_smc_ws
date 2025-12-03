#!/usr/bin/env python3
"""
Crazyswarm2 Bridge Node - GoTo Service Only

Bridge between the formation-containment controller and Crazyswarm2.
Uses only the high-level GoTo service for position commands.

Features:
- Rate limiting: GoTo commands sent at configurable rate (default 10Hz)
- Position threshold: Only sends GoTo if position changed significantly
- State machine: IDLE → TAKING_OFF → HOVERING → ACTIVE → LANDING → LANDED
- Service-based control: /formation/start, /formation/stop services

Usage:
1. Launch Crazyswarm2:
   ros2 launch crazyflie launch.py

2. Launch this bridge:
   ros2 launch formation_containment_control crazyswarm2.launch.py

3. Start formation (service call):
   ros2 service call /formation/start std_srvs/srv/Trigger

4. Stop formation (service call):
   ros2 service call /formation/stop std_srvs/srv/Trigger

Services Provided:
    /formation/start: Start the formation (takeoff + enable control)
    /formation/stop: Stop the formation (disable control + land)

Services Used (Crazyswarm2):
    /cf<id>/takeoff: Takeoff service
    /cf<id>/land: Land service  
    /cf<id>/go_to: GoTo service (main control)
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from typing import Dict, Optional
from enum import Enum, auto

# ROS2 messages
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration

# Crazyswarm2 interfaces
try:
    from crazyflie_interfaces.srv import GoTo, Takeoff, Land
    CRAZYSWARM2_AVAILABLE = True
except ImportError:
    CRAZYSWARM2_AVAILABLE = False
    print("Warning: crazyflie_interfaces not found. Running in simulation mode.")


class BridgeState(Enum):
    """State machine states for the bridge."""
    IDLE = auto()           # Drones on ground, waiting
    TAKING_OFF = auto()     # Takeoff in progress
    HOVERING = auto()       # Takeoff complete, waiting to start formation
    ACTIVE = auto()         # Formation control active
    LANDING = auto()        # Landing in progress
    LANDED = auto()         # All drones landed


class CrazyswarmBridge(Node):
    """
    Bridge between formation controller and Crazyswarm2.
    
    Uses GoTo service for high-level position control with:
    - Rate limiting (configurable, default 10Hz)
    - Position threshold (only send if changed > threshold)
    - State machine for proper sequencing
    """
    
    def __init__(self):
        super().__init__('crazyswarm_bridge_node')
        
        # Parameters
        self._declare_parameters()
        self._get_parameters()
        
        # Build drone list
        self.all_drone_ids = self.leader_ids + self.follower_ids
        self.n_drones = len(self.all_drone_ids)
        
        # State machine
        self.state = BridgeState.IDLE
        self.takeoff_start_time: Optional[float] = None
        self.drones_taking_off: set = set()
        
        # State tracking
        self.drone_states: Dict[int, np.ndarray] = {}
        self.last_goto_positions: Dict[int, np.ndarray] = {}
        self.last_goto_time: Dict[int, float] = {}
        
        # Callback groups
        self.service_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = ReentrantCallbackGroup()
        
        # Setup interface
        self._setup_interface()
        
        self.get_logger().info(
            f"Crazyswarm Bridge initialized for {self.n_drones} drones"
        )
        self.get_logger().info(f"  Leaders: {self.leader_ids}")
        self.get_logger().info(f"  Followers: {self.follower_ids}")
        self.get_logger().info(f"  GoTo rate: {self.goto_rate} Hz")
        self.get_logger().info(f"  Position threshold: {self.position_threshold} m")
        
        if not CRAZYSWARM2_AVAILABLE:
            self.get_logger().warn("Crazyswarm2 interfaces not available - simulation mode")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        self.declare_parameter('n_followers', 1)
        self.declare_parameter('n_leaders', 4)
        self.declare_parameter('follower_ids', [11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        self.declare_parameter('leader_ids', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.declare_parameter('drone_prefix', 'cf')
        self.declare_parameter('default_height', 1.0)
        
        # GoTo control parameters
        self.declare_parameter('goto_rate', 10.0)  # Hz - rate limit for GoTo commands
        self.declare_parameter('goto_duration', 0.3)  # Duration for each GoTo command
        self.declare_parameter('position_threshold', 0.05)  # meters - min change to send GoTo
        
        # Timing parameters
        self.declare_parameter('takeoff_duration', 3.0)  # seconds
        self.declare_parameter('hover_duration', 2.0)  # seconds to hover before formation
    
    def _get_parameters(self):
        """Get parameters from ROS2."""
        self.n_followers = self.get_parameter('n_followers').value
        self.n_leaders = self.get_parameter('n_leaders').value
        self.follower_ids = self.get_parameter('follower_ids').value[:self.n_followers]
        self.leader_ids = self.get_parameter('leader_ids').value[:self.n_leaders]
        self.prefix = self.get_parameter('drone_prefix').value
        self.default_height = self.get_parameter('default_height').value
        
        self.goto_rate = self.get_parameter('goto_rate').value
        self.goto_duration = self.get_parameter('goto_duration').value
        self.position_threshold = self.get_parameter('position_threshold').value
        
        self.takeoff_duration = self.get_parameter('takeoff_duration').value
        self.hover_duration = self.get_parameter('hover_duration').value
    
    def _setup_interface(self):
        """Setup ROS2 interface."""
        
        # Position command subscribers (from formation controller)
        self.cmd_subs: Dict[int, any] = {}
        
        # Odometry subscribers (state feedback from Crazyswarm2)
        self.odom_subs: Dict[int, any] = {}
        
        # Crazyswarm2 service clients
        self.goto_clients: Dict[int, any] = {}
        self.takeoff_clients: Dict[int, any] = {}
        self.land_clients: Dict[int, any] = {}
        
        for drone_id in self.all_drone_ids:
            # Subscribe to position commands from formation controller
            self.cmd_subs[drone_id] = self.create_subscription(
                PoseStamped,
                f'/{self.prefix}{drone_id}/cmd_position',
                lambda msg, did=drone_id: self._cmd_position_callback(msg, did),
                10
            )
            
            # Subscribe to odometry from Crazyswarm2 (BEST_EFFORT for compatibility)
            odom_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
            self.odom_subs[drone_id] = self.create_subscription(
                Odometry,
                f'/{self.prefix}{drone_id}/odom',
                lambda msg, did=drone_id: self._odom_callback(msg, did),
                odom_qos
            )
            
            # Initialize tracking
            self.drone_states[drone_id] = np.zeros(6)
            self.last_goto_positions[drone_id] = np.array([np.inf, np.inf, np.inf])
            self.last_goto_time[drone_id] = 0.0
            
            if CRAZYSWARM2_AVAILABLE:
                # Create service clients
                self.goto_clients[drone_id] = self.create_client(
                    GoTo,
                    f'/{self.prefix}{drone_id}/go_to',
                    callback_group=self.service_cb_group
                )
                
                self.takeoff_clients[drone_id] = self.create_client(
                    Takeoff,
                    f'/{self.prefix}{drone_id}/takeoff',
                    callback_group=self.service_cb_group
                )
                
                self.land_clients[drone_id] = self.create_client(
                    Land,
                    f'/{self.prefix}{drone_id}/land',
                    callback_group=self.service_cb_group
                )
        
        # Formation control enable publisher
        self.enable_pub = self.create_publisher(Bool, '/formation/enable', 10)
        
        # Services for external control
        self.start_srv = self.create_service(
            Trigger,
            '/formation/start',
            self._start_service_callback,
            callback_group=self.service_cb_group
        )
        
        self.stop_srv = self.create_service(
            Trigger,
            '/formation/stop',
            self._stop_service_callback,
            callback_group=self.service_cb_group
        )
        
        # State machine timer (10 Hz)
        self.state_timer = self.create_timer(
            0.1,
            self._state_machine_callback,
            callback_group=self.timer_cb_group
        )
    
    def _start_service_callback(self, request, response):
        """Handle /formation/start service call."""
        if self.state == BridgeState.IDLE:
            self.get_logger().info("Starting formation: initiating takeoff...")
            self._initiate_takeoff()
            response.success = True
            response.message = "Takeoff initiated. Formation will start after hover stabilization."
        elif self.state == BridgeState.HOVERING:
            self.get_logger().info("Already hovering, activating formation control...")
            self._activate_formation()
            response.success = True
            response.message = "Formation control activated."
        elif self.state == BridgeState.ACTIVE:
            response.success = True
            response.message = "Formation already active."
        else:
            response.success = False
            response.message = f"Cannot start from state: {self.state.name}"
        
        return response
    
    def _stop_service_callback(self, request, response):
        """Handle /formation/stop service call."""
        if self.state in [BridgeState.ACTIVE, BridgeState.HOVERING]:
            self.get_logger().info("Stopping formation: initiating landing...")
            self._initiate_landing()
            response.success = True
            response.message = "Landing initiated."
        elif self.state == BridgeState.IDLE or self.state == BridgeState.LANDED:
            response.success = True
            response.message = "Already on ground."
        else:
            response.success = False
            response.message = f"Cannot stop from state: {self.state.name}"
        
        return response
    
    def _state_machine_callback(self):
        """State machine update (runs at 10 Hz)."""
        now = self.get_clock().now().nanoseconds / 1e9
        
        if self.state == BridgeState.TAKING_OFF:
            # Check if takeoff duration has elapsed
            if self.takeoff_start_time is not None:
                elapsed = now - self.takeoff_start_time
                if elapsed >= self.takeoff_duration:
                    self.get_logger().info("Takeoff complete. Hovering...")
                    self.state = BridgeState.HOVERING
                    self.hover_start_time = now
        
        elif self.state == BridgeState.HOVERING:
            # Check if hover duration has elapsed, then auto-activate
            elapsed = now - self.hover_start_time
            if elapsed >= self.hover_duration:
                self.get_logger().info("Hover stabilization complete. Activating formation...")
                self._activate_formation()
    
    def _initiate_takeoff(self):
        """Initiate takeoff for all drones."""
        self.state = BridgeState.TAKING_OFF
        self.takeoff_start_time = self.get_clock().now().nanoseconds / 1e9
        self.drones_taking_off = set(self.all_drone_ids)
        
        # Disable formation control during takeoff
        self._set_formation_enabled(False)
        
        if CRAZYSWARM2_AVAILABLE:
            for drone_id in self.all_drone_ids:
                self._takeoff_drone(drone_id)
        else:
            self.get_logger().info("Simulation mode: Takeoff simulated")
    
    def _takeoff_drone(self, drone_id: int):
        """Takeoff a single drone."""
        if drone_id not in self.takeoff_clients:
            return
        
        client = self.takeoff_clients[drone_id]
        
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Takeoff service not available for {self.prefix}{drone_id}")
            return
        
        request = Takeoff.Request()
        request.height = self.default_height
        request.duration = Duration(
            sec=int(self.takeoff_duration),
            nanosec=int((self.takeoff_duration % 1) * 1e9)
        )
        request.group_mask = 0
        
        future = client.call_async(request)
        future.add_done_callback(
            lambda f, did=drone_id: self._takeoff_done(f, did)
        )
        self.get_logger().info(f"Takeoff sent for {self.prefix}{drone_id}")
    
    def _takeoff_done(self, future, drone_id: int):
        """Callback when takeoff service returns."""
        try:
            future.result()
            self.get_logger().debug(f"{self.prefix}{drone_id} takeoff service returned")
        except Exception as e:
            self.get_logger().error(f"Takeoff failed for {self.prefix}{drone_id}: {e}")
    
    def _activate_formation(self):
        """Activate formation control."""
        self.state = BridgeState.ACTIVE
        self._set_formation_enabled(True)
        self.get_logger().info("Formation control ACTIVE")
    
    def _initiate_landing(self):
        """Initiate landing for all drones."""
        # First disable formation control
        self._set_formation_enabled(False)
        self.state = BridgeState.LANDING
        
        if CRAZYSWARM2_AVAILABLE:
            for drone_id in self.all_drone_ids:
                self._land_drone(drone_id)
        else:
            self.get_logger().info("Simulation mode: Landing simulated")
            self.state = BridgeState.LANDED
    
    def _land_drone(self, drone_id: int):
        """Land a single drone."""
        if drone_id not in self.land_clients:
            return
        
        client = self.land_clients[drone_id]
        
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Land service not available for {self.prefix}{drone_id}")
            return
        
        request = Land.Request()
        request.height = 0.05
        request.duration = Duration(sec=3, nanosec=0)
        request.group_mask = 0
        
        future = client.call_async(request)
        future.add_done_callback(
            lambda f, did=drone_id: self._land_done(f, did)
        )
        self.get_logger().info(f"Land sent for {self.prefix}{drone_id}")
    
    def _land_done(self, future, drone_id: int):
        """Callback when landing service returns."""
        try:
            future.result()
            self.get_logger().info(f"{self.prefix}{drone_id} landed")
        except Exception as e:
            self.get_logger().error(f"Landing failed for {self.prefix}{drone_id}: {e}")
    
    def _set_formation_enabled(self, enabled: bool):
        """Publish formation enable/disable."""
        msg = Bool()
        msg.data = enabled
        self.enable_pub.publish(msg)
        status = "ENABLED" if enabled else "DISABLED"
        self.get_logger().info(f"Formation control {status}")
    
    def _cmd_position_callback(self, msg: PoseStamped, drone_id: int):
        """
        Handle position command from formation controller.
        
        Applies rate limiting and position threshold before sending GoTo.
        """
        # Only forward commands when formation is active
        if self.state != BridgeState.ACTIVE:
            return
        
        if not CRAZYSWARM2_AVAILABLE:
            return
        
        # Extract target position
        target = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # Check rate limit
        now = self.get_clock().now().nanoseconds / 1e9
        time_since_last = now - self.last_goto_time.get(drone_id, 0.0)
        min_interval = 1.0 / self.goto_rate
        
        if time_since_last < min_interval:
            return  # Rate limited
        
        # Check position threshold
        last_pos = self.last_goto_positions.get(drone_id, np.array([np.inf, np.inf, np.inf]))
        position_change = np.linalg.norm(target - last_pos)
        
        if position_change < self.position_threshold:
            return  # Position hasn't changed enough
        
        # Send GoTo command
        self._send_goto(drone_id, target)
        
        # Update tracking
        self.last_goto_positions[drone_id] = target.copy()
        self.last_goto_time[drone_id] = now
    
    def _send_goto(self, drone_id: int, target: np.ndarray):
        """Send GoTo service request to Crazyflie."""
        if drone_id not in self.goto_clients:
            return
        
        client = self.goto_clients[drone_id]
        
        if not client.service_is_ready():
            self.get_logger().debug(f"GoTo service not ready for {self.prefix}{drone_id}")
            return
        
        # Create GoTo request
        request = GoTo.Request()
        request.goal.x = float(target[0])
        request.goal.y = float(target[1])
        request.goal.z = float(target[2])
        request.yaw = 0.0
        request.duration = Duration(
            sec=0,
            nanosec=int(self.goto_duration * 1e9)
        )
        request.relative = False
        request.group_mask = 0
        
        # Async call (fire and forget for performance)
        future = client.call_async(request)
        future.add_done_callback(
            lambda f, did=drone_id: self._goto_done(f, did)
        )
    
    def _goto_done(self, future, drone_id: int):
        """Callback when GoTo completes."""
        try:
            future.result()
        except Exception as e:
            self.get_logger().debug(f"GoTo for {self.prefix}{drone_id}: {e}")
    
    def _odom_callback(self, msg: Odometry, drone_id: int):
        """Handle odometry feedback from Crazyswarm2."""
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        
        self.drone_states[drone_id] = np.array([
            pos.x, pos.y, pos.z,
            vel.x, vel.y, vel.z
        ])


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
