"""
Formation Controller Module

High-level formation and containment controller that integrates:
- Graph-based interaction network
- Leader trajectory tracking
- Follower containment control with SGASMC
- Collision avoidance
- Convex hull management

This module implements the two-layer control strategy from Figure 1:
Layer 1: Virtual leader → Leaders (trajectory tracking)
Layer 2: Leaders → Followers (containment control)

The formation vectors h = [h_{n+1}, ..., h_{n+m}] define leader positions
relative to the virtual leader:
  χ_{d,j} = χ_0 + h_j  for j = n+1, ..., n+m

Followers converge to positions inside the convex hull formed by leaders.

Supports loading custom leader offsets from file for manual formation definition.
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..core.graph_theory import InteractionNetwork, create_interaction_network
from ..core.dynamics import ReducedTrackingModel, QuadrotorState
from ..core.convex_hull import ConvexHullContainment, FormationGeometry
from .sgasmc import SGASMCController, SGASMCParameters, ContainmentErrorComputer


def load_offsets_from_file(filepath: str) -> Optional[Tuple[np.ndarray, str]]:
    """
    Load leader offsets from a YAML file.
    
    YAML format:
        formation_name: "pentagon"
        description: "Pentagon formation"
        leaders:
          - id: 1
            offset: [1.0, 0.0, 0.0, 0.0]
          - id: 2
            offset: [-1.0, 0.0, 0.0, 0.0]
    
    Args:
        filepath: Path to the YAML offsets file
        
    Returns:
        Tuple of (offsets array shape (n_leaders, 4), formation_name) or None if error
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        import yaml
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None or 'leaders' not in data:
            print(f"Error: Invalid YAML format in {filepath}. Missing 'leaders' key.")
            return None
        
        formation_name = data.get('formation_name', 'custom')
        leaders = data['leaders']
        
        offsets = []
        leader_ids = []
        
        for leader in leaders:
            leader_id = leader.get('id')
            offset = leader.get('offset')
            
            if leader_id is None or offset is None:
                print(f"Warning: Skipping invalid leader entry: {leader}")
                continue
            
            # Ensure offset has 4 elements [x, y, z, yaw]
            if len(offset) < 4:
                offset = list(offset) + [0.0] * (4 - len(offset))
            
            leader_ids.append(leader_id)
            offsets.append(offset[:4])
        
        if not offsets:
            return None
        
        # Sort by leader ID and return offsets array
        sorted_indices = np.argsort(leader_ids)
        offsets_array = np.array(offsets)[sorted_indices]
        
        return offsets_array, formation_name
        
    except Exception as e:
        print(f"Error loading offsets from {filepath}: {e}")
        return None


@dataclass
class FormationConfig:
    """Configuration for the formation controller."""
    
    # Number of agents
    n_followers: int = 4
    n_leaders: int = 4
    
    # Graph topology
    topology: str = "paper"  # "paper", "complete", "ring"
    
    # Formation geometry for leaders
    formation_type: str = "square"  # "square", "triangle", "tetrahedron", "circle", "custom"
    formation_scale: float = 1.0
    formation_height: float = 1.0
    
    # Custom offsets file path (used when formation_type is "custom")
    offsets_file: str = ""
    
    # Control parameters (from paper Section 4)
    lambda_gain: float = 3.0
    alpha: float = 4.0
    beta: float = 0.125
    
    # Safety distance for collision avoidance (γ_s from Eq. 6)
    safety_distance: float = 0.3
    
    # Control rate
    dt: float = 0.01
    
    # Maximum velocity limit (m/s) - 0 means no limit
    max_velocity: float = 0.0
    
    # Enable collision avoidance
    use_collision_avoidance: bool = True
    
    # Perturbation bound (L from paper)
    perturbation_bound: float = 1.0


@dataclass
class AgentData:
    """Data container for an individual agent."""
    agent_id: int
    agent_type: str  # "leader" or "follower"
    state: QuadrotorState = field(default_factory=QuadrotorState)
    controller: Optional[SGASMCController] = None
    desired_state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    desired_velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    control_input: np.ndarray = field(default_factory=lambda: np.zeros(4))
    tracking_error: np.ndarray = field(default_factory=lambda: np.zeros(4))
    neighbors: List[int] = field(default_factory=list)


class LeaderController:
    """
    Controller for leader agents.
    
    Leaders track desired positions relative to the virtual leader:
    χ_{d,j} = χ_0 + h_j (Equation 8)
    
    The tracking error is:
    e_t = χ_{1,0} + h_{p,j} - χ_{1,j}  (Equation 9)
    """
    
    def __init__(self, leader_id: int, 
                 formation_offset: np.ndarray,
                 params: Optional[SGASMCParameters] = None):
        """
        Initialize leader controller.
        
        Args:
            leader_id: ID of this leader
            formation_offset: Offset h_j from virtual leader [x, y, z, ψ]
            params: SGASMC parameters
        """
        self.leader_id = leader_id
        self.formation_offset = formation_offset
        self.params = params or SGASMCParameters()
        
        # Create SGASMC controller for trajectory tracking
        self.controller = SGASMCController(self.params, state_dim=4)
        
        # Dynamics model
        self.dynamics = ReducedTrackingModel()
        
        # State
        self.state = QuadrotorState()
        self.last_control = np.zeros(4)
    
    def compute_control(self,
                        current_state: np.ndarray,
                        current_velocity: np.ndarray,
                        virtual_leader_state: np.ndarray,
                        virtual_leader_velocity: np.ndarray) -> np.ndarray:
        """
        Compute control for leader to track formation position.
        
        Args:
            current_state: Leader's current state [x, y, z, ψ]
            current_velocity: Leader's current velocity
            virtual_leader_state: Virtual leader state
            virtual_leader_velocity: Virtual leader velocity
            
        Returns:
            Control input [a_x, a_y, a_z, Ω]
        """
        # Desired state: virtual leader + formation offset
        desired_state = virtual_leader_state + self.formation_offset
        desired_velocity = virtual_leader_velocity  # Leaders track VL velocity
        
        # Tracking error (Equation 9)
        error = desired_state - current_state
        error_dot = desired_velocity - current_velocity
        
        # Compute dynamics terms
        xi = current_state
        xi_dot = current_velocity
        F = self.dynamics.compute_F(xi, xi_dot)
        g = self.dynamics.compute_g(xi)
        g_inv = self.dynamics.compute_g_inverse(xi)
        
        # Compute SGASMC control
        control = self.controller.compute_control(
            xi, xi_dot, error, error_dot, F, g, g_inv
        )
        
        self.last_control = control
        return control
    
    def get_state(self) -> dict:
        """Get controller state for logging."""
        return {
            'leader_id': self.leader_id,
            'formation_offset': self.formation_offset.tolist(),
            'controller_state': self.controller.get_state()
        }


class FollowerController:
    """
    Controller for follower agents.
    
    Followers track positions inside the convex hull formed by leaders,
    using the SGASMC with collision avoidance.
    
    Desired positions are computed from Laplacian weights (Equation 16):
    ξ_{dc,i} = Σ_{j=1}^m [-L_N^{-1}L_M]_{ij} ξ_{n+j}
    """
    
    def __init__(self, follower_id: int,
                 interaction_network: InteractionNetwork,
                 params: Optional[SGASMCParameters] = None,
                 safety_distance: float = 0.3):
        """
        Initialize follower controller.
        
        Args:
            follower_id: ID of this follower
            interaction_network: Graph interaction network
            params: SGASMC parameters
            safety_distance: Minimum distance γ_s for collision avoidance
        """
        self.follower_id = follower_id
        self.network = interaction_network
        self.params = params or SGASMCParameters()
        self.safety_distance = safety_distance
        
        # Create SGASMC controller
        self.controller = SGASMCController(self.params, state_dim=4)
        
        # Error computer for containment + collision avoidance
        self.error_computer = ContainmentErrorComputer(
            interaction_network, state_dim=4
        )
        
        # Dynamics model
        self.dynamics = ReducedTrackingModel()
        
        # State
        self.last_control = np.zeros(4)
        self.last_error = np.zeros(4)
        self.min_neighbor_distance = float('inf')
    
    def compute_control(self,
                        current_state: np.ndarray,
                        current_velocity: np.ndarray,
                        leader_states: np.ndarray,
                        leader_velocities: np.ndarray,
                        all_states: Optional[np.ndarray] = None,
                        all_velocities: Optional[np.ndarray] = None,
                        use_collision_avoidance: bool = True) -> np.ndarray:
        """
        Compute control for follower containment.
        
        Args:
            current_state: Follower's current state [x, y, z, ψ]
            current_velocity: Follower's current velocity
            leader_states: All leader states, shape (m, 4)
            leader_velocities: All leader velocities, shape (m, 4)
            all_states: States of all agents (for collision avoidance)
            all_velocities: Velocities of all agents
            use_collision_avoidance: Enable collision avoidance term
            
        Returns:
            Control input [a_x, a_y, a_z, Ω]
        """
        # Compute containment error (with optional collision avoidance)
        error, error_dot = self.error_computer.compute_full_error(
            self.follower_id,
            current_state,
            current_velocity,
            leader_states,
            leader_velocities,
            all_states,
            all_velocities,
            use_collision_avoidance
        )
        
        self.last_error = error
        
        # Check minimum neighbor distance for collision warning
        if all_states is not None:
            self._update_min_distance(current_state, all_states)
        
        # Compute dynamics terms
        xi = current_state
        xi_dot = current_velocity
        F = self.dynamics.compute_F(xi, xi_dot)
        g = self.dynamics.compute_g(xi)
        g_inv = self.dynamics.compute_g_inverse(xi)
        
        # Compute SGASMC control
        control = self.controller.compute_control(
            xi, xi_dot, error, error_dot, F, g, g_inv
        )
        
        self.last_control = control
        return control
    
    def _update_min_distance(self, current_state: np.ndarray,
                            all_states: np.ndarray):
        """Update minimum distance to neighbors."""
        n = self.network.n_followers
        my_pos = current_state[:3]
        
        self.min_neighbor_distance = float('inf')
        for i, state in enumerate(all_states):
            if i != self.follower_id:  # Don't check distance to self
                other_pos = state[:3]
                dist = np.linalg.norm(my_pos - other_pos)
                self.min_neighbor_distance = min(self.min_neighbor_distance, dist)
    
    def get_desired_position(self, leader_states: np.ndarray) -> np.ndarray:
        """Get desired position inside convex hull."""
        weights = self.network.laplacian.containment_weights[self.follower_id, :]
        return weights @ leader_states
    
    def get_state(self) -> dict:
        """Get controller state for logging."""
        return {
            'follower_id': self.follower_id,
            'last_error': self.last_error.tolist(),
            'min_neighbor_distance': self.min_neighbor_distance,
            'controller_state': self.controller.get_state()
        }


class FormationController:
    """
    Main formation-containment controller.
    
    Coordinates all agents (leaders and followers) to achieve:
    1. Leader formation around virtual leader
    2. Follower containment within leader convex hull
    3. Collision avoidance between all agents
    
    This implements the complete two-layer strategy from the paper.
    """
    
    def __init__(self, config: Optional[FormationConfig] = None):
        """
        Initialize formation controller.
        
        Args:
            config: Formation configuration
        """
        self.config = config or FormationConfig()
        
        # Create interaction network
        self.network = create_interaction_network(
            self.config.n_followers,
            self.config.n_leaders,
            self.config.topology
        )
        
        # Verify network connectivity
        valid, msg = self.network.verify_connectivity()
        if not valid:
            raise ValueError(f"Invalid network topology: {msg}")
        
        # Create formation geometry
        self.formation_offsets = self._create_formation_offsets()
        
        # Create SGASMC parameters
        self.sgasmc_params = SGASMCParameters(
            lambda_gain=self.config.lambda_gain,
            alpha=self.config.alpha,
            beta=self.config.beta,
            dt=self.config.dt,
            max_velocity=self.config.max_velocity
        )
        
        # Create leader controllers
        self.leader_controllers: List[LeaderController] = []
        for i in range(self.config.n_leaders):
            offset = self.formation_offsets[i]
            # Extend to 4D if needed
            offset_4d = np.zeros(4)
            offset_4d[:len(offset)] = offset[:min(len(offset), 4)]
            
            controller = LeaderController(
                leader_id=i,
                formation_offset=offset_4d,
                params=self.sgasmc_params
            )
            self.leader_controllers.append(controller)
        
        # Create follower controllers
        self.follower_controllers: List[FollowerController] = []
        for i in range(self.config.n_followers):
            controller = FollowerController(
                follower_id=i,
                interaction_network=self.network,
                params=self.sgasmc_params,
                safety_distance=self.config.safety_distance
            )
            self.follower_controllers.append(controller)
        
        # Convex hull manager
        self.convex_hull = ConvexHullContainment(
            n_leaders=self.config.n_leaders,
            dimension=3
        )
        
        # State storage
        self.virtual_leader_state = np.zeros(4)
        self.virtual_leader_velocity = np.zeros(4)
        self.leader_states = np.zeros((self.config.n_leaders, 4))
        self.leader_velocities = np.zeros((self.config.n_leaders, 4))
        self.follower_states = np.zeros((self.config.n_followers, 4))
        self.follower_velocities = np.zeros((self.config.n_followers, 4))
    
    def _create_formation_offsets(self) -> np.ndarray:
        """
        Create formation offsets for leaders based on configuration.
        
        Supports:
        - Predefined formations: square, triangle, tetrahedron, circle, line
        - Custom formations: loaded from YAML offsets_file when formation_type is "custom"
        
        Returns:
            Array of offsets, shape (n_leaders, 3) or (n_leaders, 4)
        """
        ftype = self.config.formation_type.lower()
        scale = self.config.formation_scale
        height = self.config.formation_height
        n_leaders = self.config.n_leaders
        
        # Check for custom offsets file (YAML format)
        if ftype == "custom" or self.config.offsets_file:
            result = load_offsets_from_file(self.config.offsets_file)
            if result is not None:
                offsets, formation_name = result
                # Validate number of leaders matches
                if len(offsets) != n_leaders:
                    print(f"Warning: Offsets file has {len(offsets)} leaders, "
                          f"but config expects {n_leaders}. Using file offsets.")
                    # Adjust n_leaders to match file
                    self.config.n_leaders = len(offsets)
                print(f"Loaded '{formation_name}' formation with {len(offsets)} leaders "
                      f"from: {self.config.offsets_file}")
                return offsets[:, :3]  # Return x, y, z offsets (ignoring yaw for geometry)
            else:
                print(f"Warning: Could not load offsets from {self.config.offsets_file}. "
                      f"Using default circle formation.")
                return FormationGeometry.circle(n_leaders, scale, height)
        
        if ftype == "square" and n_leaders == 4:
            return FormationGeometry.square(scale, height)
        elif ftype == "triangle" and n_leaders == 3:
            return FormationGeometry.triangle(scale, height)
        elif ftype == "tetrahedron" and n_leaders == 4:
            return FormationGeometry.tetrahedron(scale, height)
        elif ftype == "circle":
            return FormationGeometry.circle(n_leaders, scale, height)
        elif ftype == "line":
            return FormationGeometry.line(n_leaders, scale, height)
        else:
            # Default to circle formation
            return FormationGeometry.circle(n_leaders, scale, height)
    
    def set_virtual_leader_state(self, state: np.ndarray, 
                                  velocity: np.ndarray):
        """
        Set virtual leader state.
        
        Args:
            state: Virtual leader state [x, y, z, ψ]
            velocity: Virtual leader velocity
        """
        self.virtual_leader_state = state.copy()
        self.virtual_leader_velocity = velocity.copy()
    
    def update_agent_states(self,
                           leader_states: np.ndarray,
                           leader_velocities: np.ndarray,
                           follower_states: np.ndarray,
                           follower_velocities: np.ndarray):
        """
        Update all agent states from measurements/simulation.
        
        Args:
            leader_states: shape (n_leaders, 4)
            leader_velocities: shape (n_leaders, 4)
            follower_states: shape (n_followers, 4)
            follower_velocities: shape (n_followers, 4)
        """
        self.leader_states = leader_states.copy()
        self.leader_velocities = leader_velocities.copy()
        self.follower_states = follower_states.copy()
        self.follower_velocities = follower_velocities.copy()
        
        # Update convex hull
        self.convex_hull.update_leader_states(
            leader_states[:, :3],
            leader_velocities[:, :3]
        )
    
    def compute_all_controls(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute control inputs for all agents.
        
        Returns:
            Tuple of (leader_controls, follower_controls)
            Each has shape (n_agents, 4) with [a_x, a_y, a_z, Ω]
        """
        # Compute leader controls
        leader_controls = np.zeros((self.config.n_leaders, 4))
        for i, controller in enumerate(self.leader_controllers):
            leader_controls[i] = controller.compute_control(
                self.leader_states[i],
                self.leader_velocities[i],
                self.virtual_leader_state,
                self.virtual_leader_velocity
            )
        
        # Prepare all states for collision avoidance
        all_states = np.vstack([self.follower_states, self.leader_states])
        all_velocities = np.vstack([self.follower_velocities, self.leader_velocities])
        
        # Compute follower controls
        follower_controls = np.zeros((self.config.n_followers, 4))
        for i, controller in enumerate(self.follower_controllers):
            follower_controls[i] = controller.compute_control(
                self.follower_states[i],
                self.follower_velocities[i],
                self.leader_states,
                self.leader_velocities,
                all_states,
                all_velocities,
                self.config.use_collision_avoidance
            )
        
        return leader_controls, follower_controls
    
    def check_formation_status(self) -> dict:
        """
        Check formation and containment status.
        
        Returns:
            Dictionary with status information
        """
        # Check leader tracking errors
        leader_errors = []
        for i, controller in enumerate(self.leader_controllers):
            desired = self.virtual_leader_state + controller.formation_offset
            error = np.linalg.norm(self.leader_states[i] - desired)
            leader_errors.append(error)
        
        # Check follower containment
        follower_contained = []
        follower_errors = []
        for i, controller in enumerate(self.follower_controllers):
            contained = self.convex_hull.is_point_contained(
                self.follower_states[i, :3]
            )
            follower_contained.append(contained)
            follower_errors.append(np.linalg.norm(controller.last_error))
        
        # Check minimum inter-agent distances
        min_distance = float('inf')
        all_states = np.vstack([self.follower_states, self.leader_states])
        n_total = len(all_states)
        for i in range(n_total):
            for j in range(i+1, n_total):
                dist = np.linalg.norm(all_states[i, :3] - all_states[j, :3])
                min_distance = min(min_distance, dist)
        
        return {
            'formation_achieved': max(leader_errors) < 0.1,
            'containment_achieved': all(follower_contained),
            'collision_free': min_distance >= self.config.safety_distance,
            'max_leader_error': max(leader_errors),
            'max_follower_error': max(follower_errors),
            'min_inter_agent_distance': min_distance,
            'leader_errors': leader_errors,
            'follower_contained': follower_contained,
            'follower_errors': follower_errors,
            'convex_hull_volume': self.convex_hull.get_hull_volume()
        }
    
    def change_formation(self, formation_type: str, 
                        scale: Optional[float] = None):
        """
        Change the leader formation.
        
        This implements the formation change at t=62.5s in the paper
        (square to tetrahedron).
        
        Args:
            formation_type: New formation type
            scale: New formation scale (optional)
        """
        self.config.formation_type = formation_type
        if scale is not None:
            self.config.formation_scale = scale
        
        # Update formation offsets
        self.formation_offsets = self._create_formation_offsets()
        
        # Update leader controllers with new offsets
        for i, controller in enumerate(self.leader_controllers):
            offset = self.formation_offsets[i]
            offset_4d = np.zeros(4)
            offset_4d[:len(offset)] = offset[:min(len(offset), 4)]
            controller.formation_offset = offset_4d
    
    def get_visualization_data(self) -> dict:
        """Get data for visualization."""
        hull_data = self.convex_hull.get_visualization_data()
        
        return {
            'virtual_leader': {
                'position': self.virtual_leader_state[:3].tolist(),
                'yaw': self.virtual_leader_state[3]
            },
            'leaders': {
                'positions': self.leader_states[:, :3].tolist(),
                'yaws': self.leader_states[:, 3].tolist(),
                'formation_offsets': self.formation_offsets.tolist()
            },
            'followers': {
                'positions': self.follower_states[:, :3].tolist(),
                'yaws': self.follower_states[:, 3].tolist(),
                'desired_positions': [
                    ctrl.get_desired_position(self.leader_states).tolist()
                    for ctrl in self.follower_controllers
                ]
            },
            'convex_hull': hull_data,
            'status': self.check_formation_status()
        }
    
    def reset(self):
        """Reset all controllers."""
        for controller in self.leader_controllers:
            controller.controller.reset()
        for controller in self.follower_controllers:
            controller.controller.reset()


# Example usage
if __name__ == "__main__":
    # Create formation controller with default config (4 followers, 4 leaders)
    config = FormationConfig(
        n_followers=4,
        n_leaders=4,
        topology="paper",
        formation_type="square",
        formation_scale=1.0,
        lambda_gain=3.0,
        alpha=4.0,
        beta=0.125
    )
    
    controller = FormationController(config)
    
    print("Formation Controller Initialized")
    print("=" * 50)
    print(f"Followers: {config.n_followers}")
    print(f"Leaders: {config.n_leaders}")
    print(f"Topology: {config.topology}")
    print(f"Formation: {config.formation_type}")
    print()
    
    # Set initial states
    controller.set_virtual_leader_state(
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.5, 0.0, 0.0, 0.0])
    )
    
    # Initial leader states (at formation positions)
    leader_states = np.zeros((4, 4))
    for i in range(4):
        offset = controller.formation_offsets[i]
        leader_states[i, :3] = offset
        leader_states[i, 3] = 0.0
    leader_states[:, 2] = 1.0  # Set height
    
    # Initial follower states (random inside hull)
    follower_states = np.random.uniform(-0.5, 0.5, (4, 4))
    follower_states[:, 2] = 1.0  # Set height
    follower_states[:, 3] = 0.0  # Zero yaw
    
    # Update controller
    controller.update_agent_states(
        leader_states,
        np.zeros((4, 4)),
        follower_states,
        np.zeros((4, 4))
    )
    
    # Compute controls
    leader_controls, follower_controls = controller.compute_all_controls()
    
    print("Leader controls:")
    print(leader_controls)
    print()
    print("Follower controls:")
    print(follower_controls)
    print()
    
    # Check status
    status = controller.check_formation_status()
    print("Formation Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

