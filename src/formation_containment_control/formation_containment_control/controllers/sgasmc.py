"""
Single-Gain Adaptive Sliding Mode Controller (SGASMC)

Implements the adaptive sliding mode control strategy from Section 3 of the paper.

Control Design:
--------------
1. Sliding Surface (Equation 26):
   σ_c = ė_ac + D_λ e_ac
   
   where:
   - e_ac: Collision-avoidance augmented containment error
   - D_λ: Diagonal matrix with λ_i > 0

2. Controller (Equation 28):
   U_i = g(ξ_i)^{-1} (-F_i(ξ̇_i, ξ_i) - u_i(t) + D_λ ė_c)
   
3. Auxiliary Control (Equation 29):
   u_i = -2K_c(t)|σ_c|^{1/2} sign(σ_c) - (K_c²/2)σ_c

4. Adaptive Law (Equation 30):
   K̇_c(t) = α_c^{1/2}|σ|^{1/2} - β_c^{1/2}K_c(t)
   
   where:
   - α_c: Precision parameter (larger = more aggressive adaptation)
   - β_c: Control effort parameter (smaller = less oscillation)

Stability:
---------
The paper proves practical finite-time stability using Lyapunov analysis.
The system converges to a bounded region around the origin in finite time t_c.

Parameters from paper simulation:
- α_i = 4
- β_i = 0.125
- λ_i = 3
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SGASMCParameters:
    """
    Parameters for the Single-Gain Adaptive Sliding Mode Controller.
    
    From Section 4 of the paper:
    - α_i = 4 (precision parameter)
    - β_i = 0.125 (control effort parameter)
    - λ_i = 3 (sliding surface gain)
    """
    # Sliding surface parameter λ (must be positive)
    lambda_gain: float = 3.0
    
    # Adaptive law parameters
    alpha: float = 4.0      # Precision parameter (α_c)
    beta: float = 0.125     # Control effort parameter (β_c)
    
    # Initial adaptive gain
    K_c_init: float = 1.0
    
    # Gain bounds
    K_c_min: float = 0.1
    K_c_max: float = 100.0
    
    # Integration time step
    dt: float = 0.01
    
    # Perturbation bound (L from paper)
    perturbation_bound: float = 1.0


@dataclass
class SlidingSurface:
    """
    Sliding surface for containment control.
    
    Implements equation (26): σ_c = ė_ac + D_λ e_ac
    
    The sliding surface drives the system to track the desired states.
    When σ = 0, the error dynamics become: ė_ac = -D_λ e_ac
    which ensures exponential convergence of the error.
    """
    
    # State dimension (4 for [x, y, z, ψ])
    dim: int = 4
    
    # Lambda gains (diagonal of D_λ)
    lambda_gains: np.ndarray = field(default_factory=lambda: np.ones(4) * 3.0)
    
    def __post_init__(self):
        """Ensure lambda gains are positive."""
        self.lambda_gains = np.maximum(self.lambda_gains, 0.01)
        self.D_lambda = np.diag(self.lambda_gains)
    
    def compute(self, error: np.ndarray, error_dot: np.ndarray) -> np.ndarray:
        """
        Compute sliding surface value.
        
        σ_c = ė_ac + D_λ e_ac    (Equation 26)
        
        Args:
            error: Containment error e_ac, shape (dim,)
            error_dot: Error derivative ė_ac, shape (dim,)
            
        Returns:
            Sliding surface value σ_c, shape (dim,)
        """
        return error_dot + self.D_lambda @ error
    
    def compute_derivative(self, error_dot: np.ndarray, 
                          error_ddot: np.ndarray) -> np.ndarray:
        """
        Compute sliding surface derivative.
        
        σ̇_c = ë_ac + D_λ ė_ac    (Equation 27)
        
        Args:
            error_dot: Error derivative
            error_ddot: Second error derivative
            
        Returns:
            Sliding surface derivative
        """
        return error_ddot + self.D_lambda @ error_dot


class SGASMCController:
    """
    Single-Gain Adaptive Sliding Mode Controller.
    
    This controller implements the SGASMC strategy from Section 3 of the paper,
    providing:
    - Finite-time convergence
    - Robustness to perturbations
    - No overestimation of control gains
    - Integrated collision avoidance
    
    The controller consists of:
    1. Error computation (containment + collision avoidance)
    2. Sliding surface design
    3. Auxiliary control with adaptive gain
    4. Gain adaptation law
    
    Mathematical Formulation:
    ------------------------
    Controller (Eq. 28): U = g⁻¹(-F - u + D_λ ė_c)
    Auxiliary (Eq. 29):  u = -2K_c|σ|^{1/2}sign(σ) - (K_c²/2)σ
    Adaptation (Eq. 30): K̇_c = α^{1/2}|σ|^{1/2} - β^{1/2}K_c
    """
    
    def __init__(self, params: Optional[SGASMCParameters] = None,
                 state_dim: int = 4):
        """
        Initialize SGASMC controller.
        
        Args:
            params: Controller parameters
            state_dim: Dimension of state vector (default 4 for [x,y,z,ψ])
        """
        self.params = params or SGASMCParameters()
        self.state_dim = state_dim
        
        # Initialize adaptive gain K_c(t)
        self.K_c = np.ones(state_dim) * self.params.K_c_init
        
        # Initialize sliding surface
        self.sliding_surface = SlidingSurface(
            dim=state_dim,
            lambda_gains=np.ones(state_dim) * self.params.lambda_gain
        )
        
        # Store last values for debugging/logging
        self.last_sigma = np.zeros(state_dim)
        self.last_u_aux = np.zeros(state_dim)
        self.last_control = np.zeros(state_dim)
    
    def compute_auxiliary_control(self, sigma: np.ndarray) -> np.ndarray:
        """
        Compute auxiliary control u_i(t) from Equation 29.
        
        u_i = -2K_c(t)|σ_c|^{1/2} sign(σ_c) - (K_c²/2)σ_c
        
        This is the core sliding mode control law that drives the system
        toward the sliding surface.
        
        Args:
            sigma: Sliding surface value
            
        Returns:
            Auxiliary control signal
        """
        # Compute components
        sigma_abs = np.abs(sigma)
        sigma_sqrt = np.sqrt(sigma_abs + 1e-10)  # Add small value to avoid div by zero
        sigma_sign = np.sign(sigma)
        
        # Term 1: -2K_c|σ|^{1/2}sign(σ) (discontinuous reaching term)
        term1 = -2.0 * self.K_c * sigma_sqrt * sigma_sign
        
        # Term 2: -(K_c²/2)σ (continuous stabilizing term)
        term2 = -0.5 * (self.K_c ** 2) * sigma
        
        u_aux = term1 + term2
        self.last_u_aux = u_aux
        
        return u_aux
    
    def update_adaptive_gain(self, sigma: np.ndarray) -> np.ndarray:
        """
        Update adaptive gain K_c(t) using Equation 30.
        
        K̇_c(t) = α^{1/2}|σ|^{1/2} - β^{1/2}K_c(t)
        
        This adaptation law:
        - Increases gain when sliding surface is far from zero
        - Decreases gain when close to sliding surface
        - Prevents overestimation of gains
        
        Args:
            sigma: Sliding surface value
            
        Returns:
            Updated adaptive gain
        """
        alpha_sqrt = np.sqrt(self.params.alpha)
        beta_sqrt = np.sqrt(self.params.beta)
        sigma_sqrt = np.sqrt(np.abs(sigma) + 1e-10)
        
        # Adaptive law (Equation 30)
        K_c_dot = alpha_sqrt * sigma_sqrt - beta_sqrt * self.K_c
        
        # Euler integration
        self.K_c = self.K_c + K_c_dot * self.params.dt
        
        # Apply bounds
        self.K_c = np.clip(self.K_c, self.params.K_c_min, self.params.K_c_max)
        
        return self.K_c
    
    def compute_control(self, 
                        xi: np.ndarray,
                        xi_dot: np.ndarray,
                        error: np.ndarray,
                        error_dot: np.ndarray,
                        F_dynamics: np.ndarray,
                        g_matrix: np.ndarray,
                        g_inv: np.ndarray) -> np.ndarray:
        """
        Compute full control signal using Equation 28.
        
        U_i = g(ξ_i)^{-1} (-F_i(ξ̇_i, ξ_i) - u_i(t) + D_λ ė_c)
        
        This is the complete feedback control law that:
        1. Cancels nonlinear dynamics F
        2. Applies sliding mode auxiliary control u
        3. Adds error derivative term for convergence
        
        Args:
            xi: Current state [x, y, z, ψ]
            xi_dot: Current velocity [ẋ, ẏ, ż, ψ̇]
            error: Containment error e_ac
            error_dot: Error derivative ė_ac
            F_dynamics: Nonlinear dynamics term F(ξ,ξ̇)
            g_matrix: Input mapping matrix g(ξ)
            g_inv: Inverse of g(ξ)
            
        Returns:
            Control input U = [a_x, a_y, a_z, Ω]
        """
        # Compute sliding surface
        sigma = self.sliding_surface.compute(error, error_dot)
        self.last_sigma = sigma
        
        # Compute auxiliary control (Equation 29)
        u_aux = self.compute_auxiliary_control(sigma)
        
        # Update adaptive gain (Equation 30)
        self.update_adaptive_gain(sigma)
        
        # D_λ ė_c term
        D_lambda_error_dot = self.sliding_surface.D_lambda @ error_dot
        
        # Full control (Equation 28)
        # U = g^{-1}(-F - u + D_λ ė_c)
        control_input = g_inv @ (-F_dynamics - u_aux + D_lambda_error_dot)
        
        self.last_control = control_input
        
        return control_input
    
    def compute_control_simple(self,
                               error: np.ndarray,
                               error_dot: np.ndarray) -> np.ndarray:
        """
        Simplified control computation for kinematic model.
        
        When using position/velocity control (not acceleration),
        this provides a simpler interface.
        
        Args:
            error: Position error
            error_dot: Velocity error
            
        Returns:
            Velocity command
        """
        sigma = self.sliding_surface.compute(error, error_dot)
        self.last_sigma = sigma
        
        # Auxiliary control
        u_aux = self.compute_auxiliary_control(sigma)
        
        # Update gain
        self.update_adaptive_gain(sigma)
        
        # Simple velocity command: v = -u_aux - λ*error
        velocity_cmd = -u_aux - self.params.lambda_gain * error
        
        return velocity_cmd
    
    def get_state(self) -> dict:
        """
        Get current controller state for logging/debugging.
        
        Returns:
            Dictionary with controller state
        """
        return {
            'K_c': self.K_c.copy(),
            'sigma': self.last_sigma.copy(),
            'u_aux': self.last_u_aux.copy(),
            'control': self.last_control.copy(),
            'lambda': self.params.lambda_gain,
            'alpha': self.params.alpha,
            'beta': self.params.beta
        }
    
    def reset(self):
        """Reset controller to initial state."""
        self.K_c = np.ones(self.state_dim) * self.params.K_c_init
        self.last_sigma = np.zeros(self.state_dim)
        self.last_u_aux = np.zeros(self.state_dim)
        self.last_control = np.zeros(self.state_dim)


class ContainmentErrorComputer:
    """
    Computes containment and collision-avoidance errors.
    
    Implements equations (18)-(25) from the paper:
    
    Basic containment error (Eq. 18):
    e_c = ξ_{dc,i} - ξ_i = Σ[-L_N^{-1}L_M]_{ij}ξ_{n+j} - ξ_i
    
    Collision-avoidance augmented error (Eq. 20):
    e_{ac,i} = e_{c,i} + h_{c,i}
    
    where h_{c,i} is the desired distance relative to neighbors (Eq. 23):
    h_c = (D⊗I_s)(L_ζ⊗I_s)ξ - (A⊗I_s)(L_ζ⊗I_s)ξ - (L⊗I_s)ξ
    
    Final form (Eq. 24):
    e_{ac} = (L_ζ - I_{n+m} - L)⊗I_s * ξ
    """
    
    def __init__(self, interaction_network, state_dim: int = 4):
        """
        Initialize error computer.
        
        Args:
            interaction_network: InteractionNetwork instance with Laplacian matrices
            state_dim: Dimension of state vector
        """
        self.network = interaction_network
        self.state_dim = state_dim
        
        # Precompute constant matrices
        self._compute_error_matrices()
    
    def _compute_error_matrices(self):
        """Precompute error transformation matrices."""
        n = self.network.n_followers
        m = self.network.n_leaders
        n_total = n + m
        s = self.state_dim
        
        # Error matrix from Equation 24: (L_ζ - I_{n+m} - L)
        self.error_matrix = (
            self.network.L_zeta - 
            np.eye(n_total) - 
            self.network.L
        )
        
        # Kronecker product with I_s for full state computation
        # (L_ζ - I_{n+m} - L) ⊗ I_s
        self.error_matrix_full = np.kron(self.error_matrix, np.eye(s))
        
        # Collision avoidance matrices
        self.collision_matrix, _ = self.network.get_collision_avoidance_matrices()
        self.collision_matrix_full = np.kron(self.collision_matrix, np.eye(s))
    
    def compute_containment_error(self, 
                                  follower_states: np.ndarray,
                                  leader_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute basic containment error for all followers.
        
        From Equation (18):
        e_c = ξ_{dc} - ξ_i = Σ[-L_N^{-1}L_M]_{ij}ξ_{n+j} - ξ_i
        
        Args:
            follower_states: Array of follower states, shape (n, state_dim)
            leader_states: Array of leader states, shape (m, state_dim)
            
        Returns:
            Tuple of (position_errors, velocity_errors) each shape (n, state_dim)
        """
        # Desired positions from Laplacian weights
        desired_positions = self.network.laplacian.get_all_desired_follower_positions(
            leader_states
        )
        
        # Containment error
        errors = desired_positions - follower_states
        
        return errors
    
    def compute_collision_avoidance_error(self,
                                          all_states: np.ndarray) -> np.ndarray:
        """
        Compute collision-avoidance augmented error.
        
        From Equation (24):
        e_{ac} = (L_ζ - I_{n+m} - L) ξ
        
        This adds repelling effects between agents to avoid collisions.
        
        Args:
            all_states: States of all agents, shape (n+m, state_dim)
            
        Returns:
            Augmented error for followers, shape (n, state_dim)
        """
        n = self.network.n_followers
        
        # Flatten states for matrix multiplication
        xi_flat = all_states.flatten()
        
        # Compute full error
        error_flat = self.error_matrix_full @ xi_flat
        
        # Reshape and extract follower errors
        error_full = error_flat.reshape(-1, self.state_dim)
        follower_errors = error_full[:n, :]
        
        return follower_errors
    
    def compute_full_error(self,
                           follower_idx: int,
                           follower_state: np.ndarray,
                           follower_velocity: np.ndarray,
                           leader_states: np.ndarray,
                           leader_velocities: np.ndarray,
                           all_states: Optional[np.ndarray] = None,
                           all_velocities: Optional[np.ndarray] = None,
                           use_collision_avoidance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute complete error for a single follower.
        
        Args:
            follower_idx: Index of the follower
            follower_state: Follower's current state [x,y,z,ψ]
            follower_velocity: Follower's current velocity
            leader_states: All leader states, shape (m, state_dim)
            leader_velocities: All leader velocities
            all_states: All agent states (for collision avoidance)
            all_velocities: All agent velocities
            use_collision_avoidance: Whether to use collision-free error
            
        Returns:
            Tuple of (error, error_dot) for use in SGASMC
        """
        # Basic containment error
        weights = self.network.laplacian.containment_weights[follower_idx, :]
        desired_pos = weights @ leader_states
        desired_vel = weights @ leader_velocities
        
        error = desired_pos - follower_state
        error_dot = desired_vel - follower_velocity
        
        if use_collision_avoidance and all_states is not None:
            # Add collision avoidance term
            ca_errors = self.compute_collision_avoidance_error(all_states)
            ca_error = ca_errors[follower_idx, :] if len(ca_errors) > follower_idx else np.zeros(self.state_dim)
            
            # Blend containment and collision avoidance
            # The collision avoidance error modifies the desired position
            error = error + ca_error
            
            if all_velocities is not None:
                ca_errors_vel = self.compute_collision_avoidance_error(all_velocities)
                ca_error_dot = ca_errors_vel[follower_idx, :] if len(ca_errors_vel) > follower_idx else np.zeros(self.state_dim)
                error_dot = error_dot + ca_error_dot
        
        return error, error_dot


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from core.graph_theory import create_interaction_network
    
    # Create interaction network
    network = create_interaction_network(4, 4, "paper")
    
    # Create error computer
    error_computer = ContainmentErrorComputer(network, state_dim=4)
    
    # Create controller
    params = SGASMCParameters(
        lambda_gain=3.0,
        alpha=4.0,
        beta=0.125,
        dt=0.01
    )
    controller = SGASMCController(params)
    
    # Example: simulate one control step
    print("SGASMC Controller Test")
    print("=" * 50)
    
    # Current state
    xi = np.array([0.0, 0.0, 1.0, 0.0])
    xi_dot = np.array([0.1, 0.1, 0.0, 0.0])
    
    # Error (example values)
    error = np.array([0.5, 0.3, 0.0, 0.1])
    error_dot = np.array([0.1, 0.05, 0.0, 0.02])
    
    # Dynamics (from reduced model)
    F = np.array([0.0, 0.0, 0.0, 0.0])
    g = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    g_inv = np.linalg.inv(g)
    
    # Compute control
    U = controller.compute_control(xi, xi_dot, error, error_dot, F, g, g_inv)
    
    print(f"Initial adaptive gain K_c: {controller.K_c}")
    print(f"Sliding surface σ: {controller.last_sigma}")
    print(f"Auxiliary control u: {controller.last_u_aux}")
    print(f"Control output U: {U}")
    
    # Simulate multiple steps
    print("\nSimulating 100 steps...")
    for _ in range(100):
        U = controller.compute_control(xi, xi_dot, error, error_dot, F, g, g_inv)
        # Simple update (without actual dynamics)
        error = error * 0.99
        error_dot = error_dot * 0.99
    
    print(f"Final adaptive gain K_c: {controller.K_c}")
    print(f"Final sliding surface σ: {controller.last_sigma}")

