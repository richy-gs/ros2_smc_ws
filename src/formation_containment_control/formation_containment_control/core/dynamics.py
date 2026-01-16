"""
Quadrotor Dynamics Module

Implements the reduced tracking model from Section 3 of the paper (Equations 10-13):

ẍ = a_x cos(ψ) - a_y sin(ψ) - ψ̇(v_x sin(ψ) + v_y cos(ψ))    (10)
ÿ = a_x sin(ψ) + a_y cos(ψ) + ψ̇(v_x cos(ψ) - v_y sin(ψ))    (11)
z̈ = a_z                                                        (12)
ψ̈ = Ω                                                          (13)

State vector: ξ = [x, y, z, ψ]^T
Control input: U = [a_x, a_y, a_z, Ω]^T (accelerations in body frame + yaw angular accel)

General form (Equation 14):
ξ̈_i = F(ξ_i, t) + g(ξ_i)U_i + Δ_i(t)

where:
- F(ξ,t): Nonlinear dynamics function (Equation 15)
- g(ξ): Input mapping matrix (Equation 15)
- Δ(t): Perturbations/uncertainties bounded by |Δ(t)| ≤ L

Quadrotor physical parameters from paper:
- l = 0.058 m (arm length)
- g = 9.81 m/s² (gravity)
- m = 0.060 kg (mass)
- I_xx = 3.073e-5 kg·m²
- I_yy = 3.084e-5 kg·m²
- I_zz = 5.968e-5 kg·m²
- J_r = 8.801e-8 kg·m² (rotor inertia)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class QuadrotorParameters:
    """Physical parameters of the quadrotor (Crazyflie-like)."""
    
    # Geometric parameters
    arm_length: float = 0.058  # m
    
    # Mass and inertia
    mass: float = 0.060  # kg (60g for Crazyflie)
    I_xx: float = 3.073e-5  # kg·m²
    I_yy: float = 3.084e-5  # kg·m²
    I_zz: float = 5.968e-5  # kg·m²
    J_r: float = 8.801e-8   # kg·m² (rotor inertia)
    
    # Physical constants
    gravity: float = 9.81  # m/s²
    
    # Drag coefficients (optional for more realistic model)
    k_drag_xy: float = 0.0  # Translational drag in xy
    k_drag_z: float = 0.0   # Translational drag in z
    k_drag_yaw: float = 0.0 # Rotational drag in yaw


@dataclass
class QuadrotorState:
    """
    Complete state of a quadrotor agent.
    
    Inertial frame position and orientation, plus velocities.
    """
    # Position in inertial frame
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Velocity in inertial frame
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    # Orientation (Euler angles)
    roll: float = 0.0   # φ
    pitch: float = 0.0  # θ
    yaw: float = 0.0    # ψ
    
    # Angular velocities
    roll_rate: float = 0.0   # φ̇
    pitch_rate: float = 0.0  # θ̇
    yaw_rate: float = 0.0    # ψ̇
    
    def to_position_array(self) -> np.ndarray:
        """Return position as numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z])
    
    def to_velocity_array(self) -> np.ndarray:
        """Return velocity as numpy array [vx, vy, vz]."""
        return np.array([self.vx, self.vy, self.vz])
    
    def to_reduced_state(self) -> np.ndarray:
        """Return reduced state vector [x, y, z, ψ]."""
        return np.array([self.x, self.y, self.z, self.yaw])
    
    def to_reduced_velocity(self) -> np.ndarray:
        """Return reduced velocity vector [vx, vy, vz, ψ̇]."""
        return np.array([self.vx, self.vy, self.vz, self.yaw_rate])
    
    def to_full_state(self) -> np.ndarray:
        """Return full 12-state vector [x,y,z, vx,vy,vz, φ,θ,ψ, φ̇,θ̇,ψ̇]."""
        return np.array([
            self.x, self.y, self.z,
            self.vx, self.vy, self.vz,
            self.roll, self.pitch, self.yaw,
            self.roll_rate, self.pitch_rate, self.yaw_rate
        ])
    
    @classmethod
    def from_reduced_state(cls, xi: np.ndarray, xi_dot: np.ndarray) -> 'QuadrotorState':
        """Create state from reduced state vectors [x,y,z,ψ] and [ẋ,ẏ,ż,ψ̇]."""
        return cls(
            x=xi[0], y=xi[1], z=xi[2], yaw=xi[3],
            vx=xi_dot[0], vy=xi_dot[1], vz=xi_dot[2], yaw_rate=xi_dot[3]
        )


class ReducedTrackingModel:
    """
    Reduced tracking model for quadrotor position control.
    
    Implements equations (10)-(13) from the paper.
    
    This model assumes a low-level attitude controller enables free maneuverability,
    allowing us to use a simplified position tracking model.
    
    State: ξ = [x, y, z, ψ]^T
    Velocity: ξ̇ = [ẋ, ẏ, ż, ψ̇]^T
    Control: U = [a_x, a_y, a_z, Ω]^T (body-frame accelerations + yaw angular accel)
    
    Dynamics:
    ξ̈ = F(ξ, ξ̇) + g(ξ)U + Δ(t)
    
    where:
    F(ξ,ξ̇) = [-ψ̇(ẋ sin(ψ) + ẏ cos(ψ))]    (nonlinear coupling terms)
              [ ψ̇(ẋ cos(ψ) - ẏ sin(ψ))]
              [          0               ]
              [          0               ]
              
    g(ξ) = [cos(ψ)  -sin(ψ)  0  0]    (input mapping)
           [sin(ψ)   cos(ψ)  0  0]
           [  0        0     1  0]
           [  0        0     0  1]
    """
    
    def __init__(self, params: Optional[QuadrotorParameters] = None,
                 perturbation_bound: float = 0.0):
        """
        Initialize the reduced tracking model.
        
        Args:
            params: Quadrotor physical parameters (for consistency, not used in reduced model)
            perturbation_bound: Maximum perturbation magnitude L such that |Δ(t)| ≤ L
        """
        self.params = params or QuadrotorParameters()
        self.perturbation_bound = perturbation_bound
    
    def compute_F(self, xi: np.ndarray, xi_dot: np.ndarray) -> np.ndarray:
        """
        Compute the nonlinear dynamics term F(ξ, ξ̇) from Equation (15).
        
        F(ξ,ξ̇) = [-ψ̇(ẋ sin(ψ) + ẏ cos(ψ))]
                  [ ψ̇(ẋ cos(ψ) - ẏ sin(ψ))]
                  [          0               ]
                  [          0               ]
        
        Args:
            xi: State vector [x, y, z, ψ]
            xi_dot: Velocity vector [ẋ, ẏ, ż, ψ̇]
            
        Returns:
            F vector of shape (4,)
        """
        x, y, z, psi = xi
        x_dot, y_dot, z_dot, psi_dot = xi_dot
        
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        
        F = np.array([
            -psi_dot * (x_dot * sin_psi + y_dot * cos_psi),  # ẍ coupling
             psi_dot * (x_dot * cos_psi - y_dot * sin_psi),  # ÿ coupling
            0.0,  # z̈ (no coupling)
            0.0   # ψ̈ (no coupling)
        ])
        
        return F
    
    def compute_g(self, xi: np.ndarray) -> np.ndarray:
        """
        Compute the input mapping matrix g(ξ) from Equation (15).
        
        g(ξ) = [cos(ψ)  -sin(ψ)  0  0]
               [sin(ψ)   cos(ψ)  0  0]
               [  0        0     1  0]
               [  0        0     0  1]
        
        Args:
            xi: State vector [x, y, z, ψ]
            
        Returns:
            g matrix of shape (4, 4)
        """
        psi = xi[3]
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        
        g = np.array([
            [cos_psi, -sin_psi, 0, 0],
            [sin_psi,  cos_psi, 0, 0],
            [0,        0,       1, 0],
            [0,        0,       0, 1]
        ])
        
        return g
    
    def compute_g_inverse(self, xi: np.ndarray) -> np.ndarray:
        """
        Compute the inverse of g(ξ).
        
        g⁻¹(ξ) = [ cos(ψ)  sin(ψ)  0  0]
                 [-sin(ψ)  cos(ψ)  0  0]
                 [   0       0     1  0]
                 [   0       0     0  1]
        
        Args:
            xi: State vector [x, y, z, ψ]
            
        Returns:
            g_inv matrix of shape (4, 4)
        """
        psi = xi[3]
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        
        g_inv = np.array([
            [ cos_psi, sin_psi, 0, 0],
            [-sin_psi, cos_psi, 0, 0],
            [0,        0,       1, 0],
            [0,        0,       0, 1]
        ])
        
        return g_inv
    
    def compute_dynamics(self, xi: np.ndarray, xi_dot: np.ndarray, 
                         U: np.ndarray, delta: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the full dynamics: ξ̈ = F(ξ,ξ̇) + g(ξ)U + Δ(t)
        
        Args:
            xi: State vector [x, y, z, ψ]
            xi_dot: Velocity vector [ẋ, ẏ, ż, ψ̇]
            U: Control input [a_x, a_y, a_z, Ω]
            delta: Perturbation vector (optional)
            
        Returns:
            Acceleration vector ξ̈ of shape (4,)
        """
        F = self.compute_F(xi, xi_dot)
        g = self.compute_g(xi)
        
        xi_ddot = F + g @ U
        
        if delta is not None:
            xi_ddot += delta
        
        return xi_ddot
    
    def integrate_step(self, state: QuadrotorState, U: np.ndarray, dt: float,
                       delta: Optional[np.ndarray] = None) -> QuadrotorState:
        """
        Integrate one time step using Euler method.
        
        Args:
            state: Current quadrotor state
            U: Control input [a_x, a_y, a_z, Ω]
            dt: Time step
            delta: Perturbation (optional)
            
        Returns:
            New QuadrotorState after integration
        """
        xi = state.to_reduced_state()
        xi_dot = state.to_reduced_velocity()
        
        # Compute acceleration
        xi_ddot = self.compute_dynamics(xi, xi_dot, U, delta)
        
        # Euler integration
        new_xi_dot = xi_dot + xi_ddot * dt
        new_xi = xi + xi_dot * dt + 0.5 * xi_ddot * dt**2
        
        # Normalize yaw angle
        new_xi[3] = np.mod(new_xi[3] + np.pi, 2*np.pi) - np.pi
        
        return QuadrotorState.from_reduced_state(new_xi, new_xi_dot)
    
    def integrate_rk4(self, state: QuadrotorState, U: np.ndarray, dt: float,
                      delta: Optional[np.ndarray] = None) -> QuadrotorState:
        """
        Integrate one time step using 4th-order Runge-Kutta method.
        
        Args:
            state: Current quadrotor state
            U: Control input [a_x, a_y, a_z, Ω]
            dt: Time step
            delta: Perturbation (optional)
            
        Returns:
            New QuadrotorState after integration
        """
        xi = state.to_reduced_state()
        xi_dot = state.to_reduced_velocity()
        
        def f(xi, xi_dot):
            xi_ddot = self.compute_dynamics(xi, xi_dot, U, delta)
            return xi_dot, xi_ddot
        
        # RK4 for second-order system
        k1_x, k1_v = f(xi, xi_dot)
        k2_x, k2_v = f(xi + 0.5*dt*k1_x, xi_dot + 0.5*dt*k1_v)
        k3_x, k3_v = f(xi + 0.5*dt*k2_x, xi_dot + 0.5*dt*k2_v)
        k4_x, k4_v = f(xi + dt*k3_x, xi_dot + dt*k3_v)
        
        new_xi = xi + (dt/6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        new_xi_dot = xi_dot + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        # Normalize yaw angle
        new_xi[3] = np.mod(new_xi[3] + np.pi, 2*np.pi) - np.pi
        
        return QuadrotorState.from_reduced_state(new_xi, new_xi_dot)


class QuadrotorDynamics:
    """
    Full 6-DOF quadrotor dynamics for simulation.
    
    This provides a more complete model for simulation purposes,
    while the ReducedTrackingModel is used for control design.
    
    State: [x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r]
           (position, velocity, Euler angles, body angular rates)
    
    Uses the parameters from Section 4 of the paper.
    """
    
    def __init__(self, params: Optional[QuadrotorParameters] = None):
        """
        Initialize full quadrotor dynamics.
        
        Args:
            params: Quadrotor physical parameters
        """
        self.params = params or QuadrotorParameters()
    
    def compute_rotation_matrix(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """
        Compute rotation matrix from body to inertial frame.
        
        Args:
            phi: Roll angle
            theta: Pitch angle  
            psi: Yaw angle
            
        Returns:
            3x3 rotation matrix
        """
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        
        R = np.array([
            [c_psi*c_theta, c_psi*s_theta*s_phi - s_psi*c_phi, c_psi*s_theta*c_phi + s_psi*s_phi],
            [s_psi*c_theta, s_psi*s_theta*s_phi + c_psi*c_phi, s_psi*s_theta*c_phi - c_psi*s_phi],
            [-s_theta,      c_theta*s_phi,                      c_theta*c_phi]
        ])
        
        return R
    
    def compute_angular_velocity_mapping(self, phi: float, theta: float) -> np.ndarray:
        """
        Compute mapping from body angular rates (p,q,r) to Euler angle rates (φ̇,θ̇,ψ̇).
        
        Args:
            phi: Roll angle
            theta: Pitch angle
            
        Returns:
            3x3 mapping matrix
        """
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, t_theta = np.cos(theta), np.tan(theta)
        
        W = np.array([
            [1, s_phi*t_theta, c_phi*t_theta],
            [0, c_phi,         -s_phi],
            [0, s_phi/c_theta, c_phi/c_theta]
        ])
        
        return W
    
    def dynamics(self, state: np.ndarray, thrust: float, 
                 torques: np.ndarray) -> np.ndarray:
        """
        Compute state derivative for full 6-DOF dynamics.
        
        Args:
            state: 12-element state vector
            thrust: Total thrust force (N)
            torques: [τ_φ, τ_θ, τ_ψ] torques (Nm)
            
        Returns:
            State derivative vector
        """
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        tau_phi, tau_theta, tau_psi = torques
        
        m = self.params.mass
        g = self.params.gravity
        Ixx = self.params.I_xx
        Iyy = self.params.I_yy
        Izz = self.params.I_zz
        
        # Position derivatives
        x_dot = vx
        y_dot = vy
        z_dot = vz
        
        # Velocity derivatives (translational dynamics)
        R = self.compute_rotation_matrix(phi, theta, psi)
        thrust_body = np.array([0, 0, thrust])
        thrust_inertial = R @ thrust_body
        
        vx_dot = thrust_inertial[0] / m
        vy_dot = thrust_inertial[1] / m
        vz_dot = thrust_inertial[2] / m - g
        
        # Angular velocity derivatives (rotational dynamics)
        p_dot = (tau_phi + (Iyy - Izz) * q * r) / Ixx
        q_dot = (tau_theta + (Izz - Ixx) * p * r) / Iyy
        r_dot = (tau_psi + (Ixx - Iyy) * p * q) / Izz
        
        # Euler angle derivatives
        W = self.compute_angular_velocity_mapping(phi, theta)
        euler_dot = W @ np.array([p, q, r])
        phi_dot, theta_dot, psi_dot = euler_dot
        
        return np.array([
            x_dot, y_dot, z_dot,
            vx_dot, vy_dot, vz_dot,
            phi_dot, theta_dot, psi_dot,
            p_dot, q_dot, r_dot
        ])
    
    def integrate_rk4(self, state: np.ndarray, thrust: float, 
                      torques: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate using RK4.
        
        Args:
            state: Current state
            thrust: Thrust force
            torques: Torque vector
            dt: Time step
            
        Returns:
            New state after integration
        """
        def f(s):
            return self.dynamics(s, thrust, torques)
        
        k1 = f(state)
        k2 = f(state + 0.5*dt*k1)
        k3 = f(state + 0.5*dt*k2)
        k4 = f(state + dt*k3)
        
        new_state = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize angles
        new_state[6:9] = np.mod(new_state[6:9] + np.pi, 2*np.pi) - np.pi
        
        return new_state


class VonKarmanTurbulence:
    """
    Von Kármán turbulence model for simulating wind disturbances.
    
    This model is used in Section 4 of the paper to simulate atmospheric
    disturbances affecting the MAVs.
    
    The model provides realistic representation of wind turbulence by modeling
    the power spectral density of velocity fluctuations.
    """
    
    def __init__(self, intensity: float = 0.5, 
                 scale_length: float = 3.0,
                 dt: float = 0.01):
        """
        Initialize Von Kármán turbulence generator.
        
        Args:
            intensity: Turbulence intensity (σ_w)
            scale_length: Turbulence scale length (L_w)
            dt: Sampling time step
        """
        self.intensity = intensity
        self.scale_length = scale_length
        self.dt = dt
        
        # Filter states for continuous turbulence
        self.state_u = np.zeros(2)
        self.state_v = np.zeros(2)
        self.state_w = np.zeros(2)
    
    def generate_disturbance(self, airspeed: float = 1.0) -> np.ndarray:
        """
        Generate turbulence disturbance at current time step.
        
        Args:
            airspeed: Vehicle airspeed (for frequency scaling)
            
        Returns:
            Disturbance vector [δx, δy, δz, δψ]
        """
        # White noise inputs
        noise_u = np.random.randn()
        noise_v = np.random.randn()
        noise_w = np.random.randn()
        
        # Time constant based on scale length and airspeed
        V = max(airspeed, 0.1)  # Avoid division by zero
        tau = self.scale_length / V
        
        # First-order filter for turbulence (simplified Von Kármán)
        alpha = self.dt / (tau + self.dt)
        
        # Update filter states
        self.state_u[0] = (1 - alpha) * self.state_u[0] + alpha * noise_u * self.intensity
        self.state_v[0] = (1 - alpha) * self.state_v[0] + alpha * noise_v * self.intensity
        self.state_w[0] = (1 - alpha) * self.state_w[0] + alpha * noise_w * self.intensity
        
        # Disturbance in body frame
        delta_x = self.state_u[0]
        delta_y = self.state_v[0]
        delta_z = self.state_w[0]
        delta_psi = 0.1 * (noise_u + noise_v) * self.intensity  # Small yaw disturbance
        
        return np.array([delta_x, delta_y, delta_z, delta_psi])
    
    def reset(self):
        """Reset filter states."""
        self.state_u = np.zeros(2)
        self.state_v = np.zeros(2)
        self.state_w = np.zeros(2)


# Example usage
if __name__ == "__main__":
    # Create reduced tracking model
    model = ReducedTrackingModel()
    
    # Initial state
    state = QuadrotorState(x=0.0, y=0.0, z=1.0, yaw=0.0)
    
    # Control input (accelerations)
    U = np.array([0.5, 0.0, 0.0, 0.1])  # Forward accel + yaw
    
    print("Initial state:", state.to_reduced_state())
    print("Control input:", U)
    
    # Simulate for 1 second
    dt = 0.01
    for _ in range(100):
        state = model.integrate_rk4(state, U, dt)
    
    print("Final state:", state.to_reduced_state())
    print("Final velocity:", state.to_reduced_velocity())

