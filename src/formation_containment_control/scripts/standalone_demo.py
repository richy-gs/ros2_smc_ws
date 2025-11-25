#!/usr/bin/env python3
"""
Standalone Demonstration Script

Demonstrates the formation-containment control algorithm without ROS2.
Useful for understanding the mathematics and testing the core algorithms.

This script simulates:
1. 4 followers converging into formation
2. 4 leaders tracking virtual leader trajectory
3. Collision avoidance during convergence
4. Formation change (square → tetrahedron)
5. Disturbance rejection (Von Kármán turbulence)

Run with: python3 standalone_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formation_containment_control.controllers.formation_controller import (
    FormationController, FormationConfig
)
from formation_containment_control.core.dynamics import (
    ReducedTrackingModel, QuadrotorState, VonKarmanTurbulence
)
from formation_containment_control.core.convex_hull import FormationGeometry


class FormationSimulation:
    """
    Standalone simulation of formation-containment control.
    """
    
    def __init__(self, config: FormationConfig = None):
        """Initialize simulation."""
        self.config = config or FormationConfig(
            n_followers=4,
            n_leaders=4,
            topology="paper",
            formation_type="square",
            formation_scale=1.0,
            formation_height=1.0,
            lambda_gain=3.0,
            alpha=4.0,
            beta=0.125,
            safety_distance=0.3,
            dt=0.01,
            use_collision_avoidance=True
        )
        
        # Formation controller
        self.controller = FormationController(self.config)
        
        # Dynamics models
        self.dynamics = ReducedTrackingModel()
        
        # Turbulence generators
        self.turbulence = [VonKarmanTurbulence(intensity=0.3) 
                          for _ in range(self.config.n_followers + self.config.n_leaders)]
        
        # State arrays (n x 4: [x, y, z, yaw])
        self.follower_states = np.zeros((self.config.n_followers, 4))
        self.follower_velocities = np.zeros((self.config.n_followers, 4))
        self.leader_states = np.zeros((self.config.n_leaders, 4))
        self.leader_velocities = np.zeros((self.config.n_leaders, 4))
        
        # Virtual leader
        self.vl_state = np.array([0.0, 0.0, 1.0, 0.0])
        self.vl_velocity = np.zeros(4)
        
        # Time
        self.t = 0.0
        self.dt = self.config.dt
        
        # History for plotting
        self.history = {
            'time': [],
            'vl': [],
            'leaders': [[] for _ in range(self.config.n_leaders)],
            'followers': [[] for _ in range(self.config.n_followers)],
            'errors': [],
            'adaptive_gains': [],
        }
        
        # Initialize positions (from paper Section 4)
        self._init_positions()
    
    def _init_positions(self):
        """Initialize agent positions (from paper)."""
        # Follower initial conditions (from paper)
        # F1(0) = [-0.7, 1.0, 0.0, 0]
        # F2(0) = [-2.0, 0.0, 0.0, 0]
        # F3(0) = [0.7, -1.0, 0.0, 0]
        # F4(0) = [1.0, 1.0, 0.0, 0]
        self.follower_states = np.array([
            [-0.7, 1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [0.7, -1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0]
        ])
        
        # Leader initial conditions (from paper)
        # L1(0) = [1.0, 0.0, 0.0, π]
        # L2(0) = [-1.0, 0.0, 0.0, π]
        # L3(0) = [0.0, 1.0, 0.0, π]
        # L4(0) = [0.0, -1.0, 0.0, π]
        self.leader_states = np.array([
            [1.0, 0.0, 0.0, np.pi],
            [-1.0, 0.0, 0.0, np.pi],
            [0.0, 1.0, 0.0, np.pi],
            [0.0, -1.0, 0.0, np.pi]
        ])
    
    def _virtual_leader_trajectory(self, t: float) -> tuple:
        """
        3D infinity trajectory (lemniscate) as in paper.
        
        Returns:
            Tuple of (state, velocity) each shape (4,)
        """
        period = 60.0  # seconds
        scale = 2.0
        height = 1.0
        omega = 2 * np.pi / period
        
        # Position
        x = scale * np.sin(omega * t)
        y = scale * np.sin(omega * t) * np.cos(omega * t)
        z = height + 0.3 * np.sin(omega * t / 2)
        
        # Velocity
        x_dot = scale * omega * np.cos(omega * t)
        y_dot = scale * omega * (np.cos(omega * t)**2 - np.sin(omega * t)**2)
        z_dot = 0.15 * omega * np.cos(omega * t / 2)
        
        # Yaw (follows velocity direction)
        yaw = np.arctan2(y_dot, x_dot)
        yaw_dot = 0.0
        
        state = np.array([x, y, z, yaw])
        velocity = np.array([x_dot, y_dot, z_dot, yaw_dot])
        
        return state, velocity
    
    def step(self, use_turbulence: bool = False):
        """Execute one simulation step."""
        # Update virtual leader
        self.vl_state, self.vl_velocity = self._virtual_leader_trajectory(self.t)
        
        # Update controller
        self.controller.set_virtual_leader_state(self.vl_state, self.vl_velocity)
        self.controller.update_agent_states(
            self.leader_states, self.leader_velocities,
            self.follower_states, self.follower_velocities
        )
        
        # Compute controls
        leader_controls, follower_controls = self.controller.compute_all_controls()
        
        # Apply controls with dynamics
        for i in range(self.config.n_leaders):
            xi = self.leader_states[i]
            xi_dot = self.leader_velocities[i]
            U = leader_controls[i]
            
            # Add turbulence if enabled
            delta = None
            if use_turbulence:
                delta = self.turbulence[self.config.n_followers + i].generate_disturbance()
            
            # Compute acceleration
            xi_ddot = self.dynamics.compute_dynamics(xi, xi_dot, U, delta)
            
            # Integrate
            self.leader_velocities[i] = xi_dot + xi_ddot * self.dt
            self.leader_states[i] = xi + xi_dot * self.dt + 0.5 * xi_ddot * self.dt**2
            
            # Normalize yaw
            self.leader_states[i, 3] = np.mod(self.leader_states[i, 3] + np.pi, 2*np.pi) - np.pi
        
        for i in range(self.config.n_followers):
            xi = self.follower_states[i]
            xi_dot = self.follower_velocities[i]
            U = follower_controls[i]
            
            delta = None
            if use_turbulence:
                delta = self.turbulence[i].generate_disturbance()
            
            xi_ddot = self.dynamics.compute_dynamics(xi, xi_dot, U, delta)
            
            self.follower_velocities[i] = xi_dot + xi_ddot * self.dt
            self.follower_states[i] = xi + xi_dot * self.dt + 0.5 * xi_ddot * self.dt**2
            self.follower_states[i, 3] = np.mod(self.follower_states[i, 3] + np.pi, 2*np.pi) - np.pi
        
        # Record history
        self._record_history()
        
        # Advance time
        self.t += self.dt
    
    def _record_history(self):
        """Record current state for plotting."""
        self.history['time'].append(self.t)
        self.history['vl'].append(self.vl_state.copy())
        
        for i in range(self.config.n_leaders):
            self.history['leaders'][i].append(self.leader_states[i].copy())
        
        for i in range(self.config.n_followers):
            self.history['followers'][i].append(self.follower_states[i].copy())
        
        # Compute errors
        status = self.controller.check_formation_status()
        self.history['errors'].append(status['max_follower_error'])
        
        # Record adaptive gains
        gains = [ctrl.controller.K_c.copy() 
                 for ctrl in self.controller.follower_controllers]
        self.history['adaptive_gains'].append(np.mean([np.mean(g) for g in gains]))
    
    def run(self, duration: float, turbulence_start: float = None):
        """
        Run simulation for given duration.
        
        Args:
            duration: Simulation duration in seconds
            turbulence_start: Time to start turbulence (None = no turbulence)
        """
        n_steps = int(duration / self.dt)
        
        print(f"Running simulation for {duration}s ({n_steps} steps)...")
        print(f"Configuration: {self.config.n_followers} followers, {self.config.n_leaders} leaders")
        
        for step in range(n_steps):
            use_turb = turbulence_start is not None and self.t >= turbulence_start
            self.step(use_turbulence=use_turb)
            
            # Progress indicator
            if step % (n_steps // 10) == 0:
                pct = 100 * step / n_steps
                print(f"  Progress: {pct:.0f}%")
        
        print("Simulation complete!")
    
    def plot_results(self):
        """Plot simulation results."""
        fig = plt.figure(figsize=(16, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self._plot_3d_trajectories(ax1)
        
        # Top view
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_top_view(ax2)
        
        # Error evolution
        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_errors(ax3)
        
        # Adaptive gains
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_adaptive_gains(ax4)
        
        plt.tight_layout()
        plt.savefig('formation_simulation_results.png', dpi=150)
        print("Results saved to formation_simulation_results.png")
        plt.show()
    
    def _plot_3d_trajectories(self, ax):
        """Plot 3D trajectories."""
        # Virtual leader
        vl = np.array(self.history['vl'])
        ax.plot(vl[:, 0], vl[:, 1], vl[:, 2], 'g-', linewidth=2, label='Virtual Leader')
        
        # Leaders
        colors = ['red', 'blue', 'orange', 'purple']
        for i in range(self.config.n_leaders):
            traj = np.array(self.history['leaders'][i])
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color=colors[i % len(colors)], linestyle='-', 
                   label=f'Leader {i+1}')
        
        # Followers
        for i in range(self.config.n_followers):
            traj = np.array(self.history['followers'][i])
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color='gray', linestyle='--', alpha=0.7,
                   label=f'Follower {i+1}' if i == 0 else None)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectories')
        ax.legend(loc='upper left')
    
    def _plot_top_view(self, ax):
        """Plot top view (X-Y plane)."""
        # Virtual leader
        vl = np.array(self.history['vl'])
        ax.plot(vl[:, 0], vl[:, 1], 'g-', linewidth=2, label='VL')
        
        # Final positions
        ax.scatter([self.vl_state[0]], [self.vl_state[1]], 
                  c='green', s=200, marker='*', zorder=10)
        
        colors = ['red', 'blue', 'orange', 'purple']
        for i in range(self.config.n_leaders):
            traj = np.array(self.history['leaders'][i])
            ax.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)], 
                   linestyle='-', label=f'L{i+1}')
            ax.scatter([self.leader_states[i, 0]], [self.leader_states[i, 1]], 
                      c=colors[i % len(colors)], s=100, marker='o', zorder=10)
        
        for i in range(self.config.n_followers):
            traj = np.array(self.history['followers'][i])
            ax.plot(traj[:, 0], traj[:, 1], color='gray', 
                   linestyle='--', alpha=0.7)
            ax.scatter([self.follower_states[i, 0]], [self.follower_states[i, 1]], 
                      c='gray', s=80, marker='s', zorder=10)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Top View (X-Y Plane)')
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_errors(self, ax):
        """Plot containment errors over time."""
        t = np.array(self.history['time'])
        errors = np.array(self.history['errors'])
        
        ax.plot(t, errors, 'b-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Max Containment Error (m)')
        ax.set_title('Containment Error Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    def _plot_adaptive_gains(self, ax):
        """Plot adaptive gains over time."""
        t = np.array(self.history['time'])
        gains = np.array(self.history['adaptive_gains'])
        
        ax.plot(t, gains, 'r-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Average Adaptive Gain Kc')
        ax.set_title('Adaptive Gain Evolution')
        ax.grid(True, alpha=0.3)


def main():
    """Run demonstration."""
    print("=" * 60)
    print("Formation-Containment Control Demonstration")
    print("Based on Katt & Castañeda (2025)")
    print("=" * 60)
    print()
    
    # Create simulation with paper parameters
    config = FormationConfig(
        n_followers=4,
        n_leaders=4,
        topology="paper",
        formation_type="square",
        formation_scale=1.0,
        lambda_gain=3.0,
        alpha=4.0,
        beta=0.125,
        safety_distance=0.3,
        dt=0.01
    )
    
    sim = FormationSimulation(config)
    
    # Run simulation
    # Paper simulation: 100+ seconds with turbulence at t=45s
    # For demo, use shorter duration
    sim.run(duration=30.0, turbulence_start=None)  # No turbulence for cleaner demo
    
    # Print final status
    status = sim.controller.check_formation_status()
    print()
    print("Final Formation Status:")
    print(f"  Formation Achieved: {status['formation_achieved']}")
    print(f"  Containment Achieved: {status['containment_achieved']}")
    print(f"  Collision Free: {status['collision_free']}")
    print(f"  Max Leader Error: {status['max_leader_error']:.4f} m")
    print(f"  Max Follower Error: {status['max_follower_error']:.4f} m")
    print(f"  Min Inter-Agent Distance: {status['min_inter_agent_distance']:.4f} m")
    
    # Plot results
    sim.plot_results()


if __name__ == "__main__":
    main()

