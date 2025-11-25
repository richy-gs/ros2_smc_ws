#!/usr/bin/env python3
"""
Unit Tests for Controllers

Tests the SGASMC controller and formation controller implementations.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from formation_containment_control.controllers.sgasmc import (
    SGASMCController,
    SGASMCParameters,
    SlidingSurface,
    ContainmentErrorComputer
)
from formation_containment_control.controllers.formation_controller import (
    FormationController,
    FormationConfig,
    LeaderController,
    FollowerController
)
from formation_containment_control.core.graph_theory import create_interaction_network
from formation_containment_control.core.dynamics import ReducedTrackingModel


class TestSlidingSurface:
    """Tests for SlidingSurface class."""
    
    def test_surface_computation(self):
        """Test sliding surface computation (Equation 26)."""
        surface = SlidingSurface(dim=4, lambda_gains=np.ones(4) * 3.0)
        
        error = np.array([0.1, 0.2, 0.0, 0.05])
        error_dot = np.array([0.01, 0.02, 0.0, 0.005])
        
        # σ = ė + λe
        sigma = surface.compute(error, error_dot)
        
        expected = error_dot + 3.0 * error
        np.testing.assert_array_almost_equal(sigma, expected)
    
    def test_surface_zero_on_sliding(self):
        """Test that σ=0 implies ė = -λe."""
        surface = SlidingSurface(dim=4, lambda_gains=np.ones(4) * 3.0)
        
        error = np.array([0.1, 0.2, 0.1, 0.05])
        # When on sliding surface: ė = -λe
        error_dot = -3.0 * error
        
        sigma = surface.compute(error, error_dot)
        
        np.testing.assert_array_almost_equal(sigma, np.zeros(4), decimal=10)


class TestSGASMCController:
    """Tests for SGASMC controller."""
    
    def test_auxiliary_control_computation(self):
        """Test auxiliary control u_i (Equation 29)."""
        params = SGASMCParameters(lambda_gain=3.0, alpha=4.0, beta=0.125)
        controller = SGASMCController(params)
        
        sigma = np.array([0.1, 0.2, 0.0, 0.05])
        
        u_aux = controller.compute_auxiliary_control(sigma)
        
        # u = -2K|σ|^{1/2}sign(σ) - (K²/2)σ
        K = controller.K_c
        expected_term1 = -2.0 * K * np.sqrt(np.abs(sigma) + 1e-10) * np.sign(sigma)
        expected_term2 = -0.5 * K**2 * sigma
        expected = expected_term1 + expected_term2
        
        np.testing.assert_array_almost_equal(u_aux, expected)
    
    def test_adaptive_gain_update(self):
        """Test adaptive gain update (Equation 30)."""
        params = SGASMCParameters(lambda_gain=3.0, alpha=4.0, beta=0.125, dt=0.01)
        controller = SGASMCController(params)
        
        initial_K = controller.K_c.copy()
        
        # Large sliding surface value should increase gain
        sigma = np.array([1.0, 1.0, 1.0, 1.0])
        controller.update_adaptive_gain(sigma)
        
        # Gain should increase
        assert np.all(controller.K_c >= initial_K)
    
    def test_adaptive_gain_bounded(self):
        """Test that adaptive gain stays bounded."""
        params = SGASMCParameters(K_c_min=0.1, K_c_max=10.0)
        controller = SGASMCController(params)
        
        # Run many updates with large sigma
        sigma = np.array([10.0, 10.0, 10.0, 10.0])
        for _ in range(1000):
            controller.update_adaptive_gain(sigma)
        
        # Should be bounded
        assert np.all(controller.K_c <= params.K_c_max)
        assert np.all(controller.K_c >= params.K_c_min)
    
    def test_control_computation(self):
        """Test full control computation (Equation 28)."""
        params = SGASMCParameters()
        controller = SGASMCController(params)
        
        xi = np.array([0.0, 0.0, 1.0, 0.0])
        xi_dot = np.array([0.1, 0.0, 0.0, 0.0])
        error = np.array([0.5, 0.3, 0.0, 0.1])
        error_dot = np.array([0.1, 0.05, 0.0, 0.02])
        
        F = np.zeros(4)
        g = np.eye(4)
        g_inv = np.eye(4)
        
        U = controller.compute_control(
            xi, xi_dot, error, error_dot, F, g, g_inv
        )
        
        # Control should be finite
        assert np.all(np.isfinite(U))
        
        # Control should be reasonable magnitude
        assert np.all(np.abs(U) < 100)
    
    def test_convergence(self):
        """Test that controller reduces error over time."""
        params = SGASMCParameters(lambda_gain=3.0, alpha=4.0, beta=0.125, dt=0.01)
        controller = SGASMCController(params)
        dynamics = ReducedTrackingModel()
        
        # Initial state with error
        xi = np.array([0.5, 0.3, 1.0, 0.1])
        xi_dot = np.array([0.1, 0.05, 0.0, 0.02])
        desired = np.array([0.0, 0.0, 1.0, 0.0])
        
        initial_error = np.linalg.norm(xi - desired)
        
        # Simulate for some steps
        for _ in range(500):
            error = desired - xi
            error_dot = -xi_dot  # Assuming desired is stationary
            
            F = dynamics.compute_F(xi, xi_dot)
            g = dynamics.compute_g(xi)
            g_inv = dynamics.compute_g_inverse(xi)
            
            U = controller.compute_control(
                xi, xi_dot, error, error_dot, F, g, g_inv
            )
            
            # Simple Euler integration
            xi_ddot = dynamics.compute_dynamics(xi, xi_dot, U)
            xi_dot = xi_dot + xi_ddot * params.dt
            xi = xi + xi_dot * params.dt
        
        final_error = np.linalg.norm(xi - desired)
        
        # Error should decrease
        assert final_error < initial_error


class TestContainmentErrorComputer:
    """Tests for containment error computation."""
    
    def test_basic_error_computation(self):
        """Test containment error (Equation 18)."""
        network = create_interaction_network(4, 4, "paper")
        error_computer = ContainmentErrorComputer(network, state_dim=4)
        
        # Leader positions
        leader_states = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
        ])
        
        # Follower at origin
        follower_states = np.zeros((4, 4))
        follower_states[:, 2] = 1.0  # At correct height
        
        errors = error_computer.compute_containment_error(
            follower_states, leader_states
        )
        
        assert errors.shape == (4, 4)
    
    def test_collision_avoidance_error(self):
        """Test collision avoidance error (Equation 24)."""
        network = create_interaction_network(4, 4, "paper")
        error_computer = ContainmentErrorComputer(network, state_dim=4)
        
        # All agents at different positions
        all_states = np.zeros((8, 4))
        all_states[:, 0] = np.linspace(-2, 2, 8)  # Spread along x
        all_states[:, 2] = 1.0
        
        ca_errors = error_computer.compute_collision_avoidance_error(all_states)
        
        assert ca_errors.shape == (4, 4)


class TestFormationController:
    """Tests for FormationController class."""
    
    def test_initialization(self):
        """Test formation controller initialization."""
        config = FormationConfig(
            n_followers=4,
            n_leaders=4,
            topology="paper",
            formation_type="square"
        )
        
        controller = FormationController(config)
        
        assert len(controller.leader_controllers) == 4
        assert len(controller.follower_controllers) == 4
    
    def test_formation_offsets(self):
        """Test formation offset generation."""
        config = FormationConfig(
            formation_type="square",
            formation_scale=1.0,
            n_leaders=4
        )
        
        controller = FormationController(config)
        offsets = controller.formation_offsets
        
        # Should have 4 offsets
        assert len(offsets) == 4
        
        # Offsets should be distinct
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.linalg.norm(offsets[i] - offsets[j])
                assert dist > 0.1
    
    def test_control_computation(self):
        """Test control computation for all agents."""
        config = FormationConfig(n_followers=4, n_leaders=4)
        controller = FormationController(config)
        
        # Set virtual leader state
        controller.set_virtual_leader_state(
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.5, 0.0, 0.0, 0.0])
        )
        
        # Set agent states
        leader_states = np.random.randn(4, 4) * 0.5
        leader_states[:, 2] = 1.0
        follower_states = np.random.randn(4, 4) * 0.5
        follower_states[:, 2] = 1.0
        
        controller.update_agent_states(
            leader_states,
            np.zeros((4, 4)),
            follower_states,
            np.zeros((4, 4))
        )
        
        leader_controls, follower_controls = controller.compute_all_controls()
        
        assert leader_controls.shape == (4, 4)
        assert follower_controls.shape == (4, 4)
        
        # Controls should be finite
        assert np.all(np.isfinite(leader_controls))
        assert np.all(np.isfinite(follower_controls))
    
    def test_formation_status(self):
        """Test formation status checking."""
        config = FormationConfig(n_followers=4, n_leaders=4)
        controller = FormationController(config)
        
        # Set virtual leader
        controller.set_virtual_leader_state(
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.zeros(4)
        )
        
        # Set agents at formation positions
        leader_states = np.zeros((4, 4))
        for i, offset in enumerate(controller.formation_offsets):
            leader_states[i, :3] = offset[:3]
        leader_states[:, 2] = 1.0
        
        follower_states = np.zeros((4, 4))
        follower_states[:, 2] = 1.0
        
        controller.update_agent_states(
            leader_states, np.zeros((4, 4)),
            follower_states, np.zeros((4, 4))
        )
        
        status = controller.check_formation_status()
        
        assert 'formation_achieved' in status
        assert 'containment_achieved' in status
        assert 'collision_free' in status
        assert 'min_inter_agent_distance' in status
    
    def test_formation_change(self):
        """Test dynamic formation change."""
        config = FormationConfig(formation_type="square")
        controller = FormationController(config)
        
        original_offsets = controller.formation_offsets.copy()
        
        # Change to tetrahedron
        controller.change_formation("tetrahedron")
        
        new_offsets = controller.formation_offsets
        
        # Offsets should have changed
        assert not np.allclose(original_offsets, new_offsets)


class TestLeaderController:
    """Tests for LeaderController class."""
    
    def test_tracking_control(self):
        """Test leader trajectory tracking."""
        offset = np.array([1.0, 0.0, 0.0, 0.0])
        controller = LeaderController(leader_id=0, formation_offset=offset)
        
        current_state = np.array([0.5, 0.0, 1.0, 0.0])
        current_vel = np.zeros(4)
        vl_state = np.array([0.0, 0.0, 1.0, 0.0])
        vl_vel = np.array([0.5, 0.0, 0.0, 0.0])
        
        control = controller.compute_control(
            current_state, current_vel, vl_state, vl_vel
        )
        
        assert control.shape == (4,)
        assert np.all(np.isfinite(control))


class TestFollowerController:
    """Tests for FollowerController class."""
    
    def test_containment_control(self):
        """Test follower containment control."""
        network = create_interaction_network(4, 4, "paper")
        controller = FollowerController(
            follower_id=0,
            interaction_network=network
        )
        
        # Leader states (square formation)
        leader_states = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
        ])
        leader_velocities = np.zeros((4, 4))
        
        # Follower outside convex hull
        current_state = np.array([2.0, 0.0, 1.0, 0.0])
        current_vel = np.zeros(4)
        
        control = controller.compute_control(
            current_state, current_vel,
            leader_states, leader_velocities
        )
        
        assert control.shape == (4,)
        
        # Control should push toward center
        # x-component should be negative (toward center)
        # This is indirect - control is acceleration
    
    def test_desired_position(self):
        """Test desired position computation."""
        network = create_interaction_network(4, 4, "paper")
        controller = FollowerController(follower_id=0, interaction_network=network)
        
        leader_states = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
        ])
        
        desired = controller.get_desired_position(leader_states)
        
        # Desired should be inside convex hull
        assert -1.1 <= desired[0] <= 1.1
        assert -1.1 <= desired[1] <= 1.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

