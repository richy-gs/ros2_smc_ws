#!/usr/bin/env python3
"""
Unit Tests for Graph Theory Module

Tests the graph-based multi-agent interaction network implementation,
including adjacency matrices, Laplacian computation, and containment weights.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from formation_containment_control.core.graph_theory import (
    AdjacencyMatrix,
    LaplacianMatrix,
    GraphTopology,
    InteractionNetwork,
    create_interaction_network
)


class TestAdjacencyMatrix:
    """Tests for AdjacencyMatrix class."""
    
    def test_complete_graph(self):
        """Test complete graph generation."""
        adj = GraphTopology.complete_graph(n_followers=3, n_leaders=2)
        
        assert adj.n_followers == 3
        assert adj.n_leaders == 2
        assert adj.n_agents == 5
        
        # Followers should receive from all other agents
        for i in range(3):
            neighbors = adj.get_neighbors(i)
            # Should have 4 neighbors (all except self)
            assert len(neighbors) == 4
        
        # Leaders don't receive information
        for i in range(3, 5):
            neighbors = adj.get_neighbors(i)
            assert len(neighbors) == 0
    
    def test_ring_graph(self):
        """Test ring graph generation."""
        adj = GraphTopology.ring_graph(n_followers=4, n_leaders=2)
        
        assert adj.n_followers == 4
        assert adj.n_leaders == 2
        
        # Each follower should have at least 2 neighbors (ring)
        for i in range(4):
            neighbors = adj.get_neighbors(i)
            assert len(neighbors) >= 2
    
    def test_paper_topology(self):
        """Test paper topology (Figure 1)."""
        adj = GraphTopology.paper_topology(n_followers=4, n_leaders=4)
        
        # F1 receives from F2, F4, L1, L2 (4 neighbors)
        f1_neighbors = adj.get_neighbors(0)
        assert len(f1_neighbors) == 4
        assert 1 in f1_neighbors  # F2
        assert 3 in f1_neighbors  # F4
        assert 4 in f1_neighbors  # L1
        assert 5 in f1_neighbors  # L2
        
        # Leaders don't receive
        for i in range(4, 8):
            assert len(adj.get_neighbors(i)) == 0


class TestInteractionNetwork:
    """Tests for InteractionNetwork class."""
    
    def test_laplacian_computation(self):
        """Test Laplacian matrix computation."""
        network = create_interaction_network(4, 4, "paper")
        
        # L = D - A
        D = network.degree_matrix
        A = network.adjacency.matrix
        L_computed = D - A
        
        np.testing.assert_array_almost_equal(network.L, L_computed)
    
    def test_laplacian_partitioning(self):
        """Test Laplacian matrix partitioning."""
        network = create_interaction_network(4, 4, "paper")
        
        # L_N should be 4x4
        assert network.L_N.shape == (4, 4)
        
        # L_M should be 4x4
        assert network.L_M.shape == (4, 4)
        
        # Verify partitioning matches full Laplacian
        np.testing.assert_array_almost_equal(
            network.L[:4, :4], network.L_N
        )
        np.testing.assert_array_almost_equal(
            network.L[:4, 4:], network.L_M
        )
    
    def test_assumption_1(self):
        """Test verification of Assumption 1 from paper."""
        network = create_interaction_network(4, 4, "paper")
        
        valid, msg = network.verify_connectivity()
        assert valid, f"Assumption 1 failed: {msg}"
        
        # Check L_N eigenvalues are positive
        eigenvalues = np.linalg.eigvals(network.L_N)
        assert np.all(np.real(eigenvalues) > 0)
        
        # Check containment weights
        weights = network.laplacian.containment_weights
        
        # Row sums should be 1
        row_sums = np.sum(weights, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4))
        
        # All elements should be non-negative
        assert np.all(weights >= -1e-10)
    
    def test_containment_computation(self):
        """Test desired follower position computation."""
        network = create_interaction_network(4, 4, "paper")
        
        # Leader positions (square formation)
        leader_positions = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
        ])
        
        # Compute desired follower positions
        desired = network.laplacian.get_all_desired_follower_positions(
            leader_positions
        )
        
        # Should return 4 positions (one per follower)
        assert desired.shape == (4, 4)
        
        # Desired positions should be inside convex hull
        # Check z-coordinate is within leader range
        assert np.all(desired[:, 2] >= 0.9)
        assert np.all(desired[:, 2] <= 1.1)
    
    def test_collision_avoidance_matrices(self):
        """Test collision avoidance matrix computation."""
        network = create_interaction_network(4, 4, "paper")
        
        collision_matrix, L_zeta = network.get_collision_avoidance_matrices()
        
        # Matrices should have correct dimensions
        assert collision_matrix.shape == (8, 8)
        assert L_zeta.shape == (8, 8)
        
        # L_zeta lower right should be identity
        np.testing.assert_array_almost_equal(
            L_zeta[4:, 4:], np.eye(4)
        )


class TestConvexCombination:
    """Tests for convex combination properties."""
    
    def test_convex_combination_weights(self):
        """Test that weights form valid convex combinations."""
        network = create_interaction_network(4, 4, "paper")
        
        weights = network.laplacian.containment_weights
        
        # Test for each follower
        for i in range(4):
            follower_weights = weights[i, :]
            
            # Sum to 1
            assert abs(np.sum(follower_weights) - 1.0) < 1e-10
            
            # Non-negative
            assert np.all(follower_weights >= -1e-10)
    
    def test_follower_inside_hull(self):
        """Test that desired positions are inside convex hull."""
        network = create_interaction_network(4, 4, "paper")
        
        # Square leader formation
        leaders = np.array([
            [1, 1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
        ], dtype=float)
        
        # Get desired follower positions
        for i in range(4):
            desired = network.laplacian.get_desired_follower_position(
                i, leaders
            )
            
            # x and y should be within [-1, 1]
            assert -1.1 <= desired[0] <= 1.1
            assert -1.1 <= desired[1] <= 1.1


class TestGraphVariants:
    """Tests for different graph topologies."""
    
    def test_different_sizes(self):
        """Test networks with different numbers of agents."""
        for n_followers in [2, 4, 6]:
            for n_leaders in [2, 3, 4]:
                network = create_interaction_network(
                    n_followers, n_leaders, "complete"
                )
                
                assert network.n_followers == n_followers
                assert network.n_leaders == n_leaders
                
                # Verify Assumption 1 for complete graph
                valid, _ = network.verify_connectivity()
                assert valid
    
    def test_ring_topology_connectivity(self):
        """Test that ring topology maintains connectivity."""
        network = create_interaction_network(4, 4, "ring")
        
        # Should still satisfy Assumption 1
        valid, msg = network.verify_connectivity()
        assert valid, f"Ring topology failed: {msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

