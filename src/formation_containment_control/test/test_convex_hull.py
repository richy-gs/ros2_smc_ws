#!/usr/bin/env python3
"""
Unit Tests for Convex Hull Module

Tests convex hull computation and containment verification.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from formation_containment_control.core.convex_hull import (
    compute_convex_hull,
    point_in_convex_hull,
    ConvexHullContainment,
    FormationGeometry,
    compute_convex_combination_weights
)


class TestConvexHullComputation:
    """Tests for convex hull computation."""
    
    def test_square_hull(self):
        """Test convex hull of square."""
        points = np.array([
            [1, 1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
        ], dtype=float)
        
        result = compute_convex_hull(points, dimension=2)
        
        assert result.is_valid
        assert len(result.vertices) == 4
        assert result.volume > 0  # Area in 2D
    
    def test_triangle_hull(self):
        """Test convex hull of triangle."""
        points = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [-1, 0, 0],
        ], dtype=float)
        
        result = compute_convex_hull(points, dimension=2)
        
        assert result.is_valid
        assert len(result.vertices) == 3
    
    def test_tetrahedron_hull(self):
        """Test convex hull of tetrahedron."""
        points = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        
        result = compute_convex_hull(points, dimension=3)
        
        assert result.is_valid
        assert len(result.vertices) == 4
        assert result.volume > 0
    
    def test_insufficient_points(self):
        """Test handling of insufficient points."""
        points = np.array([
            [0, 0, 0],
            [1, 1, 1],
        ], dtype=float)
        
        result = compute_convex_hull(points, dimension=3)
        
        assert not result.is_valid
        assert "Need at least" in result.error_message


class TestPointInHull:
    """Tests for point-in-hull computation."""
    
    def test_point_inside_square(self):
        """Test point inside square hull."""
        hull_points = np.array([
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ], dtype=float)
        
        # Point at center
        assert point_in_convex_hull(np.array([0.0, 0.0]), hull_points)
        
        # Point inside
        assert point_in_convex_hull(np.array([0.5, 0.5]), hull_points)
    
    def test_point_outside_square(self):
        """Test point outside square hull."""
        hull_points = np.array([
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ], dtype=float)
        
        # Point outside
        assert not point_in_convex_hull(np.array([2.0, 0.0]), hull_points)
        assert not point_in_convex_hull(np.array([0.0, 2.0]), hull_points)
    
    def test_point_on_boundary(self):
        """Test point on hull boundary."""
        hull_points = np.array([
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ], dtype=float)
        
        # Point on edge
        assert point_in_convex_hull(np.array([1.0, 0.0]), hull_points)
    
    def test_3d_hull(self):
        """Test point in 3D hull."""
        hull_points = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ], dtype=float)
        
        # Point at center
        assert point_in_convex_hull(np.array([0.0, 0.0, 0.0]), hull_points)
        
        # Point outside
        assert not point_in_convex_hull(np.array([2.0, 0.0, 0.0]), hull_points)


class TestConvexCombinationWeights:
    """Tests for convex combination weight computation."""
    
    def test_centroid_weights(self):
        """Test weights for centroid point."""
        hull_points = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
        ], dtype=float)
        
        # Centroid
        point = np.array([0.0, 0.0])
        weights = compute_convex_combination_weights(point, hull_points)
        
        assert weights is not None
        assert len(weights) == 4
        assert np.abs(np.sum(weights) - 1.0) < 1e-6
        assert np.all(weights >= -1e-6)
    
    def test_vertex_weights(self):
        """Test weights for vertex point."""
        hull_points = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
        ], dtype=float)
        
        # First vertex
        point = np.array([1.0, 0.0])
        weights = compute_convex_combination_weights(point, hull_points)
        
        assert weights is not None
        # Weight for first vertex should be ~1
        assert weights[0] > 0.9


class TestConvexHullContainment:
    """Tests for ConvexHullContainment class."""
    
    def test_initialization(self):
        """Test containment manager initialization."""
        containment = ConvexHullContainment(n_leaders=4, dimension=3)
        
        assert containment.n_leaders == 4
        assert containment.dimension == 3
    
    def test_update_leaders(self):
        """Test leader state update."""
        containment = ConvexHullContainment(n_leaders=4)
        
        positions = np.array([
            [1, 0, 1],
            [-1, 0, 1],
            [0, 1, 1],
            [0, -1, 1],
        ], dtype=float)
        
        containment.update_leader_states(positions)
        
        assert containment.hull_result is not None
        assert containment.hull_result.is_valid
    
    def test_desired_position(self):
        """Test desired position from weights."""
        containment = ConvexHullContainment(n_leaders=4)
        
        positions = np.array([
            [1, 0, 1],
            [-1, 0, 1],
            [0, 1, 1],
            [0, -1, 1],
        ], dtype=float)
        
        containment.update_leader_states(positions)
        
        # Equal weights -> centroid
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        desired = containment.get_desired_position(weights)
        
        np.testing.assert_array_almost_equal(desired, [0, 0, 1])
    
    def test_containment_check(self):
        """Test containment verification."""
        containment = ConvexHullContainment(n_leaders=4)
        
        positions = np.array([
            [1, 0, 1],
            [-1, 0, 1],
            [0, 1, 1],
            [0, -1, 1],
        ], dtype=float)
        
        containment.update_leader_states(positions)
        
        # Point inside
        assert containment.is_point_contained(np.array([0, 0, 1]))
        
        # Point outside
        assert not containment.is_point_contained(np.array([2, 0, 1]))
    
    def test_visualization_data(self):
        """Test visualization data generation."""
        containment = ConvexHullContainment(n_leaders=4)
        
        positions = np.array([
            [1, 0, 1],
            [-1, 0, 1],
            [0, 1, 1],
            [0, -1, 1],
        ], dtype=float)
        
        containment.update_leader_states(positions)
        
        viz_data = containment.get_visualization_data()
        
        assert 'vertices' in viz_data
        assert 'edges' in viz_data
        assert len(viz_data['vertices']) > 0


class TestFormationGeometry:
    """Tests for predefined formation geometries."""
    
    def test_square_formation(self):
        """Test square formation generation."""
        positions = FormationGeometry.square(scale=1.0, height=1.0)
        
        assert positions.shape == (4, 3)
        assert np.all(positions[:, 2] == 1.0)
    
    def test_triangle_formation(self):
        """Test triangle formation generation."""
        positions = FormationGeometry.triangle(scale=1.0, height=1.0)
        
        assert positions.shape == (3, 3)
        
        # Vertices should be equidistant
        d01 = np.linalg.norm(positions[0] - positions[1])
        d12 = np.linalg.norm(positions[1] - positions[2])
        d20 = np.linalg.norm(positions[2] - positions[0])
        
        np.testing.assert_almost_equal(d01, d12, decimal=5)
        np.testing.assert_almost_equal(d12, d20, decimal=5)
    
    def test_tetrahedron_formation(self):
        """Test tetrahedron formation generation."""
        positions = FormationGeometry.tetrahedron(scale=1.0, center_height=1.0)
        
        assert positions.shape == (4, 3)
    
    def test_line_formation(self):
        """Test line formation generation."""
        positions = FormationGeometry.line(n_leaders=5, spacing=1.0, height=1.0)
        
        assert positions.shape == (5, 3)
        assert np.all(positions[:, 2] == 1.0)
        assert np.all(positions[:, 1] == 0.0)
    
    def test_circle_formation(self):
        """Test circle formation generation."""
        positions = FormationGeometry.circle(n_leaders=8, radius=2.0, height=1.0)
        
        assert positions.shape == (8, 3)
        
        # All points should be at radius from center
        for p in positions:
            dist = np.sqrt(p[0]**2 + p[1]**2)
            np.testing.assert_almost_equal(dist, 2.0, decimal=5)
    
    def test_scale_parameter(self):
        """Test scale parameter effect."""
        small = FormationGeometry.square(scale=1.0)
        large = FormationGeometry.square(scale=2.0)
        
        # Large formation should be 2x the size
        small_dist = np.linalg.norm(small[0, :2] - small[1, :2])
        large_dist = np.linalg.norm(large[0, :2] - large[1, :2])
        
        np.testing.assert_almost_equal(large_dist, 2 * small_dist, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

