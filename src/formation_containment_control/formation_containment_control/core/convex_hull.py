"""
Convex Hull Module for Formation Containment

Implements convex hull computation and containment verification as described
in the paper (Equations 4-5).

The containment control ensures followers remain within the convex hull
defined by the leaders:

Co(χ_h) = {Σ_{j=n+1}^{n+m} a_j χ_j : a_j ≥ 0, Σ a_j = 1}    (Position hull)
Co(χ̇_h) = {Σ_{j=n+1}^{n+m} b_j χ̇_j : b_j ≥ 0, Σ b_j = 1}    (Velocity hull)

The convex hull represents all possible linear combinations of leader states
with non-negative coefficients that sum to 1.

Key Features:
- 2D and 3D convex hull computation
- Point-in-hull testing
- Visualization support
- Dynamic hull tracking
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.spatial import ConvexHull, Delaunay


@dataclass
class ConvexHullResult:
    """
    Result of convex hull computation.
    
    Attributes:
        vertices: Array of hull vertex positions, shape (n_vertices, dim)
        vertex_indices: Indices of vertices in original point set
        simplices: Indices of points forming the hull facets
        volume: Volume (3D) or area (2D) of the hull
        centroid: Centroid of the hull
        is_valid: Whether hull computation was successful
    """
    vertices: np.ndarray
    vertex_indices: np.ndarray
    simplices: np.ndarray
    volume: float
    centroid: np.ndarray
    is_valid: bool
    error_message: str = ""


def compute_convex_hull(points: np.ndarray, 
                        dimension: int = 3) -> ConvexHullResult:
    """
    Compute the convex hull of a set of points.
    
    This implements the convex hull computation for equation (4):
    Co(χ_h) = {Σ a_j χ_j : a_j ≥ 0, Σ a_j = 1}
    
    Args:
        points: Array of points, shape (n_points, dim)
        dimension: 2 or 3 for 2D/3D hull computation
        
    Returns:
        ConvexHullResult with hull information
    """
    if len(points) < dimension + 1:
        return ConvexHullResult(
            vertices=points,
            vertex_indices=np.arange(len(points)),
            simplices=np.array([]),
            volume=0.0,
            centroid=np.mean(points, axis=0),
            is_valid=False,
            error_message=f"Need at least {dimension+1} points for {dimension}D hull"
        )
    
    # Extract relevant dimensions
    if dimension == 2:
        hull_points = points[:, :2]
    else:
        hull_points = points[:, :3]
    
    try:
        # Handle collinear/coplanar points
        if dimension == 3:
            # Check if points are coplanar
            if len(points) >= 4:
                # Check coplanarity using SVD
                centered = hull_points - np.mean(hull_points, axis=0)
                _, s, _ = np.linalg.svd(centered)
                if s[-1] / s[0] < 1e-10:
                    # Points are coplanar, use 2D hull
                    dimension = 2
                    hull_points = points[:, :2]
        
        hull = ConvexHull(hull_points)
        
        vertices = hull_points[hull.vertices]
        centroid = np.mean(vertices, axis=0)
        
        # Pad to 3D if needed
        if dimension == 2 and points.shape[1] >= 3:
            vertices_3d = np.zeros((len(vertices), 3))
            vertices_3d[:, :2] = vertices
            vertices_3d[:, 2] = np.mean(points[:, 2])
            centroid_3d = np.zeros(3)
            centroid_3d[:2] = centroid
            centroid_3d[2] = np.mean(points[:, 2])
            vertices = vertices_3d
            centroid = centroid_3d
        
        return ConvexHullResult(
            vertices=vertices,
            vertex_indices=hull.vertices,
            simplices=hull.simplices,
            volume=hull.volume if dimension == 3 else hull.area,
            centroid=centroid,
            is_valid=True
        )
        
    except Exception as e:
        return ConvexHullResult(
            vertices=points,
            vertex_indices=np.arange(len(points)),
            simplices=np.array([]),
            volume=0.0,
            centroid=np.mean(points, axis=0),
            is_valid=False,
            error_message=str(e)
        )


def point_in_convex_hull(point: np.ndarray, 
                         hull_points: np.ndarray,
                         tolerance: float = 1e-12) -> bool:
    """
    Check if a point is inside the convex hull of given points.
    
    This is used to verify containment condition from the paper.
    
    Args:
        point: Point to test, shape (dim,)
        hull_points: Points defining the hull, shape (n_points, dim)
        tolerance: Numerical tolerance for boundary cases
        
    Returns:
        True if point is inside or on boundary of hull
    """
    if len(hull_points) < 2:
        return False
    
    try:
        # Use Delaunay triangulation for point-in-hull test
        hull = Delaunay(hull_points)
        return hull.find_simplex(point) >= 0
    except Exception:
        # Fallback: check using linear programming concept
        return _point_in_hull_lp(point, hull_points, tolerance)


def _point_in_hull_lp(point: np.ndarray, 
                      hull_points: np.ndarray,
                      tolerance: float) -> bool:
    """
    Check point in hull using linear combination test.
    
    A point p is in the convex hull of points {p1,...,pn} if and only if
    there exist coefficients a1,...,an such that:
    - p = Σ ai * pi
    - ai >= 0 for all i
    - Σ ai = 1
    
    This is the definition from equation (4) in the paper.
    """
    n_points = len(hull_points)
    dim = len(point)
    
    if n_points < dim + 1:
        return False
    
    try:
        # Set up least squares: find coefficients a such that
        # hull_points.T @ a = point and sum(a) = 1
        A = np.vstack([hull_points.T, np.ones(n_points)])
        b = np.append(point, 1.0)
        
        # Solve using least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Check if solution is valid (all coeffs >= 0 and sum to 1)
        if np.all(coeffs >= -tolerance) and abs(np.sum(coeffs) - 1.0) < tolerance:
            # Verify reconstruction
            reconstructed = hull_points.T @ coeffs
            if np.linalg.norm(reconstructed - point) < tolerance:
                return True
        
        return False
    except Exception:
        return False


def compute_convex_combination_weights(point: np.ndarray,
                                       hull_points: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the convex combination weights for a point inside the hull.
    
    Find coefficients {a_j} such that:
    point = Σ a_j * hull_points[j]
    where a_j >= 0 and Σ a_j = 1
    
    This relates to equation (4) in the paper.
    
    Args:
        point: Point inside the hull
        hull_points: Vertices of the convex hull
        
    Returns:
        Array of weights if point is inside hull, None otherwise
    """
    n_points = len(hull_points)
    
    if n_points < 2:
        return None
    
    try:
        # Set up constrained least squares
        A = np.vstack([hull_points.T, np.ones(n_points)])
        b = np.append(point, 1.0)
        
        # Solve
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Project to valid simplex (ensure non-negative and sum to 1)
        coeffs = np.maximum(coeffs, 0)
        coeffs = coeffs / np.sum(coeffs)
        
        return coeffs
    except Exception:
        return None


class ConvexHullContainment:
    """
    Convex hull containment manager for formation control.
    
    Manages the convex hull formed by leader positions and provides
    methods for:
    - Computing desired follower positions inside the hull
    - Verifying containment of followers
    - Dynamic hull updates as leaders move
    - Visualization support
    
    Mathematical basis from paper:
    - Equation (4): Position convex hull Co(χ_h)
    - Equation (5): Velocity convex hull Co(χ̇_h)
    - Equation (16): Desired follower positions using Laplacian weights
    """
    
    def __init__(self, n_leaders: int, dimension: int = 3):
        """
        Initialize containment manager.
        
        Args:
            n_leaders: Number of leader agents forming the hull
            dimension: 2 or 3 for dimensionality
        """
        self.n_leaders = n_leaders
        self.dimension = dimension
        self.leader_positions = np.zeros((n_leaders, dimension))
        self.leader_velocities = np.zeros((n_leaders, dimension))
        self.hull_result: Optional[ConvexHullResult] = None
        self.velocity_hull_result: Optional[ConvexHullResult] = None
    
    def update_leader_states(self, positions: np.ndarray, 
                            velocities: Optional[np.ndarray] = None):
        """
        Update leader positions and velocities.
        
        Args:
            positions: Leader positions, shape (n_leaders, dim)
            velocities: Leader velocities (optional), shape (n_leaders, dim)
        """
        self.leader_positions = positions[:, :self.dimension].copy()
        self.hull_result = compute_convex_hull(self.leader_positions, self.dimension)
        
        if velocities is not None:
            self.leader_velocities = velocities[:, :self.dimension].copy()
            self.velocity_hull_result = compute_convex_hull(
                self.leader_velocities, self.dimension
            )
    
    def get_desired_position(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute desired position inside hull using convex combination weights.
        
        This implements equation (4):
        χ_d = Σ a_j χ_j where a_j >= 0 and Σ a_j = 1
        
        Args:
            weights: Convex combination weights from Laplacian (-L_N^{-1} L_M)
            
        Returns:
            Desired position inside the convex hull
        """
        return weights @ self.leader_positions
    
    def get_desired_velocity(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute desired velocity inside velocity hull.
        
        This implements equation (5):
        χ̇_d = Σ b_j χ̇_j where b_j >= 0 and Σ b_j = 1
        
        Args:
            weights: Convex combination weights
            
        Returns:
            Desired velocity inside the velocity hull
        """
        return weights @ self.leader_velocities
    
    def is_point_contained(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside the position convex hull.
        
        Args:
            point: Point to test
            
        Returns:
            True if point is contained in the hull
        """
        return point_in_convex_hull(point[:self.dimension], self.leader_positions)
    
    def get_containment_violation(self, point: np.ndarray) -> float:
        """
        Compute how far a point is from being contained.
        
        Returns 0 if point is inside hull, positive distance otherwise.
        
        Args:
            point: Point to test
            
        Returns:
            Distance from hull (0 if inside)
        """
        if self.is_point_contained(point):
            return 0.0
        
        # Find closest point on hull
        if self.hull_result is None or not self.hull_result.is_valid:
            return np.linalg.norm(point - self.hull_result.centroid)
        
        # Project onto hull surface (simplified)
        centroid = self.hull_result.centroid
        direction = point[:self.dimension] - centroid[:self.dimension]
        
        # Find intersection with hull boundary
        min_dist = float('inf')
        for vertex in self.hull_result.vertices:
            dist = np.linalg.norm(point[:self.dimension] - vertex[:self.dimension])
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def get_hull_centroid(self) -> np.ndarray:
        """Get centroid of the convex hull."""
        if self.hull_result is not None:
            return self.hull_result.centroid
        return np.mean(self.leader_positions, axis=0)
    
    def get_hull_volume(self) -> float:
        """Get volume (3D) or area (2D) of the hull."""
        if self.hull_result is not None:
            return self.hull_result.volume
        return 0.0
    
    def get_visualization_data(self) -> dict:
        """
        Get data for visualizing the convex hull.
        
        Returns:
            Dictionary with vertices, edges, and faces for visualization
        """
        if self.hull_result is None or not self.hull_result.is_valid:
            return {
                'vertices': self.leader_positions,
                'edges': [],
                'faces': []
            }
        
        vertices = self.hull_result.vertices
        simplices = self.hull_result.simplices
        
        # Extract edges from simplices
        edges = set()
        for simplex in simplices:
            for i in range(len(simplex)):
                edge = tuple(sorted([simplex[i], simplex[(i+1) % len(simplex)]]))
                edges.add(edge)
        
        return {
            'vertices': vertices,
            'edges': list(edges),
            'faces': simplices.tolist() if len(simplices) > 0 else [],
            'centroid': self.hull_result.centroid
        }


class FormationGeometry:
    """
    Predefined formation geometries for leader arrangements.
    
    Provides common formation patterns used in multi-agent systems.
    """
    
    @staticmethod
    def square(scale: float = 1.0, height: float = 1.0) -> np.ndarray:
        """
        Square formation (4 leaders).
        
        L1---L2
        |     |
        L4---L3
        
        Args:
            scale: Size of the square
            height: Z coordinate
            
        Returns:
            Array of positions, shape (4, 3)
        """
        return np.array([
            [ scale,  0,      height],  # L1
            [-scale,  0,      height],  # L2
            [ 0,      scale,  height],  # L3
            [ 0,     -scale,  height],  # L4
        ])
    
    @staticmethod
    def triangle(scale: float = 1.0, height: float = 1.0) -> np.ndarray:
        """
        Equilateral triangle formation (3 leaders).
        
        Args:
            scale: Size of the triangle
            height: Z coordinate
            
        Returns:
            Array of positions, shape (3, 3)
        """
        angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
        positions = np.zeros((3, 3))
        positions[:, 0] = scale * np.cos(angles)
        positions[:, 1] = scale * np.sin(angles)
        positions[:, 2] = height
        return positions
    
    @staticmethod
    def tetrahedron(scale: float = 1.0, center_height: float = 1.0) -> np.ndarray:
        """
        Tetrahedron formation (4 leaders in 3D).
        
        From Section 4 of the paper - used after formation change at t=62.5s.
        
        Args:
            scale: Size of the tetrahedron
            center_height: Height of the centroid
            
        Returns:
            Array of positions, shape (4, 3)
        """
        # Regular tetrahedron vertices
        sqrt2 = np.sqrt(2)
        sqrt3 = np.sqrt(3)
        
        positions = np.array([
            [ 1,  0, -1/sqrt2],
            [-1,  0, -1/sqrt2],
            [ 0,  1,  1/sqrt2],
            [ 0, -1,  1/sqrt2],
        ]) * scale
        
        # Shift to center height
        positions[:, 2] += center_height
        
        return positions
    
    @staticmethod
    def line(n_leaders: int, spacing: float = 1.0, 
             height: float = 1.0) -> np.ndarray:
        """
        Line formation (variable number of leaders).
        
        Args:
            n_leaders: Number of leaders
            spacing: Distance between adjacent leaders
            height: Z coordinate
            
        Returns:
            Array of positions, shape (n_leaders, 3)
        """
        positions = np.zeros((n_leaders, 3))
        total_length = (n_leaders - 1) * spacing
        positions[:, 0] = np.linspace(-total_length/2, total_length/2, n_leaders)
        positions[:, 2] = height
        return positions
    
    @staticmethod
    def circle(n_leaders: int, radius: float = 1.0,
               height: float = 1.0) -> np.ndarray:
        """
        Circular formation.
        
        Args:
            n_leaders: Number of leaders
            radius: Radius of the circle
            height: Z coordinate
            
        Returns:
            Array of positions, shape (n_leaders, 3)
        """
        angles = np.linspace(0, 2*np.pi, n_leaders, endpoint=False)
        positions = np.zeros((n_leaders, 3))
        positions[:, 0] = radius * np.cos(angles)
        positions[:, 1] = radius * np.sin(angles)
        positions[:, 2] = height
        return positions


# Example usage
if __name__ == "__main__":
    # Create leader positions (square formation)
    leader_pos = FormationGeometry.square(scale=1.0, height=1.0)
    print("Leader positions (square formation):")
    print(leader_pos)
    
    # Create containment manager
    containment = ConvexHullContainment(n_leaders=4, dimension=3)
    containment.update_leader_states(leader_pos)
    
    print("\nConvex hull info:")
    print(f"  Volume: {containment.get_hull_volume():.4f}")
    print(f"  Centroid: {containment.get_hull_centroid()}")
    
    # Test point containment
    test_points = [
        np.array([0.0, 0.0, 1.0]),    # Inside
        np.array([0.5, 0.0, 1.0]),    # Inside
        np.array([2.0, 0.0, 1.0]),    # Outside
    ]
    
    print("\nContainment tests:")
    for p in test_points:
        contained = containment.is_point_contained(p)
        print(f"  Point {p}: {'Inside' if contained else 'Outside'}")
    
    # Test with Laplacian weights (example from paper)
    weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
    desired_pos = containment.get_desired_position(weights)
    print(f"\nDesired position with equal weights: {desired_pos}")

