"""
Formation-Containment Control Package

Implementation of:
"Collision-Free Formation-Containment Control Based on Adaptive Sliding Mode Strategy
for a Quadrotor Fleet Under Disturbances"
by Carlos Katt and Herman Castañeda (2025)

This package provides:
- Graph theory utilities for multi-agent interaction networks
- Laplacian matrix computation and partitioning
- Single-Gain Adaptive Sliding Mode Control (SGASMC)
- Convex hull containment computation
- Collision avoidance strategies
- Integration with Crazyswarm2 framework
"""

from .core.graph_theory import (
    GraphTopology,
    AdjacencyMatrix,
    LaplacianMatrix,
    InteractionNetwork
)

from .core.dynamics import (
    QuadrotorDynamics,
    ReducedTrackingModel
)

from .core.convex_hull import (
    ConvexHullContainment,
    compute_convex_hull,
    point_in_convex_hull
)

from .controllers.sgasmc import (
    SGASMCController,
    SlidingSurface
)

from .controllers.formation_controller import (
    FormationController,
    LeaderController,
    FollowerController
)

from .utils.math_utils import (
    hadamard_product,
    diagonal_matrix,
    normalize_angle,
    rotation_matrix_2d
)

__version__ = "1.0.0"
__author__ = "Carlos Katt, Herman Castañeda"

