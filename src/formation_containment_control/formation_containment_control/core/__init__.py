"""
Core modules for formation-containment control.
"""

from .graph_theory import GraphTopology, AdjacencyMatrix, LaplacianMatrix, InteractionNetwork
from .dynamics import QuadrotorDynamics, ReducedTrackingModel
from .convex_hull import ConvexHullContainment, compute_convex_hull, point_in_convex_hull

