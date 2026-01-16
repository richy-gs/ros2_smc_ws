"""
Graph Theory Module for Multi-Agent Systems

Implements graph-based interaction networks as described in Section 2.1 of:
"Collision-Free Formation-Containment Control Based on Adaptive Sliding Mode Strategy
for a Quadrotor Fleet Under Disturbances"

Mathematical Background:
-----------------------
- Graph G(V, ε): Directed graph representing information flow
- V = {v1, ..., v_{n+m}}: Set of nodes (n followers, m leaders)
- ε(i,j) ⊆ V × V: Edge set for interactions

Key Matrices:
- Adjacency Matrix A = [a_ij]: a_ij = 1 if edge exists from v_j to v_i
- Degree Matrix D = diag(d_1, ..., d_{n+m}): d_ii = Σ a_ij
- Laplacian Matrix L = D - A

Laplacian Partitioning (Equation from paper):
    L = [L_N   L_M  ]
        [0     0    ]
where:
- L_N ∈ ℜ^{n×n}: Follower-to-follower interactions
- L_M ∈ ℜ^{n×m}: Leader-to-follower interactions

Key Properties (Assumption 1):
- All eigenvalues of L_N are positive (positive-definite Hermitian matrix)
- For -L_N^{-1} L_M: row sums equal 1, all elements non-negative
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class AdjacencyMatrix:
    """
    Adjacency matrix representation for the interaction graph.
    
    The adjacency matrix A = [a_ij] where:
    - a_ij = 1 if there is an edge from v_j to v_i (j sends info to i)
    - a_ij = 0 otherwise
    - a_ii = 0 (no self-loops)
    
    Attributes:
        matrix: The adjacency matrix as numpy array
        n_followers: Number of follower agents
        n_leaders: Number of leader agents
    """
    matrix: np.ndarray
    n_followers: int
    n_leaders: int
    
    @property
    def n_agents(self) -> int:
        """Total number of agents (followers + leaders)"""
        return self.n_followers + self.n_leaders
    
    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get list of neighbors that agent_id receives information from."""
        return list(np.where(self.matrix[agent_id, :] > 0)[0])
    
    def get_edge_weight(self, i: int, j: int) -> float:
        """Get edge weight from agent j to agent i."""
        return self.matrix[i, j]
    
    def is_connected_from(self, i: int, j: int) -> bool:
        """Check if agent i receives information from agent j."""
        return self.matrix[i, j] > 0


@dataclass 
class LaplacianMatrix:
    """
    Laplacian matrix and its partitions for formation-containment control.
    
    The Laplacian L = D - A is partitioned as:
        L = [L_N   L_M  ]
            [0     0    ]
    
    where:
    - L_N ∈ ℜ^{n×n}: Follower-to-follower interactions
    - L_M ∈ ℜ^{n×m}: Leader-to-follower interactions
    
    The matrix L_ζ used in containment control (Equation 22):
        L_ζ = [0            -L_N^{-1} L_M]
              [0            I_m          ]
    
    Attributes:
        L: Full Laplacian matrix
        L_N: Follower-follower submatrix
        L_M: Leader-follower submatrix
        L_N_inv: Inverse of L_N
        L_zeta: Combined matrix for containment error computation
        n_followers: Number of followers
        n_leaders: Number of leaders
    """
    L: np.ndarray
    L_N: np.ndarray
    L_M: np.ndarray
    L_N_inv: np.ndarray
    L_zeta: np.ndarray
    n_followers: int
    n_leaders: int
    containment_weights: np.ndarray = field(init=False)
    
    def __post_init__(self):
        """Compute containment weights: -L_N^{-1} L_M"""
        # Equation: -L_N^{-1} L_M gives the convex combination weights
        # Each row sums to 1, all elements non-negative (under Assumption 1)
        self.containment_weights = -self.L_N_inv @ self.L_M
    
    def get_desired_follower_position(self, follower_id: int, 
                                      leader_positions: np.ndarray) -> np.ndarray:
        """
        Compute desired position for a follower based on leader positions.
        
        From Equation 16:
        ξ_dc,i = Σ_{j=1}^m [-L_N^{-1} L_M]_ij * ξ_{n+j}
        
        Args:
            follower_id: Index of follower (0 to n-1)
            leader_positions: Array of leader positions, shape (m, state_dim)
            
        Returns:
            Desired position for the follower
        """
        weights = self.containment_weights[follower_id, :]
        return weights @ leader_positions
    
    def get_all_desired_follower_positions(self, 
                                           leader_positions: np.ndarray) -> np.ndarray:
        """
        Compute desired positions for all followers.
        
        Args:
            leader_positions: Array of leader positions, shape (m, state_dim)
            
        Returns:
            Array of desired follower positions, shape (n, state_dim)
        """
        return self.containment_weights @ leader_positions
    
    def verify_assumption_1(self) -> Tuple[bool, str]:
        """
        Verify Assumption 1 from the paper:
        1. All eigenvalues of L_N are positive
        2. For -L_N^{-1} L_M: row sums = 1, all elements >= 0
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Check L_N eigenvalues
        eigenvalues = np.linalg.eigvals(self.L_N)
        if not np.all(np.real(eigenvalues) > 0):
            return False, f"L_N has non-positive eigenvalues: {eigenvalues}"
        
        # Check containment weights
        row_sums = np.sum(self.containment_weights, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            return False, f"Row sums of -L_N^{-1}L_M not equal to 1: {row_sums}"
        
        if np.any(self.containment_weights < -1e-10):
            return False, "Negative elements in -L_N^{-1}L_M"
        
        return True, "Assumption 1 satisfied"


class GraphTopology:
    """
    Factory class for creating common graph topologies for formation-containment.
    
    Supported topologies:
    - Complete graph (all-to-all)
    - Ring graph (circular)
    - Star graph (central leader)
    - Custom (user-defined)
    """
    
    @staticmethod
    def complete_graph(n_followers: int, n_leaders: int) -> AdjacencyMatrix:
        """
        Create a complete graph where every follower receives from all other agents
        and leaders don't receive from anyone.
        
        Args:
            n_followers: Number of follower agents
            n_leaders: Number of leader agents
            
        Returns:
            AdjacencyMatrix for complete graph
        """
        n = n_followers + n_leaders
        A = np.ones((n, n)) - np.eye(n)
        
        # Leaders don't receive information (last m rows are zero)
        A[n_followers:, :] = 0
        
        return AdjacencyMatrix(matrix=A, n_followers=n_followers, n_leaders=n_leaders)
    
    @staticmethod
    def ring_graph(n_followers: int, n_leaders: int, 
                   bidirectional: bool = True) -> AdjacencyMatrix:
        """
        Create a ring graph where followers form a ring and leaders connect to nearest followers.
        
        Args:
            n_followers: Number of follower agents
            n_leaders: Number of leader agents
            bidirectional: If True, edges go both ways between followers
            
        Returns:
            AdjacencyMatrix for ring graph
        """
        n = n_followers + n_leaders
        A = np.zeros((n, n))
        
        # Follower ring
        for i in range(n_followers):
            A[i, (i + 1) % n_followers] = 1
            if bidirectional:
                A[i, (i - 1) % n_followers] = 1
        
        # Connect each leader to followers (evenly distributed)
        for j in range(n_leaders):
            # Connect to nearest followers
            follower_idx = (j * n_followers // n_leaders) % n_followers
            A[follower_idx, n_followers + j] = 1
            if n_followers > 1:
                A[(follower_idx + 1) % n_followers, n_followers + j] = 1
        
        # Leaders don't receive
        A[n_followers:, :] = 0
        
        return AdjacencyMatrix(matrix=A, n_followers=n_followers, n_leaders=n_leaders)
    
    @staticmethod
    def paper_topology(n_followers: int = 4, n_leaders: int = 4) -> AdjacencyMatrix:
        """
        Create the topology used in the paper (Figure 1).
        
        The paper uses 4 followers (F1-F4) and 4 leaders (L1-L4) with specific
        connectivity patterns where:
        - Followers receive from neighboring followers and connected leaders
        - Leaders don't receive information
        
        Args:
            n_followers: Number of followers (default 4)
            n_leaders: Number of leaders (default 4)
            
        Returns:
            AdjacencyMatrix matching paper topology
        """
        n = n_followers + n_leaders
        A = np.zeros((n, n))
        
        # Paper topology (from Figure 1):
        # F1 receives from F2, F4, L1, L2
        # F2 receives from F1, F3, L2, L3
        # F3 receives from F2, F4, L3, L4
        # F4 receives from F1, F3, L1, L4
        
        if n_followers == 4 and n_leaders == 4:
            # Follower-follower connections (bidirectional ring)
            A[0, 1] = 1; A[0, 3] = 1  # F1 <- F2, F4
            A[1, 0] = 1; A[1, 2] = 1  # F2 <- F1, F3
            A[2, 1] = 1; A[2, 3] = 1  # F3 <- F2, F4
            A[3, 0] = 1; A[3, 2] = 1  # F4 <- F1, F3
            
            # Leader-follower connections
            A[0, 4] = 1; A[0, 5] = 1  # F1 <- L1, L2
            A[1, 5] = 1; A[1, 6] = 1  # F2 <- L2, L3
            A[2, 6] = 1; A[2, 7] = 1  # F3 <- L3, L4
            A[3, 4] = 1; A[3, 7] = 1  # F4 <- L1, L4
        else:
            # Generic topology for other configurations
            return GraphTopology.ring_graph(n_followers, n_leaders)
        
        return AdjacencyMatrix(matrix=A, n_followers=n_followers, n_leaders=n_leaders)
    
    @staticmethod
    def custom_graph(adjacency_matrix: np.ndarray, 
                     n_followers: int, n_leaders: int) -> AdjacencyMatrix:
        """
        Create a custom graph from user-provided adjacency matrix.
        
        Args:
            adjacency_matrix: Custom adjacency matrix
            n_followers: Number of followers
            n_leaders: Number of leaders
            
        Returns:
            AdjacencyMatrix wrapping the custom matrix
        """
        assert adjacency_matrix.shape[0] == n_followers + n_leaders
        assert adjacency_matrix.shape[1] == n_followers + n_leaders
        
        return AdjacencyMatrix(
            matrix=adjacency_matrix.copy(),
            n_followers=n_followers,
            n_leaders=n_leaders
        )


class InteractionNetwork:
    """
    Complete interaction network for formation-containment control.
    
    This class combines graph topology with Laplacian matrix computation
    and provides all necessary matrices for the control laws.
    
    Mathematical formulation follows Section 2.1 of the paper.
    """
    
    def __init__(self, adjacency: AdjacencyMatrix):
        """
        Initialize interaction network from adjacency matrix.
        
        Args:
            adjacency: AdjacencyMatrix defining the network topology
        """
        self.adjacency = adjacency
        self.n_followers = adjacency.n_followers
        self.n_leaders = adjacency.n_leaders
        self.n_agents = adjacency.n_agents
        
        # Compute all matrices
        self._compute_matrices()
    
    def _compute_matrices(self):
        """Compute degree matrix, Laplacian, and all partitions."""
        A = self.adjacency.matrix
        n = self.n_followers
        m = self.n_leaders
        
        # Degree matrix D = diag(d_1, ..., d_{n+m}) where d_ii = Σ_j a_ij
        self.degree_matrix = np.diag(np.sum(A, axis=1))
        
        # Laplacian L = D - A
        self.L = self.degree_matrix - A
        
        # Partition Laplacian for followers
        # L_N: Follower-follower interactions (top-left n×n block)
        self.L_N = self.L[:n, :n]
        
        # L_M: Leader-follower interactions (top-right n×m block)
        self.L_M = self.L[:n, n:n+m]
        
        # Compute inverse of L_N (needed for containment control)
        try:
            self.L_N_inv = np.linalg.inv(self.L_N)
        except np.linalg.LinAlgError:
            raise ValueError(
                "L_N is singular. Ensure graph satisfies Assumption 1: "
                "every follower must have a directed path from at least one leader."
            )
        
        # Construct L_ζ matrix (Equation 22)
        # L_ζ = [0        -L_N^{-1} L_M]
        #       [0        I_m          ]
        self.L_zeta = np.zeros((n + m, n + m))
        self.L_zeta[:n, n:] = -self.L_N_inv @ self.L_M
        self.L_zeta[n:, n:] = np.eye(m)
        
        # Create LaplacianMatrix object
        self.laplacian = LaplacianMatrix(
            L=self.L,
            L_N=self.L_N,
            L_M=self.L_M,
            L_N_inv=self.L_N_inv,
            L_zeta=self.L_zeta,
            n_followers=n,
            n_leaders=m
        )
    
    def compute_containment_error_matrix(self) -> np.ndarray:
        """
        Compute the containment error matrix from Equation 24.
        
        The anti-collision containment error uses:
        (L_ζ - I_{n+m} - L) ⊗ I_s
        
        This is a constant matrix under fixed topology.
        
        Returns:
            Containment error transformation matrix
        """
        return self.L_zeta - np.eye(self.n_agents) - self.L
    
    def get_collision_avoidance_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get matrices needed for collision avoidance error computation (Equation 23).
        
        The collision avoidance term h_{c,i} uses:
        h_c = (D ⊗ I_s)(L_ζ ⊗ I_s)ξ - (A ⊗ I_s)(L_ζ ⊗ I_s)ξ - (L ⊗ I_s)ξ
        
        Returns:
            Tuple of (D @ L_zeta - A @ L_zeta - L, L_zeta) matrices
        """
        D = self.degree_matrix
        A = self.adjacency.matrix
        
        collision_matrix = D @ self.L_zeta - A @ self.L_zeta - self.L
        return collision_matrix, self.L_zeta
    
    def verify_connectivity(self) -> Tuple[bool, str]:
        """
        Verify that the network satisfies Assumption 1:
        For any follower, there exists a directed path from at least one leader.
        
        Returns:
            Tuple of (is_valid, message)
        """
        return self.laplacian.verify_assumption_1()
    
    def print_summary(self):
        """Print summary of the interaction network."""
        print("=" * 60)
        print("Interaction Network Summary")
        print("=" * 60)
        print(f"Number of followers: {self.n_followers}")
        print(f"Number of leaders: {self.n_leaders}")
        print(f"Total agents: {self.n_agents}")
        print()
        print("Adjacency Matrix A:")
        print(self.adjacency.matrix)
        print()
        print("Degree Matrix D:")
        print(np.diag(self.degree_matrix))
        print()
        print("Laplacian Matrix L:")
        print(self.L)
        print()
        print("L_N (Follower-Follower):")
        print(self.L_N)
        print()
        print("L_M (Leader-Follower):")
        print(self.L_M)
        print()
        print("Containment weights (-L_N^{-1} L_M):")
        print(self.laplacian.containment_weights)
        print()
        
        valid, msg = self.verify_connectivity()
        print(f"Assumption 1 verification: {msg}")
        print("=" * 60)


def create_interaction_network(n_followers: int, n_leaders: int,
                               topology: str = "paper") -> InteractionNetwork:
    """
    Factory function to create an interaction network.
    
    Args:
        n_followers: Number of follower agents
        n_leaders: Number of leader agents
        topology: Type of topology ("complete", "ring", "paper", or "custom")
        
    Returns:
        InteractionNetwork instance
    """
    if topology == "complete":
        adj = GraphTopology.complete_graph(n_followers, n_leaders)
    elif topology == "ring":
        adj = GraphTopology.ring_graph(n_followers, n_leaders)
    elif topology == "paper":
        adj = GraphTopology.paper_topology(n_followers, n_leaders)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    return InteractionNetwork(adj)


# Example usage and testing
if __name__ == "__main__":
    # Create network with paper topology (4 followers, 4 leaders)
    network = create_interaction_network(4, 4, "paper")
    network.print_summary()
    
    # Test containment computation
    print("\nTesting containment computation:")
    # Example leader positions (each leader at a corner of a square)
    leader_positions = np.array([
        [1.0, 0.0, 1.0],   # L1
        [-1.0, 0.0, 1.0],  # L2
        [0.0, 1.0, 1.0],   # L3
        [0.0, -1.0, 1.0],  # L4
    ])
    
    desired_follower_pos = network.laplacian.get_all_desired_follower_positions(
        leader_positions
    )
    print("Desired follower positions (inside convex hull):")
    print(desired_follower_pos)

