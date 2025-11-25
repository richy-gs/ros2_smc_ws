"""
Mathematical Utilities for Formation Control

Provides helper functions for:
- Hadamard (element-wise) product operations (Section 2.2)
- Diagonal matrix operations
- Angle normalization
- Rotation matrices
- Quaternion/Euler conversions
"""

import numpy as np
from typing import Tuple


def hadamard_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute Hadamard (element-wise) product of two arrays.
    
    From Section 2.2 of the paper:
    (A ⊙ B)_ij = (A)_ij * (B)_ij
    
    For vectors: x ⊙ y = D_x @ y = D_y @ x
    where D_x is the diagonal matrix with x on the diagonal.
    
    Args:
        a: First array
        b: Second array (same shape as a)
        
    Returns:
        Element-wise product
    """
    return a * b


def diagonal_matrix(x: np.ndarray) -> np.ndarray:
    """
    Create diagonal matrix D_x from vector x.
    
    From Section 2.2: Any vector x can be expressed as its
    corresponding diagonal matrix D_x that has x as the main diagonal.
    
    Args:
        x: Vector to convert
        
    Returns:
        Diagonal matrix with x on diagonal
    """
    return np.diag(x)


def diagonal_sqrt(x: np.ndarray) -> np.ndarray:
    """
    Compute element-wise square root as diagonal matrix.
    
    D_x^{1/2} from Section 2.2.
    
    Args:
        x: Input vector
        
    Returns:
        Diagonal matrix with sqrt(x) on diagonal
    """
    return np.diag(np.sqrt(np.abs(x)))


def diagonal_abs(x: np.ndarray) -> np.ndarray:
    """
    Compute absolute value diagonal matrix D_{|x|}.
    
    From Section 2.2.
    
    Args:
        x: Input vector
        
    Returns:
        Diagonal matrix with |x| on diagonal
    """
    return np.diag(np.abs(x))


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-π, π].
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-π, π]
    """
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def normalize_angles(angles: np.ndarray) -> np.ndarray:
    """
    Normalize array of angles to [-π, π].
    
    Args:
        angles: Array of angles in radians
        
    Returns:
        Normalized angles
    """
    return np.mod(angles + np.pi, 2 * np.pi) - np.pi


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    Create 2D rotation matrix.
    
    R = [cos(θ)  -sin(θ)]
        [sin(θ)   cos(θ)]
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        2x2 rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s],
        [s,  c]
    ])


def rotation_matrix_3d(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create 3D rotation matrix from Euler angles (ZYX convention).
    
    R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    
    This transforms from body frame to inertial frame.
    
    Args:
        roll: Roll angle (φ) in radians
        pitch: Pitch angle (θ) in radians
        yaw: Yaw angle (ψ) in radians
        
    Returns:
        3x3 rotation matrix
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    
    return R


def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (ZYX convention).
    
    Args:
        q: Quaternion [x, y, z, w] or [w, x, y, z]
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Handle both conventions
    if len(q) == 4:
        # Assume [x, y, z, w] convention (ROS style)
        qx, qy, qz, qw = q
    else:
        raise ValueError("Quaternion must have 4 elements")
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll: Roll angle (φ) in radians
        pitch: Pitch angle (θ) in radians
        yaw: Yaw angle (ψ) in radians
        
    Returns:
        Quaternion [x, y, z, w]
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return np.array([qx, qy, qz, qw])


def sign_with_zero_handling(x: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute sign function with zero handling.
    
    sign(x) = 1 if x > 0, -1 if x < 0, 0 if x ≈ 0
    
    Args:
        x: Input array
        epsilon: Threshold for zero
        
    Returns:
        Sign of each element
    """
    result = np.sign(x)
    result[np.abs(x) < epsilon] = 0
    return result


def smooth_sign(x: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """
    Smooth approximation of sign function to reduce chattering.
    
    smooth_sign(x) = x / (|x| + ε)
    
    This is useful for reducing chattering in sliding mode control.
    
    Args:
        x: Input array
        epsilon: Smoothing parameter
        
    Returns:
        Smooth sign approximation
    """
    return x / (np.abs(x) + epsilon)


def saturate(x: np.ndarray, limit: float) -> np.ndarray:
    """
    Saturate values to [-limit, limit].
    
    Args:
        x: Input array
        limit: Saturation limit
        
    Returns:
        Saturated values
    """
    return np.clip(x, -limit, limit)


def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute Kronecker product A ⊗ B.
    
    Used in the paper for extending matrices to multiple states:
    (L ⊗ I_s) where I_s is identity of state dimension.
    
    Args:
        A: First matrix
        B: Second matrix
        
    Returns:
        Kronecker product
    """
    return np.kron(A, B)


def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix between positions.
    
    Args:
        positions: Array of positions, shape (n, dim)
        
    Returns:
        Distance matrix, shape (n, n)
    """
    n = len(positions)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def compute_min_distance(positions: np.ndarray) -> Tuple[float, int, int]:
    """
    Find minimum distance between any two positions.
    
    Args:
        positions: Array of positions, shape (n, dim)
        
    Returns:
        Tuple of (min_distance, agent_i, agent_j)
    """
    dist_matrix = compute_distance_matrix(positions)
    np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances
    
    min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    min_dist = dist_matrix[min_idx]
    
    return min_dist, min_idx[0], min_idx[1]


def interpolate_trajectory(start: np.ndarray, end: np.ndarray,
                           t: float) -> np.ndarray:
    """
    Linear interpolation between two points.
    
    Args:
        start: Starting position
        end: Ending position
        t: Interpolation parameter [0, 1]
        
    Returns:
        Interpolated position
    """
    t = np.clip(t, 0, 1)
    return start + t * (end - start)


def smooth_trajectory_point(waypoints: np.ndarray, 
                            t: float,
                            smoothing: float = 0.1) -> np.ndarray:
    """
    Get smooth trajectory point using cubic interpolation.
    
    Args:
        waypoints: Array of waypoints, shape (n_waypoints, dim)
        t: Normalized time parameter [0, n_waypoints-1]
        smoothing: Smoothing factor
        
    Returns:
        Position on smooth trajectory
    """
    n = len(waypoints)
    t = np.clip(t, 0, n - 1)
    
    idx = int(t)
    frac = t - idx
    
    if idx >= n - 1:
        return waypoints[-1]
    
    # Simple linear for now (can extend to cubic spline)
    return interpolate_trajectory(waypoints[idx], waypoints[idx + 1], frac)


# Example usage
if __name__ == "__main__":
    print("Math Utilities Test")
    print("=" * 50)
    
    # Test Hadamard product
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(f"Hadamard product {a} ⊙ {b} = {hadamard_product(a, b)}")
    
    # Test diagonal matrix
    print(f"Diagonal matrix of {a}:\n{diagonal_matrix(a)}")
    
    # Test rotation
    theta = np.pi / 4
    R = rotation_matrix_2d(theta)
    print(f"2D rotation by π/4:\n{R}")
    
    # Test quaternion conversion
    q = euler_to_quaternion(0, 0, np.pi/2)
    print(f"Quaternion for yaw=π/2: {q}")
    roll, pitch, yaw = quaternion_to_euler(q)
    print(f"Back to Euler: roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")
    
    # Test distance computation
    positions = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ])
    dist_mat = compute_distance_matrix(positions)
    print(f"Distance matrix:\n{dist_mat}")
    
    min_dist, i, j = compute_min_distance(positions)
    print(f"Minimum distance: {min_dist:.4f} between agents {i} and {j}")

