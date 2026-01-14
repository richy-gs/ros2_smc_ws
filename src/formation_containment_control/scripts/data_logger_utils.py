#!/usr/bin/env python3
"""
Data Logger Utilities

Provides utilities for logging formation control data to CSV files
with incremental run directory management.
"""

import csv
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


class DataLogger:
    """
    Data logger for formation control system.
    
    Manages CSV file writing for states, velocities, accelerations,
    errors, and adaptive gains with incremental run directories.
    """
    
    def __init__(self, log_directory: str = "test_logs", run_name: Optional[str] = None):
        """
        Initialize data logger.
        
        Args:
            log_directory: Base directory for logs
            run_name: Optional run name (if None, uses incremental numbering)
        """
        self.base_dir = Path(log_directory)
        self.run_dir = None
        self.csv_writers = {}
        self.csv_files = {}
        
        # Create run directory
        if run_name:
            self.run_dir = self.base_dir / run_name
        else:
            run_number = get_next_run_number(self.base_dir)
            self.run_dir = self.base_dir / f"run_{run_number:03d}"
        
        create_run_directory(self.run_dir)
        
        # Initialize CSV files
        self._init_csv_files()
        
        print(f"Data logger initialized. Logging to: {self.run_dir}")
    
    def _init_csv_files(self):
        """Initialize all CSV files with headers."""
        # States CSV
        self.csv_files['states'] = open(self.run_dir / 'states.csv', 'w', newline='')
        self.csv_writers['states'] = csv.writer(self.csv_files['states'])
        self.csv_writers['states'].writerow([
            'timestamp', 'agent_id', 'agent_type', 'x', 'y', 'z', 'yaw'
        ])
        
        # Velocities CSV
        self.csv_files['velocities'] = open(self.run_dir / 'velocities.csv', 'w', newline='')
        self.csv_writers['velocities'] = csv.writer(self.csv_files['velocities'])
        self.csv_writers['velocities'].writerow([
            'timestamp', 'agent_id', 'agent_type', 'vx', 'vy', 'vz', 'vyaw'
        ])
        
        # Accelerations CSV
        self.csv_files['accelerations'] = open(self.run_dir / 'accelerations.csv', 'w', newline='')
        self.csv_writers['accelerations'] = csv.writer(self.csv_files['accelerations'])
        self.csv_writers['accelerations'].writerow([
            'timestamp', 'agent_id', 'agent_type', 'ax', 'ay', 'az', 'omega'
        ])
        
        # Errors CSV
        self.csv_files['errors'] = open(self.run_dir / 'errors.csv', 'w', newline='')
        self.csv_writers['errors'] = csv.writer(self.csv_files['errors'])
        self.csv_writers['errors'].writerow([
            'timestamp', 'agent_id', 'agent_type', 'error_x', 'error_y', 'error_z', 'error_yaw', 'error_norm'
        ])
        
        # Adaptive gains CSV
        self.csv_files['adaptive_gains'] = open(self.run_dir / 'adaptive_gains.csv', 'w', newline='')
        self.csv_writers['adaptive_gains'] = csv.writer(self.csv_files['adaptive_gains'])
        self.csv_writers['adaptive_gains'].writerow([
            'timestamp', 'agent_id', 'agent_type', 'Kc_x', 'Kc_y', 'Kc_z', 'Kc_yaw'
        ])
    
    def log_state(self, timestamp: float, agent_id: int, agent_type: str, state: np.ndarray):
        """
        Log agent state.
        
        Args:
            timestamp: Current time in seconds
            agent_id: Agent ID
            agent_type: 'leader' or 'follower'
            state: State vector [x, y, z, yaw]
        """
        self.csv_writers['states'].writerow([
            timestamp, agent_id, agent_type,
            float(state[0]), float(state[1]), float(state[2]), float(state[3])
        ])
    
    def log_velocity(self, timestamp: float, agent_id: int, agent_type: str, velocity: np.ndarray):
        """
        Log agent velocity.
        
        Args:
            timestamp: Current time in seconds
            agent_id: Agent ID
            agent_type: 'leader' or 'follower'
            velocity: Velocity vector [vx, vy, vz, vyaw]
        """
        self.csv_writers['velocities'].writerow([
            timestamp, agent_id, agent_type,
            float(velocity[0]), float(velocity[1]), float(velocity[2]), float(velocity[3])
        ])
    
    def log_acceleration(self, timestamp: float, agent_id: int, agent_type: str, acceleration: np.ndarray):
        """
        Log agent acceleration (control input).
        
        Args:
            timestamp: Current time in seconds
            agent_id: Agent ID
            agent_type: 'leader' or 'follower'
            acceleration: Acceleration vector [ax, ay, az, omega]
        """
        self.csv_writers['accelerations'].writerow([
            timestamp, agent_id, agent_type,
            float(acceleration[0]), float(acceleration[1]), float(acceleration[2]), float(acceleration[3])
        ])
    
    def log_error(self, timestamp: float, agent_id: int, agent_type: str, error: np.ndarray):
        """
        Log formation error.
        
        Args:
            timestamp: Current time in seconds
            agent_id: Agent ID
            agent_type: 'leader' or 'follower'
            error: Error vector [ex, ey, ez, eyaw] or scalar
        """
        if isinstance(error, (int, float)):
            # Scalar error - log as norm only
            self.csv_writers['errors'].writerow([
                timestamp, agent_id, agent_type, 0.0, 0.0, 0.0, 0.0, float(error)
            ])
        else:
            # Vector error
            error_norm = float(np.linalg.norm(error))
            self.csv_writers['errors'].writerow([
                timestamp, agent_id, agent_type,
                float(error[0]), float(error[1]), float(error[2]), 
                float(error[3]) if len(error) > 3 else 0.0,
                error_norm
            ])
    
    def log_adaptive_gain(self, timestamp: float, agent_id: int, agent_type: str, K_c: np.ndarray):
        """
        Log adaptive gain K_c.
        
        Args:
            timestamp: Current time in seconds
            agent_id: Agent ID
            agent_type: 'leader' or 'follower'
            K_c: Adaptive gain vector [Kc_x, Kc_y, Kc_z, Kc_yaw]
        """
        self.csv_writers['adaptive_gains'].writerow([
            timestamp, agent_id, agent_type,
            float(K_c[0]), float(K_c[1]), float(K_c[2]), float(K_c[3])
        ])
    
    def write_metadata(self, metadata: Dict):
        """
        Write metadata file with run configuration.
        
        Args:
            metadata: Dictionary with run configuration
        """
        write_metadata(self.run_dir, metadata)
    
    def flush(self):
        """Flush all CSV files to disk."""
        for f in self.csv_files.values():
            f.flush()
    
    def close(self):
        """Close all CSV files."""
        for f in self.csv_files.values():
            f.close()
        print(f"Data logger closed. Data saved to: {self.run_dir}")


def get_next_run_number(base_dir: Path) -> int:
    """
    Find the next available run number.
    
    Args:
        base_dir: Base directory for logs
        
    Returns:
        Next run number (1 if no runs exist)
    """
    if not base_dir.exists():
        return 1
    
    existing_runs = []
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith('run_'):
            try:
                run_num = int(d.name.split('_')[1])
                existing_runs.append(run_num)
            except (ValueError, IndexError):
                continue
    
    if not existing_runs:
        return 1
    
    return max(existing_runs) + 1


def create_run_directory(run_dir: Path):
    """
    Create run directory structure.
    
    Args:
        run_dir: Path to run directory
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created run directory: {run_dir}")


def write_metadata(run_dir: Path, metadata: Dict):
    """
    Write metadata YAML file.
    
    Args:
        run_dir: Path to run directory
        metadata: Dictionary with configuration data
    """
    metadata_file = run_dir / 'metadata.yaml'
    
    # Add timestamp
    metadata['timestamp'] = datetime.now().isoformat()
    
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    print(f"Metadata written to: {metadata_file}")


def load_run_data(run_dir: Path) -> Dict[str, any]:
    """
    Load all data from a run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dictionary with loaded data:
            - 'states': pandas DataFrame
            - 'velocities': pandas DataFrame
            - 'accelerations': pandas DataFrame
            - 'errors': pandas DataFrame
            - 'adaptive_gains': pandas DataFrame
            - 'metadata': dict
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for loading run data. Install with: pip install pandas")
    
    run_dir = Path(run_dir)
    
    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    
    data = {}
    
    # Load CSV files
    csv_files = ['states', 'velocities', 'accelerations', 'errors', 'adaptive_gains']
    for name in csv_files:
        csv_path = run_dir / f'{name}.csv'
        if csv_path.exists():
            data[name] = pd.read_csv(csv_path)
        else:
            print(f"Warning: {name}.csv not found")
            data[name] = None
    
    # Load metadata
    metadata_path = run_dir / 'metadata.yaml'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            data['metadata'] = yaml.safe_load(f)
    else:
        print("Warning: metadata.yaml not found")
        data['metadata'] = None
    
    return data


def validate_data(data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate loaded data for completeness and consistency.
    
    Args:
        data: Dictionary from load_run_data()
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check if all data types are present
    required_keys = ['states', 'velocities', 'accelerations', 'errors', 'adaptive_gains', 'metadata']
    for key in required_keys:
        if key not in data or data[key] is None:
            issues.append(f"Missing {key} data")
    
    if data['metadata'] is None:
        issues.append("Missing metadata")
    
    # Check for empty dataframes
    for key in ['states', 'velocities', 'accelerations', 'errors', 'adaptive_gains']:
        if data.get(key) is not None and len(data[key]) == 0:
            issues.append(f"{key} data is empty")
    
    # Check for consistent number of samples (approximately)
    if data.get('states') is not None and data.get('velocities') is not None:
        state_count = len(data['states'])
        vel_count = len(data['velocities'])
        if abs(state_count - vel_count) > 10:  # Allow small differences
            issues.append(f"Inconsistent sample counts: states={state_count}, velocities={vel_count}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def list_available_runs(base_dir: Path) -> List[Tuple[str, Path]]:
    """
    List all available run directories.
    
    Args:
        base_dir: Base directory for logs
        
    Returns:
        List of tuples (run_name, run_path)
    """
    if not base_dir.exists():
        return []
    
    runs = []
    for d in sorted(base_dir.iterdir()):
        if d.is_dir():
            runs.append((d.name, d))
    
    return runs


if __name__ == "__main__":
    # Test the utilities
    print("Testing data logger utilities...")
    
    # Create a test logger
    logger = DataLogger("test_logs")
    
    # Log some test data
    timestamp = 0.0
    for i in range(5):
        timestamp += 0.1
        
        # Log leader state
        logger.log_state(timestamp, 0, 'leader', np.array([i*0.1, i*0.2, 1.0, 0.0]))
        logger.log_velocity(timestamp, 0, 'leader', np.array([0.1, 0.2, 0.0, 0.0]))
        logger.log_acceleration(timestamp, 0, 'leader', np.array([0.0, 0.0, 0.0, 0.0]))
        logger.log_error(timestamp, 0, 'leader', np.array([0.05, 0.03, 0.0, 0.0]))
        logger.log_adaptive_gain(timestamp, 0, 'leader', np.array([1.0, 1.0, 1.0, 1.0]))
    
    # Write metadata
    logger.write_metadata({
        'n_followers': 4,
        'n_leaders': 4,
        'control_rate': 50.0,
        'formation_type': 'square'
    })
    
    # Close logger
    logger.close()
    
    print("\nTest complete. Check 'test_logs' directory for output.")

