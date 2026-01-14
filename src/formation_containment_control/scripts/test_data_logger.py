#!/usr/bin/env python3
"""
Test script for data logger utilities.

This script demonstrates the data logger functionality without requiring
a full ROS2 system running.
"""

import numpy as np
import time
from pathlib import Path
from data_logger_utils import DataLogger, load_run_data, validate_data

def test_logger():
    """Test basic logger functionality."""
    print("="*60)
    print("DATA LOGGER TEST")
    print("="*60)
    
    # Create logger
    print("\n1. Creating data logger...")
    logger = DataLogger("test_logs")
    
    # Simulate data logging for 5 seconds at 10 Hz
    print("\n2. Logging simulated data (5 seconds at 10 Hz)...")
    dt = 0.1  # 10 Hz
    duration = 5.0
    n_samples = int(duration / dt)
    
    t = 0.0
    for i in range(n_samples):
        # Simulate follower motion
        for fid in range(4):
            # Circular motion
            angle = t * 0.5 + fid * np.pi / 2
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            z = 1.0
            yaw = angle
            
            state = np.array([x, y, z, yaw])
            velocity = np.array([-2.0*0.5*np.sin(angle), 2.0*0.5*np.cos(angle), 0.0, 0.5])
            acceleration = np.random.randn(4) * 0.1
            
            # Log data
            logger.log_state(t, fid, 'follower', state)
            logger.log_velocity(t, fid, 'follower', velocity)
            logger.log_acceleration(t, fid, 'follower', acceleration)
            
            # Log error (decreasing over time)
            error = np.random.randn(4) * 0.1 * (1.0 - t/duration)
            logger.log_error(t, fid, 'follower', error)
            
            # Log adaptive gain (increasing then stabilizing)
            K_c = np.ones(4) * (1.0 + 2.0 * (1.0 - np.exp(-t)))
            logger.log_adaptive_gain(t, fid, 'follower', K_c)
        
        # Simulate leader motion
        for lid in range(4):
            # Square formation around origin
            positions = [
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0]
            ]
            x, y, z = positions[lid]
            yaw = 0.0
            
            state = np.array([x, y, z, yaw])
            velocity = np.random.randn(4) * 0.05
            acceleration = np.random.randn(4) * 0.1
            
            logger.log_state(t, lid, 'leader', state)
            logger.log_velocity(t, lid, 'leader', velocity)
            logger.log_acceleration(t, lid, 'leader', acceleration)
            
            # Small formation error
            error = np.random.randn(4) * 0.05
            logger.log_error(t, lid, 'leader', error)
            
            # Leader gains
            K_c = np.ones(4) * 1.5
            logger.log_adaptive_gain(t, lid, 'leader', K_c)
        
        t += dt
        
        # Progress indicator
        if i % 10 == 0:
            print(f"  Progress: {i}/{n_samples} samples ({i/n_samples*100:.0f}%)")
    
    print(f"  Complete: {n_samples} samples logged")
    
    # Write metadata
    print("\n3. Writing metadata...")
    metadata = {
        'n_followers': 4,
        'n_leaders': 4,
        'topology': 'paper',
        'formation_type': 'square',
        'control_rate': 50.0,
        'log_rate': 10.0,
        'control_mode': 'position',
        'lambda_gain': 3.0,
        'alpha': 4.0,
        'beta': 0.125,
        'test_run': True,
        'description': 'Test data for logger validation'
    }
    logger.write_metadata(metadata)
    
    # Close logger
    print("\n4. Closing logger...")
    logger.close()
    
    print("\n5. Loading and validating data...")
    data = load_run_data(logger.run_dir)
    is_valid, issues = validate_data(data)
    
    if is_valid:
        print("  ✓ Data validation: PASSED")
    else:
        print("  ✗ Data validation: FAILED")
        for issue in issues:
            print(f"    - {issue}")
    
    # Print summary
    print("\n6. Data summary:")
    for key in ['states', 'velocities', 'accelerations', 'errors', 'adaptive_gains']:
        if data.get(key) is not None:
            print(f"  {key}: {len(data[key])} samples")
    
    print(f"\n✓ Test complete! Data saved to: {logger.run_dir}")
    print(f"\nTo analyze this data, run:")
    print(f"  python3 analyze_logs.py {logger.run_dir}")
    print("="*60)
    
    return logger.run_dir


def test_multiple_runs():
    """Test creating multiple run directories."""
    print("\n" + "="*60)
    print("TESTING MULTIPLE RUNS")
    print("="*60)
    
    base_dir = Path("test_logs_multiple")
    
    for i in range(3):
        print(f"\nCreating run {i+1}...")
        logger = DataLogger(str(base_dir))
        
        # Log minimal data
        for j in range(5):
            t = j * 0.1
            logger.log_state(t, 0, 'follower', np.array([0.0, 0.0, 1.0, 0.0]))
        
        logger.write_metadata({'run_number': i+1})
        logger.close()
    
    print(f"\n✓ Created 3 runs in: {base_dir}")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    print("\nFormation Control Data Logger Test")
    print("===================================\n")
    
    try:
        # Test basic logger
        run_dir = test_logger()
        
        # Optional: test multiple runs
        if '--multiple' in sys.argv:
            test_multiple_runs()
        
        print("\n✓ All tests completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Analyze the test data:")
        print(f"   python3 analyze_logs.py {run_dir}")
        print(f"2. Run the system with logging enabled:")
        print(f"   ros2 launch formation_containment_control logging_test.launch.py")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

