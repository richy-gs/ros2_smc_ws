# Formation Control Data Logging System

## Overview

The data logging system captures comprehensive system state data from the formation control system for post-analysis. It records position, velocity, acceleration, formation errors, and adaptive gains for all agents at configurable sampling rates.

## Features

- **Configurable Sampling Rate**: Log data at rates independent of the control loop (default 10 Hz)
- **Incremental Run Management**: Automatically creates numbered run directories (run_001, run_002, etc.)
- **Multiple Data Streams**: Separate CSV files for different data types
- **Metadata Recording**: Saves run configuration for reproducibility
- **Interactive Visualization**: Post-processing tools with agent selection
- **Minimal Performance Impact**: Separate logging callback from control loop

## Data Structure

### Directory Layout

```
formation_containment_control/
├── logs/
│   ├── run_001/
│   │   ├── states.csv              # Agent positions and orientations
│   │   ├── velocities.csv          # Agent velocities
│   │   ├── accelerations.csv       # Control inputs (accelerations)
│   │   ├── errors.csv              # Formation/containment errors
│   │   ├── adaptive_gains.csv      # K_c adaptive gain evolution
│   │   └── metadata.yaml           # Run configuration
│   ├── run_002/
│   └── ...
```

### CSV File Formats

#### states.csv
```csv
timestamp, agent_id, agent_type, x, y, z, yaw
0.100, 0, follower, 0.5, 0.3, 1.0, 0.0
0.100, 0, leader, 1.2, 0.8, 1.0, 0.1
...
```

#### velocities.csv
```csv
timestamp, agent_id, agent_type, vx, vy, vz, vyaw
0.100, 0, follower, 0.1, 0.05, 0.0, 0.02
...
```

#### accelerations.csv
```csv
timestamp, agent_id, agent_type, ax, ay, az, omega
0.100, 0, follower, 0.05, 0.03, 0.0, 0.01
...
```

#### errors.csv
```csv
timestamp, agent_id, agent_type, error_x, error_y, error_z, error_yaw, error_norm
0.100, 0, follower, 0.02, 0.01, 0.0, 0.005, 0.024
...
```

#### adaptive_gains.csv
```csv
timestamp, agent_id, agent_type, Kc_x, Kc_y, Kc_z, Kc_yaw
0.100, 0, follower, 1.5, 1.4, 1.6, 1.3
...
```

#### metadata.yaml
```yaml
n_followers: 4
n_leaders: 4
topology: paper
formation_type: square
control_rate: 50.0
log_rate: 10.0
control_mode: position
lambda_gain: 3.0
alpha: 4.0
beta: 0.125
timestamp: 2024-12-16T10:30:00
...
```

## Usage

### 1. Enabling Data Logging

#### Method A: Launch File Parameter
```bash
ros2 launch formation_containment_control simulation.launch.py \
    enable_logging:=true \
    log_rate:=10.0 \
    log_directory:=logs
```

#### Method B: Using the Test Launch File
```bash
ros2 launch formation_containment_control logging_test.launch.py
```

#### Method C: Parameter File
In your parameter YAML file:
```yaml
formation_containment:
  ros__parameters:
    enable_logging: true
    log_rate: 10.0
    log_directory: "logs"
    # ... other parameters
```

### 2. Running with Logging

Start the system normally. When logging is enabled, you'll see:
```
Data logger initialized. Logging to: logs/run_001
Metadata written to: logs/run_001/metadata.yaml
Data logging enabled at 10.0 Hz
```

The system will automatically:
- Create a new run directory
- Initialize CSV files with headers
- Start logging at the specified rate
- Flush data to disk periodically

### 3. Stopping and Saving

When you stop the node (Ctrl+C), the logger will:
- Close all CSV files properly
- Display the save location
- Ensure all data is written to disk

Output:
```
Data logger closed. Data saved to: logs/run_001
```

## Data Analysis

### Interactive Analysis Tool

The `analyze_logs.py` script provides interactive visualization:

```bash
# Interactive run selection
cd formation_containment_control/scripts
python3 analyze_logs.py

# Analyze specific run
python3 analyze_logs.py ../logs/run_001

# Save all plots
python3 analyze_logs.py ../logs/run_001 --save ../plots
```

### Interactive Menu

When you run the analyzer, you'll see:
```
==============================================================
ANALYSIS MENU
==============================================================
1. View All Agents (interactive)
2. View Leaders (interactive)
3. View Followers (interactive)
4. View Formation Errors
5. View Adaptive Gains
6. View 3D Trajectories
7. Generate All Plots
8. Save All Plots
q. Quit
==============================================================
```

### Visualization Features

#### 1. Agent State Viewer (Options 1-3)
- Interactive dropdown/button selector for agents
- Three subplots per agent:
  - **Position**: x, y, z, yaw vs time
  - **Velocity**: vx, vy, vz, vyaw vs time
  - **Acceleration**: ax, ay, az, omega vs time

#### 2. Formation Errors (Option 4)
- Two subplots:
  - Leader formation errors over time
  - Follower containment errors over time
- Shows tracking accuracy for each agent

#### 3. Adaptive Gains (Option 5)
- Evolution of K_c over time
- Separate plots for leaders and followers
- Shows adaptation behavior of SGASMC

#### 4. 3D Trajectories (Option 6)
- 3D plot of all agent paths
- Start positions marked with circles (o)
- End positions marked with stars (*)
- Leaders in solid lines, followers in dashed lines

### Python API for Custom Analysis

```python
from data_logger_utils import load_run_data
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = load_run_data('logs/run_001')

# Access DataFrames
states_df = data['states']
velocities_df = data['velocities']
accelerations_df = data['accelerations']
errors_df = data['errors']
gains_df = data['adaptive_gains']
metadata = data['metadata']

# Filter data for specific agent
follower_0 = states_df[
    (states_df['agent_id'] == 0) & 
    (states_df['agent_type'] == 'follower')
]

# Custom plotting
plt.figure(figsize=(10, 6))
plt.plot(follower_0['timestamp'], follower_0['x'], label='x')
plt.plot(follower_0['timestamp'], follower_0['y'], label='y')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.title('Follower 0 Trajectory')
plt.grid(True)
plt.show()
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_logging` | bool | false | Enable/disable data logging |
| `log_rate` | float | 10.0 | Logging frequency in Hz |
| `log_directory` | string | "logs" | Base directory for log files |

### Recommended Settings

| Scenario | log_rate | Reasoning |
|----------|----------|-----------|
| Quick tests | 5 Hz | Small files, captures main dynamics |
| Standard logging | 10 Hz | Good balance of detail and file size |
| Detailed analysis | 25 Hz | High resolution for controller analysis |
| Full fidelity | 50 Hz | Same as control rate, largest files |

## Performance Considerations

### Impact on Control Loop

The logging system is designed for minimal impact:
- **Separate Timer**: Logging runs on its own timer callback
- **No Blocking**: CSV writes are buffered
- **Periodic Flushing**: Data flushed every 50 samples
- **Error Computation**: Formation status computed less frequently (every 10 logging cycles)

### File Sizes

Approximate file sizes for 60-second run with 8 agents:

| log_rate | Total Size | states.csv | velocities.csv | accelerations.csv |
|----------|------------|------------|----------------|-------------------|
| 5 Hz | ~150 KB | ~40 KB | ~40 KB | ~40 KB |
| 10 Hz | ~300 KB | ~80 KB | ~80 KB | ~80 KB |
| 25 Hz | ~750 KB | ~200 KB | ~200 KB | ~200 KB |
| 50 Hz | ~1.5 MB | ~400 KB | ~400 KB | ~400 KB |

## Troubleshooting

### Issue: "data_logger_utils not found"
**Solution**: Make sure `data_logger_utils.py` is in the same directory as `formation_containment_node.py` (the `scripts/` directory).

### Issue: "No runs found"
**Solution**: Check that:
1. Logging was enabled during the run
2. The `log_directory` parameter points to the correct location
3. The node ran long enough to generate data

### Issue: "Missing data in CSV files"
**Possible Causes**:
- Node was terminated abruptly (kill -9)
- Insufficient disk space
- File permissions issues

**Solutions**:
- Always use Ctrl+C to stop the node gracefully
- Check available disk space
- Verify write permissions in log directory

### Issue: "Plots show empty or incomplete data"
**Solution**: Check that:
1. The agents were receiving state feedback (odometry)
2. The controller was enabled (not just started)
3. Sufficient time elapsed for data collection

## Example Workflow

### 1. Run Experiment with Logging
```bash
# Terminal 1: Launch with logging enabled
ros2 launch formation_containment_control logging_test.launch.py \
    formation_type:=square \
    log_rate:=10.0

# Let it run for 60 seconds
# Press Ctrl+C to stop

# Note the output: "Data saved to: logs/run_001"
```

### 2. Analyze Data
```bash
# Terminal 2: Run analysis
cd formation_containment_control/scripts
python3 analyze_logs.py ../logs/run_001

# Select options from menu:
# 1 - View all agents interactively
# 4 - View formation errors
# 5 - View adaptive gains
# 7 - Generate all plots
```

### 3. Save Plots for Report
```bash
python3 analyze_logs.py ../logs/run_001 --save ../experiment_plots

# This creates:
# experiment_plots/
#   ├── formation_errors.png
#   ├── adaptive_gains.png
#   ├── trajectories_3d.png
#   └── agents/
#       ├── follower_0.png
#       ├── follower_1.png
#       ├── leader_0.png
#       └── ...
```

## Advanced Usage

### Comparing Multiple Runs

```python
from data_logger_utils import load_run_data
import matplotlib.pyplot as plt

# Load multiple runs
run1 = load_run_data('logs/run_001')
run2 = load_run_data('logs/run_002')

# Compare follower 0 errors
fig, ax = plt.subplots()

for run, label in [(run1, 'Run 1'), (run2, 'Run 2')]:
    errors = run['errors']
    f0_errors = errors[
        (errors['agent_id'] == 0) & 
        (errors['agent_type'] == 'follower')
    ]
    ax.plot(f0_errors['timestamp'], f0_errors['error_norm'], label=label)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Error Norm [m]')
ax.legend()
ax.grid(True)
plt.show()
```

### Computing Custom Metrics

```python
import numpy as np
from data_logger_utils import load_run_data

data = load_run_data('logs/run_001')

# Compute average tracking error for all followers
errors = data['errors']
follower_errors = errors[errors['agent_type'] == 'follower']
avg_error = follower_errors['error_norm'].mean()
max_error = follower_errors['error_norm'].max()

print(f"Average follower error: {avg_error:.4f} m")
print(f"Maximum follower error: {max_error:.4f} m")

# Compute settling time (time to reach < 0.1m error)
for agent_id in follower_errors['agent_id'].unique():
    agent_data = follower_errors[follower_errors['agent_id'] == agent_id]
    settled = agent_data[agent_data['error_norm'] < 0.1]
    if not settled.empty:
        settling_time = settled['timestamp'].min()
        print(f"Follower {int(agent_id)} settled at t={settling_time:.2f}s")
```

### Exporting to Other Formats

```python
import pandas as pd

# Load data
from data_logger_utils import load_run_data
data = load_run_data('logs/run_001')

# Export to Excel
with pd.ExcelWriter('formation_data.xlsx') as writer:
    data['states'].to_excel(writer, sheet_name='States', index=False)
    data['velocities'].to_excel(writer, sheet_name='Velocities', index=False)
    data['errors'].to_excel(writer, sheet_name='Errors', index=False)

# Export to single CSV
combined = data['states'].merge(
    data['velocities'], 
    on=['timestamp', 'agent_id', 'agent_type'],
    how='outer'
)
combined.to_csv('combined_data.csv', index=False)
```

## Dependencies

The logging system requires:
- **ROS2 packages**: rclpy, geometry_msgs, nav_msgs
- **Python packages**: numpy, pandas, matplotlib, pyyaml

Install missing dependencies:
```bash
pip install pandas matplotlib pyyaml
```

## Best Practices

1. **Start Logging Early**: Enable logging from the beginning to capture transients
2. **Label Experiments**: Use descriptive directory names or keep notes
3. **Monitor Disk Space**: Regular logging can accumulate data
4. **Save Metadata**: The metadata.yaml file is crucial for reproducibility
5. **Backup Important Runs**: Copy significant runs to a separate location
6. **Use Version Control**: Track changes to logging code and configurations

## References

For more information about the formation control system:
- See `FULL_STATE_CONTROL_EXPLAINED.md` for control mode details
- See `CONTROL_MODE_RECOMMENDATION.md` for mode selection guidance
- See `README.md` for general system documentation

