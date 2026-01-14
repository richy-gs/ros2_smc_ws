# Example Plots and Visualizations

This document demonstrates the types of plots and visualizations available from the data logging system.

## Quick Start

After collecting data with logging enabled, use the analysis tool:

```bash
cd formation_containment_control/scripts
python3 analyze_logs.py ../logs/run_001
```

## Available Visualizations

### 1. Agent State Viewer

**Access**: Menu options 1, 2, or 3 (All Agents, Leaders, or Followers)

**Features**:
- Interactive agent selection via buttons
- Three synchronized subplots:
  - Position (x, y, z, yaw) vs time
  - Velocity (vx, vy, vz, vyaw) vs time  
  - Acceleration (ax, ay, az, omega) vs time

**Use Cases**:
- Verify individual agent tracking performance
- Debug control issues for specific drones
- Compare behavior between agents
- Validate control inputs

**Example Interpretation**:
```
Position subplot:
- x, y should show smooth trajectories following formation
- z should maintain constant height (typically 1.0m)
- yaw shows orientation tracking

Velocity subplot:
- Should be smooth without large spikes
- Magnitude indicates tracking effort
- Spikes may indicate disturbances or collisions

Acceleration subplot:
- Shows actual control commands sent to drones
- Larger values indicate aggressive control
- Should stabilize as formation is achieved
```

### 2. Formation Errors

**Access**: Menu option 4

**Features**:
- Two subplots:
  - Top: Leader formation tracking errors
  - Bottom: Follower containment errors
- Shows error norm (Euclidean distance) over time

**Use Cases**:
- Measure tracking accuracy
- Determine convergence time
- Identify problematic agents
- Validate controller tuning

**Expected Behavior**:
```
Leader Errors:
- Should decrease from initial values
- Stabilize at small values (< 0.1m typically)
- Similar magnitude across all leaders

Follower Errors:
- May start larger than leaders
- Should converge to containment region
- Final values depend on containment weights
- All followers should reach steady state
```

### 3. Adaptive Gain Evolution

**Access**: Menu option 5

**Features**:
- Two subplots showing K_c evolution:
  - Top: Leader adaptive gains
  - Bottom: Follower adaptive gains
- Shows average K_c across x, y, z, yaw dimensions

**Use Cases**:
- Verify adaptive mechanism is working
- Tune α and β parameters
- Understand controller adaptation behavior
- Debug gain saturation issues

**Expected Behavior**:
```
Typical K_c Evolution:
1. Initial value: ~1.0 (K_c_init parameter)
2. Rapid increase during transient
3. Peak during maximum error
4. Gradual decrease as error reduces
5. Stabilization at steady-state value

Healthy Patterns:
- Smooth curves without oscillations
- All agents reach similar final values
- No saturation at bounds (K_c_min, K_c_max)

Problem Indicators:
- Stuck at K_c_max: α too large or large disturbances
- Oscillating: Poor β tuning
- Never increases: α too small
```

### 4. 3D Trajectories

**Access**: Menu option 6

**Features**:
- 3D visualization of all agent paths
- Start positions: circles (o)
- End positions: stars (*)
- Leaders: solid lines
- Followers: dashed lines

**Use Cases**:
- Visualize overall formation behavior
- Verify spatial relationships
- Identify collision events
- Verify containment (followers inside leader hull)

**Interpretation**:
```
Formation Achieved:
- Leaders form geometric shape (square, circle, etc.)
- Followers remain inside leader convex hull
- Smooth trajectories without sharp turns

Issues to Look For:
- Followers outside hull: containment violated
- Crossing paths: potential collisions
- Erratic paths: control instability
- Asymmetry: weight matrix issues
```

## Generating Report-Ready Plots

### Save All Plots at Once

```bash
python3 analyze_logs.py ../logs/run_001 --save ../report_plots
```

This creates:
```
report_plots/
├── formation_errors.png
├── adaptive_gains.png
├── trajectories_3d.png
└── agents/
    ├── follower_0.png
    ├── follower_1.png
    ├── follower_2.png
    ├── follower_3.png
    ├── leader_0.png
    ├── leader_1.png
    ├── leader_2.png
    └── leader_3.png
```

### Customizing Plot Appearance

Plots use matplotlib defaults. For publications, you can modify `analyze_logs.py`:

```python
# Add at the top of the file
import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300  # High resolution for papers
```

## Example Analysis Workflow

### 1. Quick Health Check

```bash
python3 analyze_logs.py
# Select run
# Option 4 - View formation errors
# Option 5 - View adaptive gains
```

**Look for**:
- Errors converging to small values
- Gains stabilizing
- No saturation or oscillations

### 2. Detailed Agent Inspection

```bash
python3 analyze_logs.py
# Option 1 - View all agents
# Click through each agent button
```

**Check each agent**:
- Smooth position trajectories
- Reasonable velocities
- Consistent acceleration magnitudes

### 3. Formation Verification

```bash
python3 analyze_logs.py
# Option 6 - View 3D trajectories
```

**Verify**:
- Geometric formation shape
- Follower containment
- No collision events

### 4. Generate Report

```bash
python3 analyze_logs.py ../logs/run_001 --save ../experiment_results
```

Include these plots in your report/paper with captions explaining:
- Experiment parameters (from metadata.yaml)
- Observed behavior
- Performance metrics

## Computing Performance Metrics

Use the Python API for quantitative analysis:

```python
from data_logger_utils import load_run_data
import numpy as np

data = load_run_data('logs/run_001')

# Convergence time (time to reach 10cm error)
errors = data['errors']
follower_errors = errors[errors['agent_type'] == 'follower']

for agent_id in follower_errors['agent_id'].unique():
    agent_data = follower_errors[follower_errors['agent_id'] == agent_id]
    converged = agent_data[agent_data['error_norm'] < 0.1]
    if not converged.empty:
        t_converge = converged['timestamp'].min()
        print(f"Follower {int(agent_id)}: converged at {t_converge:.2f}s")

# Steady-state error (average of last 20% of data)
t_max = follower_errors['timestamp'].max()
t_cutoff = 0.8 * t_max
steady_state = follower_errors[follower_errors['timestamp'] > t_cutoff]
avg_ss_error = steady_state.groupby('agent_id')['error_norm'].mean()
print(f"\nSteady-state errors:\n{avg_ss_error}")

# Control effort (RMS acceleration)
accels = data['accelerations']
for agent_type in ['leader', 'follower']:
    type_data = accels[accels['agent_type'] == agent_type]
    rms_ax = np.sqrt((type_data['ax']**2).mean())
    rms_ay = np.sqrt((type_data['ay']**2).mean())
    rms_az = np.sqrt((type_data['az']**2).mean())
    print(f"\n{agent_type.capitalize()} RMS accelerations:")
    print(f"  ax: {rms_ax:.3f} m/s²")
    print(f"  ay: {rms_ay:.3f} m/s²")
    print(f"  az: {rms_az:.3f} m/s²")
```

## Comparing Different Experiments

### Side-by-Side Comparison

```python
from data_logger_utils import load_run_data
import matplotlib.pyplot as plt

# Load multiple runs
runs = [
    ('Square Formation', 'logs/run_001'),
    ('Circle Formation', 'logs/run_002'),
]

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

for name, path in runs:
    data = load_run_data(path)
    errors = data['errors']
    
    # Leader errors
    leader_err = errors[errors['agent_type'] == 'leader']
    leader_avg = leader_err.groupby('timestamp')['error_norm'].mean()
    axes[0].plot(leader_avg.index, leader_avg.values, label=name)
    
    # Follower errors
    follower_err = errors[errors['agent_type'] == 'follower']
    follower_avg = follower_err.groupby('timestamp')['error_norm'].mean()
    axes[1].plot(follower_avg.index, follower_avg.values, label=name)

axes[0].set_ylabel('Average Leader Error [m]')
axes[0].legend()
axes[0].grid(True)

axes[1].set_ylabel('Average Follower Error [m]')
axes[1].set_xlabel('Time [s]')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Formation Comparison')
plt.tight_layout()
plt.show()
```

## Troubleshooting Plots

### Empty or Missing Data

**Symptom**: Plots show no data or "No data available"

**Causes**:
- Controller was not enabled
- Insufficient run duration
- Agents not receiving odometry

**Solution**: Check:
```bash
# Verify data files exist and have content
ls -lh logs/run_001/
cat logs/run_001/metadata.yaml
head logs/run_001/states.csv
```

### Noisy or Erratic Plots

**Symptom**: Plots show high-frequency oscillations

**Causes**:
- Control gains too high
- Numerical issues in control loop
- Sensor noise

**Solution**:
- Check control parameters (λ, α, β)
- Enable filtering in post-processing
- Verify dt matches actual control rate

### Incomplete Trajectories

**Symptom**: 3D trajectories end abruptly

**Causes**:
- Node terminated before closing logger
- Crash or exception during run

**Solution**:
- Always stop with Ctrl+C (not kill -9)
- Check ROS logs for errors
- Verify all agents were publishing odometry

## Advanced Visualizations

### Phase Portraits

```python
# Plot velocity vs position for follower 0
states = data['states']
vels = data['velocities']

f0_states = states[(states['agent_id']==0) & (states['agent_type']=='follower')]
f0_vels = vels[(vels['agent_id']==0) & (vels['agent_type']=='follower')]

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(f0_states['x'], f0_vels['vx'])
plt.xlabel('x [m]')
plt.ylabel('vx [m/s]')
plt.title('Phase Portrait - X axis')
plt.grid(True)

plt.subplot(122)
plt.plot(f0_states['y'], f0_vels['vy'])
plt.xlabel('y [m]')
plt.ylabel('vy [m/s]')
plt.title('Phase Portrait - Y axis')
plt.grid(True)
plt.show()
```

### Animated Trajectories

```python
import matplotlib.animation as animation

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

states = data['states']
times = sorted(states['timestamp'].unique())

def update(frame):
    ax.clear()
    t = times[frame]
    current = states[states['timestamp'] <= t]
    
    for agent_id in current['agent_id'].unique():
        agent_data = current[current['agent_id'] == agent_id]
        ax.plot(agent_data['x'], agent_data['y'], agent_data['z'])
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'Time: {t:.2f}s')

ani = animation.FuncAnimation(fig, update, frames=len(times), 
                              interval=50, repeat=True)
plt.show()
```

## Tips for Publication-Quality Figures

1. **Use vector formats**: Save as PDF or SVG
   ```python
   fig.savefig('figure.pdf', format='pdf', bbox_inches='tight')
   ```

2. **Increase DPI**: For raster formats
   ```python
   fig.savefig('figure.png', dpi=300, bbox_inches='tight')
   ```

3. **Adjust font sizes**: Make text readable
   ```python
   plt.rcParams.update({'font.size': 14})
   ```

4. **Use color-blind friendly palettes**:
   ```python
   colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
   ```

5. **Add grid for readability**:
   ```python
   ax.grid(True, alpha=0.3, linestyle='--')
   ```

## Summary

The data logging system provides comprehensive visualization tools for:
- ✓ Validating controller performance
- ✓ Debugging tracking issues
- ✓ Analyzing adaptive behavior
- ✓ Generating publication figures
- ✓ Computing quantitative metrics

For more details, see `DATA_LOGGING_GUIDE.md`.

