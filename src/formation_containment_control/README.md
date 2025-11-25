# Formation-Containment Control for Quadrotor Fleet

Implementation of the control strategy from:

> **"Collision-Free Formation-Containment Control Based on Adaptive Sliding Mode Strategy for a Quadrotor Fleet Under Disturbances"**  
> Carlos Katt and Herman Castañeda, 2025  
> DOI: 10.3390/drones9100724

This ROS2 package provides a complete implementation of the paper's formation-containment control strategy, compatible with the Bitcraze Crazyswarm2 framework.

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Background](#mathematical-background)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [Testing](#testing)
9. [Examples](#examples)
10. [Future Extensions](#future-extensions)

---

## Overview

This package implements a robust formation and collision-free containment control system for a fleet of quadrotor MAVs. The key features include:

- **Two-layer control strategy**: Virtual leader → Leaders → Followers
- **Graph-based interaction networks** with Laplacian matrix computation
- **Single-Gain Adaptive Sliding Mode Control (SGASMC)** for finite-time convergence
- **Sensor-less collision avoidance** between agents
- **Convex hull containment** for followers
- **Von Kármán turbulence model** for realistic disturbance simulation
- **Crazyswarm2 integration** for both simulation and real hardware

### System Configuration (from paper)

- **8 agents**: 4 followers (F1-F4) + 4 leaders (L1-L4)
- **Control parameters**: α=4, β=0.125, λ=3
- **Safety distance**: γs = 0.3m
- **Trajectory**: 3D infinity (lemniscate)

---

## Mathematical Background

### Graph Theory (Section 2.1)

The multi-agent system uses a directed graph G(V, ε) where:
- V = {v₁, ..., v_{n+m}} : Set of n followers + m leaders
- ε(i,j) : Edge set for information flow

**Adjacency Matrix**: A = [a_ij] where a_ij = 1 if agent j sends to agent i

**Laplacian Matrix**: L = D - A, partitioned as:
```
L = [L_N   L_M ]
    [0     0   ]
```
where L_N ∈ ℜⁿˣⁿ (follower-follower) and L_M ∈ ℜⁿˣᵐ (leader-follower)

### Reduced Tracking Model (Equations 10-13)

```
ẍ = aₓcos(ψ) - aᵧsin(ψ) - ψ̇(vₓsin(ψ) + vᵧcos(ψ))
ÿ = aₓsin(ψ) + aᵧcos(ψ) + ψ̇(vₓcos(ψ) - vᵧsin(ψ))
z̈ = aᵤ
ψ̈ = Ω
```

General form: ξ̈ᵢ = F(ξᵢ,t) + g(ξᵢ)Uᵢ + Δᵢ(t)

### Containment Error (Equations 16-24)

**Desired follower position**:
```
ξ_dc,i = Σⱼ[-L_N⁻¹L_M]ᵢⱼ ξ_{n+j}
```

**Collision-avoidance augmented error**:
```
e_ac = e_c + h_c
e_ac = (L_ζ - I_{n+m} - L) ⊗ Iₛ · ξ
```

### SGASMC Controller (Equations 26-30)

**Sliding Surface**:
```
σ_c = ė_ac + D_λ e_ac
```

**Control Law**:
```
Uᵢ = g(ξᵢ)⁻¹(-Fᵢ - uᵢ(t) + D_λ ė_c)
```

**Auxiliary Control**:
```
uᵢ = -2Kc(t)|σc|^½ sign(σc) - (Kc²/2)σc
```

**Adaptive Law**:
```
K̇c(t) = α^½|σ|^½ - β^½Kc(t)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Formation-Containment Control                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │Virtual Leader│────▶│   Leaders    │────▶│  Followers   │    │
│  │  Trajectory  │     │  Controller  │     │  Controller  │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                    │                    │             │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Interaction Network (Graph)                  │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │  │
│  │  │Adjacency│  │ Degree  │  │Laplacian│  │  L_ζ    │     │  │
│  │  │   A     │  │   D     │  │   L     │  │ Matrix  │     │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    SGASMC Controller                      │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │  │
│  │  │ Sliding │  │Auxiliary│  │Adaptive │  │ Control │     │  │
│  │  │ Surface │  │ Control │  │  Gain   │  │ Output  │     │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Convex Hull Containment                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ROS2 Nodes

| Node | Description |
|------|-------------|
| `virtual_leader_node` | Generates reference trajectory (infinity, circle, etc.) |
| `formation_containment_node` | Main controller for all agents |
| `simulation_node` | Simulates quadrotor dynamics |
| `crazyswarm_bridge_node` | Interface to Crazyswarm2 |
| `visualization_node` | RViz2 visualization |

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/virtual_leader/pose` | `PoseStamped` | Virtual leader position |
| `/cf<id>/odom` | `Odometry` | Drone state feedback |
| `/cf<id>/cmd_vel` | `Twist` | Velocity commands |
| `/formation/status` | `Float64MultiArray` | Formation status |
| `/formation/visualization` | `MarkerArray` | RViz markers |

---

## Installation

### Prerequisites

- ROS2 Humble (or later)
- Python 3.8+
- NumPy, SciPy

### Build

```bash
# Clone into your ROS2 workspace
cd ~/ros2_ws/src
# (Package should already be here)

# Install dependencies
pip3 install numpy scipy

# Build
cd ~/ros2_ws
colcon build --packages-select formation_containment_control
source install/setup.bash
```

### Crazyswarm2 (Optional, for real hardware)

Follow the [Crazyswarm2 installation guide](https://imrclab.github.io/crazyswarm2/).

---

## Usage

### Simulation Mode (No Hardware Required)

```bash
# Launch complete simulation
ros2 launch formation_containment_control simulation.launch.py

# With custom parameters
ros2 launch formation_containment_control simulation.launch.py \
    n_followers:=4 \
    n_leaders:=4 \
    trajectory_type:=infinity \
    use_turbulence:=false
```

### With Crazyswarm2

```bash
# First, launch Crazyswarm2
ros2 launch crazyflie launch.py

# Then launch the controller
ros2 launch formation_containment_control crazyswarm2.launch.py

# Takeoff all drones
ros2 topic pub /formation/takeoff std_msgs/Empty "{}" --once

# Land all drones
ros2 topic pub /formation/land std_msgs/Empty "{}" --once
```

### Visualization

RViz2 is launched automatically with the simulation. Markers show:
- **Green sphere**: Virtual leader
- **Colored spheres**: Leaders (red, blue, yellow, magenta)
- **Gray spheres**: Followers
- **Cyan lines**: Convex hull
- **Colored lines**: Formation connections

---

## Configuration

### Main Configuration File

`config/formation_params.yaml`

```yaml
# Agent configuration
n_followers: 4
n_leaders: 4
topology: "paper"  # Graph topology

# Formation
formation_type: "square"
formation_scale: 1.0
formation_height: 1.0

# Control parameters (from paper)
lambda_gain: 3.0   # Sliding surface gain
alpha: 4.0         # Adaptive precision
beta: 0.125        # Control effort
safety_distance: 0.3

# Timing
control_rate: 50.0
dt: 0.02

# Trajectory
trajectory_type: "infinity"
trajectory_period: 60.0
```

### Formation Types

| Type | Description | Leaders |
|------|-------------|---------|
| `square` | Square formation (paper default) | 4 |
| `triangle` | Equilateral triangle | 3 |
| `tetrahedron` | 3D tetrahedron | 4 |
| `circle` | Circular formation | Any |
| `line` | Linear formation | Any |

### Graph Topologies

| Topology | Description |
|----------|-------------|
| `paper` | Topology from Figure 1 |
| `complete` | All-to-all connections |
| `ring` | Circular ring topology |

---

## API Reference

### Core Modules

#### `graph_theory.py`

```python
from formation_containment_control.core.graph_theory import (
    create_interaction_network,
    InteractionNetwork,
    GraphTopology
)

# Create network
network = create_interaction_network(
    n_followers=4,
    n_leaders=4,
    topology="paper"
)

# Get containment weights
weights = network.laplacian.containment_weights

# Verify connectivity
valid, msg = network.verify_connectivity()
```

#### `sgasmc.py`

```python
from formation_containment_control.controllers.sgasmc import (
    SGASMCController,
    SGASMCParameters
)

# Create controller
params = SGASMCParameters(
    lambda_gain=3.0,
    alpha=4.0,
    beta=0.125
)
controller = SGASMCController(params)

# Compute control
U = controller.compute_control(
    xi, xi_dot, error, error_dot, F, g, g_inv
)
```

#### `convex_hull.py`

```python
from formation_containment_control.core.convex_hull import (
    ConvexHullContainment,
    FormationGeometry
)

# Create containment manager
containment = ConvexHullContainment(n_leaders=4)
containment.update_leader_states(leader_positions)

# Check containment
is_inside = containment.is_point_contained(point)
```

---

## Testing

### Run All Tests

```bash
cd ~/ros2_ws
colcon test --packages-select formation_containment_control
colcon test-result --verbose
```

### Run Specific Test

```bash
# Graph theory tests
python3 -m pytest src/formation_containment_control/test/test_graph_theory.py -v

# Controller tests
python3 -m pytest src/formation_containment_control/test/test_controllers.py -v

# Convex hull tests
python3 -m pytest src/formation_containment_control/test/test_convex_hull.py -v
```

### Validation Scenarios

1. **Static Convergence**: Followers converge to formation from random initial positions
2. **Dynamic Tracking**: Formation tracks moving virtual leader
3. **Collision Avoidance**: Agents avoid collisions during convergence
4. **Formation Change**: Switch from square to tetrahedron formation
5. **Disturbance Rejection**: Maintain formation under turbulence

---

## Examples

### Example 1: Basic Simulation

```python
#!/usr/bin/env python3
from formation_containment_control.controllers.formation_controller import (
    FormationController, FormationConfig
)
import numpy as np

# Create controller
config = FormationConfig(
    n_followers=4,
    n_leaders=4,
    formation_type="square"
)
controller = FormationController(config)

# Set virtual leader
controller.set_virtual_leader_state(
    np.array([0, 0, 1, 0]),  # [x, y, z, yaw]
    np.array([0.5, 0, 0, 0])  # velocity
)

# Simulation loop
for t in range(1000):
    # Update states (from simulation or sensors)
    controller.update_agent_states(
        leader_states, leader_velocities,
        follower_states, follower_velocities
    )
    
    # Compute controls
    leader_controls, follower_controls = controller.compute_all_controls()
    
    # Apply controls to simulation/hardware
    # ...
```

### Example 2: Formation Change

```python
# Change formation at t=62.5s (as in paper)
if t >= 62.5:
    controller.change_formation("tetrahedron", scale=1.0)
```

### Example 3: Custom Graph Topology

```python
from formation_containment_control.core.graph_theory import (
    GraphTopology, InteractionNetwork
)
import numpy as np

# Custom adjacency matrix
A = np.array([
    [0, 1, 0, 0, 1, 1, 0, 0],  # F1 <- F2, L1, L2
    [1, 0, 1, 0, 0, 1, 1, 0],  # F2 <- F1, F3, L2, L3
    [0, 1, 0, 1, 0, 0, 1, 1],  # F3 <- F2, F4, L3, L4
    [1, 0, 1, 0, 1, 0, 0, 1],  # F4 <- F1, F3, L1, L4
    [0, 0, 0, 0, 0, 0, 0, 0],  # L1 (leader, no incoming)
    [0, 0, 0, 0, 0, 0, 0, 0],  # L2
    [0, 0, 0, 0, 0, 0, 0, 0],  # L3
    [0, 0, 0, 0, 0, 0, 0, 0],  # L4
])

adj = GraphTopology.custom_graph(A, n_followers=4, n_leaders=4)
network = InteractionNetwork(adj)
```

---

## Future Extensions

1. **Obstacle Avoidance**: Integrate with obstacle detection sensors
2. **Dynamic Topology**: Adapt network graph based on communication quality
3. **Heterogeneous Agents**: Support different drone types
4. **3D Formation Shapes**: More complex 3D formations
5. **Learning-based Adaptation**: Tune control parameters online
6. **Hardware Integration**: Extended Crazyswarm2 support
7. **ROS2 Services**: Formation change, parameter tuning services
8. **Logging/Analysis**: Comprehensive data logging and analysis tools

---

## References

1. Katt, C.; Castañeda, H. "Collision-Free Formation-Containment Control Based on Adaptive Sliding Mode Strategy for a Quadrotor Fleet Under Disturbances." *Drones* 2025, 9, 724.

2. Crazyswarm2: https://github.com/IMRCLab/crazyswarm2

3. Bitcraze Crazyflie: https://www.bitcraze.io/

---

## License

MIT License

## Authors

- Implementation: Based on work by Carlos Katt and Herman Castañeda
- Tecnológico de Monterrey, School of Sciences and Engineering

