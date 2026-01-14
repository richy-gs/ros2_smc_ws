# Control Mode Recommendation for Formation Containment Control

## Executive Summary

**Best Approach: Use `_cmd_full_state_changed` (Full State Control)**

Your formation controller outputs accelerations `[a_x, a_y, a_z, Ω]` where:
- `[a_x, a_y, a_z]` are linear accelerations in the world frame
- `Ω` is the yaw rate

The **Full State Control** mode is the optimal choice because it directly accepts accelerations, providing maximum control authority and best tracking performance for aggressive maneuvers.

## Available Control Modes Comparison

### 1. Full State Control (`_cmd_full_state_changed`) ⭐ **RECOMMENDED**

**Topic:** `/{prefix}{id}/cmd_full_state`  
**Message Type:** `crazyflie_interfaces/FullState`

**Advantages:**
- ✅ Directly accepts accelerations (matches controller output)
- ✅ Maximum control authority
- ✅ Best for aggressive maneuvers and precise tracking
- ✅ Provides feedforward acceleration inputs
- ✅ Supports full 6-DOF control

**Requirements:**
- Position: from current state feedback
- Velocity: from current state feedback
- Acceleration: from controller output `[a_x, a_y, a_z]`
- Yaw: from current state feedback
- Angular rates: `[0, 0, Ω]` where `Ω` is from controller output

**When to Use:**
- When you need maximum control authority
- For aggressive maneuvers
- When controller outputs accelerations (your case)
- For best tracking performance

### 2. Velocity Control (`_cmd_hover_changed`)

**Topic:** `/{prefix}{id}/cmd_hover`  
**Message Type:** `crazyflie_interfaces/Hover`

**Advantages:**
- ✅ Simpler than full state
- ✅ Good for smooth trajectories
- ✅ Maintains altitude control

**Disadvantages:**
- ❌ Requires single integration of accelerations to get velocities
- ❌ Less control authority than full state
- ❌ No feedforward acceleration

**When to Use:**
- For smooth, less aggressive maneuvers
- When you can tolerate integration errors

### 3. Position Control (`_cmd_position_changed`)

**Topic:** `/{prefix}{id}/cmd_position`  
**Message Type:** `geometry_msgs/PoseStamped` (via GoTo service)

**Advantages:**
- ✅ Simple high-level control
- ✅ Good for waypoint following

**Disadvantages:**
- ❌ Requires double integration of accelerations (accumulates errors)
- ❌ Least control authority
- ❌ No feedforward terms
- ❌ Slower response

**When to Use:**
- For simple waypoint navigation
- When precise dynamics control is not needed
- Currently used in your code (open-loop position targets)

### 4. Attitude Control (`_cmd_vel_legacy_changed`)

**Topic:** `/{prefix}{id}/cmd_vel_legacy`  
**Message Type:** `geometry_msgs/Twist`

**Disadvantages:**
- ❌ Requires complex conversion from accelerations to roll/pitch/thrust
- ❌ Not suitable for your controller output format
- ❌ Low-level control, harder to tune

**When to Use:**
- Only if you need direct attitude control
- Not recommended for your use case

## Implementation

The code has been updated to support all three main modes:
- `position`: Uses GoTo service (current default)
- `velocity`: Integrates accelerations to velocities
- `full_state`: Uses full state control (recommended)

### Configuration

Set the control mode via ROS2 parameter:

```yaml
# In your launch file or parameter file
formation_containment_node:
  ros__parameters:
    control_mode: "full_state"  # Options: "position", "velocity", "full_state"
```

Or via command line:
```bash
ros2 run formation_containment_control formation_containment_node --ros-args -p control_mode:=full_state
```

### How Full State Control Works

1. **Controller Output:** `[a_x, a_y, a_z, Ω]`
   - Accelerations from SGASMC controller
   - Yaw rate from controller

2. **State Feedback Required:**
   - Current position `[x, y, z]` from odometry
   - Current velocity `[vx, vy, vz]` from odometry
   - Current yaw `ψ` from odometry

3. **Full State Message Construction:**
   ```python
   full_state.pose.position = [x, y, z]           # From state feedback
   full_state.twist.linear = [vx, vy, vz]         # From state feedback
   full_state.acc = [a_x, a_y, a_z]               # From controller
   full_state.pose.orientation = quaternion(yaw)  # From state feedback
   full_state.twist.angular = [0, 0, Ω]           # From controller
   ```

## Performance Comparison

| Mode | Control Authority | Tracking Performance | Integration Required | Feedforward |
|------|------------------|---------------------|---------------------|-------------|
| Full State | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | None | ✅ Yes |
| Velocity | ⭐⭐⭐ | ⭐⭐⭐ | Single | ❌ No |
| Position | ⭐⭐ | ⭐⭐ | Double | ❌ No |
| Attitude | ⭐⭐⭐⭐ | ⭐⭐⭐ | Complex | ❌ No |

## Migration Guide

### Current Setup (Position Mode)
- Uses GoTo service
- Open-loop position targets
- Works but less responsive

### Recommended Setup (Full State Mode)
1. Set `# Control Mode Recommendation for Formation Containment Control

## Executive Summary

**Best Approach: Use `_cmd_full_state_changed` (Full State Control)**

Your formation controller outputs accelerations `[a_x, a_y, a_z, Ω]` where:
- `[a_x, a_y, a_z]` are linear accelerations in the world frame
- `Ω` is the yaw rate

The **Full State Control** mode is the optimal choice because it directly accepts accelerations, providing maximum control authority and best tracking performance for aggressive maneuvers.

## Available Control Modes Comparison

### 1. Full State Control (`_cmd_full_state_changed`) ⭐ **RECOMMENDED**

**Topic:** `/{prefix}{id}/cmd_full_state`  
**Message Type:** `crazyflie_interfaces/FullState`

**Advantages:**
- ✅ Directly accepts accelerations (matches controller output)
- ✅ Maximum control authority
- ✅ Best for aggressive maneuvers and precise tracking
- ✅ Provides feedforward acceleration inputs
- ✅ Supports full 6-DOF control

**Requirements:**
- Position: from current state feedback
- Velocity: from current state feedback
- Acceleration: from controller output `[a_x, a_y, a_z]`
- Yaw: from current state feedback
- Angular rates: `[0, 0, Ω]` where `Ω` is from controller output

**When to Use:**
- When you need maximum control authority
- For aggressive maneuvers
- When controller outputs accelerations (your case)
- For best tracking performance

### 2. Velocity Control (`_cmd_hover_changed`)

**Topic:** `/{prefix}{id}/cmd_hover`  
**Message Type:** `crazyflie_interfaces/Hover`

**Advantages:**
- ✅ Simpler than full state
- ✅ Good for smooth trajectories
- ✅ Maintains altitude control

**Disadvantages:**
- ❌ Requires single integration of accelerations to get velocities
- ❌ Less control authority than full state
- ❌ No feedforward acceleration

**When to Use:**
- For smooth, less aggressive maneuvers
- When you can tolerate integration errors

### 3. Position Control (`_cmd_position_changed`)

**Topic:** `/{prefix}{id}/cmd_position`  
**Message Type:** `geometry_msgs/PoseStamped` (via GoTo service)

**Advantages:**
- ✅ Simple high-level control
- ✅ Good for waypoint following

**Disadvantages:**
- ❌ Requires double integration of accelerations (accumulates errors)
- ❌ Least control authority
- ❌ No feedforward terms
- ❌ Slower response

**When to Use:**
- For simple waypoint navigation
- When precise dynamics control is not needed
- Currently used in your code (open-loop position targets)

### 4. Attitude Control (`_cmd_vel_legacy_changed`)

**Topic:** `/{prefix}{id}/cmd_vel_legacy`  
**Message Type:** `geometry_msgs/Twist`

**Disadvantages:**
- ❌ Requires complex conversion from accelerations to roll/pitch/thrust
- ❌ Not suitable for your controller output format
- ❌ Low-level control, harder to tune

**When to Use:**
- Only if you need direct attitude control
- Not recommended for your use case

## Implementation

The code has been updated to support all three main modes:
- `position`: Uses GoTo service (current default)
- `velocity`: Integrates accelerations to velocities
- `full_state`: Uses full state control (recommended)

### Configuration

Set the control mode via ROS2 parameter:

```yaml
# In your launch file or parameter file
formation_containment_node:
  ros__parameters:
    control_mode: "full_state"  # Options: "position", "velocity", "full_state"
```

Or via command line:
```bash
ros2 run formation_containment_control formation_containment_node --ros-args -p control_mode:=full_state
```

### How Full State Control Works

1. **Controller Output:** `[a_x, a_y, a_z, Ω]`
   - Accelerations from SGASMC controller
   - Yaw rate from controller

2. **State Feedback Required:**
   - Current position `[x, y, z]` from odometry
   - Current velocity `[vx, vy, vz]` from odometry
   - Current yaw `ψ` from odometry

3. **Full State Message Construction:**
   ```python
   full_state.pose.position = [x, y, z]           # From state feedback
   full_state.twist.linear = [vx, vy, vz]         # From state feedback
   full_state.acc = [a_x, a_y, a_z]               # From controller
   full_state.pose.orientation = quaternion(yaw)  # From state feedback
   full_state.twist.angular = [0, 0, Ω]           # From controller
   ```

## Performance Comparison

| Mode | Control Authority | Tracking Performance | Integration Required | Feedforward |
|------|------------------|---------------------|---------------------|-------------|
| Full State | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | None | ✅ Yes |
| Velocity | ⭐⭐⭐ | ⭐⭐⭐ | Single | ❌ No |
| Position | ⭐⭐ | ⭐⭐ | Double | ❌ No |
| Attitude | ⭐⭐⭐⭐ | ⭐⭐⭐ | Complex | ❌ No |

## Migration Guide

### Current Setup (Position Mode)
- Uses GoTo service
- Open-loop position targets
- Works but less responsive

### Recommended Setup (Full State Mode)
1. Set `control_mode: "full_state"` in parameters
2. Ensure state feedback is available (odometry)
3. Controller will automatically use full state commands
4. Better tracking and responsiveness

## Notes

⚠️ **Important:** Once you send a streaming setpoint (full_state, velocity, or hover), the Crazyflie switches to low-level control mode. You cannot use high-level commands like `goTo()` or `land()` until you reconnect or reset the drone.

## Conclusion

For your formation containment controller that outputs accelerations, **Full State Control is the best choice** because:
1. It directly uses your controller's acceleration outputs
2. It provides maximum control authority
3. It offers the best tracking performance
4. It includes feedforward acceleration terms

The implementation is ready to use - just set `control_mode: "full_state"` in your parameters!

` in parameters
2. Ensure state feedback is available (odometry)
3. Controller will automatically use full state commands
4. Better tracking and responsiveness

## Notes

⚠️ **Important:** Once you send a streaming setpoint (full_state, velocity, or hover), the Crazyflie switches to low-level control mode. You cannot use high-level commands like `goTo()` or `land()` until you reconnect or reset the drone.

## Conclusion

For your formation containment controller that outputs accelerations, **Full State Control is the best choice** because:
1. It directly uses your controller's acceleration outputs
2. It provides maximum control authority
3. It offers the best tracking performance
4. It includes feedforward acceleration terms

The implementation is ready to use - just set `control_mode: "full_state"` in your parameters!

