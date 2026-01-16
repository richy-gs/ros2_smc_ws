# Full State Control: Why Position and Velocity Are Required

## Your Question

> "If I am just sending acceleration commands, how will it act if I don't send Position and Velocity data?"

## Short Answer

**You CANNOT use Full State Control without Position and Velocity!** They are **REQUIRED**, not optional. The Crazyflie uses them as reference states for its internal controller.

## Detailed Explanation

### How Crazyflie Full State Control Works

When you send a `FullState` message to the Crazyflie, it contains:

```
FullState {
    position: [x, y, z]        ← REQUIRED: Reference position
    velocity: [vx, vy, vz]    ← REQUIRED: Feedforward velocity
    acceleration: [ax, ay, az] ← PRIMARY: Control input (from your controller)
    orientation: quaternion    ← REQUIRED: Reference orientation
    angular_rates: [ωx, ωy, ωz] ← Feedforward angular rates
}
```

### What Each Component Does

1. **Position `[x, y, z]`** (REQUIRED)
   - **Purpose:** Reference state for the internal position controller
   - **What happens if wrong:** The drone thinks it's at a different location
   - **Impact:** Large tracking errors, potential instability
   - **Source:** Must come from state feedback (odometry)

2. **Velocity `[vx, vy, vz]`** (REQUIRED)
   - **Purpose:** Feedforward term for smoother tracking
   - **What happens if wrong:** Overshoot/undershoot, oscillations
   - **Impact:** Degraded tracking performance
   - **Source:** Best from state feedback, can be integrated from acceleration (less accurate)

3. **Acceleration `[ax, ay, az]`** (PRIMARY CONTROL INPUT)
   - **Purpose:** Feedforward control input (from your SGASMC controller)
   - **What happens if wrong:** Direct control error
   - **Impact:** Immediate tracking error
   - **Source:** From your formation controller output

4. **Orientation & Angular Rates**
   - **Purpose:** Reference attitude and feedforward angular rates
   - **Source:** From state feedback (yaw) and controller (yaw rate)

### Why Position and Velocity Are Required

The Crazyflie's internal controller uses a **cascaded control structure**:

```
Your FullState Command
    ↓
[Position Reference] → Position Controller → [Velocity Reference]
    ↓                                              ↓
[Velocity Reference] → Velocity Controller → [Acceleration Reference]
    ↓                                              ↓
[Acceleration Feedforward] → Attitude Controller → [Motor Commands]
```

**Without position:**
- The position controller has no reference to track
- The drone doesn't know where it should be
- It will try to maintain zero position error from an unknown reference

**Without velocity:**
- No feedforward term for smooth motion
- The controller must rely only on position error
- Results in overshoot and oscillations

### What Happens If You Don't Send Position/Velocity?

If you send `acceleration = [1.0, 0.0, 0.0]` but `position = [0, 0, 0]` and `velocity = [0, 0, 0]`:

1. **The drone thinks it's at position [0, 0, 0]**
   - Even if it's actually at [5, 2, 1]
   - The position controller will try to keep it at [0, 0, 0]

2. **The acceleration command conflicts with position reference**
   - Acceleration says "accelerate forward"
   - Position says "stay at origin"
   - Result: Conflicting commands, poor tracking

3. **No velocity feedforward**
   - Controller must estimate velocity from position changes
   - Causes delay and overshoot

### Current Implementation

The code now:

1. **Uses state feedback (odometry) for position/velocity** (preferred)
   ```python
   # From odometry callbacks
   current_state = self.leader_states[idx]      # [x, y, z, yaw]
   current_velocity = self.leader_velocities[idx]  # [vx, vy, vz, vyaw]
   ```

2. **Falls back to integration if state feedback unavailable**
   ```python
   # If no odometry, integrate acceleration
   estimated_velocity = last_control[:3] * dt
   ```
   ⚠️ **Warning:** Integration accumulates errors over time!

3. **Uses controller output for acceleration**
   ```python
   full_state.acc.x = control[0]  # From your SGASMC controller
   full_state.acc.y = control[1]
   full_state.acc.z = control[2]
   ```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Formation Controller (Your Code)                        │
│  Output: [a_x, a_y, a_z, Ω]                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  State Feedback (Odometry)                             │
│  - Position: [x, y, z, yaw]                            │
│  - Velocity: [vx, vy, vz, vyaw]                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Full State Message Construction                        │
│  - position: from odometry                              │
│  - velocity: from odometry (or integrated)             │
│  - acceleration: from controller                        │
│  - orientation: from odometry                           │
│  - angular_rates: from controller                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Crazyflie Internal Controller                          │
│  Uses ALL components for control                        │
└─────────────────────────────────────────────────────────┘
```

### Best Practices

1. **Always use state feedback (odometry)**
   - Subscribe to `/cf{id}/odom` or `/cf{id}/pose`
   - Use actual position/velocity from sensors
   - Most accurate option

2. **If state feedback unavailable:**
   - Use integration as temporary fallback
   - Integrate: `v = v_prev + a * dt`
   - ⚠️ Errors accumulate over time
   - Consider using position control mode instead

3. **Ensure state feedback is available before using full_state mode**
   ```python
   if not has_state_feedback:
       self.get_logger().warn("No state feedback - full state control degraded")
   ```

### Comparison: With vs Without State Feedback

| Scenario | Position Source | Velocity Source | Result |
|----------|----------------|----------------|--------|
| **Ideal** | Odometry | Odometry | ✅ Best tracking, smooth motion |
| **Acceptable** | Odometry | Integrated | ⚠️ Good tracking, slight delay |
| **Poor** | Integrated | Integrated | ❌ Accumulating errors, drift |
| **Invalid** | Zero/Unknown | Zero/Unknown | ❌ Wrong reference, instability |

### Conclusion

**You MUST send position and velocity with full state control!**

- Position and velocity are **reference states**, not optional
- They come from **state feedback (odometry)**, not from your controller
- Your controller only provides **accelerations**
- Without accurate position/velocity, tracking performance degrades significantly

The current implementation handles this correctly by:
1. Using odometry for position/velocity (when available)
2. Falling back to integration (with warnings)
3. Using your controller output for accelerations

**Always ensure state feedback is available for best performance!**











