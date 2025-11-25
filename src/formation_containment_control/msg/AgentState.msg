# AgentState.msg
# Complete state of an agent (quadrotor)

std_msgs/Header header

# Agent identification
uint8 agent_id
string agent_type  # "leader" or "follower"

# Position in inertial frame [x, y, z]
geometry_msgs/Point position

# Velocity in inertial frame [vx, vy, vz]
geometry_msgs/Vector3 velocity

# Orientation (quaternion)
geometry_msgs/Quaternion orientation

# Angular velocity [roll_rate, pitch_rate, yaw_rate]
geometry_msgs/Vector3 angular_velocity

# Euler angles [roll, pitch, yaw]
float64 roll
float64 pitch
float64 yaw

# Yaw rate
float64 yaw_rate

