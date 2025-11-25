# ControllerState.msg
# State of an individual agent's controller

std_msgs/Header header

# Agent identification
uint8 agent_id
string agent_type  # "leader" or "follower"

# Position state [x, y, z, yaw]
geometry_msgs/Pose current_pose
geometry_msgs/Pose desired_pose

# Velocity state
geometry_msgs/Twist current_velocity
geometry_msgs/Twist desired_velocity

# Error states
float64[4] tracking_error      # [ex, ey, ez, e_yaw]
float64[4] tracking_error_dot  # [ex_dot, ey_dot, ez_dot, e_yaw_dot]

# Sliding surface
float64[4] sliding_surface     # sigma_c

# Adaptive gain
float64 adaptive_gain_kc       # K_c(t)

# Control output
float64[4] control_input       # [ax, ay, az, omega]

# Collision avoidance
float64 min_neighbor_distance
bool collision_warning

