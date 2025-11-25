# FormationState.msg
# State of the entire formation

std_msgs/Header header

# Number of agents
uint8 num_leaders
uint8 num_followers

# Formation status
bool formation_achieved
bool containment_achieved
bool collision_free

# Convex hull vertices (leader positions)
geometry_msgs/Point[] convex_hull_vertices

# Formation errors
float64 max_tracking_error
float64 max_containment_error
float64 min_inter_agent_distance

