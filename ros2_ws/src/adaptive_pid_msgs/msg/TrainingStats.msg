# Aggregated statistics published by pid_controller_node and rl_agent_node,
# consumed by logging_node (for rosbag2/CSV recording) and
# visualization_node (for live dashboards).

std_msgs/Header header
string source              # "pid_controller_node" or "rl_agent_node"
float64 tracking_error
float64 reward             # only meaningful when source == "rl_agent_node"
float64 kp
float64 ki
float64 kd
