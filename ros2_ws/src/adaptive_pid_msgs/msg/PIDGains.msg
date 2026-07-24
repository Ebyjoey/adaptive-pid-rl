# Current PID gain triple.
# Published by rl_agent_node (adapted gains) or reference/config at startup
# (initial gains); subscribed by pid_controller_node.

std_msgs/Header header
float64 kp
float64 ki
float64 kd
