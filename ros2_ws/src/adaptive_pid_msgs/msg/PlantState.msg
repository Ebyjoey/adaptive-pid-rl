# Full plant state as published by plant_node on /state.
# Includes both the (possibly noisy, as observed by the controller) angle
# and the disturbance estimate, so downstream nodes (rl_agent_node,
# logging_node, visualization_node) don't each need to re-derive it.

std_msgs/Header header
float64 theta            # rad, 0 = upright equilibrium
float64 theta_dot         # rad/s
float64 control_effort    # N*m, most recent commanded torque (post-saturation)
float64 disturbance_estimate  # N*m, from the residual disturbance observer
