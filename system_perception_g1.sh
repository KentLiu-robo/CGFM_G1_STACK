#!/bin/bash

# Runs on the Jetson mounted on G1: lidar driver + SLAM + domain_bridge only.
# No control, no planning, no RViz -- see system_planner_g1.sh for the other
# half (planning + G1 motion control), which runs on a separate machine on
# the same LAN.
#
# Same domain isolation as system_real_robot_g1.sh: robot's native DDS floods
# domain 0, so the autonomy stack lives on domain 1. domain_bridge_g1.yaml
# bridges domain 1 (here) -> domain 2 (planner machine), one-way.
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

source ./install/setup.bash

ros2 launch vehicle_simulator system_g1_perception.launch "$@"
