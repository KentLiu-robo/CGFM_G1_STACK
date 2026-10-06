#!/bin/bash

# Runs on this machine (off-robot, inside the autonomy_stack:jazzy container --
# see docker/run.sh): terrain analysis, local_planner, far_planner, RViz, and
# G1 motion control over unitree_sdk2 (DDS). Consumes /state_estimation +
# /registered_scan bridged in from the Jetson on G1 (system_perception_g1.sh)
# via domain_bridge_g1.yaml. /cmd_vel stays local -- g1_sdk_bridge talks to
# the robot directly over its 192.168.123.x network, it does NOT cross the
# domain_bridge.
#
# Must be on ROS_DOMAIN_ID=2 to match domain_bridge_g1.yaml's to_domain, and
# on the same LAN as the Jetson (multicast DDS discovery has to reach both
# machines) *and* have an interface on the robot's 192.168.123.x network
# (for g1_sdk_bridge's DDS control channel, independent of ROS_DOMAIN_ID).
export ROS_DOMAIN_ID=2
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

source ./install/setup.bash

# route_planner_config:=indoor|outdoor, use_far_planner:=false to disable route
# planning and drive via waypoint/smart-joystick mode only,
# network_interface:=<nic or IP> on 192.168.123.x (default eth0). This machine's
# enp4s0 has two addresses, so pass the IP (binding by name picks 10.1.1.101). Example:
#   ./system_planner_g1.sh route_planner_config:=indoor network_interface:=192.168.123.165
ros2 launch vehicle_simulator system_g1_planner.launch "$@" &

sleep 1
ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz
