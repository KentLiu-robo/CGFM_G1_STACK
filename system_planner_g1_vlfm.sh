#!/bin/bash

# Same as system_planner_g1.sh (runs inside the autonomy_stack:jazzy container
# on this machine), configured for VLFM semantic navigation
# (src/vlfm_bridge/scripts/run_vlfm_pipeline_g1.py):
#   * use_far_planner:=false        -- vlfm is the only /way_point publisher
#   * autonomyMode:=true            -- localPlanner steers toward /way_point
#   * start_path_follower:=false    -- run_vlfm_pipeline_g1.py owns /cmd_vel
#                                      during its initial 360deg scan and only
#                                      then starts pathFollower itself
#   * ROBOT_CONFIG_PATH=unitree/unitree_g1_vlfm -- forward-only, 0.25 m/s
# Normally started via `./g1_nav.sh up-vlfm` (adds the Fast DDS whitelist profile and
# start_sdk_bridge:=false, the SDK bridge living in its own g1_bridge container).
# If run by hand: the container needs
#   -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/autonomy_stack/docker/fastdds_devmachine_robotnet.xml
# and pass start_sdk_bridge:=false when g1_bridge is already running.
export ROS_DOMAIN_ID=2
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROBOT_CONFIG_PATH=unitree/unitree_g1_vlfm

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

source ./install/setup.bash

# network_interface:=<nic or IP> on 192.168.123.x -- this machine: the IP 192.168.123.165
# (enp4s0 has two addresses; binding by name picks the wrong one).
#   ./system_planner_g1_vlfm.sh network_interface:=192.168.123.165
ros2 launch vehicle_simulator system_g1_planner.launch \
  use_far_planner:=false autonomyMode:=true start_path_follower:=false "$@"
