#!/bin/bash
# G1 base-navigation run flow on the planner machine (verified 2026-09-30).
# Jetson side stays as in QUICKSTART_G1.md (perception in domain 1 + domain_bridge).
#
#   ./g1_nav.sh up            bridge + planner + rviz (bridge starts DISABLED, robot does not move)
#   ./g1_nav.sh status        containers, FSM, topic rates (read-only)
#   ./g1_nav.sh start         FSM -> 501 (3-DoF-waist regular mode)   [robot standing on the floor]
#   ./g1_nav.sh enable        forward cmd_vel to the robot            [then click a Waypoint in RViz]
#   ./g1_nav.sh stop          stop forwarding, robot keeps standing   (= disable)
#   ./g1_nav.sh sit           StandUp2Squat
#   ./g1_nav.sh damp          soft e-stop: robot goes LIMP (asks for confirmation)
#   ./g1_nav.sh restart-planner | restart-rviz   (safe: no SDK bridge inside)
#   ./g1_nav.sh up-vlfm       like up, planner in VLFM mode (no far_planner, no pathFollower;
#                             src/vlfm_bridge/scripts/run_vlfm_pipeline_g1.py drives /way_point + /cmd_vel)
#   ./g1_nav.sh restart-planner-vlfm
#   ./g1_nav.sh vlm-up | vlm-down   BLIP2-ITM :12182 + YOLO-World :12185 on the host (conda env vlfm_g1)
#   ./g1_nav.sh down          remove planner + rviz (bridge kept: removing it Damp()s the robot)
#   ./g1_nav.sh down-bridge   remove the SDK bridge (asks: robot squatting / on the gantry?)
#   ./g1_nav.sh ros2 ...      any ros2 command in the bridge container, e.g. ./g1_nav.sh ros2 topic hz /cmd_vel
#
# Why the settings below (details: QUICKSTART_G1.md "运动控制"):
#   * enp4s0 has 10.1.1.101 AND 192.168.123.165 -> bind SDK DDS to the IP, and restrict
#     ROS/Fast DDS to it (docker/fastdds_devmachine_robotnet.xml), else nothing gets through.
#   * The SDK bridge runs in its own container: its driver Damp()s the robot on exit, so
#     the planner (start_sdk_bridge:=false) can be restarted freely while the robot stands.
#   * 3-DoF-waist G1 walks under SDK velocity only in FSM 501 (not 500 / remote's 802).

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROBOT_IP_LOCAL=192.168.123.165
IMAGE=autonomy_stack:jazzy
WS=/workspace/autonomy_stack
FASTDDS_XML=$WS/docker/fastdds_devmachine_robotnet.xml
ROBOT_CONFIG=${ROBOT_CONFIG:-unitree/unitree_g1_vlfm}   # 0.25 m/s, forward only
VLFM_DIR=${VLFM_DIR:-$HOME/Taowen/vlfm_g1}
VLFM_PY=${VLFM_PY:-$HOME/anaconda3/envs/vlfm_g1/bin/python}
VLM_LOG_DIR=${VLM_LOG_DIR:-/tmp/g1_vlm}
ROS_ENV="source /opt/ros/jazzy/setup.bash && source $WS/install/setup.bash && export ROS_DOMAIN_ID=2 RMW_IMPLEMENTATION=rmw_fastrtps_cpp"

running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]; }

# ros2 command inside the bridge container
g1() {
  running g1_bridge || { echo "g1_bridge is not running (./g1_nav.sh up)"; return 1; }
  docker exec -it g1_bridge bash -c "$ROS_ENV && $*"
}

up_bridge() {
  if running g1_bridge; then echo "g1_bridge already running (left untouched)"; return; fi
  docker rm -f g1_bridge >/dev/null 2>&1 || true
  docker run -d --name g1_bridge --network host --ipc host \
    -v "$SCRIPT_DIR":$WS:rw $IMAGE bash -c "$ROS_ENV && \
    ros2 launch unitree_g1_sdk_bridge g1_sdk_control.launch.py \
      network_interface:=$ROBOT_IP_LOCAL enable_on_start:=false" >/dev/null
  echo "g1_bridge started (DISABLED)"
}

# $1 = "vlfm": vlfm is the only /way_point publisher (no far_planner) and starts
# pathFollower itself after its 360 deg scan (see system_planner_g1_vlfm.sh)
up_planner() {
  local args="use_far_planner:=true"
  [ "$1" = vlfm ] && args="use_far_planner:=false autonomyMode:=true start_path_follower:=false"
  docker rm -f autonomy_stack_planner >/dev/null 2>&1 || true
  docker run -d --name autonomy_stack_planner --network host --ipc host \
    -v "$SCRIPT_DIR":$WS:rw -e FASTRTPS_DEFAULT_PROFILES_FILE=$FASTDDS_XML \
    -e ROBOT_CONFIG_PATH=$ROBOT_CONFIG $IMAGE bash -c "$ROS_ENV && \
    ros2 launch vehicle_simulator system_g1_planner.launch $args \
      start_sdk_bridge:=false network_interface:=$ROBOT_IP_LOCAL" >/dev/null
  echo "autonomy_stack_planner started (${1:-nav} mode, config $ROBOT_CONFIG, no SDK bridge inside)"
}

vlm_up() {
  mkdir -p "$VLM_LOG_DIR"
  local m port
  for m in blip2itm:12182 yolo_world:12185; do
    port=${m#*:}
    if ss -ltn | grep -q ":$port "; then echo "${m%%:*} already listening on :$port"; continue; fi
    (cd "$VLFM_DIR" && env -u PYTHONPATH PYTHONPATH="$VLFM_DIR" nohup "$VLFM_PY" -m vlfm.vlm.${m%%:*} \
      --port "$port" >"$VLM_LOG_DIR/${m%%:*}.log" 2>&1 &)
    echo "${m%%:*} starting on :$port (log $VLM_LOG_DIR/${m%%:*}.log, ~30-60 s to load)"
  done
}

vlm_down() { pkill -f "vlfm.vlm.(blip2itm|yolo_world)" && echo "VLM servers stopped" || echo "no VLM servers running"; }

up_rviz() {
  DISPLAY=${DISPLAY:-:1} xhost +local:root >/dev/null
  docker rm -f autonomy_stack_rviz >/dev/null 2>&1 || true
  docker run -d --name autonomy_stack_rviz --network host --ipc host \
    -e DISPLAY=${DISPLAY:-:1} -e QT_X11_NO_MITSHM=1 -e FASTRTPS_DEFAULT_PROFILES_FILE=$FASTDDS_XML \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "$SCRIPT_DIR":$WS:rw $IMAGE bash -c "$ROS_ENV && \
    cd $WS && ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz" >/dev/null
  echo "autonomy_stack_rviz started on DISPLAY=${DISPLAY:-:1}"
}

fsm() {  # read-only GetFsmId / GetFsmMode
  docker run --rm --network host $IMAGE timeout 20 python3 -c '
import json
import unitree_sdk2py.core.channel as ch
ch.ChannelConfigHasInterface = """<?xml version="1.0" encoding="UTF-8" ?><CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface address="'$ROBOT_IP_LOCAL'" priority="default" multicast="default"/></Interfaces></General></Domain></CycloneDDS>"""
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.loco.g1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID as A, ROBOT_API_ID_LOCO_GET_FSM_MODE as B
ChannelFactoryInitialize(0, "enp4s0"); c = LocoClient(); c.SetTimeout(3.0); c.Init()
code, data = c._Call(A, json.dumps({}))
print("FSM id:", json.loads(data)["data"] if code == 0 else "?", "(code %s; 501 = walk under cmd_vel, 802 = remote walk-run)" % code)
' 2>&1 | grep FSM || echo "FSM query failed (robot net / SDK?)"
}

status() {
  docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -E "g1_bridge|autonomy_stack_(planner|rviz)" || echo "no containers"
  fsm
  if running autonomy_stack_planner; then
    docker exec autonomy_stack_planner bash -c "$ROS_ENV && for t in /state_estimation /registered_scan /terrain_map /way_point /path /cmd_vel; do \
      printf '%-18s' \$t; r=\$(timeout -s INT 5 ros2 topic hz \$t 2>&1 | grep average | tail -1); echo \"\${r:-NO DATA}\"; done"
  fi
  for p in 12182 12185; do ss -ltn | grep -q ":$p " && echo "VLM :$p listening"; done
  running g1_bridge && docker logs g1_bridge 2>&1 | grep -E "ENABLED|DISABLED" | tail -1 | sed 's/.*\]: /bridge: /'
}

confirm() { read -r -p "$1 [yes/NO] " a; [ "$a" = "yes" ]; }

case "$1" in
  up)              up_bridge; up_planner; up_rviz; echo "--- wait ~15 s, then: ./g1_nav.sh status" ;;
  up-vlfm)         up_bridge; up_planner vlfm; up_rviz; echo "--- wait ~15 s, then: ./g1_nav.sh status" ;;
  up-bridge)       up_bridge ;;
  restart-planner) up_planner ;;
  restart-planner-vlfm) up_planner vlfm ;;
  vlm-up)          vlm_up ;;
  vlm-down)        vlm_down ;;
  restart-rviz)    up_rviz ;;
  status)          status ;;
  start)           g1 'ros2 service call /g1_sdk_bridge/start std_srvs/srv/Trigger' ;;
  enable)          g1 'ros2 service call /g1_sdk_bridge/enable std_srvs/srv/SetBool "{data: true}"' ;;
  stop|disable)    g1 'ros2 service call /g1_sdk_bridge/enable std_srvs/srv/SetBool "{data: false}"' ;;
  sit)             g1 'ros2 service call /g1_sdk_bridge/sit std_srvs/srv/Trigger' ;;
  damp)            confirm "DAMP: the robot goes LIMP. Is it on the gantry or squatting?" \
                     && g1 'ros2 service call /g1_sdk_bridge/damp std_srvs/srv/Trigger' ;;
  down)            docker rm -f autonomy_stack_planner autonomy_stack_rviz >/dev/null 2>&1 || true
                   echo "planner + rviz removed; g1_bridge kept (use down-bridge only with the robot squatting / on the gantry)" ;;
  down-bridge)     confirm "Removing the bridge Damp()s the robot. Is it squatting or on the gantry?" \
                     && docker rm -f g1_bridge >/dev/null && echo "g1_bridge removed" ;;
  ros2)            shift; g1 "ros2 $*" ;;
  *)               sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
