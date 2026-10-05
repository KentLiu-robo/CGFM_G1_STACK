#!/bin/bash
. "$(cd "$(dirname "$0")" && pwd)/env.sh"
# Bring the whole Go2 navigation stack up in WIRELESS mode and LEAVE IT RUNNING (the main script is NOT started).
#   Jetson (Livox + RealSense) -> dev: SLAM -> unitree_control (WebRTC, auto-restart once) -> scan generation, terrain(+ext), localPlanner
#   -> health check. Nothing here can move the robot: no pathFollower, no /cmd_vel publisher.
# usage: bringup_nav_stack.sh [--no-vlm]        (VLM servers are started only if their ports are not already up)
# Logs / pids: /tmp/go2_stack/   Stop with: teardown_nav_stack.sh [--all]   Check with: health_check.sh
# The robot must be STANDING and must not be moved after this starts (SLAM initialises where it stands).
TOOLS=$(cd "$(dirname "$0")" && pwd); W=/tmp/go2_stack; mkdir -p $W; PIDF=$W/pids
J="ssh -o BatchMode=yes -o ConnectTimeout=6 ${JETSON_USER}@${JETSON_IP}"
STK="source /opt/ros/jazzy/setup.bash; source $WS_SETUP${WIFIENV:+; source $WIFIENV >/dev/null 2>&1}"
LP=$(dirname "$WS_SETUP")/local_planner
NOVLM=0; [ "$1" = "--no-vlm" ] && NOVLM=1

if pgrep -f '[l]aser_mapping_node|[l]ocalPlanner|[u]nitree_webrtc_ros/lib/unitree_webrtc_ros/unitree_control' >/dev/null; then
  echo "A navigation stack is already running on this machine -> not starting a second one. Use health_check.sh, or teardown_nav_stack.sh first."; exit 3
fi
: > $PIDF
start() { local name=$1; shift; setsid bash -c "$*" > "$W/$name.log" 2>&1 & echo $! >> $PIDF; }
WEBRTC="unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH; source /opt/ros/jazzy/setup.bash; source $WEBRTC_WS_SETUP${WIFIENV:+; source $WIFIENV >/dev/null 2>&1}; source $WEBRTC_VENV; exec ros2 launch unitree_webrtc_ros unitree_control.launch.py"

RELAY_UNIT="${GO2_RELAY_UNIT:-go2-relay}"   # Jetson systemd unit bridging the Go2 network (192.168.123.x) to WiFi; "none" = skip this check
NTP=$($J 'timedatectl show -p NTPSynchronized --value' 2>/dev/null); RELAY=$($J "systemctl is-active $RELAY_UNIT" 2>/dev/null)
echo "pre-flight: Jetson NTP=$NTP $RELAY_UNIT=$RELAY  pathFollower=$(pgrep -x pathFollower | wc -l)  detector ports=$(ss -ltn | grep -cE "$DET_RE")/$DET_N  disk free=$(df -h $HOME | tail -1 | awk '{print $4}')  Go2 ping=$(ping -c2 -W2 192.168.123.161 2>&1 | grep -oE '[0-9]+% packet loss')"
[ "$NTP" = yes ] && { [ "$RELAY_UNIT" = none ] || [ "$RELAY" = active ]; } || { echo "PRE-FLIGHT FAILED (Jetson unreachable, clock not synced, or the relay unit is down) -- not starting"; exit 1; }

if [ "$NOVLM" = 0 ] && [ "$(ss -ltn | grep -cE "$DET_RE")" -lt "$DET_N" ]; then
  echo "detector servers: starting in the background (1-3 min)"
  setsid bash -c "timeout 420 $TOOLS/$DET_LAUNCH" > $W/vlm.log 2>&1 &
fi
timeout 30 ssh -o BatchMode=yes ${JETSON_USER}@${JETSON_IP} 'mkdir -p ~/logs
setsid nohup bash -c "source /opt/ros/foxy/setup.bash; source /home/unitree/livox/ws_livox/install/setup.bash; source /home/unitree/autonomy_stack_go2/ros_env_wifi.sh; exec ros2 launch /home/unitree/livox/ws_livox/src/livox_ros_driver2/launch_ROS2/msg_MID360_launch.py" > ~/logs/livox_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
setsid nohup python3 ~/realsense_stream_server.py > ~/logs/realsense_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
echo jetson services started'
sleep 10
start slam "$STK; exec ros2 launch arise_slam_mid360 arize_slam.launch.py"
start webrtc "$WEBRTC"
if ! timeout 40 bash -c "until grep -q 'Subscribed to' $W/webrtc.log 2>/dev/null; do sleep 1; done"; then
  echo "WebRTC not connected -> restarting unitree_control once (known intermittent fault, just restart it)"
  pkill -9 -f "[u]nitree_webrtc_ros/lib/unitree_webrtc_ros/unitree_control"; sleep 2
  start webrtc "$WEBRTC"
  timeout 40 bash -c "until grep -q 'Subscribed to' $W/webrtc.log 2>/dev/null; do sleep 1; done" || echo "WebRTC STILL NOT CONNECTED after a restart"
fi
grep -hm1 "Successfully connected" $W/webrtc.log 2>/dev/null | sed 's/.*\]: /WebRTC: /'
echo "SLAM settling 25 s (robot must not be moved)..."; sleep 25
start scangen  "$STK; exec ros2 launch sensor_scan_generation sensor_scan_generation.launch"
sleep 3
start terrain  "$STK; exec ros2 launch terrain_analysis terrain_analysis.launch"
start terrainx "$STK; exec ros2 launch terrain_analysis_ext terrain_analysis_ext.launch checkTerrainConn:=true"
start localpl  "$STK; exec ros2 run local_planner localPlanner --ros-args --params-file $LP/share/local_planner/config/unitree/unitree_go2_slow.yaml -p pathFolder:=$LP/share/local_planner/paths -p vehicleLength:=0.6 -p vehicleWidth:=0.5 -p sensorOffsetX:=0.2 -p sensorOffsetY:=0.0 -p twoWayDrive:=false -p laserVoxelSize:=0.05 -p terrainVoxelSize:=0.2 -p useTerrainAnalysis:=true -p checkObstacle:=true -p checkRotObstacle:=false -p adjacentRange:=3.5 -p obstacleHeightThre:=0.05 -p groundHeightThre:=0.05 -p costHeightThre1:=0.1 -p costHeightThre2:=0.05 -p useCost:=false -p slowPathNumThre:=5 -p slowGroupNumThre:=1 -p pointPerPathThre:=2 -p minRelZ:=-0.4 -p maxRelZ:=0.3 -p maxSpeed:=0.2 -p dirWeight:=0.02 -p dirThre:=90.0 -p dirToVehicle:=false -p pathScale:=0.875 -p minPathScale:=0.675 -p pathScaleStep:=0.1 -p pathScaleBySpeed:=true -p minPathRange:=0.8 -p pathRangeStep:=0.6 -p pathRangeBySpeed:=true -p pathCropByGoal:=true -p autonomyMode:=true -p autonomySpeed:=0.2 -p joyToSpeedDelay:=2.0 -p joyToCheckObstacleDelay:=5.0 -p goalClearRange:=0.35 -p goalBehindRange:=0.35 -p freezeAng:=90.0 -p freezeTime:=0.0"
echo "nav stack settling 20 s..."; sleep 20
if [ "$NOVLM" = 0 ]; then timeout 300 bash -c 'until [ "$(ss -ltn | grep -cE "$DET_RE")" -ge "$DET_N" ]; do sleep 3; done' || echo "VLM servers did not all come up in 5 min (see /tmp/go2_stack/vlm.log)"; fi
echo; echo "================ HEALTH ================"
bash $TOOLS/health_check.sh
echo "=== stack LEFT RUNNING (main script NOT started) ==="
