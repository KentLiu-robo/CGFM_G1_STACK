#!/bin/bash
. "$(cd "$(dirname "$0")" && pwd)/env.sh"
# Stop the navigation stack (dev machine + Jetson sensors). Idempotent. Also kills a stack that was started by hand.
# usage: teardown_nav_stack.sh [--all]      --all also stops the 5 VLM model servers (they are slow to start, so default = keep them)
# Nothing here can move the robot. NOTE: it does NOT make the robot lie down or stand up.
W=/tmp/go2_stack; PIDF=$W/pids
J="ssh -o BatchMode=yes -o ConnectTimeout=6 ${JETSON_USER}@${JETSON_IP}"
[ -f $PIDF ] && for p in $(cat $PIDF); do kill -INT -- "-$p" 2>/dev/null; done
timeout 30 $J 'pkill -INT -f "[m]sg_MID360_launch.py"; sleep 2; pkill -f "[l]ivox_ros_driver2_node"; pkill -f "[r]ealsense_stream_server.py"; true' 2>/dev/null
sleep 4
[ -f $PIDF ] && { for p in $(cat $PIDF); do kill -KILL -- "-$p" 2>/dev/null; done; rm -f $PIDF; }
# anything left over (e.g. started by hand in other terminals)
pkill -f "[u]nitree_webrtc_ros/lib/unitree_webrtc_ros/unitree_control" 2>/dev/null
pkill -f "[l]aser_mapping_node|[f]eature_extraction_node|[i]mu_preintegration_node|[a]rize_slam|[s]ensor_scan_generation|[t]errainAnalysis|[l]ocalPlanner" 2>/dev/null
pkill -f "[p]ose_udp_rela[y]\.py" 2>/dev/null; pkill -f "[w]aypoint_udp_rela[y]\.py" 2>/dev/null; pkill -f "[c]mdvel_udp_rela[y]\.py" 2>/dev/null
pkill -9 -x pathFollower 2>/dev/null
[ "$1" = "--all" ] && { pkill -f 'vlfm[.]vlm[.]'; sleep 3; }
[ -n "${VLFM_REPO:-}" ] && rmdir "$VLFM_REPO/lockfiles" 2>/dev/null
sleep 2
echo "teardown: dev stack procs left=$(pgrep -af 'laser_mappin[g]|unitree_contro[l]|udp_rela[y]|arize_sla[m]|localPlanne[r]|terrainAnalysi[s]|sensor_scan_generatio[n]' | wc -l)  pathFollower=$(pgrep -x pathFollower | wc -l)  jetson livox/realsense left=$(timeout 20 $J 'pgrep -af "[l]ivox_ros_driver2|[r]ealsense_stream_server" | wc -l' 2>/dev/null)  VLM servers running=$(pgrep -af 'vlfm[.]vlm[.][a-z0-9_]* --port' | wc -l)"
