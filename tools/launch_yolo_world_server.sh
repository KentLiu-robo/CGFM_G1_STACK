#!/usr/bin/env bash
# Starts ONLY the YOLO-World detector server (vlfm.vlm.yolo_world, open-vocabulary) that the CGFM pipelines use.
# (launch_vlfm_servers.sh starts the whole VLFM server set; CGFM does not need the other four.)
#   VLFM_REPO    checkout of the vlfm repository (required)
#   VLFM_PYTHON  interpreter of the env where vlfm is installed (default: python)
#   YOLO_WORLD_PORT (default 12185), CGFM_DET_GPU (default 0)
set -eo pipefail
VLFM_REPO="${VLFM_REPO:?set VLFM_REPO to your checkout of the vlfm repository}"
VLFM_PYTHON="${VLFM_PYTHON:-python}"
PORT="${YOLO_WORLD_PORT:-12185}"
mkdir -p /tmp/cgfm_logs
cd "$VLFM_REPO"
# -u PYTHONPATH: a shell that sourced /opt/ros/*/setup.bash has ROS' python on PYTHONPATH, which breaks a conda env
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="${CGFM_DET_GPU:-0}" YOLO_WORLD_PORT="$PORT" \
  "$VLFM_PYTHON" -m vlfm.vlm.yolo_world --port "$PORT" > /tmp/cgfm_logs/yolo_world.log 2>&1 &
disown
echo "Launched YOLO-World on :$PORT (log /tmp/cgfm_logs/yolo_world.log), waiting for the port..."
until timeout 1 bash -c "echo > /dev/tcp/127.0.0.1/$PORT" 2>/dev/null; do sleep 2; done
echo "YOLO-World ready."
