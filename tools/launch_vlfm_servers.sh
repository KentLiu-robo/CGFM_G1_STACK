#!/usr/bin/env bash
# Launches the VLM model servers vlfm_bridge depends on: GroundingDINO,
# BLIP2ITM, MobileSAM, YOLOv7, YOLO-World. These are plain Flask processes
# (not ROS2 nodes) -- start this BEFORE `ros2 launch vlfm_bridge
# vlfm_bridge.launch.py` and wait for all ports to open (BLIP2ITM's first
# run downloads weights from huggingface and can take a couple of minutes).
#
# 2026-09-18: run_vlfm_pipeline.py's actual detector is now YOLO-World
# (open-vocabulary -- unlike YOLOv7 it isn't limited to the 80 COCO classes,
# so targets like "fan"/"trash can" work again). GroundingDINO/SAM/YOLOv7
# are kept running here for the older vlfm_navigator_node.py architecture
# and for manual comparisons, but the pipeline script only needs BLIP2ITM +
# YOLO-World.
#
# Verified working command (2026-09-17): GroundingDINO+MobileSAM load in
# ~15-20s; BLIP2ITM+YOLOv7 take ~60-90s longer (bigger models / first-run
# download); YOLO-World loads in ~5s but its first ever `set_classes()`
# call triggers an `ultralytics` "AutoUpdate" that installs a `clip` package
# and downloads a ~340MB CLIP checkpoint (one-time, ~25s) -- this also
# bumped Pillow 9.5.0->11.3.0 in the vlfm conda env; verified this didn't
# break GroundingDINO/BLIP2ITM imports, but if something upstream ever
# pins Pillow<10 again, that's why. GPU memory: ~5GB per pair -- split
# across both GPUs below so a single 24GB card isn't loaded with all of
# them at once.
set -eo pipefail

VLFM_REPO="${VLFM_REPO:?set VLFM_REPO to your checkout of the vlfm repository}"
VLFM_PYTHON="${VLFM_PYTHON:-python}"

export MOBILE_SAM_CHECKPOINT="${VLFM_REPO}/data/mobile_sam.pt"
export GROUNDING_DINO_CONFIG="${VLFM_REPO}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
export GROUNDING_DINO_WEIGHTS="${VLFM_REPO}/data/groundingdino_swint_ogc.pth"
export CLASSES_PATH="${VLFM_REPO}/vlfm/vlm/classes.txt"
export GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-12181}"
export BLIP2ITM_PORT="${BLIP2ITM_PORT:-12182}"
export SAM_PORT="${SAM_PORT:-12183}"
export YOLOV7_PORT="${YOLOV7_PORT:-12184}"
export YOLO_WORLD_PORT="${YOLO_WORLD_PORT:-12185}"

mkdir -p /tmp/vlfm_servers_logs
cd "$VLFM_REPO"

# -u PYTHONPATH: this dev machine's shell often has ROS2's python3.12
# site-packages on PYTHONPATH (from sourcing /opt/ros/*/setup.bash), which
# corrupts the vlfm conda env's own python3.9 import resolution if inherited.
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$VLFM_PYTHON" -m vlfm.vlm.grounding_dino --port "$GROUNDING_DINO_PORT" \
  > /tmp/vlfm_servers_logs/gdino.log 2>&1 &
disown
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$VLFM_PYTHON" -m vlfm.vlm.blip2itm --port "$BLIP2ITM_PORT" \
  > /tmp/vlfm_servers_logs/blip2itm.log 2>&1 &
disown
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 "$VLFM_PYTHON" -m vlfm.vlm.sam --port "$SAM_PORT" \
  > /tmp/vlfm_servers_logs/sam.log 2>&1 &
disown
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 "$VLFM_PYTHON" -m vlfm.vlm.yolov7 --port "$YOLOV7_PORT" \
  > /tmp/vlfm_servers_logs/yolov7.log 2>&1 &
disown
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$VLFM_PYTHON" -m vlfm.vlm.yolo_world --port "$YOLO_WORLD_PORT" \
  > /tmp/vlfm_servers_logs/yolo_world.log 2>&1 &
disown

echo "Launched. Logs in /tmp/vlfm_servers_logs/. Waiting for all ports..."
for port in "$GROUNDING_DINO_PORT" "$BLIP2ITM_PORT" "$SAM_PORT" "$YOLOV7_PORT" "$YOLO_WORLD_PORT"; do
  until timeout 1 bash -c "echo > /dev/tcp/127.0.0.1/$port" 2>/dev/null; do
    sleep 2
  done
  echo "port $port up"
done
echo "All VLM servers ready."
