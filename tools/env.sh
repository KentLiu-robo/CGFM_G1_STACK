# Central configuration of the helper scripts in tools/. Every value can be overridden from the environment
# or from configs/paths.env (copy configs/paths.example.env). Sourced by the other scripts; not meant to be run.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_here/../configs/paths.env" ] && . "$_here/../configs/paths.env"
export JETSON_IP="${GO2_JETSON_IP:-192.168.3.18}"          # on-board computer (Livox driver + RealSense server)
export JETSON_USER="${GO2_JETSON_USER:-unitree}"
export WS_SETUP="${GO2_WS_SETUP:-$HOME/GO2_STACK_dev_ws/install/setup.bash}"   # base navigation stack (ROS 2)
export WIFIENV="${GO2_ROS_ENV_SCRIPT:-}"                    # optional: DDS / CycloneDDS environment script
export WEBRTC_WS_SETUP="${GO2_WEBRTC_WS_SETUP:-$HOME/unitree_webrtc_ws/install/setup.bash}"
export WEBRTC_VENV="${GO2_WEBRTC_VENV:-$HOME/unitree_venv/bin/activate}"
export VLFM_PYTHON="${VLFM_PYTHON:-python}"                 # interpreter with vlfm, torch, open_clip installed
# Detector servers the stack scripts wait for: CGFM only needs YOLO-World (:12185). CGFM_DETECTORS=all additionally
# requires/launches the other four VLFM servers (:12181-12184) via launch_vlfm_servers.sh.
if [ "${CGFM_DETECTORS:-yolo_world}" = all ]; then
  export DET_RE=':1218[1-5] '; export DET_N=5; export DET_LAUNCH=launch_vlfm_servers.sh
else
  export DET_RE=':12185 ';     export DET_N=1; export DET_LAUNCH=launch_yolo_world_server.sh
fi
