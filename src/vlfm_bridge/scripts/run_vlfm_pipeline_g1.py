"""G1 port of the GO2 master script (VLFM_GO2_STACK run_vlfm_pipeline.py):
orchestrates the vlfm-native-mapping -> G1_STACK obstacle-avoidance loop on
the real Unitree G1, with LIVE target-object switching.

Architecture:
  RGB-D (TCP 6000, Jetson  --+
  realsense_stream_server)   +--> ObstacleMap/ValueMap (vlfm native) --> best
  pose (UDP relay from       |    frontier --> /way_point (UDP relay) -->
  /state_estimation) --------+    localPlanner (obstacle-avoidance path) -->
                                  pathFollower (autonomyMode=true) --> /cmd_vel
                                  --> g1_sdk_bridge --> unitree_sdk2 --> robot

What differs from the GO2 version:
  * ROS side runs INSIDE the autonomy_stack_planner docker container (this
    host is Ubuntu 20.04, no native Jazzy). Relays / pathFollower / ros2 CLI
    calls all go through `docker exec`; the container uses --network host, so
    the 127.0.0.1 UDP relays work unchanged. Container processes run as root,
    so they must also be KILLED via `docker exec` -- a host-side pkill
    silently fails on them. _pkill() therefore verifies the kill.
  * Stop semantics: a humanoid must NEVER be stopped by lying down / damping
    (damp = it collapses). Stop is:
      1. PRIMARY: kill pathFollower (and the scan relay) -> /cmd_vel stops;
         g1_sdk_bridge zeroes velocity by itself after cmd_timeout (0.5 s).
      2. SECONDARY: /g1_sdk_bridge/enable false -> bridge stops forwarding
         and sends an explicit zero; the robot keeps standing/balancing.
    /g1_sdk_bridge/damp is left to the human as the emergency option.
  * Camera pose uses the FULL 6-DoF /state_estimation pose (lidar and D435i
    are both rigidly mounted on the head, so the lidar pose already contains
    the gait's head pitch/roll sway) composed with a fixed sensor->camera
    extrinsic (G1_CAM_* env vars). The GO2 version only used (x, y, yaw) plus
    a fixed camera height, which is fine on a quadruped but not on a humanoid.
    G1_VLFM_POSE_MODE=planar restores the GO2 behaviour for comparison.

INITIAL SCAN: before pathFollower is started, the script rotates the robot a
full 360deg closed-loop by commanding /cmd_vel yaw rate directly through
cmdvel_udp_relay.py (linear velocity forced to 0, |wz| clamped, silent when
the stream stops). Only after the scan is pathFollower armed.

SAFETY:
  - The robot must already be standing AND the bridge enabled
    (remote: lock-stand, on the floor; then /g1_sdk_bridge/start -> FSM 501
    and /g1_sdk_bridge/enable true, see QUICKSTART_G1.md) -- done by
    the operator, step by step, before this script. This script never
    stands the robot up or enables the bridge itself.
  - Requires typed "ARM" confirmation before starting real motion.
  - Bounded runtime by default (--max-seconds), independent of Ctrl-C.
  - A human with the physical remote must be present for every real run.
  - Startup sends an initial /way_point at the robot's OWN current
    position before arming, so the first cycle can't lurch toward a stale
    default goal.

Usage (planner container must be running system_planner_g1_vlfm.sh):
    cd ~/Taowen/vlfm_g1
    env -u PYTHONPATH PYTHONPATH=$(pwd) ~/anaconda3/envs/vlfm_g1/bin/python \\
        ~/Taowen/G1_STACK/src/vlfm_bridge/scripts/run_vlfm_pipeline_g1.py \\
        --target chair --max-seconds 300 [--record]

While running, type in this terminal (Enter to submit):
    target <name>     change the exploration target, e.g. "target sofa"
    stop               graceful stop (same as Ctrl-C): kill pathFollower,
                       disable the bridge, robot stays standing
"""
import argparse
import csv
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.value_map import ValueMap
from vlfm.utils.geometry_utils import rho_theta
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.yolo_world import YOLOWorldClient, _caption_to_classes

JETSON_HOST = os.environ.get("G1_JETSON_HOST", "192.168.123.164")
CAMERA_PORT = 6000
POSE_UDP_PORT = 8765
WAYPOINT_UDP_PORT = 8766
CMDVEL_UDP_PORT = 8767
CMDVEL_LOG_UDP_PORT = 8768   # pose_udp_relay.py forwards /cmd_vel here (read-only) for cmd_vel.csv
TRAJECTORY_HZ = 20.0         # trajectory.csv sampling rate
WAYPOINT_MIN_REPUBLISH_DELTA_M = 0.05
LOOP_INTERVAL_S = 1.0
# Freshness limits, measured on THIS machine's clock at receive time (the Jetson
# clock is ~10 min off, so message stamps can't be used). Older data is reported
# as missing, so a stalled stream can never pair an old image with a new pose.
CAMERA_MAX_AGE_S = 0.5      # stream is 30 Hz
POSE_MAX_AGE_S = 0.3        # /state_estimation is 50 Hz
DATA_LOSS_STOP_S = 3.0      # motion runs stop if camera/pose stay missing this long
COMPOSITE_VIDEO_FPS = 1.5   # playback speed of the auto-generated composite video (ticks/s)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# ---------------------------------------------------------------------------
# Camera geometry. NOT CALIBRATED YET for G1 -- the defaults below are rough
# placeholders; measure them (calibrate_camera_floor.py) and export the env
# vars before any real run. The startup log prints the values in use.
#
# "full" pose mode (default): camera pose = T_map_sensor (full 6-DoF
# /state_estimation) @ T_sensor_camera. The extrinsic is expressed in the
# SLAM sensor (lidar) frame, camera frame is vlfm's (x forward, y left, z up).
# Heights are then in the SLAM map frame, whose origin is the sensor's pose at
# SLAM start, so the floor sits at G1_FLOOR_Z (= minus the sensor's height
# above the floor when SLAM initialised).
#
# "planar" mode (GO2 behaviour): only (x, y, yaw) of the pose is used, the
# camera sits G1_CAM_HEIGHT above the floor, heights are relative to it.
# ---------------------------------------------------------------------------
POSE_MODE = os.environ.get("G1_VLFM_POSE_MODE", "full")
CAM_TX = _env_float("G1_CAM_TX", 0.05)          # m, sensor frame
CAM_TY = _env_float("G1_CAM_TY", 0.0)
CAM_TZ = _env_float("G1_CAM_TZ", -0.10)
CAM_PITCH_DOWN_DEG = _env_float("G1_CAM_PITCH_DOWN_DEG", 30.0)  # + = looking down
CAM_ROLL_DEG = _env_float("G1_CAM_ROLL_DEG", 0.0)
CAM_YAW_DEG = _env_float("G1_CAM_YAW_DEG", 0.0)
FLOOR_Z = _env_float("G1_FLOOR_Z", -1.20)       # full mode: floor z in the SLAM map frame
CAMERA_HEIGHT_M = _env_float("G1_CAM_HEIGHT", 1.10)  # planar mode only
CAMERA_CALIBRATED = os.environ.get("G1_CAM_CALIBRATED", "0") == "1"

FLOOR_MARGIN_M = _env_float("G1_FLOOR_MARGIN", 0.15)  # GO2 used 0.10; gait sway on a humanoid
OBSTACLE_TOP_M = 1.5    # obstacles counted up to this height above the floor (G1 is ~1.3 m tall)
MIN_DEPTH = 0.3
MAX_DEPTH = 3.0
_FLOOR_REF = FLOOR_Z if POSE_MODE == "full" else -CAMERA_HEIGHT_M
MIN_HEIGHT = _FLOOR_REF + FLOOR_MARGIN_M
MAX_HEIGHT = _FLOOR_REF + OBSTACLE_TOP_M
MAP_SIZE = 800
PIXELS_PER_METER = 20
FRONTIER_SEARCH_RADIUS = 1.0
DINO_CONF_THRESHOLD = 0.25
MAX_BOX_AREA_FRAC = 0.7
EDGE_CROP_FRAC = 0.08

RES_DIR = os.environ.get("G1_VLFM_RES_DIR", os.path.expanduser("~/Taowen/RES_G1"))
# YOLO-World (vlfm/vlm/yolo_world.py) is open-vocabulary -- unlike YOLOv7 it
# isn't limited to the 80 COCO classes, so "fan"/"trash can" work again.
# Same ' . '-separated caption convention as the earlier GroundingDINO setup:
# a few synonym phrases per target helps recall.
DET_SYNONYMS = {
    "computer": "computer . computer tower . PC . monitor . gaming pc .",
    "fan": "fan . fans . ceiling fan . electric fan . cooling fan .",
    "chair": "chair . office chair . seat .",
    "person": "person . human .",
    "sofa": "sofa . couch .",
    # no bare "bag": it matched a black cloth bag on a tripod (2026-10-02)
    "backpack": "backpack . school bag . rucksack .",
    "school bag": "backpack . school bag . rucksack .",
}

# ROS 2 lives in the planner container (Jazzy, domain 2, Fast DDS -- see
# system_planner_g1_vlfm.sh). Paths below are inside the container, where the
# G1_STACK checkout is mounted at /workspace/autonomy_stack.
PLANNER_CONTAINER = os.environ.get("G1_PLANNER_CONTAINER", "autonomy_stack_planner")
_WS = "/workspace/autonomy_stack"
ROS_ENV_SOURCE = (
    f"source /opt/ros/jazzy/setup.bash && source {_WS}/install/setup.bash && "
    "export ROS_DOMAIN_ID=2 && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
)
LOCAL_PLANNER_YAML = f"{_WS}/install/local_planner/share/local_planner/config/unitree/unitree_g1_vlfm.yaml"
PATHFOLLOWER_MATCH = "install/local_planner/lib/local_planner/pathFollower"
POSE_RELAY_SCRIPT = f"{_WS}/src/vlfm_bridge/scripts/pose_udp_relay.py"
WAYPOINT_RELAY_SCRIPT = f"{_WS}/src/vlfm_bridge/scripts/waypoint_udp_relay.py"
CMDVEL_RELAY_SCRIPT = f"{_WS}/src/vlfm_bridge/scripts/cmdvel_udp_relay.py"


def _in_container(bash_cmd: str) -> list:
    return ["docker", "exec", PLANNER_CONTAINER, "bash", "-c", f"{ROS_ENV_SOURCE} && {bash_cmd}"]


def _ros_cli(cmd: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(_in_container(cmd), capture_output=True, text=True, timeout=timeout)


def _ros_popen(cmd: str, log_path: str = None) -> subprocess.Popen:
    out = open(log_path, "a") if log_path else subprocess.DEVNULL
    return subprocess.Popen(_in_container(f"exec {cmd}"), stdout=out, stderr=out)


def _container_pids(pattern: str) -> list:
    # pgrep run directly by docker exec (no bash -c wrapper), so its own
    # command line can't match the pattern.
    r = subprocess.run(["docker", "exec", PLANNER_CONTAINER, "pgrep", "-f", pattern],
                       capture_output=True, text=True, timeout=10.0)
    return r.stdout.split()


def _pkill(pattern: str) -> bool:
    """Kill matching processes inside the planner container and verify they
    are gone. Returns False (and prints loudly) if any survived."""
    try:
        subprocess.run(["docker", "exec", PLANNER_CONTAINER, "pkill", "-9", "-f", pattern],
                       capture_output=True, timeout=10.0)
        for _ in range(10):
            if not _container_pids(pattern):
                return True
            time.sleep(0.2)
    except subprocess.TimeoutExpired:
        pass
    print(f"\033[31m[pkill] processes matching {pattern!r} are STILL ALIVE in {PLANNER_CONTAINER}\033[0m")
    return False


def start_pathfollower(autonomy: bool, log_path: str = None) -> None:
    _pkill(PATHFOLLOWER_MATCH)
    time.sleep(1.0)
    mode = "true" if autonomy else "false"
    cmd = (
        f"ros2 run local_planner pathFollower --ros-args "
        f"--params-file {LOCAL_PLANNER_YAML} "
        f"-p useSerialPort:=false -p realRobot:=false -p autonomyMode:={mode} "
        # twoWayDrive:=false: the camera faces forward, so driving backward
        # means vlfm's perception never sees the direction of travel (GO2
        # finding, 2026-09-18). Speeds/yaw rate match the GO2 VLFM runs:
        # perception only updates at 1 Hz.
        f"-p sensorOffsetX:=0.05 -p sensorOffsetY:=0.0 -p twoWayDrive:=false "
        f"-p maxSpeed:=0.25 -p autonomySpeed:=0.25 -p maxYawRate:=30.0 "
        f"-p dirDiffThre:=1.0"
    )
    _ros_popen(cmd, log_path=log_path)
    time.sleep(2.0)


def call_bridge_enable(enable: bool) -> bool:
    """/g1_sdk_bridge/enable: false stops /cmd_vel forwarding and sends an
    explicit zero velocity -- the robot keeps standing. Never damp here."""
    data = "true" if enable else "false"
    try:
        result = _ros_cli(
            f'ros2 service call /g1_sdk_bridge/enable std_srvs/srv/SetBool "{{data: {data}}}"', timeout=8.0)
        out = result.stdout.strip() or result.stderr.strip()
        print(f"[enable {data}] {out[-300:]}")
        return result.returncode == 0 and "success=True" in out
    except subprocess.TimeoutExpired:
        print(f"[enable {data}] service call timed out")
        return False


def check_preconditions(log) -> bool:
    """Everything that must hold before ARM: container up, bridge node alive,
    no pathFollower / far_planner running (they would fight over /cmd_vel and
    /way_point)."""
    ok = True
    try:
        nodes = _ros_cli("ros2 node list", timeout=15.0).stdout
    except subprocess.TimeoutExpired:
        log(f"PRECHECK FAIL: ros2 node list timed out in {PLANNER_CONTAINER}")
        return False
    if "g1_sdk_bridge" not in nodes:
        log("PRECHECK FAIL: g1_sdk_bridge node not found (is system_planner_g1_vlfm.sh running?)")
        ok = False
    if "localPlanner" not in nodes:
        log("PRECHECK FAIL: localPlanner node not found")
        ok = False
    if _container_pids(PATHFOLLOWER_MATCH):
        log("PRECHECK FAIL: a pathFollower is already running -- use system_planner_g1_vlfm.sh "
            "(start_path_follower:=false), this script starts its own")
        ok = False
    if _container_pids("lib/far_planner/far_planner"):  # not bare "far_planner": matches use_far_planner:= args
        log("PRECHECK FAIL: far_planner is running and would also publish /way_point")
        ok = False
    return ok


class WaypointUdpSender:
    FMT = "<3d"

    def __init__(self, port: int = WAYPOINT_UDP_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dest = ("127.0.0.1", port)

    def send(self, x: float, y: float, z: float) -> None:
        self._sock.sendto(struct.pack(self.FMT, x, y, z), self._dest)


class CmdVelUdpSender:
    """Yaw-rate-only command stream to cmdvel_udp_relay.py (which forces linear
    velocity to 0, clamps |wz|, and goes silent when this stops sending)."""
    FMT = "<d"

    def __init__(self, port: int = CMDVEL_UDP_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dest = ("127.0.0.1", port)

    def send(self, wz: float) -> None:
        self._sock.sendto(struct.pack(self.FMT, float(wz)), self._dest)


class PoseUdpClient:
    FMT = "<8d"
    SIZE = struct.calcsize(FMT)

    def __init__(self, port: int = POSE_UDP_PORT):
        self._lock = threading.Lock()
        self._xy_yaw = None
        self._t_map_sensor = None
        self._t_rx = None
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", port))
        self._sock.settimeout(1.0)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            if len(data) != self.SIZE:
                continue
            _stamp, x, y, z, qx, qy, qz, qw = struct.unpack(self.FMT, data)
            yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
            t_map_sensor = np.eye(4)
            t_map_sensor[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
            t_map_sensor[:3, 3] = (x, y, z)
            with self._lock:
                self._xy_yaw = (np.array([x, y]), yaw)
                self._t_map_sensor = t_map_sensor
                self._t_rx = time.monotonic()

    def _fresh(self) -> bool:  # caller holds self._lock
        return self._t_rx is not None and time.monotonic() - self._t_rx <= POSE_MAX_AGE_S

    def age(self):
        with self._lock:
            return None if self._t_rx is None else time.monotonic() - self._t_rx

    def xy_yaw(self):
        """(xy, yaw), or None if no sample within POSE_MAX_AGE_S."""
        with self._lock:
            return self._xy_yaw if self._fresh() else None

    def pose(self):
        """(xy, yaw, 4x4 T_map_sensor) from the same sample, or None if no
        sample within POSE_MAX_AGE_S."""
        with self._lock:
            if not self._fresh():
                return None
            return self._xy_yaw[0], self._xy_yaw[1], self._t_map_sensor

    def close(self) -> None:
        self._stop.set()


class _RealSenseStreamClient:
    INTRINSICS_FMT = ">4fII f"
    INTRINSICS_SIZE = struct.calcsize(INTRINSICS_FMT)

    def __init__(self, host: str, port: int = 6000):
        self._host, self._port = host, port
        self._lock = threading.Lock()
        self._color = None
        self._depth = None
        self._t_rx = None
        self._intrinsics = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _recvall(sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self._host, self._port))
                intr_bytes = self._recvall(sock, self.INTRINSICS_SIZE)
                if intr_bytes is not None:
                    fx, fy, ppx, ppy, width, height, depth_scale = struct.unpack(
                        self.INTRINSICS_FMT, intr_bytes
                    )
                    with self._lock:
                        self._intrinsics = {"fx": fx, "fy": fy, "ppx": ppx, "ppy": ppy,
                                             "width": width, "height": height, "depth_scale": depth_scale}
                while not self._stop.is_set():
                    header = self._recvall(sock, 8)
                    if header is None:
                        break
                    color_len, depth_len = struct.unpack(">II", header)
                    color_bytes = self._recvall(sock, color_len)
                    depth_bytes = self._recvall(sock, depth_len)
                    if color_bytes is None or depth_bytes is None:
                        break
                    color = cv2.imdecode(np.frombuffer(color_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    depth = cv2.imdecode(np.frombuffer(depth_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    if color is None or depth is None:
                        continue  # corrupt frame: drop it, keep the connection
                    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._color = color
                        self._depth = depth
                        self._t_rx = time.monotonic()
                sock.close()
            except Exception:  # noqa: BLE001 -- any failure must reconnect, never kill this thread
                pass
            if not self._stop.is_set():
                time.sleep(1.0)

    def intrinsics(self):
        with self._lock:
            return self._intrinsics

    def age(self):
        with self._lock:
            return None if self._t_rx is None else time.monotonic() - self._t_rx

    def latest(self):
        """(rgb, depth), or (None, None) if no frame within CAMERA_MAX_AGE_S."""
        with self._lock:
            if self._t_rx is None or time.monotonic() - self._t_rx > CAMERA_MAX_AGE_S:
                return None, None
            return self._color, self._depth

    def close(self) -> None:
        self._stop.set()


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _build_sensor_to_camera_transform() -> np.ndarray:
    """T_sensor_camera, camera frame = vlfm's (x forward, y left, z up).
    A positive pitch about +y tilts the camera's x axis DOWN, hence
    pitch = +CAM_PITCH_DOWN_DEG."""
    t = np.eye(4)
    t[:3, :3] = _rpy_to_rot(np.deg2rad(CAM_ROLL_DEG), np.deg2rad(CAM_PITCH_DOWN_DEG), np.deg2rad(CAM_YAW_DEG))
    t[:3, 3] = (CAM_TX, CAM_TY, CAM_TZ)
    return t


_SENSOR_TO_CAMERA = _build_sensor_to_camera_transform()


def get_camera_transform(pose) -> np.ndarray:
    """tf_camera_to_episodic for vlfm, from a PoseUdpClient.pose() sample."""
    xy, yaw, t_map_sensor = pose
    if POSE_MODE == "full":
        return t_map_sensor @ _SENSOR_TO_CAMERA
    # planar (GO2 behaviour): yaw-only body pose at z=0, camera offset/tilt on top
    c, s = np.cos(yaw), np.sin(yaw)
    body_to_episodic = np.eye(4)
    body_to_episodic[0, 0], body_to_episodic[0, 1] = c, -s
    body_to_episodic[1, 0], body_to_episodic[1, 1] = s, c
    body_to_episodic[0, 3], body_to_episodic[1, 3] = xy[0], xy[1]
    cam = _SENSOR_TO_CAMERA.copy()
    cam[2, 3] = 0.0  # height is carried by MIN/MAX_HEIGHT relative to the camera
    return body_to_episodic @ cam


def normalize_depth(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32) * depth_scale
    normalized = np.clip((depth_m - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0.0, 1.0)
    normalized[depth_raw == 0] = 0.0
    edge_px = int(depth_raw.shape[1] * EDGE_CROP_FRAC)
    if edge_px > 0:
        normalized[:, :edge_px] = 0.0
        normalized[:, -edge_px:] = 0.0
    return normalized


def box_area_frac(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


TARGET_LOCK_WINDOW = 5         # sliding window of the last N perception ticks...
TARGET_LOCK_MIN_HITS = 3       # ...lock the target once at least this many of them saw it
TARGET_VERIFY_TOP_K = 2        # second-stage check on the K highest-confidence YOLO boxes per tick:
TARGET_VERIFY_MIN_ITM = 0.25   # BLIP2 ITM cosine of the box crop vs "a photo of a <target>." must reach this
                               # (G1 2026-10-02, back view of a backpack: real 0.33-0.34, tripod cloth bag 0.14,
                               #  empty floor 0.16, black cloth 0.21; ~80 ms per crop)
TARGET_LOCK_MIN_CONF = 0.30    # a tick counts as a lock hit only if its best detection reaches this
                               # (G1 2026-10-02: real backpack 0.64-0.68, tripod cloth bag false positive 0.38-0.51; back view of the real backpack only ~0.44 -> confidence alone cannot separate them, so this floor is kept loose)
TARGET_STOP_RADIUS_M = 0.5     # G1: 0.5 (GO2: 0.25) -- tall robot, a close target leaves the head camera view / gets inflated as an obstacle
TARGET_APPROACH_STANDOFF_M = 0.6  # /way_point after a lock is this far SHORT of the target (on the robot->target
                                  # line): the target itself is an obstacle to localPlanner, aiming at its centre
                                  # makes the final approach unpredictable
TARGET_GOAL_REACHED_M = 0.35   # ...and reaching that stand-off point also counts as FOUND
WALK_CHECK_INTERVAL_S = 0.1    # after a lock the stop check runs at 10 Hz (frames are still recorded at 1 Hz)
TARGET_DEPTH_PATCH_PX = 4      # half-width of the median-depth sampling patch around the box center
TARGET_MAX_DIST_M = 2.5        # a detection whose box-center depth is beyond this is 'Found but too far': not counted (far depth-based xy is too inaccurate)


def _tick_walk(idx, camera, pose_client, goal_xy, run_dir, video_writer):
    """Post-lock tick: NO detector / BLIP2 / map updates / frontier logic --
    just read the pose for the distance-to-goal check and keep recording the
    camera view (frame + depth + video) so the run stays reviewable. Returns
    (xy, dist) or None if pose/camera weren't ready."""
    xy_yaw = pose_client.xy_yaw()
    color, depth_raw = camera.latest()
    if xy_yaw is None or color is None or depth_raw is None:
        return None
    xy = xy_yaw[0]
    dist = float(np.hypot(goal_xy[0] - xy[0], goal_xy[1] - xy[1]))
    vis = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{idx:05d}.png"), depth_raw)
    cv2.putText(vis, f"WALKING TO GOAL ({goal_xy[0]:.2f},{goal_xy[1]:.2f}) dist={dist:.2f}m",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(run_dir, "frames", f"tick_{idx:05d}.jpg"), vis)
    if video_writer is not None:
        video_writer.write(vis)
    return xy, dist


def draw_detections(color_bgr, target, score, detections, valid_idxs, crop_scores=None):
    """Draws every YOLO-World box onto a copy of the frame (green if it
    passed the confidence+area filters, gray otherwise) plus the BLIP2ITM
    score, so a run can be visually audited frame-by-frame to see what the
    model actually detected."""
    vis = color_bgr.copy()
    height, width = vis.shape[:2]
    if detections is not None:
        for i, (box, logit, phrase) in enumerate(
            zip(detections.boxes, detections.logits, detections.phrases)
        ):
            x1, y1, x2, y2 = box.numpy() if hasattr(box, "numpy") else box
            pt1 = (int(x1 * width), int(y1 * height))
            pt2 = (int(x2 * width), int(y2 * height))
            ok = i in valid_idxs
            color = (0, 220, 0) if ok else (120, 120, 120)
            label = f"{phrase} {float(logit):.2f}"
            if crop_scores and i in crop_scores:
                # second-stage BLIP2 crop check: green = verified, orange = rejected
                if crop_scores[i] < TARGET_VERIFY_MIN_ITM:
                    color = (0, 140, 255)
                label += f" itm{crop_scores[i]:.2f}"
            cv2.rectangle(vis, pt1, pt2, color, 2 if ok else 1)
            cv2.putText(vis, label, (pt1[0], max(0, pt1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    header = f"target={target!r} blip_score={score:+.3f}"
    cv2.putText(vis, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return vis


def box_center_depth_m(box, depth_raw, depth_scale, width, height):
    """Median depth (m) over a small patch around the box center, or None if
    the patch has no valid depth."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2 * width)
    cy = int((y1 + y2) / 2 * height)
    y0, y1p = max(0, cy - TARGET_DEPTH_PATCH_PX), min(height, cy + TARGET_DEPTH_PATCH_PX + 1)
    x0, x1p = max(0, cx - TARGET_DEPTH_PATCH_PX), min(width, cx + TARGET_DEPTH_PATCH_PX + 1)
    patch = depth_raw[y0:y1p, x0:x1p]
    valid = patch[patch != 0]
    if valid.size == 0:
        return None
    depth_m = float(np.median(valid)) * depth_scale
    return depth_m if depth_m > 0 else None


def filter_by_distance(detections, valid_idxs, depth_raw, depth_scale, width, height, idx, log):
    """Distance gate on detections that already passed the class/confidence/
    area filters: keep only those whose box-center depth is <= TARGET_MAX_DIST_M.
    A far detection ("Found but too far") or one with no valid depth (distance
    can't be verified) is logged and NOT counted, so it can neither add a lock
    hit nor produce a far, inaccurate goal estimate."""
    kept = []
    for i in valid_idxs:
        box = detections.boxes[i]
        box = box.numpy() if hasattr(box, "numpy") else box
        d = box_center_depth_m(box, depth_raw, depth_scale, width, height)
        what = f"{detections.phrases[i]!r} conf={float(detections.logits[i]):.2f}"
        if d is None:
            log(f"  [tick {idx}] Found but no valid depth (distance can't be verified): {what} -- not counted")
        elif d > TARGET_MAX_DIST_M:
            log(f"  [tick {idx}] Found but too far: {what} dist={d:.2f}m > {TARGET_MAX_DIST_M:.1f}m -- not counted")
        else:
            kept.append(i)
    return kept


def estimate_target_xy(box, depth_raw, depth_scale, fx, fy, width, height, tf_camera_to_episodic):
    """Rough world-frame (x, y) of a detected target: box-center pixel's
    median depth (over a small patch, for robustness against single-pixel
    noise/no-return) back-projected through the pinhole model and the
    camera->episodic transform -- deliberately simple (no SAM segmentation
    or point-cloud clustering like vlfm_navigator_node.py's ObjectPointCloudMap),
    per the "just use the underlying depth nav" request. Returns None if the
    patch has no valid depth."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2 * width)
    cy = int((y1 + y2) / 2 * height)
    depth_m = box_center_depth_m(box, depth_raw, depth_scale, width, height)
    if depth_m is None:
        return None
    x_cam = (cx - width / 2) * depth_m / fx
    y_cam = (cy - height / 2) * depth_m / fy
    point_cam = np.array([depth_m, -x_cam, -y_cam, 1.0])  # (forward, left, up), matches get_point_cloud's convention
    point_world = tf_camera_to_episodic @ point_cam
    return point_world[:2]


def _wall_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds")


def _rpy_deg(r: np.ndarray):
    return (float(np.degrees(np.arctan2(r[2, 1], r[2, 2]))),
            float(np.degrees(np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)))),
            float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))))


class RunRecorder:
    """Wall-clock records of a run, for review videos (e.g. cutting the run
    together with a third-person camera):
      events.csv     -- scan steps, ARMED, waypoints, detections, LOCKED, FOUND, stop...
      timeline.csv   -- one row per perception/walk tick (capture time, pose, scores, detection)
      trajectory.csv -- pose at TRAJECTORY_HZ
      cmd_vel.csv    -- every /cmd_vel (forwarded read-only by pose_udp_relay.py)
      first_person_realtime.mp4 + realtime_frames.csv -- full-rate camera (--record-realtime)
    All times are THIS machine's clock (time.time(); the Jetson clock is ~10 min
    off, so message stamps are never used). Best effort: a failing write can
    never affect the run or the stop sequence."""

    TIMELINE_COLS = [
        "tick", "phase", "scan_step", "capture_time_iso", "capture_time_epoch", "logged_time_epoch",
        "t_since_armed_s", "x", "y", "yaw_deg", "blip_score", "frontiers", "best_frontier_x", "best_frontier_y",
        "waypoint_sent", "saw", "conf", "itm", "counted", "hits", "locked", "goal_x", "goal_y",
        "standoff_x", "standoff_y", "dist_to_target_m", "dist_to_standoff_m",
    ]

    def __init__(self, run_dir, pose_client, camera, realtime_video=False, fps=30.0):
        self._dir, self._pose, self._camera = run_dir, pose_client, camera
        self._lock = threading.RLock()  # re-entrant: the SIGINT handler may log while the main thread holds it
        self._stop = threading.Event()
        self._files = []
        self._events = self._csv("events.csv", ["time_iso", "time_epoch", "event", "detail"])
        self._timeline = self._csv("timeline.csv", self.TIMELINE_COLS)
        self._traj = self._csv("trajectory.csv", ["time_iso", "time_epoch", "pose_age_s", "x", "y", "z",
                                                  "roll_deg", "pitch_deg", "yaw_deg"])
        self._cmd = self._csv("cmd_vel.csv", ["time_iso", "time_epoch", "vx", "vy", "wz"])
        self._threads = [threading.Thread(target=self._trajectory_loop, daemon=True),
                         threading.Thread(target=self._cmdvel_loop, daemon=True)]
        self._video = self._frames = None
        if realtime_video:
            self._fps = fps
            self._frames = self._csv("realtime_frames.csv", ["video_frame", "time_iso", "time_epoch"])
            self._threads.append(threading.Thread(target=self._video_loop, daemon=True))
        for t in self._threads:
            t.start()

    def _csv(self, name, cols):
        f = open(os.path.join(self._dir, name), "w", newline="")
        w = csv.writer(f)
        w.writerow(cols)
        f.flush()
        self._files.append(f)
        return f, w

    def _write(self, fw, row):
        try:
            with self._lock:
                fw[1].writerow(row)
                fw[0].flush()
        except Exception:  # noqa: BLE001 -- recording must never break the run
            pass

    def event(self, name, detail="", ts=None):
        ts = time.time() if ts is None else ts
        self._write(self._events, [_wall_iso(ts), f"{ts:.3f}", name, detail])

    def tick(self, row):
        now = time.time()
        cap = row.pop("capture_time", None)
        row.update(logged_time_epoch=f"{now:.3f}",
                   capture_time_iso=_wall_iso(cap) if cap else "",
                   capture_time_epoch=f"{cap:.3f}" if cap else "")
        self._write(self._timeline, [row.get(c, "") for c in self.TIMELINE_COLS])

    def _trajectory_loop(self):
        while not self._stop.wait(1.0 / TRAJECTORY_HZ):
            pose, age = self._pose.pose(), self._pose.age()
            if pose is None:
                continue
            (x, y), _yaw, t_map_sensor = pose
            roll, pitch, yaw = _rpy_deg(t_map_sensor[:3, :3])
            now = time.time()
            self._write(self._traj, [_wall_iso(now), f"{now:.3f}", f"{age:.3f}", f"{x:.3f}", f"{y:.3f}",
                                     f"{t_map_sensor[2, 3]:.3f}", f"{roll:.2f}", f"{pitch:.2f}", f"{yaw:.2f}"])

    def _cmdvel_loop(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", CMDVEL_LOG_UDP_PORT))
            sock.settimeout(0.5)
        except OSError:
            return
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(64)
            except socket.timeout:
                continue
            if len(data) == struct.calcsize("<3d"):
                vx, vy, wz = struct.unpack("<3d", data)
                now = time.time()
                self._write(self._cmd, [_wall_iso(now), f"{now:.3f}", f"{vx:.3f}", f"{vy:.3f}", f"{wz:.3f}"])
        sock.close()

    def _video_loop(self):
        last, n = None, 0
        while not self._stop.is_set():
            color, _ = self._camera.latest()
            age = self._camera.age()
            if color is not None and color is not last:
                last = color
                try:
                    if self._video is None:
                        h, w = color.shape[:2]
                        self._video = cv2.VideoWriter(os.path.join(self._dir, "first_person_realtime.mp4"),
                                                      cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
                    self._video.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                    ts = time.time() - (age or 0.0)
                    self._write(self._frames, [n, _wall_iso(ts), f"{ts:.3f}"])
                    n += 1
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.005)

    def close(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        if self._video is not None:
            self._video.release()
        for f in self._files:
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass


class StdinCommands(threading.Thread):
    """Background reader for 'target <name>' / 'stop' typed into this
    terminal while the main loop runs, so the target object can change
    without restarting the whole pipeline."""

    def __init__(self):
        super().__init__(daemon=True)
        self.q: "queue.Queue[str]" = queue.Queue()

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if line:
                self.q.put(line)

    def poll(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


SCAN_STEPS = 8                 # 8 x 45deg = one full turn, then back at the start heading
SCAN_ROT_KP = 1.2              # rad/s of yaw rate per rad of yaw error
SCAN_ROT_MAX_WZ = 0.5          # rad/s (~29deg/s) -- relay clamps at 0.6 regardless
SCAN_ROT_MIN_WZ = 0.25         # rad/s, Go2's small-command deadband value -- re-check on the G1 loco controller
SCAN_ROT_TOL_DEG = 8.0         # step counts as reached within this, held for 3 loop ticks
SCAN_ROT_LOOP_HZ = 20.0
SCAN_ROT_STEP_TIMEOUT_S = 10.0
SCAN_ROT_RESPONSE_CHECK_S = 2.5    # by now the yaw error must have shrunk by >= MIN_PROGRESS...
SCAN_ROT_MIN_PROGRESS_DEG = 5.0    # ...or the rotation isn't working (wrong sign / robot not
                                   # responding / frozen pose): stop and abort the scan
SCAN_PERCEIVE_TICKS = 2        # perception ticks captured at rest after each step
SCAN_MAX_DRIFT_M = 1.0         # rotation-only must not translate; abort if it does


def _tick_perception(idx, target, camera, pose_client, obstacle_map, value_map,
                      blip2itm, detector, intr, fx, fy, fov, run_dir, log,
                      video_writer=None):
    """One perception+mapping tick, shared by the initial 360-scan and the
    main explore loop: capture a frame, save it, run detection+ITM scoring,
    update the obstacle/value maps, save map snapshots. Returns None if
    pose/camera data wasn't ready yet this tick."""
    blip_query = f"Seems like there is a {target} ahead."
    det_caption = DET_SYNONYMS.get(target, f"{target} .")

    pose = pose_client.pose()
    color, depth_raw = camera.latest()
    if pose is None or color is None or depth_raw is None:
        return None
    t_capture = time.time() - (camera.age() or 0.0)  # when this frame arrived from the Jetson
    xy, yaw, _ = pose
    depth_norm = normalize_depth(depth_raw, intr["depth_scale"])
    tf_camera_to_episodic = get_camera_transform(pose)

    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{idx:05d}.png"), depth_raw)
    depth_vis = cv2.applyColorMap(
        cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        cv2.COLORMAP_JET,
    )
    depth_vis[depth_raw == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{idx:05d}_vis.jpg"), depth_vis)

    # Defense-in-depth against a real failure mode hit live (2026-09-18): a
    # long-running YOLO-World server that's had set_classes() called with
    # many different class lists over a session can end up returning boxes
    # labeled with some OTHER class entirely (e.g. "person"/"tv"/"microwave")
    # even though it was just asked for only `det_classes` -- confirmed via a
    # fresh model instance returning zero detections on the same frame where
    # the long-lived server confidently returned unrelated classes. Restart
    # the server if this recurs; this check just stops a mislabeled box from
    # being silently treated as "found the target" regardless of the cause.
    det_classes = set(_caption_to_classes(det_caption))
    try:
        detections = detector.predict(color, caption=det_caption)
        valid_idxs = [
            i for i, (box, logit, phrase) in enumerate(
                zip(detections.boxes, detections.logits, detections.phrases)
            )
            if phrase in det_classes
            and logit >= DINO_CONF_THRESHOLD and box_area_frac(box) <= MAX_BOX_AREA_FRAC
        ]
    except Exception as e:
        log(f"  [warn] YOLO-World failed: {e}")
        detections, valid_idxs = None, []
    valid_idxs = filter_by_distance(detections, valid_idxs, depth_raw, intr["depth_scale"],
                                    intr["width"], intr["height"], idx, log)
    target_seen = len(valid_idxs) > 0

    # Second stage: YOLO-World confidence alone could not tell a backpack from a
    # black cloth bag on a tripod (overlapping 0.4-0.6 ranges with both the s and
    # l models), but BLIP2 ITM on the box crop separates them clearly.
    crop_scores = {}
    if target_seen:
        crop_query = f"a photo of a {target}."
        h_img, w_img = color.shape[:2]
        for i in sorted(valid_idxs, key=lambda i: -float(detections.logits[i]))[:TARGET_VERIFY_TOP_K]:
            bx = detections.boxes[i].numpy() if hasattr(detections.boxes[i], "numpy") else detections.boxes[i]
            x1, y1 = max(0, int(bx[0] * w_img)), max(0, int(bx[1] * h_img))
            x2, y2 = min(w_img, int(bx[2] * w_img)), min(h_img, int(bx[3] * h_img))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            try:
                crop_scores[i] = float(blip2itm.cosine(np.ascontiguousarray(color[y1:y2, x1:x2]), crop_query))
            except Exception as e:
                log(f"  [warn] BLIP2ITM crop check failed: {e}")
    verified_idxs = [i for i, c in crop_scores.items() if c >= TARGET_VERIFY_MIN_ITM]

    try:
        score = blip2itm.cosine(color, blip_query)
    except Exception as e:
        log(f"  [warn] BLIP2ITM failed: {e}")
        score = 0.0

    det_vis = draw_detections(color_bgr, target, score, detections, valid_idxs, crop_scores)
    cv2.imwrite(os.path.join(run_dir, "frames", f"tick_{idx:05d}.jpg"), det_vis)
    if video_writer is not None:
        video_writer.write(det_vis)

    try:
        obstacle_map.update_map(
            depth=depth_norm, tf_camera_to_episodic=tf_camera_to_episodic,
            min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, fx=fx, fy=fy,
            topdown_fov=fov, explore=True, update_obstacles=True,
        )
        obstacle_map.update_agent_traj(xy, yaw)
        value_map.update_map(
            values=np.array([score]), depth=depth_norm, tf_camera_to_episodic=tf_camera_to_episodic,
            min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, fov=fov,
        )
        value_map.update_agent_traj(xy, yaw)
    except IndexError as e:
        log(f"  [warn] pose outside map bounds: {e}")

    cv2.imwrite(os.path.join(run_dir, "occupancy_map", f"tick_{idx:05d}.png"), obstacle_map.visualize())
    cv2.imwrite(os.path.join(run_dir, "value_map", f"tick_{idx:05d}.png"),
                value_map.visualize(obstacle_map=obstacle_map))

    return {
        "xy": xy, "yaw": yaw, "score": score, "target_seen": target_seen,
        "valid_idxs": valid_idxs, "detections": detections,
        "crop_scores": crop_scores, "verified_idxs": verified_idxs, "t_capture": t_capture,
        "frontiers": obstacle_map.frontiers,
        "depth_raw": depth_raw, "tf_camera_to_episodic": tf_camera_to_episodic,
    }


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def rotate_to_yaw(target_yaw, pose_client, cmdvel_sender, xy0, should_stop):
    """Closed-loop in-place rotation to an absolute SLAM yaw, by commanding the
    yaw rate directly (see cmdvel_udp_relay.py for why the old waypoint-based
    turning was replaced). Yaw error is the median of the last 5 samples so a
    single SLAM glitch can't decide anything. Always leaves a zero command
    behind. Returns (outcome, final_err_rad) with outcome in
    ok | timeout | no_response | drift | stopped | no_pose."""
    dt = 1.0 / SCAN_ROT_LOOP_HZ
    tol = np.deg2rad(SCAN_ROT_TOL_DEG)
    progress_min = np.deg2rad(SCAN_ROT_MIN_PROGRESS_DEG)
    errs = deque(maxlen=5)
    t0 = time.time()
    err0, ok_n, drift_n, err = None, 0, 0, 0.0
    try:
        while True:
            if should_stop():
                return "stopped", err
            py = pose_client.xy_yaw()
            if py is None:
                if time.time() - t0 > 3.0:
                    return "no_pose", err
                time.sleep(dt)
                continue
            xy, yaw = py
            errs.append(_wrap_pi(target_yaw - yaw))
            err = float(np.median(errs))
            if err0 is None:
                err0 = abs(err)
            elapsed = time.time() - t0

            drift_n = drift_n + 1 if float(np.hypot(xy[0] - xy0[0], xy[1] - xy0[1])) > SCAN_MAX_DRIFT_M else 0
            if drift_n >= int(SCAN_ROT_LOOP_HZ * 0.5):
                return "drift", err
            ok_n = ok_n + 1 if abs(err) < tol else 0
            if ok_n >= 3:
                return "ok", err
            if (elapsed >= SCAN_ROT_RESPONSE_CHECK_S and err0 > tol + progress_min
                    and (err0 - abs(err)) < progress_min):
                return "no_response", err
            if elapsed > SCAN_ROT_STEP_TIMEOUT_S:
                return "timeout", err

            wz = float(np.clip(SCAN_ROT_KP * err, -SCAN_ROT_MAX_WZ, SCAN_ROT_MAX_WZ))
            if abs(wz) < SCAN_ROT_MIN_WZ:
                wz = float(np.sign(err)) * SCAN_ROT_MIN_WZ
            cmdvel_sender.send(wz)
            time.sleep(dt)
    finally:
        cmdvel_sender.send(0.0)


def _stop_rotation(cmdvel_sender) -> None:
    """Explicit zero-rate burst, then let the robot come to rest before the
    next camera capture (also lets cmdvel_udp_relay finish its own zero burst
    and go silent -- it must be quiet before pathFollower takes /cmd_vel)."""
    for _ in range(6):
        cmdvel_sender.send(0.0)
        time.sleep(0.05)
    time.sleep(0.6)


def perform_initial_scan(xy0, yaw0, camera, pose_client, cmdvel_sender,
                          obstacle_map, value_map, blip2itm, detector,
                          intr, fx, fy, fov, run_dir, log, target, frame_idx,
                          video_writer=None, should_stop=lambda: False, rec=None):
    """Turn a full 360 degrees in SCAN_STEPS steps before exploring, so the
    obstacle/value maps already cover every direction from the start spot.

    Runs BEFORE pathFollower is started: this function alone owns /cmd_vel
    (via cmdvel_udp_relay.py, yaw rate only), rotating closed-loop to absolute
    headings yaw0 + k*360/SCAN_STEPS and capturing/mapping SCAN_PERCEIVE_TICKS
    frames at rest after each step (sharper than capturing mid-turn). The last
    step brings the robot back to its original heading.

    Aborts (stops rotating, returns early) if the rotation doesn't respond,
    the robot drifts > SCAN_MAX_DRIFT_M, or two steps in a row time out.
    Returns (frame_idx, abort_reason); abort_reason is None only when the full
    turn completed -- the caller must not arm pathFollower otherwise."""
    log(f"Initial 360-degree scan: {SCAN_STEPS} steps of {360 / SCAN_STEPS:.0f}deg, closed-loop yaw control "
        f"(pathFollower not running; tol {SCAN_ROT_TOL_DEG:.0f}deg, max {np.rad2deg(SCAN_ROT_MAX_WZ):.0f}deg/s, "
        f"abort if drift from start exceeds {SCAN_MAX_DRIFT_M}m)...")

    def perceive(step_label):
        nonlocal frame_idx
        for _ in range(SCAN_PERCEIVE_TICKS):
            tick_t0 = time.time()
            frame_idx += 1
            result = _tick_perception(frame_idx, target, camera, pose_client, obstacle_map,
                                       value_map, blip2itm, detector, intr, fx, fy, fov,
                                       run_dir, log, video_writer)
            if result is None:
                frame_idx -= 1
            else:
                drift = float(np.hypot(result["xy"][0] - xy0[0], result["xy"][1] - xy0[1]))
                log(f"  [scan {step_label} tick {frame_idx}] pose=({result['xy'][0]:+.2f},{result['xy'][1]:+.2f}) "
                    f"yaw={np.rad2deg(result['yaw']):+.0f}deg frontiers={len(result['frontiers'])} "
                    f"drift_from_start={drift:.2f}m")
                if rec:
                    rec.tick({"tick": frame_idx, "phase": "scan", "scan_step": step_label.split("/")[0],
                              "capture_time": result["t_capture"], "x": f"{result['xy'][0]:.3f}",
                              "y": f"{result['xy'][1]:.3f}", "yaw_deg": f"{np.rad2deg(result['yaw']):.1f}",
                              "blip_score": f"{result['score']:.3f}", "frontiers": len(result["frontiers"])})
            elapsed = time.time() - tick_t0
            if elapsed < LOOP_INTERVAL_S:
                time.sleep(LOOP_INTERVAL_S - elapsed)

    if rec:
        rec.event("scan_start", f"{SCAN_STEPS} steps of {360 / SCAN_STEPS:.0f} deg from heading {np.rad2deg(yaw0):+.0f} deg")
    perceive(f"0/{SCAN_STEPS} (start heading {np.rad2deg(yaw0):+.0f}deg)")
    timeouts_in_a_row = 0
    for step in range(1, SCAN_STEPS + 1):
        target_yaw = _wrap_pi(yaw0 + step * (2 * np.pi / SCAN_STEPS))
        t_step = time.time()
        if rec:
            rec.event(f"scan_step_{step}_start", f"rotate to {np.rad2deg(target_yaw):+.0f} deg")
        outcome, err = rotate_to_yaw(target_yaw, pose_client, cmdvel_sender, xy0, should_stop)
        if rec:  # rotation commands end here; _stop_rotation then lets the robot settle
            rec.event(f"scan_step_{step}_end", f"{outcome}, final yaw error {np.rad2deg(err):+.1f} deg")
        _stop_rotation(cmdvel_sender)
        log(f"  [scan {step}/{SCAN_STEPS}] rotate to {np.rad2deg(target_yaw):+.0f}deg: {outcome} "
            f"(final yaw error {np.rad2deg(err):+.1f}deg, {time.time() - t_step:.1f}s)")
        if outcome in ("no_response", "drift", "stopped", "no_pose"):
            log(f"  [SCAN ABORTED] rotation outcome {outcome!r} -- robot left stationary.")
            if rec:
                rec.event("scan_aborted", outcome)
            return frame_idx, outcome
        timeouts_in_a_row = timeouts_in_a_row + 1 if outcome == "timeout" else 0
        if timeouts_in_a_row >= 2:
            log("  [SCAN ABORTED] two rotation steps in a row timed out -- robot left stationary.")
            if rec:
                rec.event("scan_aborted", "timeout x2")
            return frame_idx, "timeout x2"
        if step < SCAN_STEPS:
            perceive(f"{step}/{SCAN_STEPS}")
    log("Scan complete (back at the start heading).")
    if rec:
        rec.event("scan_complete")
    return frame_idx, None


def next_run_dir(base_dir: str) -> str:
    path = os.path.join(base_dir, "vlfm_pipeline_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    for sub in ("frames", "occupancy_map", "value_map", "depth"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="chair")
    parser.add_argument("--max-seconds", type=float, default=300.0,
                         help="Hard runtime cap regardless of Ctrl-C (safety net).")
    parser.add_argument("--skip-confirm", action="store_true",
                         help="Skip the typed ARM confirmation (do not use unattended).")
    parser.add_argument("--record", action="store_true",
                         help="Record the first-person camera feed (with detection "
                              "overlay) to first_person.mp4 in the run directory.")
    parser.add_argument("--record-realtime", action="store_true",
                        help="Also record the camera at full rate (~30 fps) to first_person_realtime.mp4, "
                             "with each frame's wall-clock time in realtime_frames.csv (for syncing with "
                             "a third-person video).")
    parser.add_argument("--perception-only", action="store_true",
                         help="No motion at all: no scan, no pathFollower, no cmd_vel relay, "
                              "bridge untouched. Runs the perception/mapping loop and publishes "
                              "/way_point (visible in RViz) -- for calibration and dry runs while "
                              "the robot is carried/pushed.")
    args = parser.parse_args()
    motion = not args.perception_only

    run_dir = next_run_dir(RES_DIR)
    log_path = os.path.join(run_dir, "log.txt")
    log_f = open(log_path, "a")

    def log(msg: str) -> None:
        # Colored in the terminal only (not in the file -- ANSI codes would
        # just be garbage bytes in log.txt): yellow for a tick that actually
        # saw the target, green for the final FOUND/stop line, so both stand
        # out against the frontier-exploration noise while watching a run
        # live.
        if msg.startswith("FOUND "):
            print(f"\033[32m{msg}\033[0m")
        elif ">>> saw " in msg:
            print(f"\033[33m{msg}\033[0m")
        else:
            print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    print("=" * 70)
    if motion:
        print("VLFM full-pipeline master script (G1) -- REAL ROBOT MOTION")
        print("Stop mechanism: kill pathFollower (primary), /g1_sdk_bridge/enable false (secondary).")
        print("The robot stays STANDING when stopped. /g1_sdk_bridge/damp is the human's emergency option.")
        print("Before ARM: robot standing, FSM 501 (/g1_sdk_bridge/start) and bridge enabled (enable true), done by you.")
        print("A human MUST be holding the physical remote right now.")
    else:
        print("VLFM (G1) -- PERCEPTION ONLY: no motion commands of any kind are sent.")
    print("=" * 70)
    log(f"Camera geometry: pose_mode={POSE_MODE} T_sensor_cam xyz=({CAM_TX:+.3f},{CAM_TY:+.3f},{CAM_TZ:+.3f}) "
        f"pitch_down={CAM_PITCH_DOWN_DEG:.1f}deg roll={CAM_ROLL_DEG:.1f}deg yaw={CAM_YAW_DEG:.1f}deg "
        f"floor_z={FLOOR_Z:+.3f} cam_height(planar)={CAMERA_HEIGHT_M:.2f} "
        f"-> obstacle band z in [{MIN_HEIGHT:+.2f}, {MAX_HEIGHT:+.2f}]")
    if not CAMERA_CALIBRATED:
        log("\033[31mWARNING: camera extrinsic/floor NOT calibrated (G1_CAM_CALIBRATED!=1) -- "
            "placeholder values, maps will be wrong.\033[0m")
        if motion:
            log("Refusing to move with an uncalibrated camera. Use --perception-only, or calibrate "
                "and export G1_CAM_CALIBRATED=1.")
            log_f.close()
            return
    if not check_preconditions(log):
        log("Preconditions failed -- nothing started.")
        log_f.close()
        return
    if motion and not args.skip_confirm:
        confirmation = input("Type ARM to proceed, anything else aborts: ").strip()
        if confirmation != "ARM":
            print("Aborted, nothing started.")
            return
    t_arm_confirmed = time.time() if motion else None

    state = {"target": args.target, "stop": False}
    # Re-entrant: request_stop (the SIGINT/SIGTERM handler) runs ON the main thread and
    # takes this lock -- with a plain Lock a Ctrl-C landing while the main loop holds it
    # deadlocked the process (seen as a hang in futex_wait, 2026-10-02).
    state_lock = threading.RLock()

    log("Cleaning up any stale relay/pathFollower processes...")
    _pkill("pose_udp_relay.py")
    _pkill("waypoint_udp_relay.py")
    _pkill("cmdvel_udp_relay.py")
    if not _pkill(PATHFOLLOWER_MATCH):
        log("ABORT: could not kill a stale pathFollower -- nothing started.")
        log_f.close()
        return
    time.sleep(1.0)

    log("Starting pose_udp_relay.py + waypoint_udp_relay.py"
        + (" + cmdvel_udp_relay.py (rotation-only)" if motion else "") + " ...")
    pose_relay_proc = _ros_popen(
        f"/usr/bin/python3 {POSE_RELAY_SCRIPT}", log_path=os.path.join(run_dir, "pose_udp_relay.log"))
    waypoint_relay_proc = _ros_popen(
        f"/usr/bin/python3 {WAYPOINT_RELAY_SCRIPT}", log_path=os.path.join(run_dir, "waypoint_udp_relay.log"))
    if motion:
        cmdvel_relay_proc = _ros_popen(
            f"/usr/bin/python3 {CMDVEL_RELAY_SCRIPT}", log_path=os.path.join(run_dir, "cmdvel_udp_relay.log"))
    time.sleep(2.0)

    camera = _RealSenseStreamClient(JETSON_HOST, CAMERA_PORT)
    pose_client = PoseUdpClient(POSE_UDP_PORT)
    waypoint_sender = WaypointUdpSender(WAYPOINT_UDP_PORT)
    cmdvel_sender = CmdVelUdpSender(CMDVEL_UDP_PORT)

    log("Waiting for camera intrinsics + first pose sample...")
    t0 = time.time()
    while camera.intrinsics() is None or pose_client.xy_yaw() is None:
        if time.time() - t0 > 30.0:
            log("ABORT: no camera or pose data within 30s. Not arming.")
            _pkill(PATHFOLLOWER_MATCH)
            return
        time.sleep(0.2)
    intr = camera.intrinsics()
    fx, fy, width = intr["fx"], intr["fy"], intr["width"]
    fov = 2 * np.arctan(width / (2 * fx))
    log(f"Camera OK (fx={fx:.1f}, fov={np.rad2deg(fov):.1f}deg). Pose OK.")

    video_writer = None
    if args.record:
        video_path = os.path.join(run_dir, "first_person.mp4")
        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"),
            1.0 / LOOP_INTERVAL_S, (intr["width"], intr["height"]),
        )
        log(f"Recording first-person view to {video_path}")

    start_pose = pose_client.xy_yaw()
    if start_pose is None:
        log("ABORT: pose went stale right after startup. Not arming.")
        log_f.close()
        return
    xy0, yaw0 = start_pose
    rec = RunRecorder(run_dir, pose_client, camera, realtime_video=args.record_realtime)
    rec.event("run_start", f"target={args.target!r} mode={'motion' if motion else 'perception-only'} "
                           f"max_seconds={args.max_seconds:.0f} record_realtime={args.record_realtime}")
    if t_arm_confirmed:
        rec.event("arm_confirmed", "operator typed ARM", ts=t_arm_confirmed)
    rec.event("camera_pose_ready", f"fx={fx:.1f} start pose ({xy0[0]:.2f}, {xy0[1]:.2f}, {np.rad2deg(yaw0):+.0f} deg)")
    log(f"Pinning startup goal to current position {tuple(xy0.round(3))} before arming...")
    rec.event("waypoint_pin_current_position", f"({xy0[0]:.2f}, {xy0[1]:.2f})")
    for _ in range(3):
        waypoint_sender.send(float(xy0[0]), float(xy0[1]), 0.0)
        time.sleep(0.2)

    obstacle_map = ObstacleMap(
        min_height=MIN_HEIGHT, max_height=MAX_HEIGHT, agent_radius=0.2,
        area_thresh=0.5, hole_area_thresh=-1, size=MAP_SIZE, pixels_per_meter=PIXELS_PER_METER,
    )
    value_map = ValueMap(value_channels=1, size=MAP_SIZE, obstacle_map=obstacle_map)
    blip2itm = BLIP2ITMClient(port=12182)
    detector = YOLOWorldClient(port=12185)

    stdin_reader = StdinCommands()
    stdin_reader.start()

    def request_stop(signum=None, frame=None) -> None:
        rec.event("stop_requested", f"signal {signum}")
        if motion:
            # cut /cmd_vel NOW, even if the main thread is stuck in a blocking VLM call;
            # disable the bridge in parallel (the service call can take a few seconds)
            threading.Thread(target=call_bridge_enable, args=(False,), daemon=True).start()
            _pkill(PATHFOLLOWER_MATCH)
        with state_lock:
            state["stop"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # Everything from here on (scan included) must go through the same
    # kill-pathFollower-then-disable cleanup in `finally` -- start_pathfollower
    # is called INSIDE the try so a Ctrl-C mid-scan still stops the robot
    # via the normal path instead of an uncaught KeyboardInterrupt leaving
    # it armed.
    frame_idx = 0
    last_way_point = None
    hit_window = deque(maxlen=TARGET_LOCK_WINDOW)  # (seen, world-xy estimate or None) per tick
    target_lock_xy = None
    approach_goal = None          # stand-off /way_point actually sent after a lock
    last_walk_record = 0.0
    data_lost_since = None

    def data_lost(what: str) -> bool:
        """Call when camera/pose are missing this tick. Returns True if the run
        must stop (motion runs only, after DATA_LOSS_STOP_S of continuous loss)."""
        nonlocal data_lost_since
        now = time.time()
        if data_lost_since is None:
            data_lost_since = now
            log(f"  [warn] no fresh {what} (camera age {camera.age()}, pose age {pose_client.age()})")
            return False
        if motion and now - data_lost_since > DATA_LOSS_STOP_S:
            log(f"DATA LOSS: no fresh {what} for {now - data_lost_since:.1f}s -- stopping the robot.")
            rec.event("data_loss", f"no fresh {what} for {now - data_lost_since:.1f} s")
            return True
        return False

    def should_stop() -> bool:
        with state_lock:
            return state["stop"]

    try:
        if motion:
            # The scan runs FIRST, with pathFollower deliberately NOT started: the
            # scan alone owns /cmd_vel (yaw rate only, via cmdvel_udp_relay.py).
            frame_idx, scan_abort = perform_initial_scan(
                xy0, yaw0, camera, pose_client, cmdvel_sender, obstacle_map, value_map,
                blip2itm, detector, intr, fx, fy, fov, run_dir, log, state["target"], 0,
                video_writer, should_stop, rec,
            )
            if should_stop():
                log("Stop requested during the scan -- not arming pathFollower.")
                return
            if scan_abort is not None:
                # drift / no pose / no response / repeated timeouts all mean something
                # is wrong with the robot or the data -- never go on to walk.
                log(f"Scan did not complete ({scan_abort}) -- NOT arming pathFollower, ending the run.")
                return

            # Hand /cmd_vel over to pathFollower: pin its goal to wherever the robot
            # is NOW (so it can't lurch toward a stale goal), give the relay time to
            # finish its zero burst and go silent, then arm.
            cur_pose = pose_client.xy_yaw()
            if cur_pose is None:
                log("No fresh pose after the scan -- NOT arming pathFollower, ending the run.")
                return
            cur_xy, _cur_yaw = cur_pose
            rec.event("waypoint_pin_current_position", f"({cur_xy[0]:.2f}, {cur_xy[1]:.2f})")
            for _ in range(3):
                waypoint_sender.send(float(cur_xy[0]), float(cur_xy[1]), 0.0)
                time.sleep(0.2)
            time.sleep(0.6)
            log("Starting pathFollower with autonomyMode=true (ARMED)...")
            start_pathfollower(autonomy=True, log_path=os.path.join(run_dir, "pathFollower_debug.log"))
            rec.event("armed", "pathFollower started, autonomous walking begins")

        start_time = time.time()
        log(f"{'ARMED' if motion else 'PERCEPTION ONLY'}. target={state['target']!r}. "
            f"Type 'target <name>' or 'stop' anytime. "
            f"Hard cap {args.max_seconds:.0f}s (scan time not counted against this).")

        while True:
            with state_lock:
                if state["stop"]:
                    break
            if time.time() - start_time > args.max_seconds:
                log("Max runtime reached, stopping.")
                rec.event("max_runtime", f"{args.max_seconds:.0f} s")
                break

            cmd = stdin_reader.poll()
            if cmd:
                if cmd == "stop":
                    rec.event("stop_requested", "typed 'stop'")
                    with state_lock:
                        state["stop"] = True
                    break
                elif cmd.startswith("target "):
                    new_target = cmd[len("target "):].strip()
                    if new_target:
                        with state_lock:
                            state["target"] = new_target
                        hit_window.clear()
                        target_lock_xy = None
                        approach_goal = None
                        last_way_point = None
                        log(f">>> target changed to {new_target!r}")
                        rec.event("target_changed", new_target)
                else:
                    log(f"(unrecognized command: {cmd!r})")

            loop_t0 = time.time()
            frame_idx += 1
            target = state["target"]

            if target_lock_xy is not None:
                # LOCKED: the upper layer does no more reasoning at all -- no
                # detection, no BLIP2, no map update, no frontier choice, no new
                # waypoints. localPlanner/pathFollower own getting to the goal;
                # this only checks the distance -- at 10 Hz, so the stop isn't
                # up to a whole 1 s tick (~0.25 m) late -- and records at 1 Hz.
                frame_idx -= 1  # only recorded walk ticks are numbered
                py = pose_client.xy_yaw()
                if py is None:
                    if data_lost("pose"):
                        break
                    time.sleep(WALK_CHECK_INTERVAL_S)
                    continue
                data_lost_since = None
                xy = py[0]
                dist_to_target = float(np.hypot(target_lock_xy[0] - xy[0], target_lock_xy[1] - xy[1]))
                dist_to_goal = float(np.hypot(approach_goal[0] - xy[0], approach_goal[1] - xy[1]))
                if time.time() - last_walk_record >= LOOP_INTERVAL_S:
                    last_walk_record = time.time()
                    frame_idx += 1
                    cam_age = camera.age()
                    _tick_walk(frame_idx, camera, pose_client, target_lock_xy, run_dir, video_writer)
                    rec.tick({"tick": frame_idx, "phase": "walk",
                              "capture_time": time.time() - cam_age if cam_age is not None else None,
                              "t_since_armed_s": f"{time.time() - start_time:.2f}",
                              "x": f"{xy[0]:.3f}", "y": f"{xy[1]:.3f}", "yaw_deg": f"{np.rad2deg(py[1]):.1f}",
                              "goal_x": f"{target_lock_xy[0]:.3f}", "goal_y": f"{target_lock_xy[1]:.3f}",
                              "standoff_x": f"{approach_goal[0]:.3f}", "standoff_y": f"{approach_goal[1]:.3f}",
                              "dist_to_target_m": f"{dist_to_target:.3f}", "dist_to_standoff_m": f"{dist_to_goal:.3f}"})
                    log(f"[tick {frame_idx:5d} t={time.time()-start_time:6.1f}s target={target!r}] "
                        f"pose=({xy[0]:+.2f},{xy[1]:+.2f})  Walking toward the goal "
                        f"({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}) dist={dist_to_target:.2f}m "
                        f"(stand-off point {dist_to_goal:.2f}m)")
                if dist_to_target <= TARGET_STOP_RADIUS_M or dist_to_goal <= TARGET_GOAL_REACHED_M:
                    log(f"FOUND {target!r} at ({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}), "
                        f"{dist_to_target:.2f}m away (stand-off point {dist_to_goal:.2f}m) -- stopping.")
                    rec.event("found", f"({target_lock_xy[0]:.2f}, {target_lock_xy[1]:.2f}), {dist_to_target:.2f} m away, "
                                       f"stand-off point {dist_to_goal:.2f} m; robot at ({xy[0]:.2f}, {xy[1]:.2f})")
                    with state_lock:
                        state["stop"] = True
                    break
                time.sleep(WALK_CHECK_INTERVAL_S)
                continue

            result = _tick_perception(frame_idx, target, camera, pose_client, obstacle_map,
                                       value_map, blip2itm, detector, intr, fx, fy, fov,
                                       run_dir, log, video_writer)
            if result is None:
                frame_idx -= 1
                if data_lost("camera/pose"):
                    break
                time.sleep(0.2)
                continue
            data_lost_since = None
            xy, yaw = result["xy"], result["yaw"]
            frontiers = result["frontiers"]
            status = (f"[tick {frame_idx:5d} t={time.time()-start_time:6.1f}s target={target!r}] "
                      f"pose=({xy[0]:+.2f},{xy[1]:+.2f}) score={result['score']:+.3f} "
                      f"frontiers={len(frontiers)}")
            row = {"tick": frame_idx, "phase": "explore", "capture_time": result["t_capture"],
                   "t_since_armed_s": f"{time.time() - start_time:.2f}", "x": f"{xy[0]:.3f}", "y": f"{xy[1]:.3f}",
                   "yaw_deg": f"{np.rad2deg(yaw):.1f}", "blip_score": f"{result['score']:.3f}",
                   "frontiers": len(frontiers)}

            fresh_xy = None
            lock_hit = False
            if result["target_seen"]:
                valid_idxs, detections = result["valid_idxs"], result["detections"]
                crop_scores, verified = result["crop_scores"], result["verified_idxs"]
                # lock only on a crop-verified box; fall back to the best raw box for the log line
                best_idx = max(verified or valid_idxs, key=lambda i: detections.logits[i])
                best_conf = float(detections.logits[best_idx])
                best_itm = crop_scores.get(best_idx)
                lock_hit = bool(verified) and best_conf >= TARGET_LOCK_MIN_CONF
                if lock_hit:
                    box = detections.boxes[best_idx].numpy()
                    fresh_xy = estimate_target_xy(
                        box, result["depth_raw"], intr["depth_scale"], fx, fy,
                        intr["width"], intr["height"], result["tf_camera_to_episodic"],
                    )
            hit_window.append((lock_hit, fresh_xy))
            hits = sum(1 for seen, _ in hit_window if seen)
            if result["target_seen"]:
                itm_txt = f" itm={best_itm:.3f}" if best_itm is not None else ""
                if lock_hit:
                    why = ""
                elif not verified:
                    why = f" (crop itm < {TARGET_VERIFY_MIN_ITM}, not counted)"
                else:
                    why = f" (< lock conf {TARGET_LOCK_MIN_CONF}, not counted)"
                row.update(saw=1, conf=f"{best_conf:.3f}", itm=f"{best_itm:.3f}" if best_itm is not None else "",
                           counted=int(lock_hit), hits=hits)
                rec.event("target_detected" if lock_hit else "detection_rejected",
                          f"tick {frame_idx}: conf {best_conf:.3f} itm {itm_txt.strip()[4:] or '-'} hits {hits}"
                          + (why and f" ({why.strip(' ()')})"))
                status += (f"  >>> saw {target!r} conf={best_conf:.3f}{itm_txt}{why}"
                           + f" hits={hits}/{len(hit_window)} (lock at {TARGET_LOCK_MIN_HITS} "
                           f"within the last {TARGET_LOCK_WINDOW} ticks)")

            # Lock rule: >= TARGET_LOCK_MIN_HITS detections among the last
            # TARGET_LOCK_WINDOW ticks. The goal is the MEDIAN of those ticks'
            # depth-based world-frame estimates (robust to one bad depth patch),
            # sent once as a fixed waypoint; from then on the branch above takes
            # over and the upper layer stays out of the way.
            if hits >= TARGET_LOCK_MIN_HITS:
                estimates = [p for seen, p in hit_window if seen and p is not None]
                if estimates:
                    target_lock_xy = np.median(np.array(estimates), axis=0)
                    to_target = target_lock_xy - np.asarray(xy, dtype=float)
                    dist0 = float(np.hypot(to_target[0], to_target[1]))
                    if dist0 > TARGET_APPROACH_STANDOFF_M:
                        approach_goal = target_lock_xy - to_target / dist0 * TARGET_APPROACH_STANDOFF_M
                    else:
                        approach_goal = np.asarray(xy, dtype=float).copy()  # already close: stay put
                    waypoint_sender.send(float(approach_goal[0]), float(approach_goal[1]), 0.0)
                    last_way_point = (float(approach_goal[0]), float(approach_goal[1]))
                    per_frame = " ".join(f"({e[0]:.2f},{e[1]:.2f})" for e in estimates)
                    row.update(locked=1, goal_x=f"{target_lock_xy[0]:.3f}", goal_y=f"{target_lock_xy[1]:.3f}",
                               standoff_x=f"{approach_goal[0]:.3f}", standoff_y=f"{approach_goal[1]:.3f}",
                               dist_to_target_m=f"{dist0:.3f}", waypoint_sent=1)
                    rec.event("locked", f"tick {frame_idx}: goal ({target_lock_xy[0]:.2f}, {target_lock_xy[1]:.2f}) "
                                        f"dist {dist0:.2f} m; /way_point at stand-off "
                                        f"({approach_goal[0]:.2f}, {approach_goal[1]:.2f})")
                    status += (f"  LOCKED goal=({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}) "
                               f"dist={dist0:.2f}m [median of {len(estimates)} estimates: {per_frame}] "
                               f"-> /way_point at stand-off ({approach_goal[0]:.2f},{approach_goal[1]:.2f}), "
                               f"{TARGET_APPROACH_STANDOFF_M}m short; no more upper-layer decisions")
                else:
                    status += "  [warn] enough hits but no valid depth estimate yet"

            if target_lock_xy is None and len(frontiers) > 0:
                sorted_frontiers, _ = value_map.sort_waypoints(frontiers, FRONTIER_SEARCH_RADIUS)
                best = sorted_frontiers[0]
                rho, theta = rho_theta(xy, yaw, best)
                status += f"  best_frontier=({best[0]:.2f},{best[1]:.2f}) turn={np.rad2deg(theta):+.0f}deg dist={rho:.2f}m"
                row.update(best_frontier_x=f"{best[0]:.3f}", best_frontier_y=f"{best[1]:.3f}")
                moved = last_way_point is None or np.hypot(
                    best[0]-last_way_point[0], best[1]-last_way_point[1]
                ) >= WAYPOINT_MIN_REPUBLISH_DELTA_M
                if moved:
                    waypoint_sender.send(float(best[0]), float(best[1]), 0.0)
                    last_way_point = (float(best[0]), float(best[1]))
                    status += "  [/way_point sent]"
                    row["waypoint_sent"] = 1
                    rec.event("waypoint_frontier", f"({best[0]:.2f}, {best[1]:.2f})")
            rec.tick(row)
            log(status)

            if target_lock_xy is not None:
                continue  # just locked: go straight to the 10 Hz stop check, no 1 s tick wait
            elapsed = time.time() - loop_t0
            if elapsed < LOOP_INTERVAL_S:
                time.sleep(LOOP_INTERVAL_S - elapsed)
    finally:
        stopped_ok = True
        if motion:
            for _ in range(3):  # if we die mid-rotation, leave an explicit zero yaw rate behind
                cmdvel_sender.send(0.0)
                time.sleep(0.03)
            log("Stopping: killing pathFollower first (primary stop)...")
            rec.event("stopping")
            stopped_ok = _pkill(PATHFOLLOWER_MATCH)
            rec.event("pathfollower_killed" if stopped_ok else "pathfollower_kill_NOT_confirmed")
            _pkill("cmdvel_udp_relay.py")
            time.sleep(1.0)  # let the /cmd_vel stream actually cease (bridge zeroes after 0.5 s anyway)
            log("Disabling g1_sdk_bridge (secondary stop; robot stays standing)...")
            if not call_bridge_enable(False):
                stopped_ok = False
                rec.event("bridge_disable_NOT_confirmed")
            else:
                rec.event("bridge_disabled", "robot standing still")
        _pkill("pose_udp_relay.py")
        _pkill("waypoint_udp_relay.py")
        rec.event("relays_stopped")
        rec.close()
        camera.close()
        pose_client.close()
        if video_writer is not None:
            video_writer.release()
            log(f"First-person recording saved to {os.path.join(run_dir, 'first_person.mp4')}")
        cv2.imwrite(os.path.join(run_dir, "g1_obstacle_map_final.png"), obstacle_map.visualize())
        cv2.imwrite(os.path.join(run_dir, "g1_value_map_final.png"), value_map.visualize(obstacle_map=obstacle_map))
        if video_writer is not None:
            # Robot is already stopped by now, so this (a few seconds
            # of encoding) can't delay anything safety-relevant. Turns the raw
            # first-person recording into the composite (camera + occupancy map
            # + value map, time-aligned); the raw one is kept as
            # first_person_orig.mp4. Never allowed to break cleanup.
            log("Building composite review video (camera + occupancy map + value map)...")
            try:
                out = subprocess.run(
                    [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "make_composite_video.py"),
                     run_dir, "--fps", str(COMPOSITE_VIDEO_FPS)],
                    capture_output=True, text=True, timeout=300)
                log((out.stdout.strip() or out.stderr.strip() or "(no output)")[-400:])
            except Exception as e:  # noqa: BLE001
                log(f"[warn] composite video failed ({e}); raw recording is still in first_person.mp4")
        log(f"Done. {frame_idx} ticks. Results in {run_dir}")
        if motion:
            if stopped_ok:
                log("pathFollower is dead and the bridge is disabled -- robot should be standing still.")
            else:
                log("\033[31mSTOP NOT CONFIRMED: see the [pkill]/[enable false] lines above. "
                    "Check the robot NOW; /g1_sdk_bridge/damp or the remote if it is still moving.\033[0m")
        log_f.close()


if __name__ == "__main__":
    main()
