"""CGFM real-robot pipeline, derived from the VLFM (ObjectNav) real-robot loop: BLIP-2 + ValueMap
are replaced by the CLIP scene graph + semantic-map frontier scorer from
cgfm/; outputs go to $CGFM_RES_DIR (default ./results/...); --dry-run runs perception and
inference only without ever commanding the robot.

Master script: orchestrates the full vlfm-native-mapping -> base-stack
obstacle-avoidance loop on the real robot, with LIVE target-object
switching and a hard "lie down" stop.

Architecture (all pieces individually verified in earlier sessions):
  RGB-D (TCP, Jetson) --+
                         +--> ObstacleMap/ValueMap (vlfm native) --> best
  pose (UDP relay from   |    frontier --> /way_point (UDP relay) -->
  /state_estimation) ----+    localPlanner (obstacle-avoidance path) -->
                              pathFollower (autonomyMode=true) --> /cmd_vel
                              --> unitree_control --> WebRTC --> robot

Stop mechanism, revised after two live tests:
  1. PRIMARY (reliable, twice-confirmed live): kill pathFollower FIRST.
     This cuts off /cmd_vel at the source; both live tests so far show the
     robot stopping cleanly on its own once this happens, no "keeps
     executing the last command" behavior observed.
  2. SECONDARY / best-effort: only AFTER pathFollower is dead, call the
     /liedown service (std_srvs/Trigger -> SPORT_CMD["StandDown"]) to put
     the robot into an unambiguous resting state. NOT relied upon --
     live-tested and found unreliable twice: once "Data channel is not
     open" (stale WebRTC session), once a bare timeout with zero response
     even after 60s (likely DDS-service-call starvation under this
     project's typically high nav-stack CPU load -- laser_mapping_node
     alone measured at ~52% CPU, system load average ~17 -- possibly
     compounded by the zenoh-bridge-dds.service systemd unit's auto-restart
     loop churning the DDS discovery graph; that service targets a
     long-dead Tailscale link and should be `sudo systemctl stop`ped if
     not otherwise needed).
  Earlier idea (send a /way_point at the robot's own current position
  while still armed, hoping cmd_vel settles to zero on its own) was tried
  live and rejected: linear velocity did settle, but angular.z kept
  jittering (path-direction noise on a near-zero-distance goal) even
  though it didn't visibly move the robot that one time -- not trustworthy
  as a stop primitive.

INITIAL SCAN (2026-09-19): before pathFollower is even started, the script
rotates the robot a full 360deg closed-loop by commanding /cmd_vel yaw rate
directly through cmdvel_udp_relay.py (linear velocity forced to 0, |wz|
clamped, silent when the stream stops). Only after the scan is pathFollower
armed.

SAFETY:
  - Requires typed "ARM" confirmation before starting real motion.
  - Bounded runtime by default (--max-seconds), independent of Ctrl-C.
  - A human with the physical remote must be present for every real run --
    this script does not replace that.
  - Startup sends an initial /way_point at the robot's OWN current
    position before arming, so the first cycle can't lurch toward a stale
    default goal.

Usage (needs the `vlfm` package importable, see README):
    python scripts/run_cgfm_pipeline.py \
        --target "trash can" --max-seconds 300 --record --vlm [--dry-run]

While running, type in this terminal (Enter to submit):
    target <name>     change the exploration target, e.g. "target sofa"
    stop               graceful stop (same as Ctrl-C): liedown, then exit
"""
import argparse
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

import json
import logging

import cv2
import numpy as np

import open_clip
import torch
import torch.nn.functional as _F

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.utils.geometry_utils import rho_theta
from vlfm.vlm.yolo_world import YOLOWorldClient, _caption_to_classes

# CGFM modules (../cgfm/)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "cgfm"))
from scene_graph import LightweightSceneGraph, class_color
from semantic_map import compute_semantic_map
from frontier_scorer import select_frontier_by_score
from floor_map import ObservedFloorMap
from obstacle_map_cgfm import CGFMObstacleMap
from vlm_selector import VLMTargetSelector, DEFAULT_URL as VLM_DEFAULT_URL

# CLIP (ViT-H-14) cosine of a scene-graph object to the target text must exceed
# this to feed the semantic map (contribution = score - baseline). Measured
# 2026-09-28 on real-robot crops vs "chair": chair boxes median 0.257 (p5 0.150),
# non-chair crops median 0.169, and old-target objects after a target switch
# (chair crops vs "trash can") median 0.167. At 0.16 a typical chair contributes
# ~0.10 while leaked old-target objects contribute ~0.01. MSGNav's default was 0.20.
SEMANTIC_SCORE_BASELINE = 0.16

# Frontiers closer than this (straight line) are never chosen: localPlanner treats a
# goal within goalClearRange (0.35 m) as already reached and doesn't move, so a
# frontier at the robot's feet is never cleared and can be re-selected forever. In
# the 2026-09-28 dry-run 52/170 chosen frontiers were < 0.5 m away.
MIN_FRONTIER_DIST_M = 0.8

# Scene-graph vocabulary: besides the current target, these common office classes
# (ScanNet200 names) are detected every tick and kept in the scene graph, so the
# graph describes the environment (for later VLM reasoning / life-long use), not
# just the target. The caption is fixed for a given target, so YOLO-World's
# set_classes() is not re-issued every tick. Synonyms returned by the detector are
# folded into one canonical label so the same kind of object merges.
SCENE_VOCAB = ["chair", "table", "desk", "monitor", "cabinet", "door",
               "trash can", "couch", "whiteboard", "plant"]
SCENE_SYNONYMS = {"office chair": "chair", "seat": "chair", "sofa": "couch", "potted plant": "plant"}
SG_LOCK_MIN_OBS = 3           # lock on the target from scene-graph MEMORY once a target-class node has been
                              # seen this many times -- even if never 3-in-5 consecutive ticks (2026-09-29:
                              # bin seen 6 times, consistent position, but the robot kept turning away)
FRONTIER_SWITCH_MARGIN = 0.3  # frontier hysteresis: only switch to a new frontier that is >30% better
# Frontier stall give-up (2026-10-03 run 151341: ~55 s stuck flip-flopping between two
# unreachable frontiers). While exploring (not in dry-run), if the robot's position stayed
# within FRONTIER_STALL_MOVE_M for FRONTIER_STALL_S, the frontier being pursued is
# blacklisted: no frontier within FRONTIER_BLACKLIST_RADIUS_M of it is chosen again.
FRONTIER_STALL_S = 15.0
FRONTIER_STALL_MOVE_M = 0.5
FRONTIER_BLACKLIST_RADIUS_M = 1.0
SCENE_MAX_OBJS_PER_TICK = 8   # non-target detections per tick (highest confidence first) -> bounds CLIP work

JETSON_HOST = os.environ.get("GO2_JETSON_HOST", "192.168.3.18")
CAMERA_PORT = 6000
POSE_UDP_PORT = 8765
WAYPOINT_UDP_PORT = 8766
CMDVEL_UDP_PORT = 8767
WAYPOINT_MIN_REPUBLISH_DELTA_M = 0.05
LOOP_INTERVAL_S = 1.0
# Localization-jump guard (CGFM): the robot walks <= 0.2 m/s, so the SLAM pose moves ~0.2 m per 1 s
# tick. When the WiFi sensor link stalls, SLAM's IMU preintegration fails and the pose jumps
# 0.5-1 m per tick (2026-09-30) -- the robot then chases a phantom goal. Armed runs STOP on a jump
# larger than max(POSE_JUMP_M, POSE_JUMP_SPEED_MPS * dt); dry-runs only warn (teleop can be fast).
POSE_STALE_S = 2.0        # no /state_estimation packet for this long -> SLAM output stopped
POSE_FROZEN_S = 5.0       # pose bit-identical for this long -> frozen (a live SLAM jitters at < 1 mm every
                          # message; 2026-10-02 its feature-extraction node died on an assert and the pose
                          # stayed at the last value while the run kept printing ticks)
POSE_JUMP_M = 0.5
POSE_JUMP_SPEED_MPS = 0.5
SLAM_LOG = "/tmp/go2_stack/slam.log"
SLAM_PROCS = ("feature_extract", "laser_mapping_n", "imu_preintegrat")   # process names (comm, 15 chars)


def slam_preflight():
    """None if all three SLAM processes run and none has died since bringup, else a
    reason. (2026-10-02: laser_mapping died while feature_extraction + imu_preintegration
    kept publishing an IMU-only pose that ran off 900 m -- not frozen, so only this catches it.)"""
    try:
        out = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True, timeout=5).stdout.split()
    except Exception:  # noqa: BLE001
        return None
    missing = [c for c in SLAM_PROCS if c not in out]
    if missing:
        return f"SLAM process(es) not running: {', '.join(missing)}"
    try:
        with open(SLAM_LOG, "rb") as f:
            if b"process has died" in f.read():
                return "a SLAM process died since bringup (slam.log: 'process has died')"
    except OSError:
        pass
    return None   # bringup_nav_stack.sh's SLAM log; new 'failureDetected' lines are reported
COMPOSITE_VIDEO_FPS = 1.5   # playback speed of the auto-generated composite video (ticks/s)

_CAMERA_FORWARD_OFFSET_M = 0.15
# Re-measured 2026-09-17 via my_tests-style floor-height-vs-distance
# regression against a live frame (see conversation/scratchpad
# calibrate_floor.py): at the OLD 8.6deg assumption, floor height computed
# 0.19m higher at 2m range than at 0.4m range (way off). pitch=0deg gave
# the flattest floor (-0.013m drift over 0.4-2.5m) -- the camera's actual
# mount changed (previously 8.6deg up / 0.385m off the ground) since the
# original go2_robot.py-era calibration; it now reads as level.
_CAMERA_PITCH_UP_DEG = 0.0

MIN_DEPTH = 0.3
MAX_DEPTH = 3.0
CAMERA_HEIGHT_M = 0.44  # re-measured 2026-09-17, see _CAMERA_PITCH_UP_DEG comment above
MIN_HEIGHT = -CAMERA_HEIGHT_M + 0.10
MAX_HEIGHT = 0.6
MAP_SIZE = 800
PIXELS_PER_METER = 20
FRONTIER_SEARCH_RADIUS = 1.0
DINO_CONF_THRESHOLD = 0.25
MAX_BOX_AREA_FRAC = 0.7
EDGE_CROP_FRAC = 0.08

RES_DIR = os.environ.get("CGFM_RES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "single_target"))
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
}

# PATCH (2026-09-20, wireless/router setup): every ROS 2 process spawned below (pathFollower, the UDP relays and
# the ros2 CLI calls such as /liedown) has to use the same DDS domain and CycloneDDS config as the Jetson and
# unitree_control, otherwise it silently lands in domain 0 and sees none of the wireless data. They only inherit
# the environment of the shell that started this script, so source the env script here instead of relying on
# the operator remembering to do it. GO2_ROS_ENV_SCRIPT=<path> selects another one (e.g. ros_env_mid360.sh for
# the old wired setup); GO2_ROS_ENV_SCRIPT="" adds nothing (original behaviour).
GO2_ROS_ENV_SCRIPT = os.environ.get("GO2_ROS_ENV_SCRIPT", "")
# Install space of the base navigation stack (SLAM / localPlanner / pathFollower / unitree_control), see README.
GO2_WS_SETUP = os.environ.get("GO2_WS_SETUP", os.path.expanduser("~/GO2_STACK_dev_ws/install/setup.bash"))
ROS_ENV_SOURCE = (
    "source /opt/ros/jazzy/setup.bash && "
    f"source {GO2_WS_SETUP}"
    + (f" && source {GO2_ROS_ENV_SCRIPT} >/dev/null" if GO2_ROS_ENV_SCRIPT else "")
)
LOCAL_PLANNER_YAML = os.environ.get("GO2_LOCAL_PLANNER_YAML", os.path.join(
    os.path.dirname(GO2_WS_SETUP), "local_planner", "share", "local_planner", "config", "unitree",
    "unitree_go2_slow.yaml"))
PATHFOLLOWER_MATCH = "install/local_planner/lib/local_planner/pathFollower"
POSE_RELAY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_udp_relay.py")
WAYPOINT_RELAY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoint_udp_relay.py")
CMDVEL_RELAY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmdvel_udp_relay.py")


def _ros_cli(cmd: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f"{ROS_ENV_SOURCE} && {cmd}"],
        capture_output=True, text=True, timeout=timeout,
    )


def _ros_popen(cmd: str, log_path: str = None) -> subprocess.Popen:
    # PATCH (2026-09-17): used to always silence output to DEVNULL, which
    # meant the relays' own per-message logging (useful for confirming
    # whether a given /way_point actually got sent) was unrecoverable after
    # the fact -- exactly the gap hit when diagnosing the 360-scan travelling
    # 2.9m in one direction instead of rotating. Callers that need that
    # visibility later should pass log_path.
    out = open(log_path, "a") if log_path else subprocess.DEVNULL
    return subprocess.Popen(
        ["bash", "-c", f"{ROS_ENV_SOURCE} && exec {cmd}"],
        stdout=out, stderr=out,
    )


def _pkill(pattern: str) -> None:
    subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


def start_pathfollower(autonomy: bool, log_path: str = None) -> None:
    _pkill(PATHFOLLOWER_MATCH)
    time.sleep(1.0)
    mode = "true" if autonomy else "false"
    cmd = (
        f"ros2 run local_planner pathFollower --ros-args "
        f"--params-file {LOCAL_PLANNER_YAML} "
        f"-p useSerialPort:=false -p realRobot:=false -p autonomyMode:={mode} "
        # twoWayDrive:=false (2026-09-18): with it true, dirDiff>90deg made
        # pathFollower reverse instead of turning -- but the camera faces
        # forward, so driving backward means vlfm's own perception (which
        # is what actually decides where to explore next) never sees
        # anything new in the direction of travel, even though the lidar
        # can. Forcing forward-only makes localPlanner's path selection
        # rotate toward the goal (clamped to +/-95deg per tick, see
        # localPlanner.cpp's `if (!twoWayDrive)` joyDir clamp) instead of
        # backing up to it.
        f"-p sensorOffsetX:=0.2 -p sensorOffsetY:=0.0 -p twoWayDrive:=false "
        f"-p maxSpeed:=0.2 -p autonomySpeed:=0.2 -p maxYawRate:=30.0 "
        f"-p dirDiffThre:=1.0"
    )
    # PATCH (2026-09-17): this used to discard stdout to DEVNULL, silently
    # losing the [PF_DBG] instrumentation added to pathFollower.cpp while
    # root-causing the "won't turn until about to collide" bug -- pass
    # log_path so a real run captures it.
    _ros_popen(cmd, log_path=log_path)
    time.sleep(2.0)


def call_liedown() -> bool:
    """Hard stop: SPORT_CMD StandDown via the existing /liedown service.
    Overrides whatever Move state pathFollower/unitree_control were in."""
    try:
        result = _ros_cli('ros2 service call /liedown std_srvs/srv/Trigger "{}"', timeout=8.0)
        print(f"[liedown] {result.stdout.strip() or result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("[liedown] service call timed out")
        return False


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
        self._raw = None
        self._t_rx = None          # time of the last packet
        self._t_change = None      # time the pose values last changed
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
            _stamp, x, y, _z, qx, qy, qz, qw = struct.unpack(self.FMT, data)
            yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
            now = time.time()
            with self._lock:
                self._xy_yaw = (np.array([x, y]), yaw)
                self._t_rx = now
                if (x, y, qz, qw) != self._raw:
                    self._raw, self._t_change = (x, y, qz, qw), now

    def health(self):
        """None if the pose stream looks alive, else a reason string (no packets for
        POSE_STALE_S, or values bit-identical for POSE_FROZEN_S)."""
        now = time.time()
        with self._lock:
            t_rx, t_change = self._t_rx, self._t_change
        if t_rx is None:
            return None
        if now - t_rx > POSE_STALE_S:
            return f"no SLAM pose for {now - t_rx:.1f}s (/state_estimation stopped)"
        if now - t_change > POSE_FROZEN_S:
            return f"SLAM pose frozen (unchanged for {now - t_change:.1f}s)"
        return None

    def xy_yaw(self):
        with self._lock:
            return self._xy_yaw

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
                    with self._lock:
                        self._color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                        self._depth = depth
                sock.close()
            except (socket.timeout, ConnectionRefusedError, OSError):
                pass
            if not self._stop.is_set():
                time.sleep(1.0)

    def intrinsics(self):
        with self._lock:
            return self._intrinsics

    def latest(self):
        with self._lock:
            return self._color, self._depth

    def close(self) -> None:
        self._stop.set()


def _build_camera_to_body_transform() -> np.ndarray:
    theta = np.deg2rad(_CAMERA_PITCH_UP_DEG)
    c, s = np.cos(theta), np.sin(theta)
    t = np.eye(4)
    t[0, 0], t[0, 2] = c, -s
    t[2, 0], t[2, 2] = s, c
    t[0, 3] = _CAMERA_FORWARD_OFFSET_M
    return t


_CAMERA_TO_BODY = _build_camera_to_body_transform()


def get_camera_transform(xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    body_to_episodic = np.eye(4)
    body_to_episodic[0, 0], body_to_episodic[0, 1] = c, -s
    body_to_episodic[1, 0], body_to_episodic[1, 1] = s, c
    body_to_episodic[0, 3], body_to_episodic[1, 3] = xy[0], xy[1]
    return body_to_episodic @ _CAMERA_TO_BODY


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
TARGET_STOP_RADIUS_M = 0.5     # CGFM: the goal is the object's NEAR EDGE (point-cloud near side, or where
                               # the box meets the floor); stop when the robot's body centre is this close to
                               # it (Go2 body centre -> nose ~0.3 m, so the nose ends ~0.2 m from the object)
REFINE_GATE_M = 1.0            # after lock, a fresh estimate farther than this from the goal is ignored
                               # (another object of the same class / a false detection)
REFINE_PUSHBACK_M = 0.2        # ...and a POINT-CLOUD estimate that puts the goal farther from the robot than the
                               # current goal by more than this is ignored too: at close range a mesh bin's point
                               # cloud lands on the wall behind it (2026-10-02: goal pushed 0.37 m into the corner,
                               # never reached). Ground-contact estimates are exempt: they legitimately creep
                               # forward ~0.7 m during an approach (short at long range).
# 2026-10-03 run 152524: a ground-contact lock from ~6 m came out ~2.3 m short; every later (correct)
# estimate was > REFINE_GATE_M from that goal and was ignored, and the robot stopped "FOUND" with the bin
# still ~3 m ahead. Two fixes:
#  - consistent override: if >= REGOAL_MIN_AGREE of the last REGOAL_WINDOW locked ticks had a gate-rejected
#    estimate and those estimates agree (each within REGOAL_CONSIST_M of their median), the goal itself is
#    wrong -> the median becomes the goal. Counted: ground-contact estimates, and point-cloud estimates only
#    if NEARER to the robot than the goal (a see-through cloud lands behind the object).
#  - arrival check: at the goal, if this or the previous tick still saw the target ARRIVE_VERIFY_M beyond the
#    robot (ground-contact; any method beyond ARRIVE_VERIFY_ANY_M), re-goal to it instead of FOUND
#    (at most ARRIVE_REGOAL_MAX times per lock).
REGOAL_WINDOW = 5
REGOAL_MIN_AGREE = 3
REGOAL_CONSIST_M = 0.7
ARRIVE_VERIFY_M = 1.2
ARRIVE_VERIFY_ANY_M = 2.0
ARRIVE_REGOAL_MAX = 3
REFINE_RESEND_M = 0.1          # re-send /way_point only when the refined goal moved at least this much
TARGET_DEPTH_PATCH_PX = 4      # half-width of the median-depth sampling patch around the box center


SCENE_MAX_DIST_M = 3.0         # non-target (scene vocabulary) detections: = scene graph MAX_DEPTH_M, so they
                               # always have depth points (no synthetic furniture nodes from 3-4 m away)
DETECT_MAX_DIST_M = 4.0        # target detections farther than this are ignored (CGFM; VLFM uses 2.5 m box-centre depth).
                               # Distance = the NEARER of box-centre depth and ground-contact range, because a
                               # see-through mesh bin's centre depth is the wall behind it (read 2.5-3.6 m at ~2 m).
SG_IMAGE_SIZE = (448, 336)    # evidence frames saved for the VLM (4:3, ~190 Qwen2.5-VL tokens each)


def save_evidence_frame(run_dir, idx, color_rgb, touched):
    """Save this tick's camera frame as scene-graph evidence: the raw frame,
    downscaled (sg_images/tick_N.jpg), plus which nodes it shows and where
    (tick_N.json). Ids are drawn only when the frame is sent to the VLM
    (render_evidence), so marks always use current ids and only listed nodes."""
    w, h = SG_IMAGE_SIZE
    im = cv2.resize(cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_AREA)
    base = os.path.join(run_dir, "sg_images", f"tick_{idx:05d}")
    cv2.imwrite(base + ".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(base + ".json", "w") as f:
        json.dump([[int(nid), lab, [round(float(v), 4) for v in box]] for nid, lab, box in touched], f)


_SOM_FONT = None


def render_evidence(run_dir, key, labels_by_id, resolve):
    """Evidence frame `key` with every node that is in labels_by_id (after id
    resolution) outlined and tagged '#id' in its class colour (Set-of-Mark).
    Big boxes are drawn first so a small object inside a big box (a trash can
    under a desk) keeps its own outline and tag on top; tags are a bold TrueType
    font with a dark outline so the VLM can read the number.
    Saved to vlm_inputs/key.jpg (what the VLM actually saw); returns that path."""
    global _SOM_FONT
    from PIL import Image, ImageDraw, ImageFont
    if _SOM_FONT is None:
        _SOM_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    base = os.path.join(run_dir, "sg_images", key)
    im = cv2.imread(base + ".jpg")
    try:
        with open(base + ".json") as f:
            marks = json.load(f)
    except (OSError, ValueError):
        marks = []
    h, w = im.shape[:2]
    best = {}   # several detections of a frame can resolve to one node (merged since): keep its biggest box
    for nid, _lab, (x1, y1, x2, y2) in marks:
        nid = resolve(nid)
        if nid in labels_by_id:
            b = (nid, int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
            if nid not in best or (b[3] - b[1]) * (b[4] - b[2]) > (best[nid][3] - best[nid][1]) * (best[nid][4] - best[nid][2]):
                best[nid] = b
    boxes = list(best.values())
    boxes.sort(key=lambda b: -(b[3] - b[1]) * (b[4] - b[2]))
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for nid, x1, y1, x2, y2 in boxes:
        b, g, r = (int(c) for c in class_color(labels_by_id[nid]))
        d.rectangle((x1, y1, x2, y2), outline=(0, 0, 0), width=4)
        d.rectangle((x1, y1, x2, y2), outline=(r, g, b), width=2)
        tag = f"#{nid}"
        l, t, rr, bb = d.textbbox((0, 0), tag, font=_SOM_FONT)
        tw, th = rr - l, bb - t
        tx = min(max(x1, 0), w - tw - 8)
        ty = y1 - th - 8 if y1 - th - 8 >= 0 else y1 + 2      # above the box, else just inside
        d.rectangle((tx, ty, tx + tw + 8, ty + th + 6), fill=(r, g, b), outline=(0, 0, 0), width=1)
        d.text((tx + 4 - l, ty + 3 - t), tag, font=_SOM_FONT, fill=(255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0))
    im = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    out = os.path.join(run_dir, "vlm_inputs", f"{key}.jpg")
    cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def crop_node(run_dir, nid, scene_graph, pad=0.2):
    """Crop of node `nid` for the VLM's crop check: its biggest box (boxes cut by the
    image border count a quarter) over all saved
    raw evidence frames (sg_images/; frames/ carries the detection overlay), padded by
    `pad` of the box size. Saved to vlm_inputs/verify_<id>.jpg; returns the path or None."""
    best = None
    for key in scene_graph.node_images(nid):
        try:
            with open(os.path.join(run_dir, "sg_images", f"{key}.json")) as f:
                marks = json.load(f)
        except (OSError, ValueError):
            continue
        for mid, _lab, box in marks:
            if scene_graph.resolve_id(mid) != scene_graph.resolve_id(nid):
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            if box[0] < 0.01 or box[2] > 0.99 or box[3] > 0.99:
                area *= 0.25        # cut off by the image border: only half an object, a poor crop
            if best is None or area >= best[0]:
                best = (area, key, box)
    if best is None:
        return None
    _, key, (x1, y1, x2, y2) = best
    im = cv2.imread(os.path.join(run_dir, "sg_images", f"{key}.jpg"))
    if im is None:
        return None
    h, w = im.shape[:2]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    c = im[max(0, int((y1 - py) * h)):min(h, int((y2 + py) * h) + 1),
           max(0, int((x1 - px) * w)):min(w, int((x2 + px) * w) + 1)]
    if c.size == 0:
        return None
    out = os.path.join(run_dir, "vlm_inputs", f"verify_{nid}.jpg")
    cv2.imwrite(out, c, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def _tick_walk_refine(idx, target, camera, pose_client, goal_xy, detector, scene_graph,
                      intr, fx, fy, run_dir, video_writer, log,
                      obstacle_map=None, floor_map=None, fov=None):
    """Post-lock tick (CGFM): read the pose, re-detect the target and estimate its
    near edge (for goal refinement by the caller), and keep recording. The scene
    graph and the occupancy/floor maps keep being updated for the record (review
    video, final export) but nothing here or in the caller reads them back while
    locked: the goal refinement uses only this frame's detection + depth, and
    there is no frontier / semantic / VLM work. Returns None if pose/camera
    weren't ready, else dict(xy, yaw, cam_xy, est, method, conf).""" 
    xy_yaw = pose_client.xy_yaw()
    color, depth_raw = camera.latest()
    if xy_yaw is None or color is None or depth_raw is None:
        return None
    xy, yaw = xy_yaw
    tf_camera_to_episodic = get_camera_transform(xy, yaw)
    detections, valid_idxs, scene_idxs = detect_scene(color, depth_raw, target, detector, intr, idx, log)
    if scene_idxs:
        try:
            touched = scene_graph.update(color, detections, scene_idxs, depth_raw, intr["depth_scale"],
                                         tf_camera_to_episodic, fx, fy, timestamp=round(time.time(), 2),
                                         image_key=f"tick_{idx:05d}")
            if touched:
                save_evidence_frame(run_dir, idx, color, touched)
        except Exception as e:
            log(f"  [warn] scene_graph.update failed: {e}")
    if obstacle_map is not None:
        try:
            obstacle_map.update_map(
                depth=normalize_depth(depth_raw, intr["depth_scale"]), tf_camera_to_episodic=tf_camera_to_episodic,
                min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, fx=fx, fy=fy,
                topdown_fov=fov, explore=True, update_obstacles=True,
            )
            obstacle_map.update_agent_traj(xy, yaw)
            if floor_map is not None:
                floor_map.update(depth_raw, intr["depth_scale"], tf_camera_to_episodic, fx, fy, xy)
        except IndexError as e:
            log(f"  [warn] pose outside map bounds: {e}")
        cv2.imwrite(os.path.join(run_dir, "occupancy_map", f"tick_{idx:05d}.png"), obstacle_map.visualize())
    est, method, conf = None, "not seen", None
    if valid_idxs:
        best_idx, est, method = estimate_target_edge(detections, valid_idxs, depth_raw, intr, fx, fy,
                                                     tf_camera_to_episodic, scene_graph)
        conf = float(detections.logits[best_idx])
    dist = float(np.hypot(goal_xy[0] - xy[0], goal_xy[1] - xy[1]))
    vis = draw_detections(cv2.cvtColor(color, cv2.COLOR_RGB2BGR), target, len(scene_graph), detections,
                          valid_idxs, scene_idxs)
    cv2.putText(vis, f"WALKING TO GOAL ({goal_xy[0]:.2f},{goal_xy[1]:.2f}) dist={dist:.2f}m",
                (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{idx:05d}.png"), depth_raw)
    cv2.imwrite(os.path.join(run_dir, "frames", f"tick_{idx:05d}.jpg"), vis)
    if video_writer is not None:
        video_writer.write(vis)
    return {"xy": xy, "yaw": yaw, "cam_xy": tf_camera_to_episodic[:2, 3], "est": est, "method": method, "conf": conf}


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


def draw_detections(color_bgr, target, n_objs, detections, valid_idxs, scene_idxs=()):
    """Draws every YOLO-World box onto a copy of the frame (green: target that
    passed the filters, orange: other vocabulary object kept for the scene
    graph, gray: rejected) plus the number of CGFM scene-graph objects, so a run
    can be visually audited frame-by-frame to see what the model detected."""
    vis = color_bgr.copy()
    height, width = vis.shape[:2]
    if detections is not None:
        for i, (box, logit, phrase) in enumerate(
            zip(detections.boxes, detections.logits, detections.phrases)
        ):
            x1, y1, x2, y2 = box.numpy() if hasattr(box, "numpy") else box
            pt1 = (int(x1 * width), int(y1 * height))
            pt2 = (int(x2 * width), int(y2 * height))
            ok = i in valid_idxs or i in scene_idxs
            color = (0, 220, 0) if i in valid_idxs else ((0, 165, 255) if ok else (120, 120, 120))
            cv2.rectangle(vis, pt1, pt2, color, 2 if ok else 1)
            label = f"{phrase} {float(logit):.2f}"
            cv2.putText(vis, label, (pt1[0], max(0, pt1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    header = f"target={target!r} scene_graph_objs={n_objs}"
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


GROUND_CONTACT_MIN_ROWS_BELOW_HORIZON = 15   # box bottom must be this far below the image centre row
GROUND_CONTACT_MAX_BOTTOM_FRAC = 0.97        # box bottom touching the image edge = base cut off, unusable


def ground_contact_xy(box, fx, fy, width, height, tf_camera_to_episodic):
    """Depth-free world (x, y) of a floor-standing object: intersect the ray through
    the box's bottom-centre pixel with the floor (camera CAMERA_HEIGHT_M above it,
    level mount, see _CAMERA_PITCH_UP_DEG). Used when the depth camera does not see
    the object at all -- e.g. a mesh wastebasket, whose IR pattern passes through
    the holes so the box holds only floor/wall depth. None if the bottom edge is
    near/above the horizon or cut off by the image border."""
    x1, y1, x2, y2 = [float(v) for v in box]
    u = (x1 + x2) / 2 * width
    v = y2 * height
    if y2 >= GROUND_CONTACT_MAX_BOTTOM_FRAC or v - height / 2 < GROUND_CONTACT_MIN_ROWS_BELOW_HORIZON:
        return None
    fwd = CAMERA_HEIGHT_M * fy / (v - height / 2)
    point_cam = np.array([fwd, -(u - width / 2) * fwd / fx, -CAMERA_HEIGHT_M, 1.0])
    return (tf_camera_to_episodic @ point_cam)[:2]


def detection_dist_m(box, depth_raw, intr):
    """Range of a detection for the distance gate: the nearer of the box-centre depth
    and the ground-contact range (bottom-edge ray meets the floor). None if neither exists."""
    box = [float(v) for v in (box.numpy() if hasattr(box, "numpy") else box)]
    w, h = intr["width"], intr["height"]
    cands = [box_center_depth_m(box, depth_raw, intr["depth_scale"], w, h)]
    v = box[3] * h
    if box[3] < GROUND_CONTACT_MAX_BOTTOM_FRAC and v - h / 2 >= GROUND_CONTACT_MIN_ROWS_BELOW_HORIZON:
        cands.append(CAMERA_HEIGHT_M * intr["fy"] / (v - h / 2))
    cands = [c for c in cands if c is not None]
    return min(cands) if cands else None


def scene_caption(target):
    """(target class names, full caption): target synonyms first, then SCENE_VOCAB."""
    target_classes = _caption_to_classes(DET_SYNONYMS.get(target, f"{target} ."))
    classes = list(dict.fromkeys(target_classes + SCENE_VOCAB))
    return target_classes, " . ".join(classes) + " ."


def detect_scene(color, depth_raw, target, detector, intr, idx, log):
    """One YOLO-World pass over target + SCENE_VOCAB.
    Returns (detections, target_idxs, scene_idxs):
      target_idxs: target-class boxes passing the usual gates (confidence, box
                   area, range <= DETECT_MAX_DIST_M (see detection_dist_m), logged if too far);
      scene_idxs:  target_idxs + up to SCENE_MAX_OBJS_PER_TICK other vocabulary
                   boxes passing the same gates.
    detections.phrases are rewritten to canonical labels (target synonyms -> the
    target name, SCENE_SYNONYMS -> their canonical class)."""
    target_classes, det_caption = scene_caption(target)
    # Defense-in-depth against a real failure mode hit live (2026-09-18): a
    # long-running YOLO-World server that's had set_classes() called with
    # many different class lists over a session can end up returning boxes
    # labeled with some OTHER class entirely (e.g. "person"/"tv"/"microwave")
    # even though it was just asked for only `det_classes` -- confirmed via a
    # fresh model instance returning zero detections on the same frame where
    # the long-lived server confidently returned unrelated classes. Restart
    # the server if this recurs; this check just stops a mislabeled box from
    # being silently treated as "found the target" regardless of the cause.
    det_classes = set(target_classes)
    vocab = set(SCENE_VOCAB) | set(SCENE_SYNONYMS)
    try:
        detections = detector.predict(color, caption=det_caption)
    except Exception as e:
        log(f"  [warn] YOLO-World failed: {e}")
        return None, [], []
    raw = list(detections.phrases)

    def passes(i):
        return (float(detections.logits[i]) >= DINO_CONF_THRESHOLD
                and box_area_frac(detections.boxes[i]) <= MAX_BOX_AREA_FRAC)

    def near(i, limit=DETECT_MAX_DIST_M):
        d = detection_dist_m(detections.boxes[i], depth_raw, intr)
        return d is None or d <= limit     # no range at all: let the scene graph decide

    target_idxs = [i for i, p in enumerate(raw) if p in det_classes and passes(i)]
    far = [i for i in target_idxs if not near(i)]
    for i in far:
        log(f"  [tick {idx}] Found but too far: {detections.phrases[i]!r} conf={float(detections.logits[i]):.2f} "
            f"dist={detection_dist_m(detections.boxes[i], depth_raw, intr):.2f}m > {DETECT_MAX_DIST_M:.1f}m -- ignored")
    target_idxs = [i for i in target_idxs if i not in far]
    other = []
    for i, p in enumerate(raw):
        if p in det_classes or p not in vocab or not passes(i) or not near(i, SCENE_MAX_DIST_M):
            continue
        other.append(i)
    other = sorted(other, key=lambda i: float(detections.logits[i]), reverse=True)[:SCENE_MAX_OBJS_PER_TICK]
    detections.phrases = [target if p in det_classes else SCENE_SYNONYMS.get(p, p) for p in raw]
    return detections, target_idxs, target_idxs + other


def estimate_target_edge(detections, valid_idxs, depth_raw, intr, fx, fy, tf_camera_to_episodic, scene_graph):
    """World (x, y) of the NEAR EDGE of the best single-object detection, from two
    independent measurements of the side facing the robot:
      - point cloud: the closest part of the scene graph's floor-free foreground cluster;
      - ground contact: where the box's bottom edge meets the floor (depth-free).
    When both exist the NEARER one is used: for a see-through object (mesh bin) the
    point cloud lands on whatever is behind it (measured 0.55-1.02 m farther), and a
    chair's wheel base is its real front edge on the floor; erring near only stops
    the robot a little early. Returns (best_idx, xy or None, method)."""
    single_idxs = LightweightSceneGraph.drop_group_boxes(detections, valid_idxs) or valid_idxs
    best_idx = max(single_idxs, key=lambda i: detections.logits[i])
    cloud = scene_graph.estimate_box_xy(
        detections.boxes[best_idx], depth_raw, intr["depth_scale"], tf_camera_to_episodic,
        fx, fy, intr["width"], intr["height"], near_edge=True,
    )
    ground = ground_contact_xy(detections.boxes[best_idx].numpy(), fx, fy, intr["width"], intr["height"],
                               tf_camera_to_episodic)
    if cloud is None and ground is None:
        return best_idx, None, "none"
    if cloud is None:
        return best_idx, ground, "ground-contact"
    if ground is None:
        return best_idx, cloud, "point-cloud"
    cam = tf_camera_to_episodic[:2, 3]
    if np.linalg.norm(ground - cam) < np.linalg.norm(cloud - cam):
        return best_idx, ground, "ground-contact"
    return best_idx, cloud, "point-cloud"


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
SCAN_ROT_MIN_WZ = 0.25         # rad/s, stay above the Go2's small-command deadband while |err| > tol
SCAN_ROT_TOL_DEG = 8.0         # step counts as reached within this, held for 3 loop ticks
SCAN_ROT_LOOP_HZ = 20.0
SCAN_ROT_STEP_TIMEOUT_S = 10.0
SCAN_ROT_RESPONSE_CHECK_S = 2.5    # by now the yaw error must have shrunk by >= MIN_PROGRESS...
SCAN_ROT_MIN_PROGRESS_DEG = 5.0    # ...or the rotation isn't working (wrong sign / robot not
                                   # responding / frozen pose): stop and abort the scan
SCAN_PERCEIVE_TICKS = 2        # perception ticks captured at rest after each step
SCAN_MAX_DRIFT_M = 1.0         # rotation-only must not translate; abort if it does


def _tick_perception(idx, target, camera, pose_client, obstacle_map, scene_graph,
                      detector, intr, fx, fy, fov, run_dir, log,
                      video_writer=None, floor_map=None):
    """One perception+mapping tick, shared by the initial 360-scan and the
    main explore loop: capture a frame, save it, run YOLO-World detection,
    update the obstacle map + scene graph, save snapshots. Returns None if
    pose/camera data wasn't ready yet this tick."""
    xy_yaw = pose_client.xy_yaw()
    color, depth_raw = camera.latest()
    if xy_yaw is None or color is None or depth_raw is None:
        return None
    xy, yaw = xy_yaw
    depth_norm = normalize_depth(depth_raw, intr["depth_scale"])
    tf_camera_to_episodic = get_camera_transform(xy, yaw)

    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{idx:05d}.png"), depth_raw)
    depth_vis = cv2.applyColorMap(
        cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        cv2.COLORMAP_JET,
    )
    depth_vis[depth_raw == 0] = (0, 0, 0)
    cv2.imwrite(os.path.join(run_dir, "depth", f"tick_{idx:05d}_vis.jpg"), depth_vis)

    detections, valid_idxs, scene_idxs = detect_scene(color, depth_raw, target, detector, intr, idx, log)
    target_seen = len(valid_idxs) > 0

    if scene_idxs and scene_graph is not None:
        try:
            touched = scene_graph.update(
                color, detections, scene_idxs,
                depth_raw, intr["depth_scale"],
                tf_camera_to_episodic, fx, fy, timestamp=round(time.time(), 2),
                image_key=f"tick_{idx:05d}",
            )
            if touched:
                save_evidence_frame(run_dir, idx, color, touched)
        except Exception as e:
            log(f"  [warn] scene_graph.update failed: {e}")

    n_objs = len(scene_graph) if scene_graph is not None else 0
    det_vis = draw_detections(color_bgr, target, n_objs, detections, valid_idxs, scene_idxs)
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
        if floor_map is not None:
            floor_map.update(depth_raw, intr["depth_scale"], tf_camera_to_episodic, fx, fy, xy)
    except IndexError as e:
        log(f"  [warn] pose outside map bounds: {e}")

    cv2.imwrite(os.path.join(run_dir, "occupancy_map", f"tick_{idx:05d}.png"), obstacle_map.visualize())

    return {
        "xy": xy, "yaw": yaw, "n_objs": n_objs, "target_seen": target_seen,
        "valid_idxs": valid_idxs, "detections": detections,
        "frontiers": obstacle_map.frontiers,
        "depth_raw": depth_raw, "tf_camera_to_episodic": tf_camera_to_episodic,
        "color": color,
    }


def _make_semantic_map_vis(sem_map: np.ndarray, obstacle_map, seen_mask=None) -> np.ndarray:
    """PLASMA heatmap of semantic scores overlaid on explored free space
    (fog-of-war explored area OR observed floor, i.e. what the diffusion uses).

    Same base colours as the top-down / scene-graph panels: unexplored white,
    agent-radius padding gray, obstacles black. Explored free space: PLASMA
    colormap, brighter = higher semantic score toward the current target.
    """
    size = obstacle_map._map.shape[0]
    vis = np.full((size, size, 3), 255, dtype=np.uint8)
    seen = obstacle_map.explored_area.astype(bool)
    if seen_mask is not None:
        seen = seen | seen_mask
    vis[obstacle_map._navigable_map == 0] = (150, 150, 150)
    vis[obstacle_map._map == 1] = (0, 0, 0)

    if sem_map.max() > 1e-6:
        norm = np.clip(sem_map / sem_map.max(), 0, 1)
    else:
        norm = np.zeros_like(sem_map)

    color_img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    mask = seen & obstacle_map._navigable_map.astype(bool)
    vis[mask] = color_img[mask]

    return cv2.flip(vis, 0)


def _make_scene_graph_vis(
    obstacle_map,
    nodes: list,
    robot_xy: np.ndarray,
    frontiers: np.ndarray,
    target: str = None,
) -> np.ndarray:
    """Draw scene graph objects and frontiers on a copy of the occupancy map.

    nodes: (score, world_xy, label, id) from scene_graph.get_scored_labeled().
    Each object is a dot in its class colour (scene_graph.class_color), no text;
    the target class is drawn larger with a thick outline. The colour -> class
    legend is added by the composite video (from scene_graph_legend.json), not
    here, so it doesn't change the maps' shared crop window.
    Frontiers: small blue circles. Robot: filled cyan circle.

    Draws in pre-flip map space (same convention as obstacle_map.visualize())
    then flips at the end, so objects land at correct world positions.
    """
    vis = np.ones((*obstacle_map._map.shape[:2], 3), dtype=np.uint8) * 255
    vis[obstacle_map.explored_area == 1] = (200, 255, 200)
    vis[obstacle_map._navigable_map == 0] = (150, 150, 150)
    vis[obstacle_map._map == 1] = (0, 0, 0)

    # Frontiers: small blue circles
    if len(frontiers) > 0:
        for f in frontiers:
            px = obstacle_map._xy_to_px(np.array([[f[0], f[1]]]))[0]
            cv2.circle(vis, (int(px[0]), int(px[1])), 5, (200, 80, 0), 2)

    # Scene graph objects: one colour per class; target class on top, larger
    for node in sorted(nodes, key=lambda n: n[2] == target):
        world_xy, label = node[1], node[2]
        px = obstacle_map._xy_to_px(np.asarray(world_xy).reshape(1, 2))[0]
        cx, cy = int(px[0]), int(px[1])
        is_target = label == target
        r = 10 if is_target else 7
        cv2.circle(vis, (cx, cy), r, class_color(label), -1)
        cv2.circle(vis, (cx, cy), r, (0, 0, 0), 3 if is_target else 1)

    # Robot position: filled cyan circle
    if robot_xy is not None:
        px = obstacle_map._xy_to_px(np.array([[robot_xy[0], robot_xy[1]]]))[0]
        cv2.circle(vis, (int(px[0]), int(px[1])), 7, (255, 220, 0), -1)
        cv2.circle(vis, (int(px[0]), int(px[1])), 7, (0, 0, 0), 1)

    return cv2.flip(vis, 0)


def _write_legend(path: str, nodes: list, target: str = None) -> None:
    """scene_graph_legend.json: [[label, [b, g, r], is_target], ...] for the classes
    currently in the graph (target first) -- read by make_cgfm_composite_video.py."""
    labels = sorted({n[2] for n in nodes}, key=lambda l: (l != target, l))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump([[l, list(class_color(l)), l == target] for l in labels], f)
    os.replace(tmp, path)


class _NoopSender:
    """Stand-in for WaypointUdpSender/CmdVelUdpSender in --dry-run: drops everything."""

    def send(self, *args) -> None:
        pass


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


SCAN_RESULT = {"outcome": None}   # last initial-scan outcome, read before arming


def perform_initial_scan(xy0, yaw0, camera, pose_client, cmdvel_sender,
                          obstacle_map, scene_graph, detector,
                          intr, fx, fy, fov, run_dir, log, target, frame_idx,
                          video_writer=None, should_stop=lambda: False, floor_map=None):
    """Turn a full 360 degrees in SCAN_STEPS steps before exploring, so the
    obstacle/value maps already cover every direction from the start spot.

    Runs BEFORE pathFollower is started: this function alone owns /cmd_vel
    (via cmdvel_udp_relay.py, yaw rate only), rotating closed-loop to absolute
    headings yaw0 + k*360/SCAN_STEPS and capturing/mapping SCAN_PERCEIVE_TICKS
    frames at rest after each step (sharper than capturing mid-turn). The last
    step brings the robot back to its original heading.

    Aborts (stops rotating, returns early) if the rotation doesn't respond,
    the robot drifts > SCAN_MAX_DRIFT_M, or two steps in a row time out.
    Returns the updated frame_idx."""
    log(f"Initial 360-degree scan: {SCAN_STEPS} steps of {360 / SCAN_STEPS:.0f}deg, closed-loop yaw control "
        f"(pathFollower not running; tol {SCAN_ROT_TOL_DEG:.0f}deg, max {np.rad2deg(SCAN_ROT_MAX_WZ):.0f}deg/s, "
        f"abort if drift from start exceeds {SCAN_MAX_DRIFT_M}m)...")

    def perceive(step_label):
        nonlocal frame_idx
        for _ in range(SCAN_PERCEIVE_TICKS):
            tick_t0 = time.time()
            frame_idx += 1
            result = _tick_perception(frame_idx, target, camera, pose_client, obstacle_map,
                                       scene_graph, detector, intr, fx, fy, fov,
                                       run_dir, log, video_writer, floor_map=floor_map)
            if result is None:
                frame_idx -= 1
            else:
                drift = float(np.hypot(result["xy"][0] - xy0[0], result["xy"][1] - xy0[1]))
                log(f"  [scan {step_label} tick {frame_idx}] pose=({result['xy'][0]:+.2f},{result['xy'][1]:+.2f}) "
                    f"yaw={np.rad2deg(result['yaw']):+.0f}deg frontiers={len(result['frontiers'])} "
                    f"drift_from_start={drift:.2f}m")
            elapsed = time.time() - tick_t0
            if elapsed < LOOP_INTERVAL_S:
                time.sleep(LOOP_INTERVAL_S - elapsed)

    perceive(f"0/{SCAN_STEPS} (start heading {np.rad2deg(yaw0):+.0f}deg)")
    timeouts_in_a_row = 0
    for step in range(1, SCAN_STEPS + 1):
        target_yaw = _wrap_pi(yaw0 + step * (2 * np.pi / SCAN_STEPS))
        t_step = time.time()
        outcome, err = rotate_to_yaw(target_yaw, pose_client, cmdvel_sender, xy0, should_stop)
        _stop_rotation(cmdvel_sender)
        log(f"  [scan {step}/{SCAN_STEPS}] rotate to {np.rad2deg(target_yaw):+.0f}deg: {outcome} "
            f"(final yaw error {np.rad2deg(err):+.1f}deg, {time.time() - t_step:.1f}s)")
        if outcome in ("no_response", "drift", "stopped", "no_pose"):
            SCAN_RESULT["outcome"] = outcome
            log(f"  [SCAN ABORTED] rotation outcome {outcome!r} -- robot left stationary.")
            if pose_client.health():
                log(f"  [SCAN] {pose_client.health()} -- the 'no_response' is SLAM, not the robot.")
            return frame_idx
        timeouts_in_a_row = timeouts_in_a_row + 1 if outcome == "timeout" else 0
        if timeouts_in_a_row >= 2:
            log("  [SCAN ABORTED] two rotation steps in a row timed out -- robot left stationary.")
            return frame_idx
        if step < SCAN_STEPS:
            perceive(f"{step}/{SCAN_STEPS}")
    log("Scan complete (back at the start heading).")
    SCAN_RESULT["outcome"] = "ok"
    return frame_idx


def next_run_dir(base_dir: str) -> str:
    path = os.path.join(base_dir, "cgfm_pipeline_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    for sub in ("frames", "occupancy_map", "semantic_map", "scene_graph", "depth", "sg_images", "vlm_inputs"):
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
    parser.add_argument("--vlm", action="store_true",
                        help="VLM gatekeeper: the target is locked only when the VLM, given the scene graph as "
                             "text (node id + class), names the id of the target (MSGNav-style). Needs the server "
                             "from launch_cgfm_vlm_server.sh; if it is down/fails, the detection rules are used.")
    parser.add_argument("--vlm-url", default=VLM_DEFAULT_URL, help="OpenAI-compatible base URL of the VLM server.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Perception/inference only, NO motion: no initial scan rotation, no "
                              "waypoint/cmdvel relays, no pathFollower, no /liedown. Robot never "
                              "receives a command; decisions are only logged/visualised.")
    args = parser.parse_args()

    run_dir = next_run_dir(RES_DIR)
    log_path = os.path.join(run_dir, "log.txt")
    log_f = open(log_path, "a")

    # CGFM module diagnostics ([FrontierScore] per-frontier scores, [SceneGraph]
    # new objects, [SemanticMap] stats) go through `logging`; send them to
    # cgfm_debug.log in the run dir (not the terminal). Only these prefixes are
    # kept so third-party INFO logs don't flood the file.
    _cgfm_prefixes = ("[tick", "[FrontierScore]", "[SceneGraph]", "[SemanticMap]")
    _dbg = logging.FileHandler(os.path.join(run_dir, "cgfm_debug.log"))
    _dbg.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _dbg.addFilter(lambda rec: str(rec.getMessage()).startswith(_cgfm_prefixes))
    logging.getLogger().addHandler(_dbg)
    logging.getLogger().setLevel(logging.INFO)

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
    if args.dry_run:
        print("CGFM full-pipeline master script -- DRY RUN (NO MOTION)")
        print("Only the pose relay is started; no scan rotation, no waypoint/cmdvel relay,")
        print("no pathFollower, no /liedown. Decisions are logged but never sent.")
    else:
        print("CGFM full-pipeline master script -- REAL ROBOT MOTION")
        print("Stop mechanism: kill pathFollower (primary), /liedown best-effort (secondary).")
        print("A human MUST be holding the physical remote right now.")
    print("=" * 70)
    if not args.skip_confirm and not args.dry_run:
        confirmation = input("Type ARM to proceed, anything else aborts: ").strip()
        if confirmation != "ARM":
            print("Aborted, nothing started.")
            return

    state = {"target": args.target, "stop": False}
    state_lock = threading.Lock()

    log("Cleaning up any stale relay/pathFollower processes...")
    _pkill("pose_udp_relay.py")
    _pkill("waypoint_udp_relay.py")
    _pkill("cmdvel_udp_relay.py")
    _pkill(PATHFOLLOWER_MATCH)
    time.sleep(1.0)

    if args.dry_run:
        # DRY RUN: only the pose relay (ROS -> vlfm, read-only). The two relays
        # that can reach /way_point or /cmd_vel are never started, and the
        # senders are no-ops, so nothing leaves this process toward the robot.
        log("[DRY-RUN] Starting pose_udp_relay.py only (no waypoint/cmdvel relay) ...")
        pose_relay_proc = _ros_popen(
            f"/usr/bin/python3 {POSE_RELAY_SCRIPT}", log_path=os.path.join(run_dir, "pose_udp_relay.log"))
    else:
        log("Starting pose_udp_relay.py + waypoint_udp_relay.py + cmdvel_udp_relay.py (rotation-only) ...")
        pose_relay_proc = _ros_popen(
            f"/usr/bin/python3 {POSE_RELAY_SCRIPT}", log_path=os.path.join(run_dir, "pose_udp_relay.log"))
        waypoint_relay_proc = _ros_popen(
            f"/usr/bin/python3 {WAYPOINT_RELAY_SCRIPT}", log_path=os.path.join(run_dir, "waypoint_udp_relay.log"))
        cmdvel_relay_proc = _ros_popen(
            f"/usr/bin/python3 {CMDVEL_RELAY_SCRIPT}", log_path=os.path.join(run_dir, "cmdvel_udp_relay.log"))
    time.sleep(2.0)

    camera = _RealSenseStreamClient(JETSON_HOST, CAMERA_PORT)
    pose_client = PoseUdpClient(POSE_UDP_PORT)
    if args.dry_run:
        waypoint_sender = _NoopSender()
        cmdvel_sender = _NoopSender()
    else:
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

    xy0, yaw0 = pose_client.xy_yaw()
    log(f"Pinning startup goal to current position {tuple(xy0.round(3))} before arming...")
    for _ in range(3):
        waypoint_sender.send(float(xy0[0]), float(xy0[1]), 0.0)
        time.sleep(0.2)

    # CGFM-only subclass: fixes the swapped cone centre in vlfm's clear-view reveal
    # (see src/cgfm/obstacle_map_cgfm.py); the original VLFM pipeline is unaffected.
    obstacle_map = CGFMObstacleMap(
        min_height=MIN_HEIGHT, max_height=MAX_HEIGHT, agent_radius=0.2,
        area_thresh=0.5, hole_area_thresh=-1, size=MAP_SIZE, pixels_per_meter=PIXELS_PER_METER,
    )
    # CGFM: cells where floor was actually observed; widens the (fog-of-war)
    # explored area that semantics diffuse over (see src/cgfm/floor_map.py).
    floor_map = ObservedFloorMap(obstacle_map, MIN_HEIGHT, MIN_DEPTH, MAX_DEPTH)
    vlm = None
    vlm_events_path = os.path.join(run_dir, "vlm_events.jsonl")

    def vlm_event(tick, kind, **kw):
        """Tick-stamped VLM interaction record (asked / none / picked / gone /
        failed / target / found), drawn by make_cgfm_composite_video.py."""
        try:
            with open(vlm_events_path, "a") as f:
                f.write(json.dumps({"tick": int(tick), "kind": kind, **kw}) + "\n")
        except OSError:
            pass

    if args.vlm:
        vlm = VLMTargetSelector(args.vlm_url, log_path=os.path.join(run_dir, "vlm_log.jsonl"))
        vlm_event(0, "on" if vlm.available else "unavailable", target=args.target, url=args.vlm_url)
        log(f"VLM gatekeeper: {'ON' if vlm.available else 'REQUESTED BUT SERVER NOT REACHABLE'} "
            f"({args.vlm_url}) -- " + ("the target is locked only on the VLM's pick from the scene graph"
                                       if vlm.available else "falling back to the detection lock rules"))
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    clip_model = clip_model.cuda().eval()
    scene_graph = LightweightSceneGraph(clip_model, clip_preprocess, device="cuda", min_z=MIN_HEIGHT,
                                        camera_height=CAMERA_HEIGHT_M)
    cached_target: str = ""
    cached_text_feat = None

    # Scene graph export (every perception/walk tick, and at the end): JSON for
    # tools / a VLM, plain text ready for a prompt. Robot-relative fields use the
    # pose of the tick that triggered the export.
    run_t0 = time.time()
    last_pose = None

    def export_graph(xy, yaw):
        nonlocal last_pose
        if xy is not None:
            last_pose = (xy, yaw)
        try:
            scene_graph.export(os.path.join(run_dir, "scene_graph.json"), os.path.join(run_dir, "scene_graph.txt"),
                               robot_xy=xy, robot_yaw=yaw, target=state["target"], text_feat=cached_text_feat,
                               stamp=round(time.time() - run_t0, 1))
        except Exception as e:  # noqa: BLE001
            log(f"  [warn] scene-graph export failed: {e}")
    detector = YOLOWorldClient(port=12185)

    stdin_reader = StdinCommands()
    stdin_reader.start()

    def request_stop(signum=None, frame=None) -> None:
        _pkill(PATHFOLLOWER_MATCH)  # cut /cmd_vel NOW, even if the main thread is stuck in a blocking VLM call
        with state_lock:
            state["stop"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # Everything from here on (scan included) must go through the same
    # kill-pathFollower-then-liedown cleanup in `finally` -- start_pathfollower
    # is called INSIDE the try so a Ctrl-C mid-scan still stops the robot
    # via the normal path instead of an uncaught KeyboardInterrupt leaving
    # it armed.
    frame_idx = 0
    last_way_point = None
    hit_window = deque(maxlen=TARGET_LOCK_WINDOW)  # (seen, world-xy estimate or None) per tick
    target_lock_xy = None
    target_best_range = None   # camera->goal range of the estimate the current goal came from
    last_frontier_xy = None    # frontier pursued last tick (hysteresis)
    frontier_blacklist = []    # frontiers the robot stalled on (never chosen again)
    locked_ticks = 0           # ticks since the current lock (0 = not locked)
    regoal_window = deque(maxlen=REGOAL_WINDOW)   # per locked tick: gate-rejected estimate or None
    recent_seen = deque(maxlen=2)                 # per locked tick: (est, method, range) or None
    arrive_regoals = 0
    stall_hist = deque()       # (time, xy) while exploring, for the stall check
    def should_stop() -> bool:
        with state_lock:
            return state["stop"]

    try:
        # Set before the 360 scan too (it runs before the main loop, which re-sets them every tick):
        # otherwise every class could get synthetic ground-contact nodes during the scan (2026-10-03
        # run 160003: desk/chairs placed in free space from far, short ground-contact estimates).
        scene_graph.synthetic_labels = {state["target"]}
        scene_graph.protected_labels = {state["target"]}
        why = slam_preflight()
        if why and not args.dry_run:
            log(f"NOT STARTING: {why}. Restart the nav stack (teardown + bringup) first.")
            return
        if why:
            log(f"  [warn] {why} (dry-run: continuing)")
        if args.dry_run:
            log("[DRY-RUN] Skipping the initial 360-degree scan rotation and pathFollower. "
                "Move/turn the robot with the remote if you want it to see more.")
        else:
            # The scan runs FIRST, with pathFollower deliberately NOT started: the
            # scan alone owns /cmd_vel (yaw rate only, via cmdvel_udp_relay.py).
            frame_idx = perform_initial_scan(
                xy0, yaw0, camera, pose_client, cmdvel_sender, obstacle_map, scene_graph,
                detector, intr, fx, fy, fov, run_dir, log, state["target"], 0,
                video_writer, should_stop, floor_map=floor_map,
            )
            if should_stop():
                log("Stop requested during the scan -- not arming pathFollower.")
                return
            time.sleep(POSE_STALE_S)   # let a just-died stream show up as stale
            why = pose_client.health() or slam_preflight()
            if not why and SCAN_RESULT["outcome"] in ("drift", "no_pose"):
                why = (f"the scan aborted with {SCAN_RESULT['outcome']!r} (the pose moved > {SCAN_MAX_DRIFT_M} m "
                       f"while only turning, or vanished) -- localization is wrong")
            if why:
                log(f"NOT ARMING: {why}. Check /tmp/go2_stack/slam.log and restart the nav stack.")
                return
            if SCAN_RESULT["outcome"] == "no_response":
                log("  [warn] the scan got no rotation response but SLAM looks healthy -- if the robot does not "
                    "move, the WebRTC motion link is stale: restart unitree_control and test /hello.")

            # Hand /cmd_vel over to pathFollower: pin its goal to wherever the robot
            # is NOW (so it can't lurch toward a stale goal), give the relay time to
            # finish its zero burst and go silent, then arm.
            cur_xy, _cur_yaw = pose_client.xy_yaw()
            for _ in range(3):
                waypoint_sender.send(float(cur_xy[0]), float(cur_xy[1]), 0.0)
                time.sleep(0.2)
            time.sleep(0.6)
            log("Starting pathFollower with autonomyMode=true (ARMED)...")
            start_pathfollower(autonomy=True, log_path=os.path.join(run_dir, "pathFollower_debug.log"))

        start_time = time.time()
        jump_ref = {"xy": None, "t": None}
        try:
            slam_log_pos = os.path.getsize(SLAM_LOG)
        except OSError:
            slam_log_pos = None

        def pose_jump(xy_now):
            """Localization-jump check for this tick. Returns a reason string if the pose
            jumped (armed: the caller stops the run), else None. Also reports new SLAM
            'failureDetected' lines (warning only)."""
            nonlocal slam_log_pos
            msg = pose_client.health()
            now = time.time()
            if jump_ref["xy"] is not None:
                dt = now - jump_ref["t"]
                jump = float(np.hypot(xy_now[0] - jump_ref["xy"][0], xy_now[1] - jump_ref["xy"][1]))
                allowed = max(POSE_JUMP_M, POSE_JUMP_SPEED_MPS * dt)
                if msg is None and jump > allowed:
                    msg = (f"localization jump {jump:.2f}m in {dt:.1f}s (> {allowed:.2f}m): "
                           f"({jump_ref['xy'][0]:.2f},{jump_ref['xy'][1]:.2f}) -> ({xy_now[0]:.2f},{xy_now[1]:.2f})")
            jump_ref["xy"], jump_ref["t"] = (float(xy_now[0]), float(xy_now[1])), now
            if slam_log_pos is not None:
                try:
                    with open(SLAM_LOG, "rb") as f:
                        f.seek(slam_log_pos)
                        new = f.read()
                        slam_log_pos = f.tell()
                    if b"process has died" in new and msg is None:
                        msg = "a SLAM process died (slam.log: 'process has died')"
                    n_fail = new.count(b"failureDetected")
                    if n_fail:
                        log(f"  [warn] SLAM reported failureDetected x{n_fail} (IMU preintegration reset) -- "
                            f"watch the pose")
                except OSError:
                    pass
            return msg

        def stop_on_jump(msg):
            """True if the run must stop because of a localization jump."""
            if msg is None:
                return False
            if args.dry_run:
                log(f"  [warn] {msg} (dry-run: not stopping)")
                return False
            log(f"STOPPING: {msg} -- SLAM is unreliable, not sending more goals. "
                f"Check /tmp/go2_stack/slam.log and restart the nav stack.")
            if vlm is not None:
                vlm_event(frame_idx, "failed", error="SLAM fault -- run stopped", unavailable=False)
            with state_lock:
                state["stop"] = True
            return True
        log(f"{'DRY-RUN (no motion)' if args.dry_run else 'ARMED'}. target={state['target']!r}. "
            f"Type 'target <name>' or 'stop' anytime. "
            f"Hard cap {args.max_seconds:.0f}s (scan time not counted against this).")

        while True:
            with state_lock:
                if state["stop"]:
                    break
            if time.time() - start_time > args.max_seconds:
                log("Max runtime reached, stopping.")
                break

            cmd = stdin_reader.poll()
            if cmd:
                if cmd == "stop":
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
                        target_best_range = None
                        last_frontier_xy = None
                        if vlm is not None:
                            vlm.reset()
                        last_way_point = None
                        cached_target = ""   # force text-feat re-encode on next tick
                        # scene_graph intentionally NOT cleared: history persists
                        log(f">>> target changed to {new_target!r}")
                        if vlm is not None:
                            vlm_event(frame_idx, "target", target=new_target)
                else:
                    log(f"(unrecognized command: {cmd!r})")

            loop_t0 = time.time()
            frame_idx += 1
            target = state["target"]
            scene_graph.protected_labels = {target}   # never merged into another class
            scene_graph.synthetic_labels = {target}   # only the target may get ground-contact (synthetic) nodes
            if target_lock_xy is None:
                locked_ticks = 0

            if target_lock_xy is not None:
                # LOCKED: no exploration, no frontier / semantic / VLM decision.
                # localPlanner/pathFollower own getting to the goal. The occupancy map
                # and scene graph are still updated and drawn (record only, never read
                # back while locked).
                # CGFM: the target is still detected each tick, and a sighting from
                # CLOSER than the one the goal came from (closer = more accurate)
                # replaces the goal -- unless it is > REFINE_GATE_M away from it
                # (another object / false detection) or would move the goal more than
                # REFINE_PUSHBACK_M farther from the robot (point cloud seeing through). Not seen -> goal unchanged.
                walk = _tick_walk_refine(frame_idx, target, camera, pose_client, target_lock_xy, detector,
                                         scene_graph, intr, fx, fy, run_dir, video_writer, log,
                                         obstacle_map=obstacle_map, floor_map=floor_map, fov=fov)
                if walk is None:
                    frame_idx -= 1
                    time.sleep(LOOP_INTERVAL_S)
                    continue
                xy = walk["xy"]
                if stop_on_jump(pose_jump(xy)):
                    break
                if locked_ticks == 0:          # a new lock: start its re-goal bookkeeping afresh
                    regoal_window.clear()
                    recent_seen.clear()
                    arrive_regoals = 0
                locked_ticks += 1
                note = ""
                est = walk["est"]
                gate_est = None
                if est is not None:
                    rng = float(np.linalg.norm(est - walk["cam_xy"]))
                    jump = float(np.linalg.norm(est - target_lock_xy))
                    if jump > REFINE_GATE_M:
                        if (walk["method"] == "ground-contact"
                                or np.linalg.norm(est - xy) < np.linalg.norm(target_lock_xy - xy)):
                            gate_est = np.asarray(est, dtype=float)
                        regoal_window.append(gate_est)
                        cons = [e for e in regoal_window if e is not None]
                        med = np.median(np.array(cons), axis=0) if len(cons) >= REGOAL_MIN_AGREE else None
                        if med is not None and max(float(np.linalg.norm(e - med)) for e in cons) <= REGOAL_CONSIST_M:
                            old = target_lock_xy
                            target_lock_xy = med
                            target_best_range = float(np.linalg.norm(med - walk["cam_xy"]))
                            regoal_window.clear()
                            waypoint_sender.send(float(med[0]), float(med[1]), 0.0)
                            last_way_point = (float(med[0]), float(med[1]))
                            note = (f"  [RE-GOAL] {len(cons)} consistent estimates disagree with the goal by "
                                    f">{REFINE_GATE_M}m -> goal ({old[0]:.2f},{old[1]:.2f}) -> ({med[0]:.2f},{med[1]:.2f}) "
                                    f"[/way_point sent]")
                        else:
                            note = (f"  [ignored {walk['method']} est ({est[0]:.2f},{est[1]:.2f}): "
                                    f"{jump:.2f}m from goal > {REFINE_GATE_M}m; "
                                    f"{len(cons)}/{REGOAL_MIN_AGREE} for re-goal]")
                    elif (walk["method"] == "point-cloud"
                          and np.linalg.norm(est - xy) - np.linalg.norm(target_lock_xy - xy) > REFINE_PUSHBACK_M):
                        note = (f"  [ignored {walk['method']} est ({est[0]:.2f},{est[1]:.2f}): "
                                f"{np.linalg.norm(est - xy) - np.linalg.norm(target_lock_xy - xy):.2f}m farther "
                                f"from the robot than the goal > {REFINE_PUSHBACK_M}m (see-through?)]")
                    elif target_best_range is not None and rng >= target_best_range:
                        note = (f"  [{walk['method']} est ({est[0]:.2f},{est[1]:.2f}) from range {rng:.2f}m, "
                                f"not closer than {target_best_range:.2f}m -> goal kept]")
                    else:
                        old = target_lock_xy
                        target_lock_xy, target_best_range = np.asarray(est, dtype=float), rng
                        note = (f"  refine goal ({old[0]:.2f},{old[1]:.2f}) -> ({est[0]:.2f},{est[1]:.2f}) "
                                f"from range {rng:.2f}m [{walk['method']}]")
                        if last_way_point is None or np.hypot(target_lock_xy[0] - last_way_point[0],
                                                              target_lock_xy[1] - last_way_point[1]) >= REFINE_RESEND_M:
                            waypoint_sender.send(float(target_lock_xy[0]), float(target_lock_xy[1]), 0.0)
                            last_way_point = (float(target_lock_xy[0]), float(target_lock_xy[1]))
                            note += " [/way_point sent]"
                elif walk["method"] == "none":
                    note = "  [target seen, but no position estimate]"
                if gate_est is None and not (est is not None and jump > REFINE_GATE_M):
                    regoal_window.append(None)
                recent_seen.append((np.asarray(est, dtype=float), walk["method"], rng) if est is not None else None)
                dist_to_target = float(np.hypot(target_lock_xy[0] - xy[0], target_lock_xy[1] - xy[1]))
                export_graph(xy, walk["yaw"])
                # Redraw the scene-graph and semantic-field tiles every tick (no frontiers while
                # locked) so the review video is not held at the lock tick; drawing only, no decision uses them.
                if cached_text_feat is not None:
                    try:
                        sg_nodes = scene_graph.get_scored_labeled(cached_text_feat)
                        _write_legend(os.path.join(run_dir, "scene_graph_legend.json"), sg_nodes, target)
                        cv2.imwrite(
                            os.path.join(run_dir, "scene_graph", f"tick_{frame_idx:05d}.png"),
                            _make_scene_graph_vis(obstacle_map, sg_nodes, xy, np.empty((0, 2)), target),
                        )
                        # Semantic field tile too (own variable: the explore branch's sem_map is untouched)
                        walk_sem = compute_semantic_map(obstacle_map, scene_graph.get_scored_objects(cached_text_feat),
                                                        seen_mask=floor_map.mask,
                                                        semantic_score_baseline=SEMANTIC_SCORE_BASELINE)
                        cv2.imwrite(
                            os.path.join(run_dir, "semantic_map", f"tick_{frame_idx:05d}.png"),
                            _make_semantic_map_vis(walk_sem, obstacle_map, floor_map.mask),
                        )
                    except Exception as e:
                        log(f"  [warn] scene-graph vis save failed: {e}")
                log(f"[tick {frame_idx:5d} t={time.time()-start_time:6.1f}s target={target!r}] "
                    f"pose=({xy[0]:+.2f},{xy[1]:+.2f}) yaw={np.rad2deg(walk['yaw']):+.0f}deg  Walking toward the goal "
                    f"({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}) dist={dist_to_target:.2f}m"
                    + (f"  seen conf={walk['conf']:.2f}" if walk["conf"] is not None else "  (target not seen)")
                    + note)
                if dist_to_target <= TARGET_STOP_RADIUS_M:
                    far = [r for r in recent_seen if r is not None and (
                        (r[1] == "ground-contact" and r[2] > ARRIVE_VERIFY_M) or r[2] > ARRIVE_VERIFY_ANY_M)]
                    if far and arrive_regoals < ARRIVE_REGOAL_MAX:
                        e, m, r = far[-1]
                        arrive_regoals += 1
                        old = target_lock_xy
                        target_lock_xy, target_best_range = e.copy(), r
                        regoal_window.clear()
                        recent_seen.clear()
                        waypoint_sender.send(float(e[0]), float(e[1]), 0.0)
                        last_way_point = (float(e[0]), float(e[1]))
                        log(f"  [NOT FOUND YET] at the goal ({old[0]:.2f},{old[1]:.2f}) but {target!r} was just seen "
                            f"{r:.2f}m ahead ({m}) -> re-goal ({e[0]:.2f},{e[1]:.2f}) "
                            f"[{arrive_regoals}/{ARRIVE_REGOAL_MAX}] [/way_point sent]")
                        elapsed = time.time() - loop_t0
                        if elapsed < LOOP_INTERVAL_S:
                            time.sleep(LOOP_INTERVAL_S - elapsed)
                        continue
                    if vlm is not None:
                        vlm_event(frame_idx, "found", target=target, dist=round(dist_to_target, 2))
                    log(f"FOUND {target!r} at ({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}), "
                        f"{dist_to_target:.2f}m away -- stopping early.")
                    with state_lock:
                        state["stop"] = True
                    break
                elapsed = time.time() - loop_t0
                if elapsed < LOOP_INTERVAL_S:
                    time.sleep(LOOP_INTERVAL_S - elapsed)
                continue

            result = _tick_perception(frame_idx, target, camera, pose_client, obstacle_map,
                                       scene_graph, detector, intr, fx, fy, fov,
                                       run_dir, log, video_writer, floor_map=floor_map)
            if result is None:
                frame_idx -= 1
                time.sleep(LOOP_INTERVAL_S)
                continue
            xy, yaw = result["xy"], result["yaw"]
            if stop_on_jump(pose_jump(xy)):
                break
            frontiers = result["frontiers"]
            status = (f"[tick {frame_idx:5d} t={time.time()-start_time:6.1f}s target={target!r}] "
                      f"pose=({xy[0]:+.2f},{xy[1]:+.2f}) yaw={np.rad2deg(yaw):+.0f}deg sg_objs={result['n_objs']} "
                      f"frontiers={len(frontiers)}")

            fresh_xy = None
            if result["target_seen"]:
                valid_idxs, detections = result["valid_idxs"], result["detections"]
                # CGFM: pick among single-object boxes (drop boxes around a group of
                # objects) and estimate the position from the scene graph's floor-free
                # foreground point cloud, not the box-centre depth (which, for a chair,
                # often hits the wall behind it through the gaps).
                best_idx, fresh_xy, est_method = estimate_target_edge(
                    detections, valid_idxs, result["depth_raw"], intr, fx, fy,
                    result["tf_camera_to_episodic"], scene_graph,
                )
                if est_method == "none":
                    status_note = "  [no point-cloud or ground-contact estimate]"
                elif est_method == "ground-contact":
                    # No point cloud, or it lies farther (e.g. seen through a mesh bin).
                    status_note = f"  est=ground-contact({fresh_xy[0]:.2f},{fresh_xy[1]:.2f})"
                else:
                    old_xy = estimate_target_xy(
                        detections.boxes[best_idx].numpy(), result["depth_raw"], intr["depth_scale"], fx, fy,
                        intr["width"], intr["height"], result["tf_camera_to_episodic"],
                    )
                    status_note = (f"  est=near-edge({fresh_xy[0]:.2f},{fresh_xy[1]:.2f})"
                                   + (f" box-centre=({old_xy[0]:.2f},{old_xy[1]:.2f})" if old_xy is not None else ""))
            hit_window.append((bool(result["target_seen"]), fresh_xy))
            hits = sum(1 for seen, _ in hit_window if seen)
            if result["target_seen"]:
                status += (f"  >>> saw {target!r} conf={detections.logits[best_idx]:.3f} "
                           f"hits={hits}/{len(hit_window)} (lock at {TARGET_LOCK_MIN_HITS} "
                           f"within the last {TARGET_LOCK_WINDOW} ticks)" + status_note)

            # Lock rule: >= TARGET_LOCK_MIN_HITS detections among the last
            # TARGET_LOCK_WINDOW ticks. The goal is the MEDIAN of those ticks'
            # depth-based world-frame estimates (robust to one bad depth patch),
            # sent once as a fixed waypoint; from then on the branch above takes
            # over and the upper layer stays out of the way.
            vlm_on = vlm is not None and vlm.available
            if hits >= TARGET_LOCK_MIN_HITS and vlm_on:
                pass   # VLM gatekeeper: detections only feed the scene graph; the VLM decides the lock
            elif hits >= TARGET_LOCK_MIN_HITS and args.dry_run:
                estimates = [p for seen, p in hit_window if seen and p is not None]
                if estimates:
                    would_xy = np.median(np.array(estimates), axis=0)
                    status += (f"  [DRY-RUN] WOULD LOCK goal=({would_xy[0]:.2f},{would_xy[1]:.2f}) "
                               f"dist={float(np.hypot(would_xy[0]-xy[0], would_xy[1]-xy[1])):.2f}m (not sent)")
            elif hits >= TARGET_LOCK_MIN_HITS:
                estimates = [p for seen, p in hit_window if seen and p is not None]
                if estimates:
                    target_lock_xy = np.median(np.array(estimates), axis=0)
                    target_best_range = float(np.linalg.norm(
                        target_lock_xy - result["tf_camera_to_episodic"][:2, 3]))
                    waypoint_sender.send(float(target_lock_xy[0]), float(target_lock_xy[1]), 0.0)
                    last_way_point = (float(target_lock_xy[0]), float(target_lock_xy[1]))
                    dist0 = float(np.hypot(target_lock_xy[0] - xy[0], target_lock_xy[1] - xy[1]))
                    per_frame = " ".join(f"({e[0]:.2f},{e[1]:.2f})" for e in estimates)
                    status += (f"  LOCKED goal=({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}) "
                               f"dist={dist0:.2f}m [median of {len(estimates)} estimates: {per_frame}] "
                               f"-> handed to nav stack; no more exploration, goal only refined by closer sightings")
                else:
                    status += "  [warn] enough hits but no valid depth estimate yet"

            # CGFM: lock from scene-graph memory -- a target node seen >= SG_LOCK_MIN_OBS
            # times (possibly on non-consecutive ticks) is enough; goal = its near edge.
            if target_lock_xy is None and not vlm_on:
                cam_xy = result["tf_camera_to_episodic"][:2, 3]
                sg_hit = scene_graph.best_node_edge(target, cam_xy, SG_LOCK_MIN_OBS)
                if sg_hit is not None:
                    edge_xy, nid, nobs = sg_hit
                    dist0 = float(np.hypot(edge_xy[0] - xy[0], edge_xy[1] - xy[1]))
                    if args.dry_run:
                        status += (f"  [DRY-RUN] WOULD LOCK from scene graph #{nid} (seen {nobs} times) "
                                   f"goal=({edge_xy[0]:.2f},{edge_xy[1]:.2f}) dist={dist0:.2f}m (not sent)")
                    else:
                        target_lock_xy = np.asarray(edge_xy, dtype=float)
                        target_best_range = float(np.linalg.norm(target_lock_xy - cam_xy))
                        waypoint_sender.send(float(target_lock_xy[0]), float(target_lock_xy[1]), 0.0)
                        last_way_point = (float(target_lock_xy[0]), float(target_lock_xy[1]))
                        last_frontier_xy = None
                        status += (f"  LOCKED goal=({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}) dist={dist0:.2f}m "
                                   f"from scene graph #{nid} (seen {nobs} times) -> handed to nav stack; "
                                   f"no more exploration, goal only refined by closer sightings")

            # CGFM VLM gatekeeper (MSGNav-style): the VLM gets the scene graph as text
            # (confirmed node id + class) and names the target's id, or null. A picked id
            # is turned into a world goal here (its near edge) -- the VLM never outputs
            # coordinates. "null" -> keep exploring (frontier choice below is unchanged).
            if vlm is not None and target_lock_xy is None:
                cam_xy = result["tf_camera_to_episodic"][:2, 3]
                res = vlm.poll()
                if res is not None:
                    if not res["ok"]:
                        vlm_event(frame_idx, "failed", error=(res["error"] or "")[:120], unavailable=not vlm.available)
                        status += f"  [VLM] call failed: {res['error'][:80]}"
                        if not vlm.available:
                            status += " -> VLM marked unavailable, falling back to the detection lock rules"
                    elif res["target"] != target:
                        pass   # answer to a question about a previous target
                    elif res["object_id"] is not None and res.get("verified") is not True:
                        oid = res["object_id"]
                        label = dict(res["id_labels"]).get(oid, "?")
                        why = ("the crop check named something else" if res.get("verified") is False
                               else f"crop check unavailable ({res.get('verify_raw')})")
                        vlm_event(frame_idx, "rejected", target=target, object_id=oid, label=label,
                                  latency_s=res["latency_s"], raw=res["raw"], verify_raw=res.get("verify_raw"),
                                  crop=os.path.basename(res["crop"]) if res.get("crop") else None)
                        status += (f"  [VLM] picked #{oid} ({label}) but {why}: {res.get('verify_raw')!r} "
                                   f"-> not locked, keep exploring")
                    elif res["object_id"] is None:
                        vlm_event(frame_idx, "none", target=target, n_objects=len(res["id_labels"]),
                                  latency_s=res["latency_s"], raw=res["raw"])
                        status += (f"  [VLM] no {target!r} among {len(res['id_labels'])} scene-graph objects "
                                   f"({res['latency_s']:.2f}s) -> keep exploring")
                    else:
                        oid = res["object_id"]
                        label = dict(res["id_labels"]).get(oid, "?")
                        edge_xy = scene_graph.node_edge(oid, cam_xy)
                        if edge_xy is None:
                            vlm_event(frame_idx, "gone", object_id=oid, label=label)
                            status += f"  [VLM] picked #{oid} ({label}) but it no longer exists (merged) -> ask again"
                            vlm.reset()
                        else:
                            dist0 = float(np.hypot(edge_xy[0] - xy[0], edge_xy[1] - xy[1]))
                            vlm_event(frame_idx, "picked", target=target, object_id=oid, label=label,
                                      latency_s=res["latency_s"], raw=res["raw"], dist=round(dist0, 2),
                                      goal=[round(float(edge_xy[0]), 2), round(float(edge_xy[1]), 2)],
                                      locked=not args.dry_run, verify_raw=res.get("verify_raw"),
                                      crop=os.path.basename(res["crop"]) if res.get("crop") else None)
                            if args.dry_run:
                                status += (f"  [DRY-RUN] WOULD LOCK VLM pick #{oid} ({label}) "
                                           f"goal=({edge_xy[0]:.2f},{edge_xy[1]:.2f}) dist={dist0:.2f}m (not sent)")
                            else:
                                target_lock_xy = np.asarray(edge_xy, dtype=float)
                                target_best_range = float(np.linalg.norm(target_lock_xy - cam_xy))
                                waypoint_sender.send(float(target_lock_xy[0]), float(target_lock_xy[1]), 0.0)
                                last_way_point = (float(target_lock_xy[0]), float(target_lock_xy[1]))
                                last_frontier_xy = None
                                status += (f"  LOCKED goal=({target_lock_xy[0]:.2f},{target_lock_xy[1]:.2f}) "
                                           f"dist={dist0:.2f}m VLM picked scene-graph #{oid} ({label}) "
                                           f"({res['latency_s']:.2f}s) -> handed to nav stack")
                if target_lock_xy is None and vlm.available:
                    objs = scene_graph.id_labels()
                    # MSGNav image edges: fewest saved frames covering every edge and node
                    keys, edge_keys, node_keys = scene_graph.evidence_images([i for i, _ in objs])
                    keys = [k for k in keys if os.path.exists(os.path.join(run_dir, "sg_images", f"{k}.jpg"))]
                    kidx = {k: n for n, k in enumerate(keys)}
                    e_idx = {e: [kidx[k] for k in ks if k in kidx] for e, ks in edge_keys.items()}
                    n_idx = {n: [kidx[k] for k in ks if k in kidx] for n, ks in node_keys.items()}
                    lab_by_id = dict(objs)
                    paths = [render_evidence(run_dir, k, lab_by_id, scene_graph.resolve_id) for k in keys]
                    if vlm.maybe_query(target, objs, paths, e_idx, n_idx, current_view=result["color"],
                                       crop_fn=lambda nid: crop_node(run_dir, nid, scene_graph)):
                        vlm_event(frame_idx, "asked", target=target, objects=[[i, lab] for i, lab in objs],
                                  images=keys, relations=[[a, b, idx] for (a, b), idx in sorted(e_idx.items())])
                        status += (f"  [VLM] asked about {len(objs)} objects, {len(e_idx)} relations, "
                                   f"{len(keys)} evidence image(s) + current view")

            # Compute semantic map + scene graph visualization every tick
            # (even when locked, so the review video shows full history)
            if target != cached_target:
                cached_text_feat = scene_graph.encode_text(target)
                cached_target = target
            scored_objects = scene_graph.get_scored_objects(cached_text_feat) if cached_text_feat is not None else []
            export_graph(xy, yaw)
            sg_nodes = scene_graph.get_scored_labeled(cached_text_feat) if cached_text_feat is not None else []
            try:
                _write_legend(os.path.join(run_dir, "scene_graph_legend.json"), sg_nodes, target)
            except OSError as e:
                log(f"  [warn] legend write failed: {e}")
            sem_map = compute_semantic_map(obstacle_map, scored_objects, seen_mask=floor_map.mask,
                                           semantic_score_baseline=SEMANTIC_SCORE_BASELINE)
            try:
                cv2.imwrite(
                    os.path.join(run_dir, "semantic_map", f"tick_{frame_idx:05d}.png"),
                    _make_semantic_map_vis(sem_map, obstacle_map, floor_map.mask),
                )
                cv2.imwrite(
                    os.path.join(run_dir, "scene_graph", f"tick_{frame_idx:05d}.png"),
                    _make_scene_graph_vis(obstacle_map, sg_nodes, xy, frontiers, target),
                )
            except Exception as e:
                log(f"  [warn] vis save failed: {e}")

            if target_lock_xy is None and len(frontiers) > 0:
                logging.info(f"[tick {frame_idx}] pose=({xy[0]:+.2f},{xy[1]:+.2f}) frontiers={len(frontiers)}")
                if not args.dry_run:
                    now = time.time()
                    stall_hist.append((now, np.asarray(xy, dtype=float)))
                    while stall_hist and now - stall_hist[0][0] > FRONTIER_STALL_S:
                        stall_hist.popleft()
                    span = now - stall_hist[0][0]
                    spread = max(float(np.linalg.norm(p - stall_hist[0][1])) for _, p in stall_hist)
                    if span >= FRONTIER_STALL_S - LOOP_INTERVAL_S and spread < FRONTIER_STALL_MOVE_M \
                            and last_frontier_xy is not None:
                        frontier_blacklist.append(np.asarray(last_frontier_xy, dtype=float))
                        status += (f"  [STALL] moved <{FRONTIER_STALL_MOVE_M}m in {span:.0f}s -> blacklisted frontier "
                                   f"({last_frontier_xy[0]:.2f},{last_frontier_xy[1]:.2f}) "
                                   f"(blacklist size {len(frontier_blacklist)})")
                        last_frontier_xy = None
                        stall_hist.clear()
                best_idx, sel = select_frontier_by_score(obstacle_map, xy, sem_map, seen_mask=floor_map.mask,
                                                         min_frontier_dist_m=MIN_FRONTIER_DIST_M, return_info=True,
                                                         prefer_xy=last_frontier_xy,
                                                         switch_margin=FRONTIER_SWITCH_MARGIN,
                                                         exclude_xy=frontier_blacklist,
                                                         exclude_radius_m=FRONTIER_BLACKLIST_RADIUS_M)
                best = frontiers[best_idx]
                last_frontier_xy = np.asarray(best, dtype=float)
                rho, theta = rho_theta(xy, yaw, best)
                status += f"  best_frontier=({best[0]:.2f},{best[1]:.2f}) turn={np.rad2deg(theta):+.0f}deg dist={rho:.2f}m"
                status += f"  sel={sel['mode']} score={sel['best_score']:.3f} cand={sel['n_candidates']}/{sel['n_frontiers']}"
                moved = last_way_point is None or np.hypot(
                    best[0]-last_way_point[0], best[1]-last_way_point[1]
                ) >= WAYPOINT_MIN_REPUBLISH_DELTA_M
                if moved:
                    waypoint_sender.send(float(best[0]), float(best[1]), 0.0)
                    last_way_point = (float(best[0]), float(best[1]))
                    status += "  [/way_point NOT sent: dry-run]" if args.dry_run else "  [/way_point sent]"
            log(status)

            elapsed = time.time() - loop_t0
            if elapsed < LOOP_INTERVAL_S:
                time.sleep(LOOP_INTERVAL_S - elapsed)
    finally:
        for _ in range(3):  # if we die mid-rotation, leave an explicit zero yaw rate behind
            cmdvel_sender.send(0.0)
            time.sleep(0.03)
        log("Stopping: killing pathFollower first (primary, reliable stop)...")
        _pkill(PATHFOLLOWER_MATCH)
        _pkill("cmdvel_udp_relay.py")
        time.sleep(1.0)  # let the /cmd_vel stream actually cease
        if args.dry_run:
            log("[DRY-RUN] Skipping /liedown (robot was never commanded).")
        else:
            log("Best-effort /liedown (secondary -- NOT guaranteed under load, see module docstring)...")
            call_liedown()
        camera.close()
        pose_client.close()
        if video_writer is not None:
            video_writer.release()
            log(f"First-person recording saved to {os.path.join(run_dir, 'first_person.mp4')}")
        cv2.imwrite(os.path.join(run_dir, "go2_obstacle_map_final.png"), obstacle_map.visualize())
        try:
            export_graph(*(last_pose if last_pose is not None else (None, None)))
            scene_graph.save_features(os.path.join(run_dir, "scene_graph_features.npz"))
            log(f"Scene graph: {len(scene_graph)} confirmed objects -> scene_graph.json / scene_graph.txt / "
                f"scene_graph_features.npz")
        except Exception as e:  # noqa: BLE001
            log(f"  [warn] final scene-graph export failed: {e}")
        if video_writer is not None:
            # Robot is already stopped/lying down by now, so this (a few seconds
            # of encoding) can't delay anything safety-relevant. Turns the raw
            # first-person recording into the composite (camera + occupancy map
            # + value map, time-aligned); the raw one is kept as
            # first_person_orig.mp4. Never allowed to break cleanup.
            log("Building composite review video (camera + occupancy map)...")
            try:
                out = subprocess.run(
                    [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "make_cgfm_composite_video.py"),
                     run_dir, "--fps", str(COMPOSITE_VIDEO_FPS)],
                    capture_output=True, text=True, timeout=300)
                log((out.stdout.strip() or out.stderr.strip() or "(no output)")[-400:])
            except Exception as e:  # noqa: BLE001
                log(f"[warn] composite video failed ({e}); raw recording is still in first_person.mp4")
        log(f"Done. {frame_idx} ticks. Results in {run_dir}")
        log("pathFollower is dead -> /cmd_vel has stopped (primary stop, confirmed reliable). "
            "/liedown above may or may not have actually landed the robot -- check the "
            "[liedown] line and the robot itself; don't assume it's lying down.")
        log_f.close()


if __name__ == "__main__":
    main()
