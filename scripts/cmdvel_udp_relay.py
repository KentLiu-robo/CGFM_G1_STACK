#!/usr/bin/env python3
"""ROTATION-ONLY /cmd_vel relay for the CGFM pipelines' initial 360-degree
scan. Listens on a local UDP socket for a single yaw rate (wz, rad/s) and
republishes it as a geometry_msgs/TwistStamped on /cmd_vel, which
unitree_control turns into SPORT_CMD["Move"].

WHY a relay: same reason as pose_udp_relay.py / waypoint_udp_relay.py -- the
vlfm conda env's rclpy doesn't work here, so this node runs under ROS's own
python and publishes on the vlfm script's behalf.

WHY the scan needs it: the old scan asked pathFollower to "turn to face a
waypoint 0.3m away" via localPlanner, which turned out to be unreliable
(localPlanner zeroes goals closer than goalBehindRange=0.35m that are >90deg
off-heading, its direction weight is tiny so it prefers straight paths, and
pathFollower zeroes rotation for 1-point paths). Commanding the yaw rate
directly, closed-loop on the SLAM yaw, is the only dependable way.

SAFETY (by construction, not by convention):
  * linear.x / linear.y are ALWAYS 0 -- this relay cannot make the robot
    translate, whatever bytes arrive on the socket.
  * |wz| is hard-clamped to MAX_WZ; NaN/inf packets are dropped.
  * It only publishes while commands are arriving (last packet < ACTIVE_S
    ago). When the stream stops it publishes a few zero-velocity messages and
    then goes SILENT -- it must not keep publishing /cmd_vel once pathFollower
    takes over, or the two would fight.
  * Never touches /joy, autonomyMode, or pathFollower.

Usage:
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 cmdvel_udp_relay.py [--port 8767]
"""
import argparse
import math
import socket
import struct
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node

PACK_FMT = "<d"      # wz [rad/s]
MAX_WZ = 0.6         # rad/s hard clamp
ACTIVE_S = 0.4       # commands older than this mean "stream stopped"
ZERO_BURST = 6       # zero-velocity messages published after the stream stops
PUBLISH_HZ = 20.0


class CmdVelUdpRelay(Node):
    def __init__(self, port: int) -> None:
        super().__init__("cmdvel_udp_relay")
        self._pub = self.create_publisher(TwistStamped, "/cmd_vel", 5)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", port))
        self._sock.setblocking(False)
        self._wz = 0.0
        self._last_rx = 0.0
        self._zeros_left = 0
        self._was_active = False
        self.create_timer(1.0 / PUBLISH_HZ, self._tick)
        self.get_logger().info(
            f"Listening on udp://127.0.0.1:{port} -> /cmd_vel (yaw rate only, |wz|<={MAX_WZ} rad/s, "
            "linear velocity forced to 0, silent when idle)."
        )

    def _publish(self, wz: float) -> None:
        msg = TwistStamped()
        msg.header.frame_id = "vehicle"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.angular.z = float(wz)
        self._pub.publish(msg)

    def _tick(self) -> None:
        while True:
            try:
                data, _ = self._sock.recvfrom(1024)
            except BlockingIOError:
                break
            if len(data) != struct.calcsize(PACK_FMT):
                continue
            (wz,) = struct.unpack(PACK_FMT, data)
            if not math.isfinite(wz):
                continue
            self._wz = max(-MAX_WZ, min(MAX_WZ, wz))
            self._last_rx = time.monotonic()

        active = (time.monotonic() - self._last_rx) < ACTIVE_S
        if active:
            self._was_active = True
            self._zeros_left = ZERO_BURST
            self._publish(self._wz)
        elif self._was_active:
            # stream just stopped: make sure the last nonzero command is overridden
            if self._zeros_left > 0:
                self._publish(0.0)
                self._zeros_left -= 1
            if self._zeros_left == 0:
                self._was_active = False
                self._wz = 0.0
                self.get_logger().info("command stream stopped -> zero burst sent, going silent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    rclpy.init()
    node = CmdVelUdpRelay(args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # best effort: leave the robot with a zero command on the way out
        try:
            node._publish(0.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
