#!/usr/bin/env python3
"""Relays /state_estimation (nav_msgs/Odometry) to a plain UDP socket on
localhost, as (stamp_sec: f8, x,y,z: f8 each, qx,qy,qz,qw: f8 each).

WHY: vlfm's own mapping code (ObstacleMap/ValueMap) lives in the `vlfm`
conda env (python3.9, needs frontier_exploration+numba+torch1.12), which
has no working rclpy build here (its C extension doesn't match this
system's ROS Jazzy build). Rather than fight that mismatch, this node runs
under ROS's own python (which already has rclpy) and just relays pose over
UDP -- the same pattern this project already uses for the D435i camera
stream (plain TCP instead of a ROS2 Image topic, see realsense_stream_server.py
+ _RealSenseStreamClient in go2_robot.py/vlfm_navigator_node.py).

UDP (not TCP) because this is localhost, one-directional, ~10-20Hz, and an
occasional dropped pose sample is harmless for mapping (same tradeoff as
this project's SensorDataQoS fix elsewhere) -- much simpler than managing a
reconnecting TCP server here.

Usage:
    source /opt/ros/jazzy/setup.bash
    python3 pose_udp_relay.py [--port 8765]
"""
import argparse
import socket
import struct

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

PACK_FMT = "<8d"  # stamp_sec, x, y, z, qx, qy, qz, qw


class PoseUdpRelay(Node):
    def __init__(self, port: int) -> None:
        super().__init__("pose_udp_relay")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dest = ("127.0.0.1", port)
        self.create_subscription(Odometry, "/state_estimation", self._on_odom, 10)
        self.get_logger().info(f"Relaying /state_estimation -> udp://127.0.0.1:{port}")

    def _on_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        packet = struct.pack(PACK_FMT, stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        self._sock.sendto(packet, self._dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    rclpy.init()
    node = PoseUdpRelay(args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
