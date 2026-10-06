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

Also forwards /cmd_vel (geometry_msgs/TwistStamped, whoever publishes it:
the scan's rotation relay or pathFollower) as (vx, vy, wz: f8 each) to
--cmdvel-port, so run_vlfm_pipeline_g1.py can log the commanded velocity
(cmd_vel.csv) for review videos. Read-only: it never publishes anything.

Usage:
    source /opt/ros/jazzy/setup.bash
    python3 pose_udp_relay.py [--port 8765] [--cmdvel-port 8768]
"""
import argparse
import socket
import struct

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

PACK_FMT = "<8d"  # stamp_sec, x, y, z, qx, qy, qz, qw
CMDVEL_FMT = "<3d"  # vx, vy, wz


class PoseUdpRelay(Node):
    def __init__(self, port: int, cmdvel_port: int) -> None:
        super().__init__("pose_udp_relay")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dest = ("127.0.0.1", port)
        self._cmdvel_dest = ("127.0.0.1", cmdvel_port)
        self.create_subscription(Odometry, "/state_estimation", self._on_odom, 10)
        self.create_subscription(TwistStamped, "/cmd_vel", self._on_cmdvel, 10)
        self.get_logger().info(f"Relaying /state_estimation -> udp://127.0.0.1:{port}, "
                               f"/cmd_vel (read-only) -> udp://127.0.0.1:{cmdvel_port}")

    def _on_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        packet = struct.pack(PACK_FMT, stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        self._sock.sendto(packet, self._dest)

    def _on_cmdvel(self, msg: TwistStamped) -> None:
        t = msg.twist
        self._sock.sendto(struct.pack(CMDVEL_FMT, t.linear.x, t.linear.y, t.angular.z), self._cmdvel_dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cmdvel-port", type=int, default=8768)
    args = parser.parse_args()

    rclpy.init()
    node = PoseUdpRelay(args.port, args.cmdvel_port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
