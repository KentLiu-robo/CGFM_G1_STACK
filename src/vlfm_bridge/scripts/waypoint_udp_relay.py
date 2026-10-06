#!/usr/bin/env python3
"""Reverse of pose_udp_relay.py: listens on a local UDP socket for a
(x, y, z) point and republishes it as a geometry_msgs/PointStamped on
/way_point, which GO2_STACK's local_planner (localPlanner.cpp, subGoal
subscription) consumes for obstacle-avoidance path planning.

WHY a relay instead of publishing directly from the vlfm-conda-env script:
same reason as pose_udp_relay.py -- that env's rclpy build doesn't work
here, so this node runs under ROS's own python and does the actual
publish on the vlfm script's behalf.

SAFETY: this only ever publishes to /way_point. It does not touch /joy or
autonomyMode, and does not start pathFollower -- per this project's
established safety design, localPlanner will compute an obstacle-avoided
path toward this point, but nothing converts that path into real robot
motion unless pathFollower is separately, deliberately started (e.g. via
arm_movement_briefly_go2stack.sh's bounded window). Verified before writing
this: `ros2 node list` currently shows no /pathFollower node running.

Usage:
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 waypoint_udp_relay.py [--port 8766]
"""
import argparse
import socket
import struct

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node

PACK_FMT = "<3d"  # x, y, z


class WaypointUdpRelay(Node):
    def __init__(self, port: int) -> None:
        super().__init__("waypoint_udp_relay")
        self._pub = self.create_publisher(PointStamped, "/way_point", 5)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", port))
        self._sock.setblocking(False)
        self.create_timer(0.05, self._poll)  # 20Hz poll, plenty for ~1Hz waypoint updates
        self.get_logger().info(
            f"Listening on udp://127.0.0.1:{port}, republishing to /way_point "
            "(does NOT touch /joy or autonomyMode -- publish-only, see docstring)."
        )

    def _poll(self) -> None:
        while True:
            try:
                data, _ = self._sock.recvfrom(1024)
            except BlockingIOError:
                return
            if len(data) != struct.calcsize(PACK_FMT):
                continue
            x, y, z = struct.unpack(PACK_FMT, data)
            msg = PointStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x, msg.point.y, msg.point.z = x, y, z
            self._pub.publish(msg)
            self.get_logger().info(f"/way_point <- ({x:.2f}, {y:.2f}, {z:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    rclpy.init()
    node = WaypointUdpRelay(args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
