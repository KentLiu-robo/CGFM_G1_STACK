"""Arrival-time monitor for /laser_odometry and /livox/imu (run inside the domain-42 dev env).
usage: python3 odom_monitor.py <seconds>
Prints one line per 10 s bin plus every gap > 0.5 s, then an overall summary."""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message

DUR = float(sys.argv[1])
BIN = 10.0
TOPICS = {"odom": "/laser_odometry", "imu": "/livox/imu"}

rclpy.init()
node = Node("wifi_health_monitor")

# wait for the topics to be discovered, then subscribe with their real types
deadline = time.monotonic() + 45
types = {}
while time.monotonic() < deadline and len(types) < len(TOPICS):
    rclpy.spin_once(node, timeout_sec=0.5)
    known = dict(node.get_topic_names_and_types())
    for k, t in TOPICS.items():
        if t in known and k not in types:
            types[k] = known[t][0]
missing = [TOPICS[k] for k in TOPICS if k not in types]
if missing:
    print("NOT DISCOVERED:", missing, flush=True)
    sys.exit(2)

arr = {k: [] for k in TOPICS}
t0 = time.monotonic()
for k in TOPICS:
    node.create_subscription(get_message(types[k]), TOPICS[k],
                             (lambda kk: lambda m: arr[kk].append(time.monotonic() - t0))(k),
                             qos_profile_sensor_data)
print(f"types: { {TOPICS[k]: types[k] for k in types} }", flush=True)

next_bin = BIN
while rclpy.ok() and time.monotonic() - t0 < DUR:
    rclpy.spin_once(node, timeout_sec=0.05)
    now = time.monotonic() - t0
    if now >= next_bin:
        lo = next_bin - BIN
        parts = []
        for k in TOPICS:
            ts = [t for t in arr[k] if lo <= t < next_bin]
            gaps = [b - a for a, b in zip(ts, ts[1:])]
            parts.append(f"{TOPICS[k]:16s} {len(ts)/BIN:6.1f} Hz  max gap {max(gaps) if gaps else float('nan'):5.2f}s")
        print(f"t={lo:4.0f}-{next_bin:3.0f}s  " + " | ".join(parts), flush=True)
        next_bin += BIN

print("\n== summary ==", flush=True)
for k in TOPICS:
    ts = arr[k]
    if len(ts) < 3:
        print(f"{TOPICS[k]}: only {len(ts)} messages"); continue
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    big = [(round(a, 1), round(b - a, 2)) for a, b in zip(ts, ts[1:]) if b - a > 0.5]
    s = sorted(gaps)
    print(f"{TOPICS[k]}: {len(ts)} msgs in {DUR:.0f}s = {len(ts)/DUR:.2f} Hz | gap median {s[len(s)//2]*1000:.0f} ms, "
          f"p99 {s[int(len(s)*0.99)]*1000:.0f} ms, max {max(gaps):.2f} s | gaps>0.5s: {len(big)} {big[:12]}", flush=True)
node.destroy_node()
rclpy.shutdown()
