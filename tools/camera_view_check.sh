#!/bin/bash
. "$(cd "$(dirname "$0")" && pwd)/env.sh"
# SAFETY GATE before starting the stack / a run: grab ONE live camera frame + depth clearance numbers, so the robot's posture and
# surroundings can be LOOKED AT (open the PNG). Moves nothing. Starts the RealSense server on the Jetson only if it is not running
# already, and stops it again only if this script started it.
# usage: camera_view_check.sh      -> writes /tmp/go2_stack/view.png (left = colour, right = depth colour-map) and prints clearance
W=/tmp/go2_stack; mkdir -p $W
J="ssh -o BatchMode=yes -o ConnectTimeout=6 ${JETSON_USER}@${JETSON_IP}"
STARTED=0
if [ "$($J 'pgrep -f "[r]ealsense_stream_server" | wc -l')" = 0 ]; then
  timeout 30 ssh -o BatchMode=yes ${JETSON_USER}@${JETSON_IP} 'mkdir -p ~/logs; setsid nohup python3 ~/realsense_stream_server.py > ~/logs/realsense_look.log 2>&1 < /dev/null &
echo "realsense started for the check"'; STARTED=1
fi
for i in $(seq 1 15); do nc -z -w1 $JETSON_IP 6000 2>/dev/null && break; sleep 1; done
env -u PYTHONPATH $VLFM_PYTHON - <<'EOF'
import socket, struct, cv2, numpy as np
W = "/tmp/go2_stack"
s = socket.create_connection((__import__("os").environ["JETSON_IP"], 6000), timeout=5); s.settimeout(5)
def rx(n):
    b = bytearray()
    while len(b) < n:
        p = s.recv(n - len(b))
        if not p: raise EOFError
        b.extend(p)
    return bytes(b)
rx(28); fr = []; col = None
for _ in range(12):
    c, d = struct.unpack(">II", rx(8)); cb = rx(c); db = rx(d)
    col = cv2.imdecode(np.frombuffer(cb, np.uint8), cv2.IMREAD_COLOR); fr.append(cv2.imdecode(np.frombuffer(db, np.uint8), cv2.IMREAD_UNCHANGED))
dep = np.median(np.stack(fr), axis=0).astype(np.uint16); m = dep * 0.001; h, w = dep.shape; valid = (dep > 0) & (dep < 6000)
vis = cv2.applyColorMap(cv2.normalize(np.clip(dep, 0, 4000), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET); vis[dep == 0] = (0, 0, 0)
cv2.imwrite(W + "/view.png", np.hstack([col, vis]))
print(f"valid depth {valid.mean()*100:.0f}%; closer than 1.0 m: {(valid & (m < 1.0)).mean()*100:.1f}%; closer than 0.6 m: {(valid & (m < 0.6)).mean()*100:.1f}%")
cols = [("left", 0, w // 4), ("c-left", w // 4, w // 2), ("c-right", w // 2, 3 * w // 4), ("right", 3 * w // 4, w)]; rows = [("upper", 0, h // 3), ("middle", h // 3, 2 * h // 3), ("lower", 2 * h // 3, h)]
print("5th-percentile valid depth (m) per region      " + "".join(f"{n:>9s}" for n, _, _ in cols) + "   (lower row ~0.75 m = the floor itself)")
for rn, r0, r1 in rows:
    line = f"  {rn:8s}                                  "
    for cn, c0, c1 in cols:
        v = m[r0:r1, c0:c1][valid[r0:r1, c0:c1]]; line += f"{np.percentile(v, 5):9.2f}" if v.size > 50 else f"{'-':>9s}"
    print(line)
print("image: /tmp/go2_stack/view.png")
EOF
[ $STARTED = 1 ] && timeout 20 $J 'pkill -f "[r]ealsense_stream_server.py"; sleep 1; rm -f ~/logs/realsense_look.log; echo "realsense stopped again"'
