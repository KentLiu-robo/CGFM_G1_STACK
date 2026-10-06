"""Rebuild wall-clock timeline.csv / events.csv for a run directory written
BEFORE run_vlfm_pipeline_g1.py logged real timestamps itself (runs up to
2026-10-05). Only ADDS files to the run directory; nothing existing is touched.

Time sources (all the planner machine's own clock, so they agree):
  * per-tick capture time  = mtime of depth/tick_N.png (written right after the
    frame is grabbed, before YOLO/BLIP2 run)
  * per-tick logged time   = mtime of frames/tick_N.jpg (written ~0.2-0.4 s later,
    about when the log line for that tick was produced)
  * relay logs             = ROS timestamps of waypoint_udp_relay.log (every
    /way_point sent) and cmdvel_udp_relay.log (end of each scan rotation)
  * log.txt                = everything else (pose, scores, detections, lock);
    explore ticks carry t = seconds since ARMED, which anchors ARMED.

Needs the ORIGINAL run directory: copies that do not preserve mtimes (zip keeps
only 2 s resolution, plain `cp` resets them) make the per-tick times useless.

usage: python3 reconstruct_timeline.py <run_dir>
"""
import csv
import os
import re
import sys
from datetime import datetime

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUM = r"[+-]?\d+(?:\.\d+)?"


def iso(ts):
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds") if ts else ""


def mtime(path):
    return os.stat(path).st_mtime_ns / 1e9 if os.path.exists(path) else None


def parse_log(lines):
    ticks, scan_steps, other = {}, [], []
    for raw in lines:
        line = ANSI.sub("", raw.rstrip("\n"))
        m = re.match(rf"\s*\[scan (\d+)/\d+.*? tick (\d+)\] pose=\(({NUM}),({NUM})\) yaw=({NUM})deg "
                     rf"frontiers=(\d+) drift_from_start=({NUM})m", line)
        if m:
            ticks[int(m.group(2))] = {"phase": "scan", "scan_step": int(m.group(1)),
                                      "x": m.group(3), "y": m.group(4), "yaw_deg": m.group(5),
                                      "frontiers": m.group(6), "scan_drift_m": m.group(7)}
            continue
        m = re.match(rf"\s*\[scan (\d+)/\d+\] rotate to ({NUM})deg: (\w+) \(final yaw error ({NUM})deg, ({NUM})s\)", line)
        if m:
            scan_steps.append({"step": int(m.group(1)), "target_yaw_deg": m.group(2), "outcome": m.group(3),
                               "final_err_deg": m.group(4), "duration_s": float(m.group(5))})
            continue
        m = re.match(rf"\[tick +(\d+) t= *({NUM})s target='([^']*)'\] pose=\(({NUM}),({NUM})\)(.*)", line)
        if m:
            idx, rest = int(m.group(1)), m.group(6)
            t = {"t_since_armed_s": m.group(2), "target": m.group(3), "x": m.group(4), "y": m.group(5)}
            w = re.search(rf"Walking toward the goal \(({NUM}),({NUM})\) dist=({NUM})m(?: \(stand-off point ({NUM})m\))?", rest)
            if w:
                t.update(phase="walk", goal_x=w.group(1), goal_y=w.group(2), dist_to_target_m=w.group(3),
                         dist_to_standoff_m=w.group(4) or "")
            else:
                t["phase"] = "explore"
                s = re.search(rf"score=({NUM}) frontiers=(\d+)", rest)
                if s:
                    t.update(blip_score=s.group(1), frontiers=s.group(2))
                d = re.search(rf">>> saw '[^']*' conf=({NUM})(?: itm=({NUM}))?(.*?) hits=(\d+)/(\d+)", rest)
                if d:
                    t.update(saw=1, conf=d.group(1), itm=d.group(2) or "",
                             counted=0 if "not counted" in d.group(3) else 1, hits=d.group(4))
                f = re.search(rf"best_frontier=\(({NUM}),({NUM})\)", rest)
                if f:
                    t.update(best_frontier_x=f.group(1), best_frontier_y=f.group(2))
                if "[/way_point sent]" in rest:
                    t["waypoint_sent"] = 1
                lk = re.search(rf"LOCKED goal=\(({NUM}),({NUM})\) dist=({NUM})m.*?stand-off \(({NUM}),({NUM})\)", rest)
                if lk:
                    t.update(locked=1, goal_x=lk.group(1), goal_y=lk.group(2), dist_to_target_m=lk.group(3),
                             standoff_x=lk.group(4), standoff_y=lk.group(5))
            ticks[idx] = t
            continue
        if line.startswith(("FOUND", "Max runtime", "DATA LOSS", "Scan did not complete", "Stop requested",
                            "Scan complete", "ARMED", "STOP NOT CONFIRMED")):
            other.append(line)
    return ticks, scan_steps, other


def relay_lines(path, pattern):
    out = []
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            m = re.match(r"\[INFO\] \[(\d+\.\d+)\] \[\w+\]: (.*)", line.strip())
            if m and re.search(pattern, m.group(2)):
                out.append((float(m.group(1)), m.group(2)))
    return out


def main(run_dir):
    run_dir = os.path.abspath(run_dir)
    ticks, scan_steps, other = parse_log(open(os.path.join(run_dir, "log.txt"), errors="replace"))
    for idx, t in ticks.items():
        t["capture_time"] = mtime(os.path.join(run_dir, "depth", f"tick_{idx:05d}.png"))
        t["logged_time"] = mtime(os.path.join(run_dir, "frames", f"tick_{idx:05d}.jpg"))

    events = []

    def ev(ts, event, detail="", source=""):
        if ts:
            events.append((ts, event, detail, source))

    for ts, msg in relay_lines(os.path.join(run_dir, "pose_udp_relay.log"), "Relaying"):
        ev(ts, "relays_started", "pose/waypoint relays up", "pose_udp_relay.log")

    # scan rotations: each "command stream stopped" closes one rotate_to_yaw step
    stops = relay_lines(os.path.join(run_dir, "cmdvel_udp_relay.log"), "command stream stopped")
    for step, (ts, _msg) in zip(scan_steps, stops):
        ev(ts - step["duration_s"], f"scan_step_{step['step']}_start", f"rotate to {step['target_yaw_deg']} deg",
           "cmdvel_udp_relay.log - logged duration (approx.)")
        ev(ts, f"scan_step_{step['step']}_end",
           f"{step['outcome']}, final yaw error {step['final_err_deg']} deg, {step['duration_s']} s", "cmdvel_udp_relay.log")
    if len(stops) != len(scan_steps):
        print(f"[warn] {len(scan_steps)} scan steps in log.txt but {len(stops)} stream-stopped lines in the cmdvel relay log")

    scan_ticks = [i for i, t in ticks.items() if t["phase"] == "scan"]
    if scan_ticks:
        ev(ticks[min(scan_ticks)]["capture_time"], "scan_first_frame", f"tick {min(scan_ticks)}", "depth/ mtime")

    explore = sorted(i for i, t in ticks.items() if t["phase"] == "explore")
    if explore and ticks[explore[0]]["logged_time"]:
        first = ticks[explore[0]]
        ev(first["logged_time"] - float(first["t_since_armed_s"]), "armed",
           "pathFollower started, autonomous walking begins", f"frames/ mtime of tick {explore[0]} - its t=")

    waypoints = relay_lines(os.path.join(run_dir, "waypoint_udp_relay.log"), r"/way_point <-")
    armed_ts = next((e[0] for e in events if e[1] == "armed"), None)
    standoffs = {(t.get("standoff_x"), t.get("standoff_y")) for t in ticks.values() if t.get("locked")}
    for ts, msg in waypoints:
        m = re.search(rf"\(({NUM}), ({NUM}), {NUM}\)", msg)
        x, y = (m.group(1), m.group(2)) if m else ("", "")
        if any(abs(float(x) - float(sx)) < 0.01 and abs(float(y) - float(sy)) < 0.01 for sx, sy in standoffs if sx):
            kind = "waypoint_standoff"
        elif armed_ts and ts < armed_ts:
            kind = "waypoint_pin_current_position"
        else:
            kind = "waypoint_frontier"
        ev(ts, kind, f"({x}, {y})", "waypoint_udp_relay.log")

    for i in sorted(ticks):
        t = ticks[i]
        if t.get("saw"):
            ev(t["capture_time"], "target_detected" if t.get("counted") else "detection_rejected",
               f"tick {i}: conf {t['conf']} itm {t.get('itm') or '-'} hits {t['hits']}", "log.txt + depth/ mtime")
        if t.get("locked"):
            ev(t["logged_time"], "locked", f"tick {i}: goal ({t['goal_x']}, {t['goal_y']}) dist {t['dist_to_target_m']} m, "
               f"stand-off ({t['standoff_x']}, {t['standoff_y']})", "log.txt + frames/ mtime")
    last = max(ticks) if ticks else None
    for line in other:
        if line.startswith(("FOUND", "Max runtime", "DATA LOSS", "Scan did not complete", "Stop requested")):
            ts = ticks[last]["logged_time"] if last else None
            ev(ts, "run_end_reason", line, f"log.txt (~ time of the last tick {last}, within ~0.1 s for FOUND)")
    ev(mtime(os.path.join(run_dir, "g1_obstacle_map_final.png")), "stopped_and_final_maps_saved",
       "pathFollower killed + bridge disabled happen just before this", "g1_obstacle_map_final.png mtime")
    ev(mtime(os.path.join(run_dir, "first_person.mp4")), "composite_video_written", "", "first_person.mp4 mtime")

    cols = ["tick", "phase", "scan_step", "capture_time_iso", "capture_time_epoch", "logged_time_epoch",
            "t_since_armed_s", "x", "y", "yaw_deg", "blip_score", "frontiers", "best_frontier_x", "best_frontier_y",
            "waypoint_sent", "saw", "conf", "itm", "counted", "hits", "locked", "goal_x", "goal_y",
            "standoff_x", "standoff_y", "dist_to_target_m", "dist_to_standoff_m", "scan_drift_m"]
    with open(os.path.join(run_dir, "timeline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in sorted(ticks):
            t = ticks[i]
            row = dict(t, tick=i, capture_time_iso=iso(t["capture_time"]),
                       capture_time_epoch=f"{t['capture_time']:.3f}" if t["capture_time"] else "",
                       logged_time_epoch=f"{t['logged_time']:.3f}" if t["logged_time"] else "")
            w.writerow([row.get(c, "") for c in cols])
    events.sort()
    with open(os.path.join(run_dir, "events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_iso", "time_epoch", "t_from_first_event_s", "event", "detail", "source"])
        t0 = events[0][0] if events else 0
        for ts, e, d, s in events:
            w.writerow([iso(ts), f"{ts:.3f}", f"{ts - t0:.2f}", e, d, s])
    print(f"timeline.csv: {len(ticks)} ticks | events.csv: {len(events)} events -> {run_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
