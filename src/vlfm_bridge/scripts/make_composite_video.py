"""Builds a composite review video for a run_vlfm_pipeline.py run directory:
first-person camera view on top, occupancy map + value map side by side (equal
halves) underneath, one video frame per tick, time-aligned by tick index.

Both maps use the SAME crop window (union of everything drawn across the whole
run, padded, made square) so they stay co-registered and don't jitter between
frames. After the target is locked the pipeline stops updating the maps
(upper layer does no more reasoning), so the last map is held and labeled.

Usage:
    python make_composite_video.py /home/tommy/Taowen/RES/vlfm_pipeline_XXXX \
        [--out first_person.mp4] [--fps 1] [--margin 40] [--keep-orig]
"""
import argparse
import glob
import os
import re
import shutil
import subprocess

import cv2
import numpy as np

PANEL_W = 960                      # video width
FP_H = 720                         # first-person panel (4:3 at PANEL_W)
MAP_SIDE = PANEL_W // 2            # each map: 480x480, the two split the strip evenly
BG = (24, 24, 24)


def tick_of(path):
    return int(re.search(r"tick_(\d+)", os.path.basename(path)).group(1))


def load_by_tick(folder, ext):
    return {tick_of(p): p for p in glob.glob(os.path.join(folder, f"tick_*.{ext}"))
            if not p.endswith("_vis.jpg")}


def union_crop(paths, margin):
    """Square crop window covering every non-white pixel of all given maps."""
    x0 = y0 = 10**9
    x1 = y1 = -1
    h = w = None
    for p in paths:
        im = cv2.imread(p)
        h, w = im.shape[:2]
        nz = np.argwhere(np.any(im != 255, axis=2))
        if len(nz) == 0:
            continue
        (ya, xa), (yb, xb) = nz.min(0), nz.max(0)
        x0, y0, x1, y1 = min(x0, xa), min(y0, ya), max(x1, xb), max(y1, yb)
    if x1 < 0:
        return 0, 0, w, h
    side = int(max(x1 - x0, y1 - y0) + 1 + 2 * margin)
    side = min(side, h, w)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    xs = int(np.clip(cx - side // 2, 0, w - side))
    ys = int(np.clip(cy - side // 2, 0, h - side))
    return xs, ys, side, side


def label(img, text, color=(255, 255, 255), org=(10, 24), scale=0.7):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="first_person.mp4")
    ap.add_argument("--fps", type=float, default=1.0, help="ticks per second of video time (1 = real time)")
    ap.add_argument("--margin", type=int, default=40, help="map crop padding in map pixels")
    ap.add_argument("--keep-orig", action="store_true", default=True,
                    help="if --out already exists, keep it as <name>_orig.mp4 (default on)")
    a = ap.parse_args()

    frames = load_by_tick(os.path.join(a.run_dir, "frames"), "jpg")
    occ = load_by_tick(os.path.join(a.run_dir, "occupancy_map"), "png")
    val = load_by_tick(os.path.join(a.run_dir, "value_map"), "png")
    ticks = sorted(frames)
    assert ticks and occ and val, "need frames/, occupancy_map/, value_map/ in the run dir"
    last_map_tick = max(min(max(occ), max(val)), 0)
    frozen_after = last_map_tick if last_map_tick < ticks[-1] else None

    xs, ys, cw, ch = union_crop(list(occ.values()) + list(val.values()), a.margin)
    print(f"{len(ticks)} ticks, map crop window x={xs} y={ys} size={cw}x{ch} -> {MAP_SIDE}x{MAP_SIDE}"
          + (f", maps frozen after tick {frozen_after}" if frozen_after else ""))

    out_path = os.path.join(a.run_dir, a.out)
    if os.path.exists(out_path) and a.keep_orig:
        orig = out_path[:-4] + "_orig.mp4"
        if not os.path.exists(orig):
            shutil.move(out_path, orig)
            print("original kept as", orig)
    H = FP_H + MAP_SIDE
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{PANEL_W}x{H}", "-framerate", str(a.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-r", "30", out_path],
        stdin=subprocess.PIPE)

    def latest_at_or_before(d, t):
        k = max((x for x in d if x <= t), default=min(d))
        return k, d[k]

    def map_tile(path, title, held_from):
        im = cv2.imread(path)[ys:ys + ch, xs:xs + cw]
        im = cv2.resize(im, (MAP_SIDE, MAP_SIDE), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(im, (0, 0), (MAP_SIDE - 1, MAP_SIDE - 1), (160, 160, 160), 1)
        label(im, title)
        if held_from is not None:
            label(im, f"held from tick {held_from} (no upper-layer updates)", (0, 200, 255),
                  (10, MAP_SIDE - 12), 0.5)
        return im

    for t in ticks:
        fp = cv2.resize(cv2.imread(frames[t]), (PANEL_W, FP_H), interpolation=cv2.INTER_CUBIC)
        label(fp, f"tick {t}/{ticks[-1]}", (255, 255, 255), (10, FP_H - 18))
        # timeline bar along the bottom edge of the first-person panel
        bar_y = FP_H - 6
        cv2.rectangle(fp, (0, bar_y), (PANEL_W, FP_H), (60, 60, 60), -1)
        cv2.rectangle(fp, (0, bar_y), (int(PANEL_W * (t - ticks[0] + 1) / len(ticks)), FP_H), (0, 200, 0), -1)
        if frozen_after:
            fx = int(PANEL_W * (frozen_after - ticks[0] + 1) / len(ticks))
            cv2.line(fp, (fx, bar_y - 6), (fx, FP_H), (0, 200, 255), 2)

        ko, po = latest_at_or_before(occ, t)
        kv, pv = latest_at_or_before(val, t)
        tiles = [map_tile(po, "OCCUPANCY MAP", ko if ko != t else None),
                 map_tile(pv, "VALUE MAP", kv if kv != t else None)]
        canvas = np.full((H, PANEL_W, 3), BG, np.uint8)
        canvas[:FP_H] = fp
        canvas[FP_H:, :MAP_SIDE] = tiles[0]
        canvas[FP_H:, MAP_SIDE:] = tiles[1]
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out_path, f"({PANEL_W}x{H}, {len(ticks)} ticks @ {a.fps} tick/s)")


if __name__ == "__main__":
    main()
