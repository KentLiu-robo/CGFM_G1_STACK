#!/usr/bin/env python3
"""LIFE-LONG demo video for a make_cgfm_lifelong run (derived from make_cgfm_demo_front.py, which stays untouched).
Same layout (hand-held third-person video time-synced with the first-person camera, VLM card, three maps, event
feed, slim timeline), but for a multi-target mission: header title "Life-long Object Navigation", a mission strip
with the target queue and its progress, per-leg phases (360 scan / searching / approaching / RECALL from the scene
graph / found), and a timeline split into legs.

Usage:
    python make_cgfm_demo_lifelong.py RUN_DIR --front RUN_DIR/front.MOV --front-start HH:MM:SS.sss \\
        --panels PANELS_DIR [--front-hold SECONDS] [--front-rate 1.0] [--post-speed 1.5] [--out demo_life_long.mp4]
(panels: make_cgfm_composite_video.py RUN --out /elsewhere.mp4 --panels-dir PANELS_DIR; run dir needs log.txt,
frames/ with original mtimes, vlm_events.jsonl, vlm_inputs/.)
"""
import argparse
import bisect
import datetime as dt
import json
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_cgfm_composite_video import (_font, _gradient, _grad_rrect, _grad_text, _spaced, _thumb,  # noqa: E402
                                       _mix, _STATUS, VLM_MODEL_NAME, class_color, load_vlm_events, vlm_state_at,
                                       C_CANVAS, C_BG, C_CARD, C_CARD_EDGE, C_TEXT, C_MUTED, C_SUBTLE, C_GREEN,
                                       C_AMBER, C_RED, C_GRAY, C_KEY, C_SHADOW, G1, G2)

W, H = 1920, 1080
M = 16
TOP = 64
TL_H = 96
BOTTOM = H - M - TL_H - M                            # 952: bottom of the content area
LW = 820                                             # left column width
FRONT_BOX = (M, TOP, LW, LW * 9 // 16)               # 820 x 461
VLM_BOX = (M, TOP + FRONT_BOX[3] + M, LW, BOTTOM - (TOP + FRONT_BOX[3] + M))
RX = M + LW + M                                      # right column x
RW = W - M - RX                                      # 1052
MAP = (RW - 2 * M) // 3                              # 340
MAP_Y = BOTTOM - MAP
FP_H = MAP_Y - M - TOP                               # 532
FP_W = FP_H * 4 // 3                                 # 709 (camera is 4:3)
FP_BOX = (RX, TOP, FP_W, FP_H)
EV_BOX = (RX + FP_W + M, TOP, RW - FP_W - M, FP_H)
MAP_BOXES = [(RX + i * (MAP + M), MAP_Y, MAP, MAP) for i in range(3)]
TL_BOX = (M, BOTTOM + M, W - 2 * M, TL_H)
RADIUS = 16

C_TEAL = (0, 168, 150)
PHASES = [("scan", "360° scan", (111, 76, 255)), ("explore", "Searching", G2),
          ("approach", "Approaching", C_AMBER), ("recall", "Recall", C_TEAL)]
PHASE_COL = {k: c for k, _, c in PHASES}
PHASE_COL.update({"found": C_GREEN, "done": C_GREEN})
PHASE_NAME = {k: n for k, n, _ in PHASES}
EV_COL = {"found": C_GREEN, "lock": C_AMBER, "saw": C_AMBER, "regoal": C_RED, "stall": C_RED,
          "scan": PHASES[0][2], "liedown": C_SUBTLE, "recall": C_TEAL, "next": G1}
LEG_COL = [G2, C_AMBER, G1, C_TEAL]


def wall(s, day):
    h, m, rest = s.split(":")
    return dt.datetime.combine(day, dt.time(int(h), int(m))).timestamp() + float(rest)


def parse_run(run):
    """Per-tick wall time + phase from frames/ mtimes and log.txt; legs (one per target) + events from log.txt and
    vlm_events.jsonl. Returns t_of, phase_of, events, legs, first_explore, vlm."""
    frames = sorted(f for f in os.listdir(os.path.join(run, "frames")) if f.endswith(".jpg"))
    t_of = {int(f[5:10]): os.path.getmtime(os.path.join(run, "frames", f)) for f in frames}
    phase_of, events, legs = {}, [], []
    first_explore = last_tick = None
    for line in open(os.path.join(run, "log.txt")):
        m = re.search(r"tick\s+(\d+)\]?.*?pose=", line)
        if not m:
            if line.startswith("ARMED"):
                tg = re.search(r"target='(.*?)'", line)
                legs.append({"target": tg[1] if tg else "?", "start": min(t_of), "recall": False,
                             "lock": None, "found": None})
            elif line.startswith(">>> target") and legs:
                tg = re.search(r"(?:next in queue|changed by user): '(.*?)'", line)
                name = tg[1] if tg else "?"
                mm = re.search(r"Memory check: (\d+) objects in the scene graph, (\d+) labelled", line)
                n_known = int(mm[2]) if mm else 0
                before = any(l["target"] == name and l["found"] is not None for l in legs)
                legs.append({"target": name, "start": (last_tick or 0) + 1, "recall": False, "lock": None,
                             "found": None})
                if before:
                    txt = f"Next target: {name}. Found earlier -> go straight back"
                elif n_known:
                    txt = f"Next target: {name}. {n_known} in the scene graph -> ask the VLM"
                else:
                    txt = f"Next target: {name}. Never seen -> explore frontiers"
                events.append((last_tick, "next", txt))
            elif line.startswith("FOUND") and legs:
                d = re.search(r"([\d.]+)m away", line)
                legs[-1]["found"] = last_tick
                events.append((last_tick, "found", f"FOUND {legs[-1]['target']}, {d[1] if d else '?'} m from the goal"))
            continue
        k = last_tick = int(m[1])
        if "[scan" in line:
            phase_of.setdefault(k, "scan")
        elif "IDLE" in line:
            phase_of.setdefault(k, "done")
        elif "Walking toward" in line:
            phase_of.setdefault(k, "recall" if legs and legs[-1]["recall"] else "approach")
            if "[RE-GOAL]" in line:
                events.append((k, "regoal", "Goal corrected from consistent closer sightings"))
            if "[NOT FOUND YET]" in line:
                events.append((k, "regoal", "Target still ahead at the goal -> new goal"))
        else:
            phase_of.setdefault(k, "explore")
            if first_explore is None:
                first_explore = k
                events.append((k, "explore", "Scan done -> exploring frontiers guided by the semantic field"))
            if "[STALL]" in line:
                events.append((k, "stall", "Stalled on a frontier -> blacklisted"))
            if "LOCKED" in line and legs:
                legs[-1]["lock"] = k
                if "RECALL of" in line:
                    legs[-1]["recall"] = True
                    dd = re.search(r"dist=([\d.]+)m", line)
                    events.append((k, "recall", f"RECALL: {legs[-1]['target']} was found earlier -> straight back "
                                                f"({dd[1] if dd else '?'} m), no VLM gate"))
    if phase_of:
        events.insert(0, (min(phase_of), "scan", "Start: 360° scan in 8 steps of 45°"))
    vlm = load_vlm_events(run) or []
    for e in vlm:
        if e["kind"] == "none":
            events.append((e["tick"], "vlm", f"VLM: no {e.get('target', 'target')} among {e.get('n_objects', '?')} graph objects"))
        elif e["kind"] == "picked" and e.get("raw") != "recall":      # recalls are reported from the log above
            how = " (accepted: picked repeatedly)" if str(e.get("verify_raw", "")).startswith("accepted by votes") else ""
            events.append((e["tick"], "lock", f"VLM picks #{e['object_id']} {e.get('label', '')}{how} -> goal locked "
                                             f"{e.get('dist', 0):.1f} m away"))
        elif e["kind"] == "rejected":
            events.append((e["tick"], "vlm", f"VLM pick #{e['object_id']} rejected by the crop check"))
    for lg in legs:
        if lg["found"] is not None:
            phase_of[lg["found"]] = "found"
    events = [(t_of.get(k, 0.0), kind, text) for k, kind, text in events if k is not None]
    events.sort(key=lambda e: e[0])
    return t_of, phase_of, events, legs, first_explore, vlm


def rounded_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
    return m


def card(base, box, r=RADIUS, fill=C_BG):
    """Borderless card: soft violet-tinted shadow + white rounded fill."""
    x, y, w, h = box
    sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((x + 1, y + 5, x + w + 1, y + h + 5), r, fill=C_SHADOW + (34,))
    base.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)))
    ImageDraw.Draw(base).rounded_rectangle((x, y, x + w, y + h), r, fill=fill + (255,))


def wrap(d, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        nxt = (cur + " " + w_).strip()
        if d.textlength(nxt, font=font) <= width or not cur:
            cur = nxt
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def vlm_card(events, t, run_dir, w, h):
    """Wide VLM question/answer card for tick t: INPUT (objects + evidence images) -> OUTPUT (answer)."""
    img = Image.new("RGB", (w, h), C_BG)
    d = ImageDraw.Draw(img)
    pad = 22
    # header: gradient badge, title, model, status pill on the right
    bw = d.textlength("VLM", font=_font("bold", 14)) + 20
    _grad_rrect(img, (pad, 18, pad + bw, 44), radius=13)
    d.text((pad + 10, 31), "VLM", font=_font("bold", 14), fill=(255, 255, 255), anchor="lm")
    d.text((pad + bw + 12, 31), "Target check", font=_font("bold", 21), fill=C_TEXT, anchor="lm")
    tx = pad + bw + 12 + d.textlength("Target check", font=_font("bold", 21)) + 14
    d.text((tx, 33), VLM_MODEL_NAME, font=_font("mono", 13), fill=C_MUTED, anchor="lm")
    if events is None:
        d.text((w / 2, h / 2), "VLM not used in this run", font=_font("medium", 18), fill=C_MUTED, anchor="mm")
        return img
    st = vlm_state_at(events, t)
    ans = st["answer"] or {}
    picked = ans.get("object_id") if st["status"] in ("picked", "locked", "found") else None
    text, col = _STATUS.get(st["status"], (st["status"], C_GRAY))
    if picked is not None:
        text = {"picked": f"Picked #{picked} · dry-run", "locked": f"Locked #{picked} · navigating",
                "found": f"Reached #{picked}"}[st["status"]]
    f_pill = _font("medium", 15)
    pw = d.textlength(text, font=f_pill) + 40
    d.rounded_rectangle((w - pad - pw, 16, w - pad, 46), 15, fill=_mix(col, C_BG, 0.14))
    d.ellipse((w - pad - pw + 13, 26, w - pad - pw + 23, 36), fill=col)
    d.text((w - pad - pw + 30, 31), text, font=f_pill, fill=_mix(col, C_TEXT, 0.75), anchor="lm")
    _grad_rrect(img, (pad, 58, w - pad, 59), radius=1)

    # INPUT (left) and OUTPUT (right) sub-cards, connected by an arrow
    top, bot = 74, h - 18
    out_w = 236
    in_x0, in_x1 = pad, w - pad - out_w - 44
    out_x0, out_x1 = w - pad - out_w, w - pad
    d.rounded_rectangle((in_x0, top, in_x1, bot), 12, fill=C_CARD)
    d.rounded_rectangle((out_x0, top, out_x1, bot), 12, fill=C_CARD)
    ay = (top + bot) // 2
    _grad_rrect(img, (in_x1 + 8, ay - 1, out_x0 - 14, ay + 1), radius=1)
    d.polygon([(out_x0 - 14, ay - 7), (out_x0 - 14, ay + 7), (out_x0 - 5, ay)], fill=G2)

    # INPUT: header line
    _grad_text(img, (in_x0 + 16, top + 12), "INPUT", _font("bold", 13), spacing=1.6)
    objs = st["objects"]
    def n_(n, word):
        return f"{n} {word}" + ("" if n == 1 else "s")
    head = f"{n_(len(objs), 'object')} · {n_(len(st['relations']), 'relation')} · {n_(len(st['images']), 'image')}"
    d.text((in_x0 + 86, top + 13), head, font=_font("regular", 14), fill=C_SUBTLE)
    if st["asked_tick"] is not None:
        meta = f"call {st['calls']} · tick {st['asked_tick']}"
        d.text((in_x1 - 16, top + 21), meta, font=_font("mono", 13), fill=C_MUTED, anchor="rm")
    # object grid (3 columns)
    thumbs = list(st["images"]) if run_dir else []
    th_w, th_h = 104, 78
    strip_h = th_h + 20 if thumbs else 0
    gy0, row_h = top + 46, 28
    rows = max(1, (bot - strip_h - gy0 - 8) // row_h)
    cols = 2 if len(objs) <= 2 * rows else 3     # two wide columns while they fit: full class names
    cap = rows * cols
    shown = sorted(objs, key=lambda o: (o[0] != picked, o[1] != st["target"]))
    more = len(shown) - cap
    if more > 0:
        shown = shown[:cap - 1]
    col_w = (in_x1 - in_x0 - 24) // cols
    if not objs:
        d.text((in_x0 + 18, gy0 + 4), "no confirmed objects yet", font=_font("regular", 15), fill=C_SUBTLE)
    for i, (oid, lab) in enumerate(shown):
        c, r_ = i % cols, i // cols
        x0, y = in_x0 + 12 + c * col_w, gy0 + r_ * row_h
        is_pick, is_tgt = oid == picked, lab == st["target"]
        if is_pick:
            d.rounded_rectangle((x0, y - 2, x0 + col_w - 8, y + row_h - 5), 8, fill=_mix(G1, C_CARD, 0.14))
            _grad_rrect(img, (x0, y - 2, x0 + 3, y + row_h - 5), radius=2)
        b, g, r = class_color(lab)
        cy = y + (row_h - 6) // 2
        d.ellipse((x0 + 10, cy - 6, x0 + 22, cy + 6), fill=(r, g, b))
        d.text((x0 + 30, cy), f"#{oid}", font=_font("mono", 14), fill=C_TEXT if is_pick else C_MUTED, anchor="lm")
        name = lab
        nf = _font("bold" if (is_pick or is_tgt) else "regular", 16)
        while d.textlength(name, font=nf) > col_w - 82 and len(name) > 3:
            name = name[:-2] + "…"
        d.text((x0 + 72, cy), name, font=nf, fill=C_TEXT if (is_pick or is_tgt) else _mix(C_TEXT, C_CARD, 0.8),
               anchor="lm")
    if more > 0:
        i = len(shown)
        x0, y = in_x0 + 12 + (i % cols) * col_w, gy0 + (i // cols) * row_h
        d.text((x0 + 30, y + (row_h - 6) // 2), f"+ {more + 1} more", font=_font("regular", 14), fill=C_SUBTLE,
               anchor="lm")
    # evidence images actually sent to the VLM
    if thumbs:
        sy = bot - th_h - 14
        d.text((in_x0 + 16, sy - 12), "evidence images", font=_font("regular", 12), fill=C_SUBTLE, anchor="lm")
        slots = (in_x1 - in_x0 - 32 + 8) // (th_w + 8)
        shown_k = thumbs[:slots - 1] if len(thumbs) > slots else thumbs
        x = in_x0 + 16
        for i, k in enumerate(shown_k):
            tt = _thumb(os.path.join(run_dir, "vlm_inputs", f"{k}.jpg"), th_w, th_h)
            if tt is not None:
                img.paste(tt[0], (x, sy), tt[1])
            d.rounded_rectangle((x + 4, sy + 4, x + 22, sy + 22), 5, fill=(0, 0, 0))
            d.text((x + 13, sy + 13), str(i), font=_font("bold", 12), fill=(255, 255, 255), anchor="mm")
            x += th_w + 8
        if len(thumbs) > len(shown_k):
            d.text((x + 6, sy + th_h / 2), f"+{len(thumbs) - len(shown_k)}", font=_font("medium", 16), fill=C_MUTED,
                   anchor="lm")

    # OUTPUT
    _grad_text(img, (out_x0 + 16, top + 12), "OUTPUT", _font("bold", 13), spacing=1.6)
    jx, jy = out_x0 + 16, top + 50
    if st["status"] == "asking":
        d.text((jx, jy), "waiting for the answer …", font=_font("regular", 15), fill=C_AMBER)
    elif ans.get("error"):
        d.text((jx, jy), str(ans["error"])[:24], font=_font("mono", 13), fill=C_RED)
    elif ans.get("raw") == "recall":
        d.text((jx, jy), "MEMORY RECALL", font=_font("bold", 17), fill=C_TEAL)
        d.text((jx, jy + 30), f"-> #{ans.get('object_id')} {ans.get('label', '')}", font=_font("medium", 17), fill=C_TEXT)
        for n_, ln in enumerate(("found earlier in", "this run: go back", "no VLM needed")):
            d.text((jx, jy + 64 + 20 * n_), ln, font=_font("regular", 14), fill=C_MUTED)
    elif ans.get("raw"):
        oid = ans.get("object_id")
        d.text((jx, jy), '{"object_id":', font=_font("mono", 17), fill=C_KEY)
        d.text((jx + 12, jy + 26), (str(oid) if oid is not None else "null") + "}",
               font=_font("monobold", 26), fill=G1 if oid is not None else C_AMBER)
        if oid is not None and ans.get("label"):
            d.text((jx, jy + 72), f"→ #{oid} {ans['label']}", font=_font("medium", 16),
                   fill=C_AMBER if st["status"] == "rejected" else C_TEXT)
            if ans.get("verify_raw") is not None:
                vr = str(ans["verify_raw"]).strip().rstrip(".").lower()
                vr = "too small" if vr.startswith("too small") else vr[:16]
                d.text((jx, jy + 98), f"crop check: {vr}", font=_font("regular", 14),
                       fill=C_AMBER if st["status"] == "rejected" else C_MUTED)
        elif oid is None:
            d.text((jx, jy + 72), "→ no target in", font=_font("regular", 15), fill=C_MUTED)
            d.text((jx, jy + 94), "   the scene graph", font=_font("regular", 15), fill=C_MUTED)
    else:
        d.text((jx, jy), "—", font=_font("mono", 18), fill=C_SUBTLE)
    sy = bot - 64
    d.line((out_x0 + 16, sy - 10, out_x1 - 16, sy - 10), fill=C_CARD_EDGE, width=1)
    lat = f"{ans['latency_s']:.2f}s" if ans.get("latency_s") is not None and ans.get("raw") != "recall" else "—"
    for i, (val, lab) in enumerate(((lat, "latency"), (str(st["calls"]), "calls"))):
        x0 = out_x0 + 16 + i * 112
        _grad_text(img, (x0, sy - 2), val, _font("monobold", 24))
        _spaced(d, (x0, sy + 34), lab.upper(), _font("medium", 11), C_SUBTLE, spacing=1.2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--front", required=True, help="third-person video")
    ap.add_argument("--front-start", required=True,
                    help="wall clock (HH:MM:SS.sss) at which the demo starts (the panels start here). With the default "
                         "--front-hold 0 this is also the third-person video's first frame; with --front-hold H that "
                         "first frame is held for H seconds, so the video starts playing at front-start + H")
    ap.add_argument("--panels", required=True, help="dir from make_cgfm_composite_video.py --panels-dir")
    ap.add_argument("--liedown", default=None, help="wall clock of the /liedown command (adds an event)")
    ap.add_argument("--post-speed", type=float, default=1.0,
                    help="play everything AFTER the 360-degree scan this many times faster (third-person video and "
                         "panels together, so their alignment is unchanged)")
    ap.add_argument("--front-hold", type=float, default=0.0,
                    help="hold the third-person video's first frame for this many seconds at the start (the panels "
                         "keep running in real time), then play it")
    ap.add_argument("--blend", action="store_true",
                    help="cross-fade the first-person camera and the three maps between consecutive ticks "
                         "(ticks are ~1.3 s apart; gaps > 2.5 s, e.g. scan rotations, switch without a fade)")
    ap.add_argument("--front-rate", type=float, default=1.0,
                    help="video seconds per wall-clock second of the third-person recording (the phone video of the "
                         "2026-10-04 run ran ~1.4%% slower than the PC clock: 0.986; fit from 4 anchor events)")
    ap.add_argument("--fp-lag", type=float, default=0.15,
                    help="seconds between a first-person frame being captured and its file being saved (frames/ mtime)")
    ap.add_argument("--title", default="Go2 · Life-long Object Navigation")
    ap.add_argument("--subtitle", default="scene graph memory + VLM   ·   real robot   ·   fully autonomous")
    ap.add_argument("--scan-speed", type=int, default=2,
                    help="play the initial 360-degree scan this many times faster (1 = real time)")
    ap.add_argument("--out", default="demo_front.mp4")
    a = ap.parse_args()
    run = a.run_dir
    t_of, phase_of, events, legs, first_explore, vlm = parse_run(run)
    ticks = sorted(t for t in t_of if t in phase_of)
    day = dt.date.fromtimestamp(t_of[ticks[0]])
    t0 = wall(a.front_start, day)
    if a.liedown:
        events.append((wall(a.liedown, day), "liedown", "Robot lies down (/liedown)"))
        events.sort(key=lambda e: e[0])

    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=r_frame_rate,nb_frames", "-of", "json", a.front],
                                      capture_output=True, text=True).stdout)["streams"][0]
    num, den = map(int, probe["r_frame_rate"].split("/"))
    fps = num / den
    n_front = int(probe["nb_frames"])
    n_hold = int(round(a.front_hold * fps))
    span = (t0, t0 + (n_front + n_hold) / fps / a.front_rate)
    print(f"{len(ticks)} ticks, front {n_front} frames @ {fps:.2f} fps, "
          f"front covers {dt.datetime.fromtimestamp(span[0]):%H:%M:%S.%f}"[:-3] + f" + {n_front / fps:.1f}s")

    # per-tick panels (resized once) and VLM cards (rendered once per tick)
    def load(k, name, size):
        p = os.path.join(a.panels, f"tick_{k:05d}_{name}.png")
        return Image.open(p).convert("RGB").resize(size, Image.LANCZOS)
    pan = {k: {"fp": load(k, "fp", (FP_W, FP_H)), **{n: load(k, n, (MAP, MAP)) for n in ("occ", "sgr", "sem")}}
           for k in ticks}
    vcard = {k: vlm_card(vlm or None, k, run, VLM_BOX[2], VLM_BOX[3]) for k in ticks}

    # ---- static layer --------------------------------------------------------------------------
    base = Image.new("RGBA", (W, H))
    base.paste(_gradient(W, H, C_CANVAS[0], C_CANVAS[1], diag=0.5).convert("RGBA"))
    for box in (FRONT_BOX, VLM_BOX, FP_BOX, EV_BOX, TL_BOX, *MAP_BOXES):
        card(base, box)
    d = ImageDraw.Draw(base)
    base.paste(_gradient(40, 40, G1, G2).resize((10, 40)), (M, 12))
    title = a.title
    d.text((M + 22, 32), title, font=_font("bold", 28), fill=C_TEXT, anchor="lm")
    d.text((M + 22 + d.textlength(title, font=_font("bold", 28)) + 18, 34), a.subtitle,
           font=_font("regular", 17), fill=C_MUTED, anchor="lm")
    d.text((EV_BOX[0] + 20, EV_BOX[1] + 26), "Events", font=_font("bold", 20), fill=C_TEXT, anchor="lm")
    # slim timeline: [title + legend] [bar: phase segments, leg labels below, event markers above]
    tx0, tx1 = TL_BOX[0] + 330, TL_BOX[0] + TL_BOX[2] - 30
    ty = TL_BOX[1] + TL_BOX[3] // 2 + 2
    d.text((TL_BOX[0] + 22, TL_BOX[1] + 22), "Life-long mission timeline", font=_font("bold", 18), fill=C_TEXT,
           anchor="lm")
    lx, ly = TL_BOX[0] + 22, TL_BOX[1] + 50
    for n, (p, name, col) in enumerate(PHASES):
        px_, py_ = lx + (n % 2) * 150, ly + (n // 2) * 22
        d.ellipse((px_, py_ - 5, px_ + 10, py_ + 5), fill=col)
        d.text((px_ + 16, py_), name, font=_font("regular", 14), fill=C_MUTED, anchor="lm")

    def x_of(t):
        return tx0 + (tx1 - tx0) * (t - span[0]) / (span[1] - span[0])

    # merged phase runs over time: (t_start, t_end, phase)
    segs = []
    for n_, k_ in enumerate(ticks):
        p_ = phase_of[k_]
        e_ = t_of[ticks[n_ + 1]] if n_ + 1 < len(ticks) else span[1]
        if segs and segs[-1][2] == p_:
            segs[-1][1] = e_
        else:
            segs.append([t_of[k_], e_, p_])
    scan_end = t_of.get(first_explore, span[1])
    for s0, s1, p in segs:
        s0, s1 = max(s0, span[0]), min(s1, span[1])           # the video starts a few s after the run did
        if s1 > s0:
            w_ = max(x_of(s1) - x_of(s0), 2)
            d.rounded_rectangle((x_of(s0), ty - 5, x_of(s0) + w_, ty + 5), min(5, w_ / 2), fill=PHASE_COL[p] + (60,))
    marks = {"lock": ("goal locked", C_AMBER), "found": ("found", C_GREEN), "recall": ("recall", C_TEAL),
             "liedown": ("lies down", C_SUBTLE), "regoal": ("re-goal", C_RED), "stall": ("stall", C_RED)}
    last_right = -1e9
    for t, kind, _ in events:
        if span[0] <= t <= span[1] and kind in marks:
            lab, col = marks[kind]
            x = x_of(t)
            d.polygon([(x, ty - 8), (x - 5, ty - 16), (x + 5, ty - 16)], fill=col)
            tw = d.textlength(lab, font=_font("regular", 13))
            lx_ = max(x - tw / 2, last_right + 8)          # labels of close events must not overlap
            if lx_ + tw > TL_BOX[0] + TL_BOX[2] - 8:
                lx_ = TL_BOX[0] + TL_BOX[2] - 8 - tw
            d.text((lx_, ty - 18), lab, font=_font("regular", 13), fill=col, anchor="lb")
            last_right = lx_ + tw
    for sec in range(0, int(span[1] - span[0]) + 1, 10):
        d.text((x_of(span[0] + sec), ty + 12), f"{sec}s", font=_font("mono", 12), fill=C_SUBTLE, anchor="mt")
    for i_, lg in enumerate(legs):                          # one labelled band per target
        s0 = t_of.get(lg["start"], span[0])
        s1 = t_of.get(lg["found"], span[1]) if lg["found"] is not None else span[1]
        xa, xb = x_of(max(s0, span[0])), x_of(min(s1, span[1]))
        lab = f"{i_ + 1}  {lg['target']}" + ("  ·  recall" if lg["recall"] else "")
        d.line((xa + 2, ty + 27, xb - 2, ty + 27), fill=LEG_COL[i_ % len(LEG_COL)], width=3)
        d.text(((xa + xb) / 2, ty + 31), lab, font=_font("bold", 13), fill=LEG_COL[i_ % len(LEG_COL)], anchor="mt")
    masks = {(w_, h_): rounded_mask(w_, h_, RADIUS) for (_, _, w_, h_) in
             (FRONT_BOX, VLM_BOX, FP_BOX, MAP_BOXES[0])}
    static = base.convert("RGB")

    # ---- per frame -----------------------------------------------------------------------------
    fw, fh = FRONT_BOX[2], FRONT_BOX[3]
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.front, "-vf", f"scale={fw}:{fh}:flags=lanczos",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    out_path = os.path.join(run, a.out)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-framerate", f"{num}/{den}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    f_ev, f_ev_new, f_mono = _font("regular", 15), _font("medium", 15), _font("mono", 13)
    ev_w = EV_BOX[2] - 20 - 34 - 16
    i = 0                                           # output-frame index: drives the wall clock of the panels
    speed_acc = 0.0
    n_total = n_front + n_hold
    tc = [t_of[k_] - a.fp_lag for k_ in ticks]      # capture time of every tick

    def tp_frames():
        n_ = fw * fh * 3
        b0 = dec.stdout.read(n_)
        if len(b0) < n_:
            return
        for _ in range(n_hold):                     # paused: first frame repeated while the panels run on
            yield b0, True
        yield b0, False
        while True:
            b_ = dec.stdout.read(n_)
            if len(b_) < n_:
                return
            yield b_, False

    for buf, held in tp_frames():
        now = span[0] + i / fps / a.front_rate
        speed = 1.0
        if not held:
            if a.scan_speed > 1 and now < scan_end:
                speed = float(a.scan_speed)                   # the 360-degree scan
            elif a.post_speed > 1 and now >= scan_end:
                speed = a.post_speed                          # everything after the scan
        fast = speed > 1
        if fast:                                  # speed up: keep 1 of every `speed` frames (fractional speeds ok)
            speed_acc += 1.0 / speed
            if speed_acc < 1.0:
                i += 1
                continue
            speed_acc -= 1.0
        # first-person panel = the tick whose CAPTURE time is nearest to now (not "the last one already past":
        # ticks are ~1.3 s apart, so that rule made the first-person view lag the third-person view by ~0.7 s)
        cur = now >= tc[0] - 0.5
        k0 = k1 = None
        al = 0.0
        if a.blend and tc[0] <= now < tc[-1]:
            n_ = bisect.bisect_right(tc, now) - 1
            gap = tc[n_ + 1] - tc[n_]
            if gap <= 2.5:
                al = (now - tc[n_]) / gap
            else:                                   # long gap (scan rotation): no fade, switch at the midpoint
                al = 0.0 if now - tc[n_] < gap / 2 else 1.0
            k0, k1 = ticks[n_], ticks[n_ + 1]
            k = k0 if al < 0.5 else k1
        else:
            k = min(ticks, key=lambda kk: abs(t_of[kk] - a.fp_lag - now)) if cur else ticks[0]
            k0 = k1 = k

        def panel(name):
            if k0 == k1 or al <= 0.02:
                return pan[k0][name]
            if al >= 0.98:
                return pan[k1][name]
            return Image.blend(pan[k0][name], pan[k1][name], al)
        phase = phase_of.get(k, "scan") if cur else "scan"
        cur_leg = max([i_ for i_, lg in enumerate(legs) if lg["start"] <= k] or [0])
        for i_, lg in enumerate(legs):                     # keep "Found" on screen for 3 ticks after each find
            if lg["found"] is not None and lg["found"] <= k < lg["found"] + 3:
                phase, cur_leg = "found", i_
        tgt = legs[cur_leg]["target"] if legs else ""
        img = static.copy()
        img.paste(Image.frombuffer("RGB", (fw, fh), buf), FRONT_BOX[:2], masks[(fw, fh)])
        img.paste(vcard[k], VLM_BOX[:2], masks[VLM_BOX[2:]])
        img.paste(panel("fp"), FP_BOX[:2], masks[FP_BOX[2:]])
        for box, n in zip(MAP_BOXES, ("occ", "sgr", "sem")):
            img.paste(panel(n), box[:2], masks[box[2:]])
        d = ImageDraw.Draw(img)
        tw = d.textlength("Third-person view", font=_font("medium", 15))
        d.rounded_rectangle((FRONT_BOX[0] + 14, FRONT_BOX[1] + 14, FRONT_BOX[0] + 38 + tw, FRONT_BOX[1] + 44), 15,
                            fill=(255, 255, 255))
        d.text((FRONT_BOX[0] + 26, FRONT_BOX[1] + 29), "Third-person view", font=_font("medium", 15), fill=C_TEXT,
               anchor="lm")
        if fast:
            sp = f"▶▶ {speed:g}× speed"
            sw = d.textlength(sp, font=_font("bold", 15))
            x1 = FRONT_BOX[0] + FRONT_BOX[2] - 14
            _grad_rrect(img, (x1 - sw - 24, FRONT_BOX[1] + 14, x1, FRONT_BOX[1] + 44), radius=15)
            d.text((x1 - sw - 12, FRONT_BOX[1] + 29), sp, font=_font("bold", 15), fill=(255, 255, 255), anchor="lm")
        # header: phase + clock
        el = now - span[0]
        clock = f"{int(el // 60):02d}:{el % 60:05.2f}"
        cw_txt = d.textlength(clock, font=_font("monobold", 24))
        d.text((W - M - 6, 32), clock, font=_font("monobold", 24), fill=C_TEXT, anchor="rm")
        pn = {"scan": "360° scan", "explore": f"Searching · {tgt}", "approach": f"Approaching · {tgt}",
              "recall": f"Recalling · {tgt}", "found": f"Found · {tgt}", "done": "Mission complete"}[phase]
        pw = d.textlength(pn, font=_font("bold", 18)) + 32
        px = W - M - 6 - cw_txt - 18 - pw
        d.rounded_rectangle((px, 15, px + pw, 49), 17, fill=PHASE_COL[phase])
        d.text((px + pw / 2, 32), pn, font=_font("bold", 18), fill=(255, 255, 255), anchor="mm")
        # mission strip: the target queue and its progress (done = green + check, current = outlined)
        f_chip = _font("medium", 16)
        items = []
        for i_, lg in enumerate(legs):
            dn = lg["found"] is not None and k >= lg["found"]
            items.append((f"{i_ + 1}  {lg['target']}", dn, i_ == cur_leg and not dn and phase != "done"))
        widths = [d.textlength(t_, font=f_chip) + 26 + (18 if dn else 0) for t_, dn, _ in items]
        cx_ = px - 22 - sum(widths) - 8 * (len(items) - 1)
        for (t_, dn, is_cur), w_ in zip(items, widths):
            col = C_GREEN if dn else (PHASE_COL.get(phase, G2) if is_cur else C_GRAY)
            d.rounded_rectangle((cx_, 16, cx_ + w_, 48), 16, fill=_mix(col, C_BG, 0.16), outline=col if is_cur else None,
                                width=2)
            d.text((cx_ + 13, 32), t_, font=f_chip, fill=_mix(col, C_TEXT, 0.7) if (dn or is_cur) else C_MUTED, anchor="lm")
            if dn:
                ox, oy = cx_ + w_ - 24, 32
                d.line([(ox, oy), (ox + 4, oy + 5), (ox + 12, oy - 6)], fill=C_GREEN, width=3)
            cx_ += w_ + 8
        # timeline: elapsed part of each phase in its own colour, then the cursor
        for s0, s1f, p in segs:
            s0 = max(s0, span[0])
            s1 = min(s1f, now)
            if s1 > s0:
                d.rounded_rectangle((x_of(s0), ty - 5, max(x_of(s0) + 10, x_of(s1)), ty + 5), 5, fill=PHASE_COL[p])
        xc = x_of(now)
        d.ellipse((xc - 7, ty - 7, xc + 7, ty + 7), fill=(255, 255, 255), outline=PHASE_COL[phase], width=3)
        # event feed: newest at the bottom, as many (wrapped) events as fit
        past = [e for e in events if e[0] <= now]
        blocks = []
        for t, kind, text in reversed(past):
            lines = wrap(d, text, f_ev, ev_w)
            blocks.append((t, kind, lines))
            if sum(18 + 20 * len(b[2]) + 10 for b in blocks) > EV_BOX[3] - 60:
                blocks.pop()
                break
        y = EV_BOX[1] + 56
        for j, (t, kind, lines) in enumerate(reversed(blocks)):
            latest = j == len(blocks) - 1
            new = now - t < 2.0
            d.ellipse((EV_BOX[0] + 20, y + 4, EV_BOX[0] + 30, y + 14), fill=EV_COL.get(kind, G2))
            d.text((EV_BOX[0] + 38, y + 9), f"{max(0.0, t - span[0]):5.1f}s", font=f_mono, fill=C_SUBTLE, anchor="lm")
            y += 20
            for ln in lines:
                d.text((EV_BOX[0] + 38, y + 9), ln, font=f_ev_new if new else f_ev,
                       fill=C_TEXT if (new or latest) else C_MUTED, anchor="lm")
                y += 20
            y += 10
        enc.stdin.write(img.tobytes())
        i += 1
        if i % 300 == 0:
            print(f"  {i}/{n_total} frames", flush=True)
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print("wrote", out_path, f"({W}x{H}, scan at {a.scan_speed}x)")


if __name__ == "__main__":
    main()
