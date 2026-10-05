"""Builds a composite review video for a CGFM run directory.

Layout (one frame per tick, time-aligned by tick index), 960x800:
  ┌───────────────────────────┬──────────────┐
  │ first-person camera       │ VLM target   │
  │ (628x468 card)            │ check        │
  │                           │ (308x468)    │
  ├────────────┬──────────────┼──────────────┤
  │ Top-down   │ Scene graph  │Semantic field│
  │   view     │  (objects)   │              │
  │ (308x308)  │  (308x308)   │  (308x308)   │
  └────────────┴──────────────┴──────────────┘
Every panel is a rounded white card (violet->blue gradient edge, soft shadow)
on a pale lavender-blue gradient canvas; accents use the same gradient.

VLM panel (from vlm_events.jsonl written by run_cgfm_pipeline.py --vlm): status
chip, target, the scene-graph object list last sent to the VLM (class colours as
in the scene-graph legend; the picked object highlighted), the raw answer and
latency. Runs without --vlm show "VLM not used in this run".

All three map panels share the same square crop window (union of every
non-background pixel across the whole run, padded) so they stay
co-registered and don't jitter. Before a semantic / scene-graph image exists
(e.g. during the initial scan) that panel is a black "not yet" placeholder;
after the last saved image (e.g. once the target is locked) it holds that
image, labelled "held tick N".

Usage:
    python make_cgfm_composite_video.py /path/to/cgfm_pipeline_XXXX \\
        [--out first_person.mp4] [--fps 1] [--margin 40] [--keep-orig]
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess

import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cgfm"))
try:
    from palette import class_color       # same class colours as the scene-graph panel / legend
except ImportError:                       # keep the video tool usable on its own
    def class_color(label):
        return (160, 160, 160)

OUT_W, OUT_H = 960, 800   # output video size
CELL = 320                 # grid unit: bottom row = three 320x320 cells, top row = 640x480 + 320x480
TOP_H = 480
GUT = 6                    # every panel is inset GUT px in its cell -> 12 px gaps between the rounded cards
MAP_SIDE = CELL - 2 * GUT  # 308 px map-tile content
FP_W, FP_H = 2 * CELL - 2 * GUT, TOP_H - 2 * GUT   # 628x468 first-person card (camera 4:3, ~unscaled)
VLM_W = CELL - 2 * GUT     # 308 px VLM panel (height FP_H)
CARD_R = 14                # card corner radius
SHADOW_PAD = 12
BG = (255, 255, 255)       # fill for a missing map image

# ---- VLM panel design (drawn with Pillow for anti-aliased type and rounded cards) ----
_FONT_DIR_SANS = "/usr/share/fonts/opentype/noto"
_FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_FONT_MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
_FONTS = {}


def _font(kind, size):
    """kind: regular / medium / bold / serif / serifmed / serifsemi / mono / monobold.
    Falls back to Pillow's default."""
    key = (kind, size)
    if key not in _FONTS:
        path = {"regular": f"{_FONT_DIR_SANS}/NotoSansCJK-Regular.ttc",
                "medium": f"{_FONT_DIR_SANS}/NotoSansCJK-Medium.ttc",
                "bold": f"{_FONT_DIR_SANS}/NotoSansCJK-Bold.ttc",
                "serif": f"{_FONT_DIR_SANS}/NotoSerifCJK-Regular.ttc",
                "serifmed": f"{_FONT_DIR_SANS}/NotoSerifCJK-Medium.ttc",
                "serifsemi": f"{_FONT_DIR_SANS}/NotoSerifCJK-SemiBold.ttc",
                "mono": _FONT_MONO, "monobold": _FONT_MONO_B}[kind]
        try:
            _FONTS[key] = ImageFont.truetype(path, size)
        except OSError:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


# palette (RGB): Qwen-style violet -> blue gradient accents, white cards on a pale lavender-blue canvas
G1 = (111, 76, 255)                # gradient start (violet)
G2 = (58, 134, 255)                # gradient end (blue)
C_CANVAS = ((244, 241, 255), (232, 242, 255))   # background gradient, top-left -> bottom-right
C_SHADOW = (72, 60, 160)           # card shadow tint
C_BG = (255, 255, 255)             # card / panel background
C_CARD = (247, 246, 255)           # sub-card (lavender)
C_CARD_EDGE = (228, 225, 250)
C_TEXT = (27, 25, 56)              # indigo ink
C_MUTED = (96, 96, 130)
C_SUBTLE = (148, 150, 182)
C_KEY = G2                         # JSON keys
C_GREEN = (18, 170, 120)
C_AMBER = (236, 146, 28)
C_BLUE = G2
C_RED = (234, 72, 100)
C_GRAY = (148, 150, 182)
C_DOT_EDGE = (60, 58, 92)          # outline of class-colour dots

# status -> (label, RGB colour)
_STATUS = {
    "off": ("VLM not used", C_GRAY),
    "unavailable": ("VLM unavailable", C_RED),
    "waiting": ("Waiting for objects", C_GRAY),
    "asking": ("Asking VLM", C_AMBER),
    "none": ("Not found · exploring", C_BLUE),
    "picked": ("Target picked · dry-run", C_GREEN),
    "locked": ("Locked · navigating", C_GREEN),
    "found": ("Target reached", C_GREEN),
    "failed": ("Call failed", C_RED),
    "gone": ("Pick merged · asking again", C_AMBER),
    "rejected": ("Crop check failed · exploring", C_AMBER),
}
VLM_MODEL_NAME = "Qwen2.5-VL-3B"


def load_vlm_events(run_dir):
    path = os.path.join(run_dir, "vlm_events.jsonl")
    if not os.path.exists(path):
        return None
    ev = []
    with open(path) as f:
        for line in f:
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
    return sorted(ev, key=lambda e: e.get("tick", 0))      # stable: keeps order within a tick


def vlm_state_at(events, t):
    """Fold the VLM events up to tick t into what the panel shows."""
    st = {"mode": "on", "target": None, "calls": 0, "objects": [], "asked_tick": None,
          "status": "waiting", "answer": None, "images": [], "relations": []}
    for e in events:
        if e.get("tick", 0) > t:
            break
        k = e["kind"]
        if k in ("on", "unavailable"):
            st["mode"], st["target"] = k, e.get("target")
            if k == "unavailable":
                st["status"] = "unavailable"
        elif k == "target":
            st.update(target=e["target"], objects=[], asked_tick=None, answer=None, status="waiting")
        elif k == "asked":
            st.update(calls=st["calls"] + 1, objects=e.get("objects", []), asked_tick=e["tick"],
                      status="asking", target=e.get("target", st["target"]),
                      images=e.get("images", []), relations=e.get("relations", []))
        elif k == "none":
            st.update(status="none", answer=e)
        elif k == "picked":
            st.update(status="locked" if e.get("locked") else "picked", answer=e)
        elif k == "gone":
            st.update(status="gone", answer=e)
        elif k == "rejected":
            st.update(status="rejected", answer=e)
        elif k == "failed":
            st.update(status="unavailable" if e.get("unavailable") else "failed", answer=e)
        elif k == "found":
            st["status"] = "found"
    return st


def tick_of(path):
    return int(re.search(r"tick_(\d+)", os.path.basename(path)).group(1))


def load_by_tick(folder, ext):
    return {tick_of(p): p for p in glob.glob(os.path.join(folder, f"tick_*.{ext}"))
            if not p.endswith("_vis.jpg")}


def union_crop(paths, margin):
    """Square crop window covering every non-background pixel across all maps."""
    x0 = y0 = 10**9
    x1 = y1 = -1
    h = w = None
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        # occupancy maps: white background; semantic maps: dark background
        # Use "any pixel deviates from corners" as non-background heuristic
        bg = im[0, 0]
        nz = np.argwhere(np.any(im != bg, axis=2))
        if len(nz) == 0:
            continue
        (ya, xa), (yb, xb) = nz.min(0), nz.max(0)
        x0, y0, x1, y1 = min(x0, xa), min(y0, ya), max(x1, xb), max(y1, yb)
    if h is None:
        return 0, 0, 100, 100
    if x1 < 0:
        return 0, 0, w, h
    side = int(max(x1 - x0, y1 - y0) + 1 + 2 * margin)
    side = min(side, h, w)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    xs = int(np.clip(cx - side // 2, 0, w - side))
    ys = int(np.clip(cy - side // 2, 0, h - side))
    return xs, ys, side, side


def _mix(c, bg, a):
    """Colour c blended over bg with opacity a (RGB)."""
    return tuple(int(round(a * ci + (1 - a) * bi)) for ci, bi in zip(c, bg))


def _spaced(d, xy, text, font, fill, spacing=1.2):
    """Small-caps style label with letter spacing."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + spacing
    return x


def _gradient(w, h, c1=G1, c2=G2, diag=0.35):
    """w x h RGB image going c1 -> c2 left to right (tilted by `diag`)."""
    x = np.linspace(0.0, 1.0, max(int(w), 1))[None, :]
    y = np.linspace(0.0, 1.0, max(int(h), 1))[:, None]
    t = np.clip((1 - diag) * x + diag * y, 0, 1)[..., None]
    arr = (1 - t) * np.array(c1, float) + t * np.array(c2, float)
    return Image.fromarray(arr.astype(np.uint8))


def _grad_rrect(img, box, radius, width=0):
    """Rounded rectangle filled (width=0) or stroked with the violet->blue gradient."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = x1 - x0 + 1, y1 - y0 + 1
    m = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(m)
    if width:
        md.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, outline=255, width=width)
    else:
        md.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    img.paste(_gradient(w, h), (x0, y0), m)


def _grad_text(img, xy, text, font, spacing=0.0):
    """Text filled with the violet->blue gradient; returns the x after the text."""
    d = ImageDraw.Draw(img)
    w = int(sum(d.textlength(ch, font=font) + spacing for ch in text)) + 2
    h = int(font.size * 1.6) + 2
    m = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(m)
    cx = 0.0
    for ch in text:
        md.text((cx, 0), ch, font=font, fill=255)
        cx += md.textlength(ch, font=font) + spacing
    img.paste(_gradient(w, h, diag=0.0), (int(xy[0]), int(xy[1])), m)
    return xy[0] + cx


def _card(d, box, fill=C_CARD, edge=C_CARD_EDGE, r=10):
    """Soft rounded lavender sub-card with a hairline edge."""
    d.rounded_rectangle(box, radius=r, fill=fill, outline=edge, width=1)


def _section(img, xy, text):
    """Small-caps section label in the brand gradient."""
    return _grad_text(img, xy, text, _font("bold", 10), spacing=1.4)


_THUMBS = {}


def _thumb(path, w, h):
    """Rounded thumbnail of an image the VLM was shown (cached)."""
    key = (path, w, h)
    if key not in _THUMBS:
        im = cv2.imread(path) if path and os.path.exists(path) else None
        if im is None:
            _THUMBS[key] = None
        else:
            th = Image.fromarray(cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=5, fill=255)
            _THUMBS[key] = (th, m)
    return _THUMBS[key]


def vlm_panel(events, t, run_dir=None):
    """VLM panel (VLM_W x FP_H) for tick t, as a BGR array."""
    W, H = VLM_W, FP_H
    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)
    pad = 12

    # header: gradient badge + title, model on the right, gradient rule
    bx0, by0 = pad, 12
    bw = d.textlength("VLM", font=_font("bold", 11)) + 16
    _grad_rrect(img, (bx0, by0, bx0 + bw, by0 + 20), radius=10)
    d.text((bx0 + 8, by0 + 2), "VLM", font=_font("bold", 11), fill=(255, 255, 255))
    d.text((bx0 + bw + 8, by0 - 1), "Target check", font=_font("bold", 15), fill=C_TEXT)
    mw = d.textlength(VLM_MODEL_NAME, font=_font("mono", 10))
    d.text((W - pad - mw, by0 + 5), VLM_MODEL_NAME, font=_font("mono", 10), fill=C_MUTED)
    _grad_rrect(img, (pad, 41, W - pad, 42), radius=1)

    if events is None:
        _card(d, (pad, 60, W - pad, 140))
        msg, sub = "VLM not used in this run", "run with --vlm to enable"
        d.text(((W - d.textlength(msg, font=_font("medium", 14))) / 2, 84), msg, font=_font("medium", 14), fill=C_MUTED)
        d.text(((W - d.textlength(sub, font=_font("regular", 11))) / 2, 108), sub, font=_font("regular", 11), fill=C_SUBTLE)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    st = vlm_state_at(events, t)
    ans = st["answer"] or {}
    picked = ans.get("object_id") if st["status"] in ("picked", "locked", "found") else None

    # target
    _spaced(d, (pad, 52), "TARGET", _font("medium", 10), C_SUBTLE, spacing=1.4)
    d.text((pad, 63), str(st["target"] or "-"), font=_font("bold", 20), fill=C_TEXT)

    # status pill (semantic colour)
    text, col = _STATUS.get(st["status"], (st["status"], C_GRAY))
    if picked is not None:
        text = {"picked": f"Picked #{picked} · dry-run", "locked": f"Locked #{picked} · navigating",
                "found": f"Reached #{picked}"}[st["status"]]
    f_pill = _font("medium", 12)
    pw = d.textlength(text, font=f_pill) + 34
    py = 98
    d.rounded_rectangle((pad, py, pad + pw, py + 24), radius=12, fill=_mix(col, C_BG, 0.12),
                        outline=_mix(col, C_BG, 0.45), width=1)
    d.ellipse((pad + 11, py + 8, pad + 19, py + 16), fill=col)
    d.text((pad + 25, py + 3), text, font=f_pill, fill=_mix(col, C_TEXT, 0.75))

    # layout: OUTPUT card anchored to the bottom, INPUT card fills the space above it
    out_h = 118
    out_top = H - pad - out_h
    in_top, row_h = 134, 21
    in_bot = out_top - 26                      # room for the flow connector
    objs = st["objects"]
    thumbs = [k for k in st["images"]] if run_dir else []
    strip_h = 56 if thumbs else 0
    rows_fit = max(1, (in_bot - strip_h - in_top - 40) // row_h)
    cols = 2 if len(objs) > rows_fit else 1    # many objects: two compact columns
    capacity = rows_fit * cols
    if len(objs) > capacity:
        capacity -= cols                       # keep a line for "+ N more"
    _card(d, (pad, in_top, W - pad, in_bot))
    x = _section(img, (pad + 12, in_top + 9), "INPUT")
    head = f"{len(objs)} obj · {len(st['relations'])} rel · {len(st['images'])} img"
    d.text((x + 6, in_top + 8), head, font=_font("regular", 11), fill=C_SUBTLE)
    if st["asked_tick"] is not None:
        meta = f"call {st['calls']} · t{st['asked_tick']}"
        d.text((W - pad - 12 - d.textlength(meta, font=_font("mono", 10)), in_top + 9), meta,
               font=_font("mono", 10), fill=C_MUTED)
    y0 = in_top + 34
    if not objs:
        d.text((pad + 12, y0 + 2), "no confirmed objects yet", font=_font("regular", 12), fill=C_SUBTLE)
    col_w = (W - 2 * pad) // cols
    # the pick first, then target-class objects, so they never fold into "+ N more"
    shown = sorted(objs, key=lambda o: (o[0] != picked, o[1] != st["target"]))[:capacity]
    per_col = -(-len(shown) // cols) if shown else 0
    for k, (oid, lab) in enumerate(shown):
        c, r_ = divmod(k, per_col) if cols == 2 else (0, k)
        x0, y = pad + c * col_w, y0 + r_ * row_h
        x1 = x0 + col_w
        is_pick, is_tgt_cls = oid == picked, lab == st["target"]
        if is_pick:
            d.rounded_rectangle((x0 + 6, y - 1, x1 - 6, y + row_h - 3), radius=6, fill=_mix(G1, C_CARD, 0.10),
                                outline=_mix(G1, C_CARD, 0.35), width=1)
            _grad_rrect(img, (x0 + 6, y - 1, x0 + 9, y + row_h - 3), radius=2)
        b, g, r = class_color(lab)
        cy = y + (row_h - 4) // 2
        d.ellipse((x0 + 16, cy - 5, x0 + 26, cy + 5), fill=(r, g, b), outline=C_DOT_EDGE)
        id_w = 38 if cols == 1 else 30
        d.text((x0 + 33, y + 1), f"#{oid}", font=_font("mono", 12 if cols == 1 else 11),
               fill=C_TEXT if is_pick else C_MUTED)
        name_font = _font("bold" if (is_pick or is_tgt_cls) else "regular", 13 if cols == 1 else 12)
        name = lab
        max_name_w = (x1 - 12) - (x0 + 33 + id_w) - (46 if (is_pick and cols == 1) else 4)
        while d.textlength(name, font=name_font) > max_name_w and len(name) > 3:
            name = name[:-2] + "…"
        d.text((x0 + 33 + id_w, y), name, font=name_font,
               fill=C_TEXT if (is_pick or is_tgt_cls) else _mix(C_TEXT, C_CARD, 0.8))
        if is_pick and cols == 1:
            tag = "match"
            tw = d.textlength(tag, font=_font("medium", 10))
            tx = x1 - 18 - tw - 10
            _grad_rrect(img, (tx, y + 1, tx + tw + 10, y + 16), radius=7)
            d.text((tx + 5, y + 1), tag, font=_font("medium", 10), fill=(255, 255, 255))
    if len(objs) > capacity:
        d.text((pad + 33, y0 + per_col * row_h), f"+ {len(objs) - capacity} more",
               font=_font("regular", 11), fill=C_SUBTLE)

    # evidence images actually sent (+ current view), as a thumbnail strip
    if thumbs:
        sy0 = in_bot - strip_h + 4
        d.line((pad + 12, sy0 - 4, W - pad - 12, sy0 - 4), fill=C_CARD_EDGE, width=1)
        tw_, th_ = 52, 39
        slots = (W - 2 * pad - 24 + 6) // (tw_ + 6)
        shown_k = thumbs[:slots - 1] if len(thumbs) > slots else thumbs
        tx = pad + 12
        for i, k in enumerate(shown_k):
            tt = _thumb(os.path.join(run_dir, "vlm_inputs", f"{k}.jpg"), tw_, th_)
            if tt is not None:
                img.paste(tt[0], (tx, sy0 + 2), tt[1])
                _grad_rrect(img, (tx, sy0 + 2, tx + tw_ - 1, sy0 + 2 + th_ - 1), radius=5, width=1)
            d.text((tx + 3, sy0 + 3), str(i), font=_font("bold", 9), fill=(255, 255, 255))
            tx += tw_ + 6
        if len(thumbs) > len(shown_k):
            d.text((tx + 2, sy0 + 14), f"+{len(thumbs) - len(shown_k)}", font=_font("medium", 11), fill=C_MUTED)

    # flow connector
    cx, fy = W // 2, in_bot + 6
    _grad_rrect(img, (cx - 1, fy, cx, fy + 12), radius=0)
    d.polygon([(cx - 5, fy + 12), (cx + 5, fy + 12), (cx, fy + 18)], fill=G2)
    d.text((cx + 10, fy + 2), VLM_MODEL_NAME, font=_font("mono", 10), fill=C_MUTED)

    # OUTPUT card
    out_bot = out_top + out_h
    _card(d, (pad, out_top, W - pad, out_bot))
    _section(img, (pad + 12, out_top + 9), "OUTPUT")
    jx, jy = pad + 12, out_top + 28
    if st["status"] == "asking":
        d.text((jx, jy), "waiting for the answer …", font=_font("regular", 12), fill=C_AMBER)
    elif ans.get("error"):
        d.text((jx, jy), str(ans["error"])[:36], font=_font("mono", 11), fill=C_RED)
    elif ans.get("raw"):
        oid = ans.get("object_id")
        parts = [("{", C_MUTED), ('"object_id"', C_KEY), (": ", C_MUTED),
                 (str(oid) if oid is not None else "null", G1 if oid is not None else C_AMBER), ("}", C_MUTED)]
        x = jx
        for txt, col in parts:
            fnt = _font("monobold" if col in (G1, C_AMBER) else "mono", 14)
            d.text((x, jy), txt, font=fnt, fill=col)
            x += d.textlength(txt, font=fnt)
        if oid is not None and ans.get("label"):
            check = ""
            if ans.get("verify_raw") is not None:
                vr = str(ans["verify_raw"]).strip().rstrip(".").lower()
                check = " · crop: " + ("too small" if vr.startswith("too small") else vr[:14])
            d.text((jx, jy + 22), f"→ #{oid} {ans['label']}{check}", font=_font("regular", 12),
                   fill=C_AMBER if st["status"] == "rejected" else C_TEXT)
        elif oid is None:
            d.text((jx, jy + 22), "→ no target in the scene graph", font=_font("regular", 12), fill=C_MUTED)
    else:
        d.text((jx, jy), "—", font=_font("mono", 14), fill=C_SUBTLE)
    # stats (gradient figures)
    sy = out_top + 80
    d.line((pad + 12, sy - 7, W - pad - 12, sy - 7), fill=C_CARD_EDGE, width=1)
    lat = f"{ans['latency_s']:.2f}s" if ans.get("latency_s") is not None else "—"
    for i, (val, lab) in enumerate(((lat, "latency"), (str(st["calls"]), "calls"))):
        x0 = pad + 12 + i * 110
        _grad_text(img, (x0, sy - 4), val, _font("monobold", 16))
        _spaced(d, (x0, sy + 19), lab.upper(), _font("medium", 9), C_SUBTLE, spacing=1.2)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ---- shared styling for the map tiles / camera overlays / card composition ----
TITLE_H = 26


def _to_pil(im_bgr):
    return Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))


def _to_bgr(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def pill(d, xy, text, font, fg=G1, bg=C_BG, edge=C_CARD_EDGE):
    x, y = xy
    tw = d.textlength(text, font=font)
    d.rounded_rectangle((x, y, x + tw + 16, y + 18), radius=9, fill=bg, outline=edge, width=1)
    d.text((x + 8, y + 2), text, font=font, fill=fg)


def style_tile(im_bgr, title, held_from=None):
    """White title bar (gradient dot + title, hairline rule) and a 'HELD t N'
    pill when the panel shows an older image. The rounded card edge itself is
    added when the tile is placed on the canvas."""
    img = _to_pil(im_bgr)
    d = ImageDraw.Draw(img)
    W, H = img.size
    d.rectangle((0, 0, W, TITLE_H - 1), fill=C_BG)
    _grad_rrect(img, (12, 9, 20, 17), radius=4)
    d.text((26, 3), title, font=_font("bold", 13), fill=C_TEXT)
    d.line((0, TITLE_H - 1, W, TITLE_H - 1), fill=C_CARD_EDGE, width=1)
    if held_from is not None:
        pill(d, (10, H - 28), f"HELD · t{held_from}", _font("mono", 10))
    return _to_bgr(img)


def placeholder(title, message):
    """White 'not yet' tile with a centred lavender card."""
    img = Image.new("RGB", (MAP_SIDE, MAP_SIDE), C_BG)
    d = ImageDraw.Draw(img)
    cy = MAP_SIDE // 2 + TITLE_H // 2
    _card(d, (24, cy - 30, MAP_SIDE - 24, cy + 30))
    f = _font("medium", 13)
    d.text(((MAP_SIDE - d.textlength(message, font=f)) / 2, cy - 10), message, font=f, fill=C_MUTED)
    return style_tile(_to_bgr(img), title)


def draw_legend(im_bgr, entries):
    """Colour -> class key inside the scene-graph tile: a white card with a
    hairline edge, placed in the emptiest corner below the title bar."""
    if not entries:
        return im_bgr
    img = _to_pil(im_bgr)
    d = ImageDraw.Draw(img)
    f, fb, row = _font("regular", 11), _font("bold", 11), 16
    texts = [(lab, " target" if is_tgt else "") for lab, _, is_tgt in entries]
    w = int(max(d.textlength(a, font=fb if b else f) + d.textlength(b, font=f) for a, b in texts)) + 30
    h = 8 + row * len(entries)
    top = TITLE_H + 6
    cands = [(MAP_SIDE - w - 8, top), (8, top), (MAP_SIDE - w - 8, MAP_SIDE - h - 8), (8, MAP_SIDE - h - 8)]
    arr = np.asarray(img)

    def emptiness(xy):
        x, y = xy
        patch = arr[y:y + h, x:x + w]
        return float(((patch.min(axis=2) > 235) | (patch.max(axis=2) < 45)).mean())

    x, y = max(cands, key=emptiness)
    overlay = img.copy()
    ImageDraw.Draw(overlay).rounded_rectangle((x, y, x + w, y + h), radius=8, fill=C_BG)
    img = Image.blend(img, overlay, 0.92)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((x, y, x + w, y + h), radius=8, outline=C_CARD_EDGE, width=1)
    for k, ((lab, col, is_tgt), (a, b)) in enumerate(zip(entries, texts)):
        cy = y + 12 + k * row
        bb, gg, rr = (int(c) for c in col)
        d.ellipse((x + 8, cy - 5, x + 18, cy + 5), fill=(rr, gg, bb), outline=C_DOT_EDGE, width=2 if is_tgt else 1)
        d.text((x + 24, cy - 8), a, font=fb if is_tgt else f, fill=C_TEXT)
        if b:
            d.text((x + 24 + d.textlength(a, font=fb), cy - 8), b, font=f, fill=G1)
    return _to_bgr(img)


def camera_overlay(fp_bgr, t, t_last, frac, frozen_frac=None):
    """Tick pill + gradient progress bar on the first-person panel."""
    img = _to_pil(fp_bgr)
    d = ImageDraw.Draw(img)
    W, H = img.size
    pill(d, (12, H - 36), f"TICK {t} / {t_last}", _font("monobold", 11))
    d.rectangle((0, H - 6, W, H), fill=C_CARD_EDGE)
    if frac > 0:
        _grad_rrect(img, (0, H - 6, max(1, int(W * frac)), H - 1), radius=0)
    if frozen_frac is not None:
        fx = int(W * frozen_frac)
        d.line((fx, H - 12, fx, H), fill=C_TEXT, width=2)
    return _to_bgr(img)


_CARD_CACHE = {}


def _card_masks(w, h):
    """(rounded mask, blurred shadow mask) for a w x h card, cached."""
    if (w, h) not in _CARD_CACHE:
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=CARD_R, fill=255)
        sh = Image.new("L", (w + 2 * SHADOW_PAD, h + 2 * SHADOW_PAD), 0)
        ImageDraw.Draw(sh).rounded_rectangle((SHADOW_PAD, SHADOW_PAD + 3, SHADOW_PAD + w - 1, SHADOW_PAD + h + 2),
                                             radius=CARD_R, fill=70)
        _CARD_CACHE[(w, h)] = (mask, sh.filter(ImageFilter.GaussianBlur(6)))
    return _CARD_CACHE[(w, h)]


def place_card(canvas, content_bgr, cell_xy):
    """Put a panel on the canvas as a rounded card: soft violet-tinted shadow,
    rounded corners, gradient border."""
    content = _to_pil(content_bgr)
    w, h = content.size
    x, y = cell_xy[0] + GUT, cell_xy[1] + GUT
    mask, shadow = _card_masks(w, h)
    canvas.paste(C_SHADOW, (x - SHADOW_PAD, y - SHADOW_PAD, x - SHADOW_PAD + shadow.width,
                            y - SHADOW_PAD + shadow.height), shadow)
    canvas.paste(content, (x, y), mask)
    _grad_rrect(canvas, (x, y, x + w - 1, y + h - 1), radius=CARD_R, width=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="first_person.mp4")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="ticks per second of video time (1 = real-time tick rate)")
    ap.add_argument("--margin", type=int, default=40,
                    help="map crop padding in map pixels")
    ap.add_argument("--keep-orig", action="store_true", default=True,
                    help="rename existing --out to <name>_orig.mp4 before writing")
    ap.add_argument("--panels-dir", default=None,
                    help="also save each tick's panels (camera, 3 map tiles) as tick_N_{fp,occ,sgr,sem}.png "
                         "for other layouts (make_cgfm_demo_front.py)")
    a = ap.parse_args()
    if a.panels_dir:
        os.makedirs(a.panels_dir, exist_ok=True)

    frames = load_by_tick(os.path.join(a.run_dir, "frames"), "jpg")
    occ    = load_by_tick(os.path.join(a.run_dir, "occupancy_map"), "png")
    sem    = load_by_tick(os.path.join(a.run_dir, "semantic_map"),  "png")
    sgr    = load_by_tick(os.path.join(a.run_dir, "scene_graph"),   "png")

    ticks = sorted(frames)
    assert ticks and occ, "need frames/ and occupancy_map/ in the run dir"

    # Crop window: derive from all three map types combined
    all_map_paths = list(occ.values()) + list(sem.values()) + list(sgr.values())
    xs, ys, cw, ch = union_crop(all_map_paths or list(occ.values()), a.margin)
    print(f"{len(ticks)} ticks, crop {xs},{ys} size {cw}x{ch} → {MAP_SIDE}x{MAP_SIDE}")

    last_occ_tick = max(occ) if occ else ticks[-1]
    frozen_after = last_occ_tick if last_occ_tick < ticks[-1] else None

    out_path = os.path.join(a.run_dir, a.out)
    if os.path.exists(out_path) and a.keep_orig:
        orig = out_path[:-4] + "_orig.mp4"
        if not os.path.exists(orig):
            shutil.move(out_path, orig)
            print("original kept as", orig)

    canvas_bg = _gradient(OUT_W, OUT_H, C_CANVAS[0], C_CANVAS[1], diag=0.5)
    vlm_events = load_vlm_events(a.run_dir)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{OUT_W}x{OUT_H}", "-framerate", str(a.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-r", "30", out_path],
        stdin=subprocess.PIPE)

    legend_path = os.path.join(a.run_dir, "scene_graph_legend.json")
    legend = []
    if os.path.exists(legend_path):
        with open(legend_path) as f:
            legend = json.load(f)          # [[label, [b, g, r], is_target], ...]

    def latest_at_or_before(d, t, fallback=None):
        valid = [x for x in d if x <= t]
        if not valid:
            return None, fallback
        k = max(valid)
        return k, d[k]

    def map_tile(path, title, held_from=None):
        im = cv2.imread(path)
        if im is None:
            im = np.full((ch, cw, 3), BG, dtype=np.uint8)
        else:
            im = im[ys:ys + ch, xs:xs + cw]
        im = cv2.resize(im, (MAP_SIDE, MAP_SIDE), interpolation=cv2.INTER_AREA)
        return style_tile(im, title, held_from)

    for t in ticks:
        # --- first-person panel ---
        fp = cv2.resize(cv2.imread(frames[t]), (FP_W, FP_H), interpolation=cv2.INTER_AREA)
        fp = camera_overlay(fp, t, ticks[-1], (t - ticks[0] + 1) / len(ticks),
                            (frozen_after - ticks[0] + 1) / len(ticks) if frozen_after else None)

        # --- three map panels ---
        ko, po = latest_at_or_before(occ, t)
        ks, ps = latest_at_or_before(sem, t)
        kg, pg = latest_at_or_before(sgr, t)

        # Semantic / scene-graph not computed yet (e.g. during the initial scan):
        # black placeholder, never another panel's image.
        tile_occ = map_tile(po, "Top-down view",  ko if ko != t else None)
        tile_sem = (map_tile(ps, "Semantic field", ks if ks != t else None) if ps
                    else placeholder("Semantic field", "No semantic map yet"))
        tile_sgr = (draw_legend(map_tile(pg, "Scene graph", kg if kg != t else None), legend) if pg
                    else placeholder("Scene graph", "No scene graph yet"))

        if a.panels_dir:
            for name, im in (("fp", fp), ("occ", tile_occ), ("sgr", tile_sgr), ("sem", tile_sem)):
                cv2.imwrite(os.path.join(a.panels_dir, f"tick_{t:05d}_{name}.png"), im)
        canvas = canvas_bg.copy()
        place_card(canvas, fp, (0, 0))
        place_card(canvas, vlm_panel(vlm_events, t, a.run_dir), (2 * CELL, 0))
        place_card(canvas, tile_occ, (0, TOP_H))
        place_card(canvas, tile_sgr, (CELL, TOP_H))
        place_card(canvas, tile_sem, (2 * CELL, TOP_H))
        ff.stdin.write(_to_bgr(canvas).tobytes())

    ff.stdin.close()
    ff.wait()
    print("wrote", out_path, f"({OUT_W}x{OUT_H}, {len(ticks)} ticks @ {a.fps} tick/s)")


if __name__ == "__main__":
    main()
