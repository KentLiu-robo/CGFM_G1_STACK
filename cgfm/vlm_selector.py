"""VLM gatekeeper for target selection (MSGNav-style "object N" decision).

The VLM sees the multimodal scene graph the MSGNav way: the id and class of
every confirmed node, the "seen together" edges with the images that show them
(evidence frames picked by a KSS-style set cover so every edge and every node
appears at least once; objects are outlined and labelled #id in them), plus the
current camera view -- and answers which id (if any) is the target. The caller
maps that id to a world position (scene_graph.node_edge) for the nav stack; the
VLM never produces coordinates.

Queries run in a background thread so the 1 Hz control loop never waits on the
model. The same (target, object list, edge set) is asked only once: after a "none"
answer the VLM is asked again only when a node or an edge appears / merges /
is relabelled, or the target changes.

Served by vLLM's OpenAI-compatible API (see launch_cgfm_vlm_server.sh).
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from typing import List, Optional, Tuple

import cv2
import requests

DEFAULT_URL = "http://127.0.0.1:8101/v1"
DEFAULT_MODEL = "qwen2.5-vl-3b"
VERIFY_SMALL_PX = 48           # a crop with a shorter side than this is too small to judge (a bin ~4 m away is
                               # ~32 px: the model answered "box"/"chair" 13 times, 2026-10-02). Then the pick is
                               # accepted only if the detector's label for the node is the target itself.
VERIFY_MIN_SIDE = 224          # crops are upscaled to at least this short side for the crop check
MAX_IMAGES = 31                 # evidence frames per prompt (+1 current view = the server's 32-image limit)
IMAGE_SIZE = (448, 336)


def build_prompt(target: str, id_labels: List[Tuple[int, str]], evidence: Optional[dict] = None,
                 n_images: int = 0, current_view: bool = False) -> str:
    """Text part of the prompt. evidence: {"edges": {(a, b): [image idx]},
    "nodes": {id: [image idx]}} with indices into the attached image list."""
    lab = dict(id_labels)
    objs = "; ".join(f"{i}: {c}" for i, c in id_labels)
    lines = [f"A robot is searching for: {target}.",
             f"Scene graph objects (id: category): {objs}."]
    if evidence:
        edges = evidence.get("edges") or {}
        if edges:
            lines.append("Relations (objects seen together in one image, and the images that show them):")
            for (a, b), idx in sorted(edges.items()):
                lines.append(f"- {a} {lab.get(a, '?')} & {b} {lab.get(b, '?')}: "
                             + (", ".join(f"Image {i}" for i in idx) or "no image attached"))
        nodes = evidence.get("nodes") or {}
        if nodes:
            lines.append("Where each object can be seen: " + "; ".join(
                f"{n} {lab.get(n, '?')}: " + (", ".join(f"Image {i}" for i in idx) or "-")
                for n, idx in sorted(nodes.items())) + ".")
    if n_images:
        lines.append(f"Images 0-{n_images - 1} follow; in them every scene-graph object is outlined and "
                     "labelled with its id (#N). The category names come from an object detector and may be "
                     "wrong, so check the images.")
    if current_view:
        lines.append("The last image is the robot's current camera view.")
    if n_images:
        # Wording picked on 13 recorded questions (2026-09-30): stricter variants ("answer null if
        # the target has no box of its own") made the 3B model refuse real targets (9-10/13).
        lines.append("Only a box drawn tightly around the target itself counts; a larger box around "
                     "furniture next to or above the target does not.")
    lines.append("If one of these objects is what the robot is searching for, give its id; otherwise null.")
    lines.append('Reply with JSON only: {"object_id": <id or null>}')
    return "\n".join(lines)


# Crop check as a multiple-choice question: on 15 recorded crops (2026-10-02) yes/no ("Is it a trash can?")
# got 12/15 -- the 3B model says "Yes" to a white stool and a black PC tower -- open naming 10/15 (bins
# called "Cup"), this choice list 14/15 (only miss: a desk crop with the bin under it).
VERIFY_CHOICES = ["chair", "couch", "desk", "table", "cabinet", "monitor", "computer", "box", "other"]


def build_verify_prompt(target: str) -> str:
    choices = [target] + [c for c in VERIFY_CHOICES if c != target]
    return ("What is the object in the middle of this image? Choose one: " + ", ".join(choices)
            + ". Answer with the choice only.")


def verify_says_target(answer: str, target: str) -> bool:
    return target.lower() in answer.strip().lower()


def _b64_jpeg(path_or_rgb, resize: bool = True) -> Optional[str]:
    if isinstance(path_or_rgb, str):
        im = cv2.imread(path_or_rgb)
    else:
        im = cv2.cvtColor(path_or_rgb, cv2.COLOR_RGB2BGR)
    if im is None:
        return None
    if resize and (im.shape[1], im.shape[0]) != IMAGE_SIZE:
        im = cv2.resize(im, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def parse_answer(text: str, valid_ids) -> Optional[int]:
    """object_id from the model's reply if it is one of valid_ids, else None."""
    m = re.search(r'"?object_id"?\s*:\s*(null|None|-?\d+)', text)
    if not m or m.group(1) in ("null", "None"):
        return None
    oid = int(m.group(1))
    return oid if oid in valid_ids else None


class VLMTargetSelector:
    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, log_path: Optional[str] = None,
                 timeout_s: float = 10.0, max_consecutive_failures: int = 3):
        self._url = url.rstrip("/")
        self._model = model
        self._log_path = log_path
        self._timeout = timeout_s
        self._max_fail = max_consecutive_failures
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[dict] = None
        self._asked = None            # (target, id_labels) of the last completed/in-flight query
        self._fails = 0
        self._n_calls = 0
        self.available = self._health()

    # ------------------------------------------------------------------
    def _health(self) -> bool:
        try:
            r = requests.get(f"{self._url}/models", timeout=3.0)
            return r.ok and any(m.get("id") == self._model for m in r.json().get("data", []))
        except Exception:  # noqa: BLE001
            return False

    def reset(self) -> None:
        """Forget what was asked (call on target change)."""
        with self._lock:
            self._asked = None
            self._result = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def maybe_query(self, target: str, id_labels: List[Tuple[int, str]], image_paths: Optional[List[str]] = None,
                    edges: Optional[dict] = None, nodes: Optional[dict] = None, current_view=None,
                    crop_fn=None) -> bool:
        """Start a background query if the VLM is up, idle, there is something to
        ask about, and this exact (target, object list, edge set) has not been asked
        yet. image_paths: evidence frames (Image 0..); edges / nodes map to indices
        into them; current_view: RGB array appended last. crop_fn(id) -> crop image
        path (or None): when given, a picked id is double-checked with a multiple-choice
        question on that crop (result["verified"]). Returns True if started."""
        key = (target, tuple(id_labels), tuple(sorted(edges or {})))
        if not self.available or not id_labels or self.busy() or key == self._asked:
            return False
        self._asked = key
        self._thread = threading.Thread(
            target=self._run,
            args=(target, list(id_labels), list(image_paths or []), dict(edges or {}), dict(nodes or {}), current_view,
                  crop_fn),
            daemon=True)
        self._thread.start()
        return True

    def poll(self) -> Optional[dict]:
        """The finished query's result (once), else None:
        {target, id_labels, object_id (int or None), raw, latency_s, ok,
         verified (True / False / None = not checked), verify_raw, crop}."""
        with self._lock:
            r, self._result = self._result, None
        return r

    # ------------------------------------------------------------------
    def _run(self, target: str, id_labels: List[Tuple[int, str]], image_paths: List[str], edges: dict,
             nodes: dict, current_view, crop_fn=None) -> None:
        truncated = max(0, len(image_paths) - MAX_IMAGES)
        image_paths = image_paths[:MAX_IMAGES]
        if truncated:   # drop references to images that were not attached
            edges = {e: [i for i in idx if i < MAX_IMAGES] for e, idx in edges.items()}
            nodes = {n: [i for i in idx if i < MAX_IMAGES] for n, idx in nodes.items()}
        imgs = [_b64_jpeg(pth) for pth in image_paths]
        cur = _b64_jpeg(current_view) if current_view is not None else None
        prompt = build_prompt(target, id_labels, {"edges": edges, "nodes": nodes}, len(imgs), cur is not None)
        content = [{"type": "text", "text": prompt}]
        for i, b64 in enumerate(imgs):
            content.append({"type": "text", "text": f"Image {i}:"})
            if b64 is not None:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        if cur is not None:
            content.append({"type": "text", "text": "Current camera view:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur}"}})
        body = {"model": self._model, "temperature": 0.0, "max_tokens": 20,
                "messages": [{"role": "user", "content": content}]}
        t0 = time.time()
        raw, ok, err, usage = "", False, None, None
        try:
            r = requests.post(f"{self._url}/chat/completions", json=body, timeout=self._timeout)
            r.raise_for_status()
            j = r.json()
            raw = j["choices"][0]["message"]["content"]
            usage = j.get("usage")
            ok = True
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        oid = parse_answer(raw, {i for i, _ in id_labels}) if ok else None
        verified, verify_raw, crop = None, None, None
        if oid is not None and crop_fn is not None:
            try:
                crop = crop_fn(oid)
            except Exception as e:  # noqa: BLE001
                crop, verify_raw = None, f"crop failed: {type(e).__name__}: {e}"
            if crop is not None:
                im = cv2.imread(crop)
                if im is not None and min(im.shape[:2]) < VERIFY_SMALL_PX:
                    label = dict(id_labels).get(oid, "?")
                    verified = label == target
                    verify_raw = (f"too small to check ({im.shape[1]}x{im.shape[0]} px); detector label "
                                  f"{label!r} -> {'accepted' if verified else 'rejected'}")
                else:
                    verified, verify_raw = self._verify(target, crop)
        latency = round(time.time() - t0, 3)
        with self._lock:
            self._n_calls += 1
            if ok:
                self._fails = 0
            else:
                self._fails += 1
                self._asked = None            # allow a retry of the same question
                if self._fails >= self._max_fail:
                    self.available = False
            self._result = {"target": target, "id_labels": id_labels, "object_id": oid, "raw": raw,
                            "latency_s": latency, "ok": ok, "error": err, "n_images": len(imgs),
                            "truncated_images": truncated, "usage": usage,
                            "verified": verified, "verify_raw": verify_raw, "crop": crop}
        if self._log_path:
            try:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps({"call": self._n_calls, "t": round(t0, 2), "prompt": prompt, "raw": raw,
                                        "images": image_paths, "current_view": cur is not None,
                                        "truncated_images": truncated, "object_id": oid, "ok": ok,
                                        "crop": crop, "verified": verified, "verify_raw": verify_raw,
                                        "error": err, "latency_s": latency, "usage": usage}) + "\n")
            except OSError:
                pass

    def _verify(self, target: str, crop_path: str):
        """Multiple-choice check of one object crop: True if the model names the target.
        Returns (True/False, raw) or (None, error)."""
        im = cv2.imread(crop_path)
        if im is None:
            return None, "crop unreadable"
        s = VERIFY_MIN_SIDE / min(im.shape[:2])
        if s > 1:
            im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        b64 = _b64_jpeg(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), resize=False)
        body = {"model": self._model, "temperature": 0.0, "max_tokens": 8,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": build_verify_prompt(target)}]}]}
        try:
            r = requests.post(f"{self._url}/chat/completions", json=body, timeout=self._timeout)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
        return verify_says_target(raw, target), raw
