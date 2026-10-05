"""Lightweight scene graph for CGFM navigation.

Stores detected objects as CLIP-embedded nodes backed by a small 3D point
cloud. No SAM — YOLO-World boxes only — but object association follows the
two reference implementations instead of the original single-pixel/0.5 m rule:

  * VLFM  (vlfm/mapping/object_point_cloud_map.py)
      - back-project the detection region to a point cloud and keep only the
        largest DBSCAN cluster (the object, not the wall behind it);
      - without masks, YOLO-World sometimes returns one box around a GROUP of
        objects next to the per-object boxes; a box that contains another
        same-class box is dropped so the group is not fused into one object;
      - objects the depth camera cannot see (a mesh wastebasket: the IR pattern
        passes through the holes, so the box holds only floor/wall depth and no
        cluster survives) get SYNTHETIC points instead: a small cylinder at the
        box's ground contact (bottom edge meets the floor), sized from the box
        width; objects involving synthetic points merge by centroid distance;
      - floor points are removed first (height filter, like ObstacleMap's
        min_height), otherwise the floor links neighbouring objects into one cloud;
      - objects are keyed by class name (a chair never merges into a table).
  * MSGNav / ConceptGraph (src/multimodal_3d_scene_graph.py, cfg/concept_graph_default.yaml)
      - a detection joins an existing object when
            spatial_overlap + clip_cosine > SIM_THRESHOLD   (1.2, phys_bias 0)
      - merged CLIP features are the detection-count-weighted mean;
      - periodic object-object merge when overlap and visual similarity are both high;
      - objects need >= OBJ_MIN_DETECTIONS sightings before they are used.

Usage (unchanged)
-----------------
    sg = LightweightSceneGraph(clip_model, clip_preprocess, device='cuda', min_z=MIN_HEIGHT)
    sg.update(color_rgb, detections, valid_idxs, depth_raw,
              depth_scale, tf_camera_to_episodic, fx, fy)
    text_feat = sg.encode_text('chair')          # (1, D) tensor
    scored = sg.get_scored_objects(text_feat)    # [(score, world_xy), ...]
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

CROP_PAD_PX = 20            # pixel padding around bbox before CLIP crop

# --- point cloud extraction (VLFM-style) ---
BOX_SHRINK = 0.15           # ignore this fraction of the box on each side (background bleeds in at the edges)
PIX_STRIDE = 2              # subsample pixels inside the box
MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 3.0
DBSCAN_EPS_M = 0.10         # MSGNav dbscan_eps
DBSCAN_MIN_POINTS = 10
MIN_CLUSTER_POINTS = 30     # smaller clusters are too unreliable to become an object
FOREGROUND_MIN_FRAC = 0.3   # nearest cluster wins if it has >= this fraction of the largest cluster's points
VOXEL_M = 0.05              # downsample voxel for stored object clouds
MAX_OBJ_POINTS = 3000

# --- association (MSGNav/ConceptGraph-style) ---
OVERLAP_RADIUS_M = 0.10     # a detection point "overlaps" if an object point is within this radius
SIM_THRESHOLD = 1.2         # spatial_overlap + clip_cosine must exceed this to merge
GATE_DIST_M = 1.5           # never compare against objects whose centroids are farther than this
MERGE_OVERLAP_THRESH = 0.5  # periodic object-object merge: overlap ...
MERGE_VISUAL_THRESH = 0.8   # ... and CLIP cosine both above these
# Periodic duplicate merge (same class), in addition to the overlap rule above.
# Measured 2026-09-29 on a real run: fragments of one big object (cabinet, couch,
# a chair seen from two sides) have footprints that TOUCH (gap 0-0.02 m, CLIP
# 0.68-0.86); one object recorded twice can sit apart but looks the same (trash
# can: 0.6 m apart, CLIP 0.96); distinct neighbours keep a gap and/or lower CLIP.
DUP_TOUCH_GAP_M = 0.05      # footprints (xy) closer than this ...
DUP_TOUCH_CLIP = 0.70       # ... and CLIP >= this -> same object
DUP_SAME_DIST_M = 1.0       # centres closer than this ...
DUP_SAME_CLIP = 0.90        # ... and CLIP >= this -> same object
# Cross-class duplicate merge: one object detected under two labels (chair/couch,
# desk/chair). Measured 2026-09-30 by replaying 6 runs: such duplicates have centres
# 0.05-0.18 m apart, point overlap (max of both ways) 0.44-0.99, CLIP 0.86-0.96;
# the closest distinct different-class neighbours (trash can beside a couch or a
# desk, chair at a desk) sit >= 0.30 m apart. The merged node takes the majority
# label; the navigation target's class is never merged away (protected_labels).
# Sparse objects (a mesh trash can: depth returns 14-36 points) never overlap or
# touch reliably; one bin was recorded twice 0.17 m apart on 2026-09-30. For them,
# like synthetic nodes, being close with a similar look is enough (same class only).
SPARSE_POINTS = 50
SPARSE_MERGE_DIST_M = 0.30
SPARSE_MERGE_CLIP = 0.70
NEAR_MISS_LOG_M = 0.50      # log close pairs that were NOT merged (for threshold tuning)
XCLASS_CENTRE_M = 0.25
XCLASS_OVERLAP = 0.40
XCLASS_CLIP = 0.85
MERGE_INTERVAL = 5          # run the object-object merge every N update() calls
OBJ_MIN_DETECTIONS = 2      # objects seen fewer times are not scored / drawn
NEAR_EDGE_FRAC = 0.10       # estimate_box_xy(near_edge=True): closest 10% of the object's points
SYNTH_MIN_ROWS_BELOW_HORIZON = 15   # synthetic points need the box bottom this far below the image centre row
SYNTH_MAX_BOTTOM_FRAC = 0.97        # ... and not touching the image bottom (base cut off)
SYNTH_RADIUS_RANGE_M = (0.10, 0.50) # cylinder radius = half the box width in metres, clipped
SYNTH_HEIGHT_RANGE_M = (0.15, 1.00) # cylinder height from the box height in metres, clipped
SYNTH_STEP_M = 0.05                 # grid spacing of the synthetic points
from palette import CLASS_COLORS_BGR, class_color  # noqa: F401  (re-exported)

EDGE_DIST_M = 2.0                   # MSGNav-style edge: two nodes seen in the SAME frame, centres closer than this
NEAR_RELATION_M = 1.0               # export(): node pairs closer than this get a 'near' relation
SYNTH_TO_REAL_MERGE_M = 3.0         # a synthetic node of a synthetic_labels class (the target) merges into a same-class
                                    # node WITH depth points this close: far ground-contact positions are short by up to
                                    # ~1.6 m (one bin was two nodes 1.7 m apart, 2026-10-02) and 2.4 m (run 160003,
                                    # 2026-10-03; was 2.0 m); the depth node is the truth
SYNTH_MERGE_DIST_M = 0.8            # objects involving synthetic points: same object if centroids closer than this
GROUP_BOX_CONTAIN = 0.8    # box B is "inside" box A if >= this fraction of B's area lies in A ...
GROUP_BOX_AREA_RATIO = 1.5  # ... and A is at least this much larger -> A is a group box, dropped
# A box that holds another, smaller detection of a DIFFERENT class (a stool pushed under a desk)
# has that object's pixels masked out before back-projection: otherwise the nearest-cluster rule
# gives the outer box the inner object's points (2026-10-03 run 135824: desk nodes on stools).
INNER_BOX_CONTAIN = 0.8    # inner box: >= this fraction of its area inside the outer box ...
# Same-class periodic merge: point clouds that nearly coincide are one object whatever CLIP says
# (one couch from two sides: overlap 0.98 but CLIP 0.53, run 135824).
SAME_CLASS_OVERLAP_ONLY = 0.8
# Periodic merges must not grow a node beyond a plausible size for its class (2026-10-03 run 160003:
# a desk merged into a chair, and that blob -- overlap of a small cloud inside a big one is ~1.0 --
# then swallowed the neighbouring chairs: a table with ~10 chairs became ONE 'chair'). Size = 5-95 %
# extent of the merged xy footprint along its main axis. Cross-class: the larger of the two caps.
# Not applied when either node is synthetic (ground-contact cylinders, sized from the box).
MERGE_SIZE_CAP_M = {"chair": 1.0, "trash can": 0.7, "monitor": 1.0, "plant": 1.0, "door": 1.5,
                    "cabinet": 2.5, "whiteboard": 2.5, "desk": 2.5, "table": 2.5, "couch": 2.5}
MERGE_SIZE_CAP_DEFAULT_M = 2.0
# Class pairs never merged across classes: a chair at a desk is the normal case, not one object
# with two labels (user decision 2026-10-03).
NO_XCLASS_MERGE = {frozenset({"chair", "desk"})}


class LightweightSceneGraph:
    """Set of scene-graph nodes, one per distinct 3D object instance."""

    def __init__(self, clip_model, clip_preprocess, device: str = "cuda",
                 min_z: Optional[float] = None, camera_height: Optional[float] = None):
        """min_z: world-frame height below which points are floor and are
        discarded before clustering (pass the ObstacleMap's min_height).
        camera_height: camera height above the floor (level mount); enables the
        synthetic ground-contact points for objects depth cannot see."""
        self._clip = clip_model
        self._min_z = min_z
        self._camera_height = camera_height
        self._preprocess = clip_preprocess
        self._device = device
        self._objects: List[dict] = []
        self._n_updates = 0
        self._next_id = 1
        # MSGNav-style multimodal edges: (a, b) with a < b -> image keys of the frames
        # where both nodes were observed together (the "image edges"); and node -> image
        # keys where it was observed. Image keys name frames the caller saved.
        self._edges: dict = {}
        self._node_imgs: dict = {}
        self._alias: dict = {}          # merged-away id -> surviving id
        # classes never merged into another class (the caller sets the nav target here)
        self.protected_labels: set = set()
        # classes that may get synthetic ground-contact points when depth sees nothing in the
        # box. Meant for the see-through mesh bin (the target); furniture is depth-visible, and
        # without this limit every chair/desk 3-4 m away (beyond MAX_DEPTH_M) became a misplaced
        # synthetic node (19 of 43 nodes in run 20261002_122931). None = all classes.
        self.synthetic_labels: Optional[set] = None
        self._near_miss_logged: dict = {}   # (id, id) -> last logged stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored objects (call only on full episode reset)."""
        self._objects.clear()
        self._n_updates = 0
        self._edges.clear()
        self._node_imgs.clear()
        self._alias.clear()
        self._near_miss_logged.clear()

    def update(
        self,
        color_rgb: np.ndarray,
        detections,
        valid_idxs: list,
        depth_raw: np.ndarray,
        depth_scale: float,
        tf_camera_to_episodic: np.ndarray,
        fx: float,
        fy: float,
        timestamp: Optional[float] = None,
        image_key: Optional[str] = None,
    ) -> List[Tuple[int, str, Tuple[float, float, float, float]]]:
        """Ingest one frame's worth of valid detections.

        Args:
            color_rgb:               (H, W, 3) uint8 RGB image.
            detections:              YOLO-World result with .boxes (normalised
                                     xyxy), .logits, .phrases.
            valid_idxs:              Indices into detections that passed all
                                     prior gates.
            depth_raw:               (H, W) uint16 or float depth array.
            depth_scale:             Metres per raw depth unit.
            tf_camera_to_episodic:   (4,4) camera→world homogeneous transform.
            fx, fy:                  Focal lengths in pixels.
            timestamp:               Observation time (s), kept as first/last seen.
            image_key:               Name of this frame as saved by the caller. When
                                     given, the frame is recorded as evidence for every
                                     node seen in it and for every edge between two of
                                     them closer than EDGE_DIST_M (MSGNav image edges).

        Returns [(node id, class, normalised xyxy box), ...] for the detections
        that landed in the graph this frame (ids after any merge), so the caller
        can draw id marks on the saved frame.
        """
        if not valid_idxs or detections is None:
            return []
        touched = []

        H, W = color_rgb.shape[:2]
        kept = _drop_group_boxes(detections, valid_idxs)
        for i in kept:
            box = detections.boxes[i].numpy() if hasattr(detections.boxes[i], "numpy") else np.asarray(detections.boxes[i])
            x1n, y1n, x2n, y2n = [float(v) for v in box]

            points = self._box_to_world_cloud(
                (x1n, y1n, x2n, y2n), depth_raw, depth_scale, fx, fy, W, H, tf_camera_to_episodic,
                exclude=_inner_boxes(detections, kept, i),
            )
            synthetic = False
            phrase = str(detections.phrases[i]) if hasattr(detections, "phrases") else ""
            if points is None and (self.synthetic_labels is None or phrase in self.synthetic_labels):
                points = self._ground_contact_points(
                    (x1n, y1n, x2n, y2n), fx, fy, W, H, tf_camera_to_episodic
                )
                synthetic = points is not None
            if points is None:
                continue

            clip_ft = self._crop_clip(color_rgb, x1n, y1n, x2n, y2n, W, H)
            if clip_ft is None:
                continue

            nid = self._add_or_merge(points, clip_ft, phrase, synthetic=synthetic,
                                     conf=float(detections.logits[i]), t=timestamp)
            touched.append((nid, phrase, (x1n, y1n, x2n, y2n)))

        self._n_updates += 1
        if self._n_updates % MERGE_INTERVAL == 0:
            self._merge_objects()
        touched = [(self._resolve(nid), lab, box) for nid, lab, box in touched]
        if image_key is not None and touched:
            self._record_frame(image_key, sorted({nid for nid, _, _ in touched}))
        return touched

    # ---- multimodal edges -------------------------------------------------------
    def resolve_id(self, nid: int) -> int:
        """Current id of a node that may have been merged since (id aliases)."""
        return self._resolve(nid)

    def _resolve(self, nid: int) -> int:
        while nid in self._alias:
            nid = self._alias[nid]
        return nid

    def _record_frame(self, key: str, ids: List[int]) -> None:
        pos = {o["id"]: o["world_xy"] for o in self._objects}
        for nid in ids:
            lst = self._node_imgs.setdefault(nid, [])
            if key not in lst:
                lst.append(key)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                if a in pos and b in pos and np.linalg.norm(pos[a] - pos[b]) < EDGE_DIST_M:
                    lst = self._edges.setdefault((a, b), [])
                    if key not in lst:
                        lst.append(key)

    def _remap(self, old: int, new: int) -> None:
        """Node `old` was merged into `new`: move its edges / images."""
        self._alias[old] = new
        imgs = self._node_imgs.pop(old, [])
        dst = self._node_imgs.setdefault(new, [])
        dst.extend(k for k in imgs if k not in dst)
        for (a, b) in [e for e in self._edges if old in e]:
            keys = self._edges.pop((a, b))
            a2, b2 = (new if a == old else a), (new if b == old else b)
            if a2 == b2:
                continue
            e2 = (min(a2, b2), max(a2, b2))
            dst = self._edges.setdefault(e2, [])
            dst.extend(k for k in keys if k not in dst)

    def edges_among(self, ids) -> dict:
        """{(a, b): [image keys]} for edges whose both ends are in ids."""
        ids = set(ids)
        return {e: list(k) for e, k in self._edges.items() if e[0] in ids and e[1] in ids}

    def node_images(self, nid: int) -> List[str]:
        """All saved-frame keys where node `nid` (after merges) was observed, oldest first."""
        return list(self._node_imgs.get(self.resolve_id(nid), []))

    def evidence_images(self, ids) -> Tuple[List[str], dict, dict]:
        """Greedy set cover (MSGNav KSS, extended to nodes): the fewest saved frames
        such that every edge among `ids` AND every node in `ids` appears in at least
        one of them. Returns (image keys in selection order,
        {edge: [selected keys showing it]}, {node id: [selected keys showing it]})."""
        ids = list(ids)
        edges = self.edges_among(ids)
        covers = {}
        for e, keys in edges.items():
            for k in keys:
                covers.setdefault(k, set()).add(("e", e))
        for nid in ids:
            for k in self._node_imgs.get(nid, []):
                covers.setdefault(k, set()).add(("n", nid))
        universe = set().union(*covers.values()) if covers else set()
        chosen = []
        while universe:
            # most newly covered items; ties -> most recent frame (keys sort by tick)
            k = max(covers, key=lambda c: (len(covers[c] & universe), c))
            gain = covers[k] & universe
            if not gain:
                break
            chosen.append(k)
            universe -= gain
        sel = set(chosen)
        edge_imgs = {e: [k for k in keys if k in sel] for e, keys in edges.items()}
        node_imgs = {n: [k for k in self._node_imgs.get(n, []) if k in sel] for n in ids}
        return chosen, edge_imgs, node_imgs

    def estimate_box_xy(
        self,
        box,
        depth_raw: np.ndarray,
        depth_scale: float,
        tf_camera_to_episodic: np.ndarray,
        fx: float,
        fy: float,
        width: int,
        height: int,
        near_edge: bool = False,
    ) -> Optional[np.ndarray]:
        """World (x, y) of the object in one detection box, from the same
        floor-free, foreground DBSCAN cluster the scene graph uses (robust to the
        box centre hitting the background through a chair's gaps). None if the
        box yields no reliable cluster.

        near_edge=False: median of the cluster (object centre).
        near_edge=True:  median of the cluster's points nearest the camera
                         (closest NEAR_EDGE_FRAC of them), i.e. the side of the
                         object facing the robot -- what an approach should stop at
                         (VLFM's ObjectPointCloudMap likewise targets the closest point).
        """
        b = box.numpy() if hasattr(box, "numpy") else np.asarray(box)
        pts = self._box_to_world_cloud(
            tuple(float(v) for v in b), depth_raw, depth_scale, fx, fy, width, height, tf_camera_to_episodic
        )
        if pts is None:
            return None
        if not near_edge:
            return np.median(pts[:, :2], axis=0)
        rng = np.linalg.norm(pts[:, :2] - tf_camera_to_episodic[:2, 3][None, :], axis=1)
        near = pts[rng <= np.percentile(rng, NEAR_EDGE_FRAC * 100)]
        return np.median(near[:, :2], axis=0)

    def _ground_contact_points(
        self,
        box: Tuple[float, float, float, float],
        fx: float,
        fy: float,
        W: int,
        H: int,
        tf_camera_to_episodic: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Synthetic world points for an object the depth camera does not see: a
        vertical cylinder standing where the box's bottom edge meets the floor
        (camera self._camera_height above a level floor), pushed back by its
        radius along the viewing ray (the contact point is the object's front),
        radius / height from the box size at that range. None if disabled or the
        bottom edge is near the horizon / cut off by the image border."""
        if self._camera_height is None:
            return None
        x1n, y1n, x2n, y2n = box
        v = y2n * H
        if y2n >= SYNTH_MAX_BOTTOM_FRAC or v - H / 2 < SYNTH_MIN_ROWS_BELOW_HORIZON:
            return None
        h = self._camera_height
        fwd = h * fy / (v - H / 2)
        u = (x1n + x2n) / 2 * W
        left = -(u - W / 2) * fwd / fx
        radius = float(np.clip((x2n - x1n) * W * fwd / fx / 2, *SYNTH_RADIUS_RANGE_M))
        height = float(np.clip((y2n - y1n) * H * fwd / fy, *SYNTH_HEIGHT_RANGE_M))
        ray = np.array([fwd, left]) / np.hypot(fwd, left)
        cx, cy = np.array([fwd, left]) + radius * ray
        g = np.arange(-radius, radius + 1e-6, SYNTH_STEP_M)
        gx, gy = np.meshgrid(g, g)
        disk = np.stack([gx.ravel(), gy.ravel()], axis=1)
        disk = disk[np.hypot(disk[:, 0], disk[:, 1]) <= radius] + np.array([cx, cy])
        floor_z = -h                                    # camera frame: up is +z
        zs = np.arange(floor_z + 0.12, floor_z + height + 1e-6, 0.10)
        pts_cam = np.array([[x, y, z, 1.0] for z in zs for x, y in disk])
        return (tf_camera_to_episodic @ pts_cam.T).T[:, :3]

    @staticmethod
    def drop_group_boxes(detections, idxs: list) -> list:
        """Public wrapper: drop boxes that contain another same-class box."""
        return _drop_group_boxes(detections, idxs)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a text query to a normalised CLIP feature (1, D)."""
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        tokens = tokenizer([text]).to(self._device)
        with torch.no_grad():
            feat = self._clip.encode_text(tokens)
            feat = F.normalize(feat.float(), dim=-1)
        return feat

    def get_scored_objects(
        self, text_feat: torch.Tensor
    ) -> List[Tuple[float, np.ndarray]]:
        """Cosine similarity of every confirmed object to text_feat.

        Only objects seen at least OBJ_MIN_DETECTIONS times are returned.

        Returns:
            List of (score, world_xy) sorted by score descending.
        """
        results = []
        for obj in self._objects:
            if obj["count"] < OBJ_MIN_DETECTIONS:
                continue
            ft = obj["clip_ft"].to(self._device)  # (1, D)
            score = float((ft @ text_feat.T).squeeze())
            results.append((score, obj["world_xy"].copy()))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def get_scored_labeled(self, text_feat: torch.Tensor) -> List[Tuple[float, np.ndarray, str, int]]:
        """Like get_scored_objects but (score, world_xy, label, id) for display."""
        out = []
        for obj in self._objects:
            if obj["count"] < OBJ_MIN_DETECTIONS:
                continue
            score = float((obj["clip_ft"].to(self._device) @ text_feat.T).squeeze())
            out.append((score, obj["world_xy"].copy(), obj["phrase"], obj["id"]))
        out.sort(key=lambda x: x[0], reverse=True)
        return out

    def best_node_edge(self, label: str, from_xy, min_obs: int) -> Optional[Tuple[np.ndarray, int, int]]:
        """The node of class `label` seen most often (>= min_obs), as
        (near-edge xy seen from from_xy, node id, n_obs), or None. Near edge =
        median of its points closest to from_xy (NEAR_EDGE_FRAC), the same
        notion estimate_box_xy(near_edge=True) uses for a single detection."""
        cands = [o for o in self._objects if o["phrase"] == label and o["count"] >= min_obs]
        if not cands:
            return None
        o = max(cands, key=lambda c: (c["count"], c["last_seen"] or 0.0))
        rng = np.linalg.norm(o["points"][:, :2] - np.asarray(from_xy, dtype=float)[None, :2], axis=1)
        near = o["points"][rng <= np.percentile(rng, NEAR_EDGE_FRAC * 100)]
        return np.median(near[:, :2], axis=0), o["id"], o["count"]

    def id_labels(self) -> List[Tuple[int, str]]:
        """(id, class) of every confirmed node, by id -- the whole scene graph as
        the VLM sees it (text only, to keep the prompt tiny)."""
        return sorted((o["id"], o["phrase"]) for o in self._objects if o["count"] >= OBJ_MIN_DETECTIONS)

    def node_edge(self, node_id: int, from_xy) -> Optional[np.ndarray]:
        """World (x, y) of the near edge (seen from from_xy) of the node with this
        id, or None if it no longer exists (e.g. merged into another node)."""
        for o in self._objects:
            if o["id"] == node_id:
                rng = np.linalg.norm(o["points"][:, :2] - np.asarray(from_xy, dtype=float)[None, :2], axis=1)
                near = o["points"][rng <= np.percentile(rng, NEAR_EDGE_FRAC * 100)]
                return np.median(near[:, :2], axis=0)
        return None

    def nodes(self, text_feat: Optional[torch.Tensor] = None, confirmed_only: bool = True) -> List[dict]:
        """Plain-data view of the graph nodes (for export / VLM prompts)."""
        out = []
        for o in self._objects:
            if confirmed_only and o["count"] < OBJ_MIN_DETECTIONS:
                continue
            ext = np.ptp(o["points"][:, :2], axis=0) if len(o["points"]) > 1 else np.zeros(2)
            n = {
                "id": o["id"],
                "label": o["phrase"],
                "xy": [round(float(o["world_xy"][0]), 3), round(float(o["world_xy"][1]), 3)],
                "extent_xy_m": [round(float(ext[0]), 2), round(float(ext[1]), 2)],
                "n_obs": int(o["count"]),
                "mean_conf": round(o["conf_sum"] / max(1, o["count"]), 3),
                "first_seen": o["first_seen"],
                "last_seen": o["last_seen"],
                "depth_unseen": bool(o["synthetic"]),
            }
            if text_feat is not None:
                n["target_clip_score"] = round(float((o["clip_ft"].to(self._device) @ text_feat.T).squeeze()), 4)
            out.append(n)
        out.sort(key=lambda d: d["id"] if d["id"] is not None else 1 << 30)
        return out

    def export(self, json_path: str, txt_path: Optional[str] = None, robot_xy=None, robot_yaw: Optional[float] = None,
               target: Optional[str] = None, text_feat: Optional[torch.Tensor] = None,
               stamp: Optional[float] = None, near_dist_m: float = NEAR_RELATION_M) -> None:
        """Write the confirmed graph as JSON (and optionally a plain-text summary
        for a VLM prompt). Relations: MSGNav-style 'seen_together' edges (both
        nodes in one saved frame, centres < EDGE_DIST_M) with their image keys;
        every node also lists the frames it was seen in. Robot-relative distance/bearing added when the robot
        pose is given (bearing: +left / -right of the robot's heading, degrees).
        Files are replaced atomically (tmp + rename) so a reader never sees a
        half-written file."""
        nodes = self.nodes(text_feat)
        if robot_xy is not None and robot_yaw is not None:
            for n in nodes:
                dx, dy = n["xy"][0] - float(robot_xy[0]), n["xy"][1] - float(robot_xy[1])
                n["dist_from_robot_m"] = round(float(np.hypot(dx, dy)), 2)
                n["bearing_from_robot_deg"] = round(float(np.rad2deg(
                    (np.arctan2(dy, dx) - robot_yaw + np.pi) % (2 * np.pi) - np.pi)), 0)
        byid = {n["id"]: n for n in nodes}
        rel = []
        for (a, b), keys in sorted(self.edges_among(byid).items()):
            d = float(np.hypot(byid[a]["xy"][0] - byid[b]["xy"][0], byid[a]["xy"][1] - byid[b]["xy"][1]))
            rel.append({"a": a, "b": b, "relation": "seen_together", "dist_m": round(d, 2), "images": keys})
        for n in nodes:
            n["images"] = list(self._node_imgs.get(n["id"], []))
        doc = {
            "stamp": stamp,
            "frame": "SLAM map frame of this run (x, y in metres; resets when the nav stack restarts)",
            "robot": None if robot_xy is None else {
                "xy": [round(float(robot_xy[0]), 3), round(float(robot_xy[1]), 3)],
                "yaw_deg": None if robot_yaw is None else round(float(np.rad2deg(robot_yaw)), 1)},
            "current_target": target,
            "n_nodes": len(nodes),
            "nodes": nodes,
            "relations": rel,
        }
        _atomic_write(json_path, json.dumps(doc, indent=1))
        if txt_path is not None:
            _atomic_write(txt_path, _describe(doc))

    def save_features(self, npz_path: str) -> None:
        """CLIP features + ids/labels/positions of ALL nodes (for later re-use,
        e.g. re-scoring against a new target without re-observing)."""
        if not self._objects:
            return
        np.savez_compressed(
            npz_path,
            ids=np.array([o["id"] for o in self._objects]),
            labels=np.array([o["phrase"] for o in self._objects]),
            xy=np.stack([o["world_xy"] for o in self._objects]),
            n_obs=np.array([o["count"] for o in self._objects]),
            clip=torch.cat([o["clip_ft"].cpu() for o in self._objects]).numpy(),
        )

    def __len__(self) -> int:
        """Number of confirmed objects (>= OBJ_MIN_DETECTIONS sightings)."""
        return sum(1 for o in self._objects if o["count"] >= OBJ_MIN_DETECTIONS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _box_to_world_cloud(
        self,
        box: Tuple[float, float, float, float],
        depth_raw: np.ndarray,
        depth_scale: float,
        fx: float,
        fy: float,
        W: int,
        H: int,
        tf_camera_to_episodic: np.ndarray,
        exclude: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> Optional[np.ndarray]:
        """Back-project the (shrunk) box to a world-frame cloud and keep the
        largest DBSCAN cluster, as VLFM does for its object masks. Pixels inside
        any `exclude` box (normalised xyxy, other objects held in this box) are skipped.

        Returns (N, 3) world points or None if too few reliable points.
        """
        x1n, y1n, x2n, y2n = box
        bw, bh = x2n - x1n, y2n - y1n
        c0 = int((x1n + BOX_SHRINK * bw) * W); c1 = int((x2n - BOX_SHRINK * bw) * W)
        r0 = int((y1n + BOX_SHRINK * bh) * H); r1 = int((y2n - BOX_SHRINK * bh) * H)
        c0, r0 = max(0, c0), max(0, r0)
        c1, r1 = min(W, c1), min(H, r1)
        if c1 - c0 < 2 or r1 - r0 < 2:
            return None

        rows, cols = np.mgrid[r0:r1:PIX_STRIDE, c0:c1:PIX_STRIDE]
        d = depth_raw[rows, cols].astype(np.float32) * depth_scale
        ok = (d > MIN_DEPTH_M) & (d < MAX_DEPTH_M)
        for ex1, ey1, ex2, ey2 in exclude or ():
            ok &= ~((cols >= ex1 * W) & (cols < ex2 * W) & (rows >= ey1 * H) & (rows < ey2 * H))
        if ok.sum() < MIN_CLUSTER_POINTS:
            return None
        d, u, v = d[ok], cols[ok].astype(np.float32), rows[ok].astype(np.float32)

        # Camera convention matches get_point_cloud / estimate_target_xy: (forward, left, up)
        x_cam = (u - W / 2) * d / fx
        y_cam = (v - H / 2) * d / fy
        pts_cam = np.stack([d, -x_cam, -y_cam, np.ones_like(d)], axis=1)
        pts = (tf_camera_to_episodic @ pts_cam.T).T[:, :3]
        if self._min_z is not None:
            above = pts[:, 2] > self._min_z
            pts, d = pts[above], d[above]
            if len(pts) < MIN_CLUSTER_POINTS:
                return None

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        labels = np.asarray(pcd.cluster_dbscan(DBSCAN_EPS_M, DBSCAN_MIN_POINTS))
        keep = labels >= 0
        if not keep.any():
            return None
        lab, cnt = np.unique(labels[keep], return_counts=True)
        if cnt.max() < MIN_CLUSTER_POINTS:
            return None
        # Without masks the box also contains what is behind the object (e.g. the
        # wall seen through a chair's gaps), and that cluster can be the largest.
        # Take the NEAREST sizeable cluster (the foreground object), in the spirit
        # of VLFM's closest-point target estimate.
        big = lab[cnt >= max(MIN_CLUSTER_POINTS, FOREGROUND_MIN_FRAC * cnt.max())]
        best = min(big, key=lambda l: float(np.median(d[labels == l])))
        pts = pts[labels == best]
        return _voxel_down(pts)

    def _crop_clip(
        self,
        color_rgb: np.ndarray,
        x1n: float,
        y1n: float,
        x2n: float,
        y2n: float,
        W: int,
        H: int,
    ) -> Optional[torch.Tensor]:
        """Crop bbox with padding, run through CLIP, return normalised (1,D) tensor."""
        x1 = max(0, int(x1n * W) - CROP_PAD_PX)
        y1 = max(0, int(y1n * H) - CROP_PAD_PX)
        x2 = min(W, int(x2n * W) + CROP_PAD_PX)
        y2 = min(H, int(y2n * H) + CROP_PAD_PX)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = color_rgb[y1:y2, x1:x2]
        pil_img = Image.fromarray(crop)
        img_tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._clip.encode_image(img_tensor)
            feat = F.normalize(feat.float(), dim=-1)
        return feat.cpu()

    def _add_or_merge(self, points: np.ndarray, clip_ft: torch.Tensor, phrase: str,
                      synthetic: bool = False, conf: float = 0.0, t: Optional[float] = None) -> int:
        """Join the best-matching same-class object (overlap + CLIP > SIM_THRESHOLD) or add a new one."""
        centroid = np.median(points[:, :2], axis=0)
        best_idx, best_sim = -1, SIM_THRESHOLD
        for i, obj in enumerate(self._objects):
            if phrase not in obj["votes"]:
                continue
            d = np.linalg.norm(centroid - obj["world_xy"])
            if d > GATE_DIST_M:
                continue
            if synthetic or obj["synthetic"]:
                # Synthetic points only approximate the object's footprint: point
                # overlap is meaningless, same object if the centroids are close.
                spatial = 1.0 if d < SYNTH_MERGE_DIST_M else 0.0
            else:
                spatial = _overlap(points, obj["tree"])
            visual = float((clip_ft @ obj["clip_ft"].T).squeeze())
            sim = spatial + visual
            if sim > best_sim:
                best_idx, best_sim = i, sim

        new = _make_object(points, clip_ft, phrase, count=1, synthetic=synthetic,
                           meta={"id": None, "first_seen": t, "last_seen": t, "conf_sum": conf})
        if best_idx >= 0:
            self._objects[best_idx] = _merge(self._objects[best_idx], new)
            return self._objects[best_idx]["id"]
        else:
            new["id"] = self._next_id
            self._next_id += 1
            self._objects.append(new)
            logging.info(
                f"[SceneGraph] added {phrase!r} #{new['id']}: "
                f"xy=({centroid[0]:.2f},{centroid[1]:.2f}) pts={len(points)}"
                + (" [synthetic: depth does not see it, ground-contact position]" if synthetic else "")
            )
            return new["id"]

    def _merge_objects(self) -> None:
        """Periodic object-object merge. Same class:
          - ConceptGraph merge_objects: high point overlap and high CLIP similarity;
          - fragments of one object: footprints touch (DUP_TOUCH_*);
          - the same object recorded twice: near and near-identical CLIP (DUP_SAME_*);
          - sparse objects (few depth points): near and similar CLIP (SPARSE_*).
        Different classes (one object, two labels): same place, overlapping points and
        near-identical CLIP (XCLASS_*), unless one of the classes is protected."""
        merged = True
        while merged:
            merged = False
            n = len(self._objects)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = self._objects[i], self._objects[j]
                    cross = a["phrase"] != b["phrase"]
                    if cross and (self.protected_labels & {a["phrase"], b["phrase"]}):
                        continue
                    synth_to_real = (not cross and a["synthetic"] != b["synthetic"]
                                     and self.synthetic_labels is not None and a["phrase"] in self.synthetic_labels)
                    gate = NEAR_MISS_LOG_M if cross else (max(GATE_DIST_M, SYNTH_TO_REAL_MERGE_M) if synth_to_real
                                                          else GATE_DIST_M)
                    if np.linalg.norm(a["world_xy"] - b["world_xy"]) > gate:
                        continue
                    if a["synthetic"] or b["synthetic"]:
                        overlap = 1.0 if np.linalg.norm(a["world_xy"] - b["world_xy"]) < SYNTH_MERGE_DIST_M else 0.0
                    else:
                        overlap = max(_overlap(a["points"], b["tree"]), _overlap(b["points"], a["tree"]))
                    visual = float((a["clip_ft"] @ b["clip_ft"].T).squeeze())
                    centre = float(np.linalg.norm(a["world_xy"] - b["world_xy"]))
                    sparse = min(len(a["points"]), len(b["points"])) < SPARSE_POINTS
                    if cross:
                        same = (frozenset({a["phrase"], b["phrase"]}) not in NO_XCLASS_MERGE
                                and centre < XCLASS_CENTRE_M and overlap >= XCLASS_OVERLAP and visual >= XCLASS_CLIP)
                    else:
                        same = (
                            (overlap > MERGE_OVERLAP_THRESH and visual > MERGE_VISUAL_THRESH)
                            or overlap >= SAME_CLASS_OVERLAP_ONLY
                            or (visual >= DUP_SAME_CLIP and centre < DUP_SAME_DIST_M)
                            or (visual >= DUP_TOUCH_CLIP and _footprint_gap(a["points"], b["points"]) <= DUP_TOUCH_GAP_M)
                            or (sparse and visual >= SPARSE_MERGE_CLIP and centre < SPARSE_MERGE_DIST_M)
                            or (synth_to_real and centre < SYNTH_TO_REAL_MERGE_M)
                        )
                    if same and not (a["synthetic"] or b["synthetic"]):
                        cap = max(MERGE_SIZE_CAP_M.get(a["phrase"], MERGE_SIZE_CAP_DEFAULT_M),
                                  MERGE_SIZE_CAP_M.get(b["phrase"], MERGE_SIZE_CAP_DEFAULT_M))
                        size = _footprint_extent(np.vstack([a["points"], b["points"]]))
                        if size > cap:
                            same = False
                            logging.info(f"[SceneGraph] size-capped (not merged) #{a['id']} {a['phrase']!r} + "
                                         f"#{b['id']} {b['phrase']!r}: merged extent {size:.2f} m > {cap:.1f} m "
                                         f"(centre {centre:.2f} m, overlap {overlap:.2f}, CLIP {visual:.2f})")
                    if not same and centre < NEAR_MISS_LOG_M:
                        self._log_near_miss(a, b, cross, centre, overlap, visual)
                    if same:
                        overlap_only = (not cross and overlap >= SAME_CLASS_OVERLAP_ONLY
                                        and not (overlap > MERGE_OVERLAP_THRESH and visual > MERGE_VISUAL_THRESH))
                        if cross or sparse or synth_to_real or overlap_only:
                            kind = ("cross-class" if cross else "synthetic->depth" if synth_to_real
                                    else "sparse" if sparse else "overlap-only")
                            logging.info(f"[SceneGraph] {kind} merge "
                                         f"#{a['id']} {a['phrase']!r} + "
                                         f"#{b['id']} {b['phrase']!r} (centre {centre:.2f} m, overlap "
                                         f"{overlap:.2f}, CLIP {visual:.2f}, pts {len(a['points'])}/"
                                         f"{len(b['points'])})")
                        ids = [x for x in (a["id"], b["id"]) if x is not None]
                        self._objects[i] = _merge(a, b)
                        del self._objects[j]
                        if len(ids) == 2 and ids[0] != ids[1]:
                            self._remap(max(ids), min(ids))
                        merged = True
                        break
                if merged:
                    break

    def _log_near_miss(self, a: dict, b: dict, cross: bool, centre: float, overlap: float, visual: float) -> None:
        """Log a close pair that stayed separate; again only when its numbers change."""
        if a["id"] is None or b["id"] is None:
            return
        key = tuple(sorted((a["id"], b["id"])))
        gap = _footprint_gap(a["points"], b["points"])
        stats = (round(centre, 1), round(overlap, 1), round(visual, 2), round(gap, 1))
        if self._near_miss_logged.get(key) == stats:
            return
        self._near_miss_logged[key] = stats
        logging.info(f"[SceneGraph] near-miss (not merged, {'cross' if cross else 'same'}-class) "
                     f"#{a['id']} {a['phrase']!r} / #{b['id']} {b['phrase']!r}: centre {centre:.2f} m, "
                     f"overlap {overlap:.2f}, CLIP {visual:.2f}, xy gap {gap:.2f} m, "
                     f"pts {len(a['points'])}/{len(b['points'])}, seen {a['count']}/{b['count']}")


def _footprint_extent(points: np.ndarray) -> float:
    """5-95 % extent (m) of the xy footprint along its main axis (outlier-robust object length)."""
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    if len(xy) < 3:
        return 0.0
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    proj = xy @ vt[0]
    lo, hi = np.percentile(proj, [5, 95])
    return float(hi - lo)


def _footprint_gap(pa: np.ndarray, pb: np.ndarray) -> float:
    """Smallest horizontal (xy) distance between two point sets."""
    d, _ = cKDTree(pb[:, :2]).query(pa[:, :2], k=1)
    return float(np.min(d))


def _drop_group_boxes(detections, idxs: list) -> list:
    """Drop boxes that contain another same-class box (a box around several objects)."""
    def box(i):
        b = detections.boxes[i]
        return [float(v) for v in (b.numpy() if hasattr(b, "numpy") else b)]

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    keep = []
    for i in idxs:
        a = box(i)
        is_group = False
        for j in idxs:
            if j == i or detections.phrases[j] != detections.phrases[i]:
                continue
            b = box(j)
            inter = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
            if area(b) > 0 and inter / area(b) >= GROUP_BOX_CONTAIN and area(a) >= GROUP_BOX_AREA_RATIO * area(b):
                is_group = True
                break
        if not is_group:
            keep.append(i)
    return keep


def _inner_boxes(detections, idxs: list, i: int) -> list:
    """Boxes of other-class detections (among idxs) that lie inside box i and are smaller."""
    def box(k):
        b = detections.boxes[k]
        return [float(v) for v in (b.numpy() if hasattr(b, "numpy") else b)]

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    a = box(i)
    out = []
    for j in idxs:
        if j == i or detections.phrases[j] == detections.phrases[i]:
            continue
        b = box(j)
        inter = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
        if 0 < area(b) < area(a) and inter / area(b) >= INNER_BOX_CONTAIN:
            out.append(tuple(b))
    return out


def _voxel_down(points: np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pts = np.asarray(pcd.voxel_down_sample(VOXEL_M).points)
    if len(pts) > MAX_OBJ_POINTS:
        pts = pts[np.random.choice(len(pts), MAX_OBJ_POINTS, replace=False)]
    return pts


def _overlap(points: np.ndarray, tree: cKDTree) -> float:
    """Fraction of `points` with a neighbour in `tree` within OVERLAP_RADIUS_M (ConceptGraph 'overlap')."""
    d, _ = tree.query(points, k=1, distance_upper_bound=OVERLAP_RADIUS_M)
    return float(np.isfinite(d).mean())


def _make_object(points: np.ndarray, clip_ft: torch.Tensor, phrase: str, count: int,
                 synthetic: bool = False, meta: Optional[dict] = None) -> dict:
    meta = meta or {}
    return {
        "points": points,
        "tree": cKDTree(points),
        "world_xy": np.median(points[:, :2], axis=0),
        "clip_ft": clip_ft,
        "phrase": phrase,
        "count": count,
        "votes": dict(meta.get("votes") or {phrase: count}),   # label -> detections under it
        "synthetic": synthetic,   # True while ALL of its points are synthetic
        "id": meta.get("id"),
        "first_seen": meta.get("first_seen"),
        "last_seen": meta.get("last_seen"),
        "conf_sum": meta.get("conf_sum", 0.0),
    }


def _merge(a: dict, b: dict) -> dict:
    """Merge b into a: union of voxelised points, count-weighted mean CLIP feature."""
    n_a, n_b = a["count"], b["count"]
    ft = F.normalize((a["clip_ft"] * n_a + b["clip_ft"] * n_b) / (n_a + n_b), dim=-1)
    if a["synthetic"] != b["synthetic"]:
        # A real depth cluster exists for this object: keep only the real points
        # (the synthetic ones were a stand-in).
        points = a["points"] if not a["synthetic"] else b["points"]
    else:
        points = _voxel_down(np.concatenate([a["points"], b["points"]], axis=0))
    ids = [x for x in (a["id"], b["id"]) if x is not None]
    firsts = [x for x in (a["first_seen"], b["first_seen"]) if x is not None]
    lasts = [x for x in (a["last_seen"], b["last_seen"]) if x is not None]
    meta = {"id": min(ids) if ids else None,
            "first_seen": min(firsts) if firsts else None,
            "last_seen": max(lasts) if lasts else None,
            "conf_sum": a["conf_sum"] + b["conf_sum"],
            "votes": {k: a["votes"].get(k, 0) + b["votes"].get(k, 0) for k in {**a["votes"], **b["votes"]}}}
    # majority label; ties keep a's
    phrase = max(meta["votes"], key=lambda k: (meta["votes"][k], k == a["phrase"]))
    return _make_object(points, ft, phrase, n_a + n_b,
                        synthetic=a["synthetic"] and b["synthetic"], meta=meta)


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _describe(doc: dict) -> str:
    """Plain-text scene description for a VLM prompt."""
    lines = []
    r = doc.get("robot")
    lines.append(f"Scene graph ({doc['n_nodes']} objects) in the robot's map frame (metres)."
                 + (f" Robot at ({r['xy'][0]:.2f}, {r['xy'][1]:.2f}), heading {r['yaw_deg']:.0f} deg." if r else ""))
    if doc.get("current_target"):
        lines.append(f"Current navigation target: {doc['current_target']}.")
    lines.append("Objects:")
    for n in doc["nodes"]:
        where = ""
        if "dist_from_robot_m" in n:
            b = n["bearing_from_robot_deg"]
            a, lr = abs(b), ("left" if b > 0 else "right")
            side = ("ahead" if a < 20 else f"ahead-{lr}" if a < 60 else lr if a < 120
                    else f"behind-{lr}" if a < 160 else "behind")
            where = f", {n['dist_from_robot_m']:.1f} m {side} (bearing {b:+.0f} deg from the robot's heading)"
        extra = "; not visible to the depth camera, position approximate" if n["depth_unseen"] else ""
        score = f", similarity to target {n['target_clip_score']:.2f}" if "target_clip_score" in n else ""
        lines.append(f"- #{n['id']} {n['label']} at ({n['xy'][0]:.2f}, {n['xy'][1]:.2f}){where}; "
                     f"seen {n['n_obs']} times{score}{extra}.")
    if doc["relations"]:
        lines.append("Relations:")
        names = {n["id"]: f"#{n['id']} {n['label']}" for n in doc["nodes"]}
        for e in doc["relations"]:
            imgs = ", ".join(e.get("images", [])[-3:])
            lines.append(f"- {names[e['a']]} and {names[e['b']]} were seen together ({e['dist_m']:.1f} m apart"
                         + (f"; images {imgs}" if imgs else "") + ").")
    return "\n".join(lines) + "\n"
