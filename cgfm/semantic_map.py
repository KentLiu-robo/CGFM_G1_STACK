"""Semantic map diffusion over obstacle_map's explored free space.

Adapted from MSGNav/src/tsdf_planner.py (origin/CGFM branch):
_run_semantic_map_diffusion + _multi_source_geodesic_weights + _build_grid_graph_8conn.

All TSDF / Habitat references removed. Works directly with:
  obstacle_map._navigable_map  (bool H×W, indexed [mc, mr] = [col, row])
  obstacle_map.explored_area   (bool H×W, same indexing)
  obstacle_map.pixels_per_meter

Coordinate convention (same as frontier_scorer.py):
  _xy_to_px returns [row, col] but maps are indexed [col, row].
  Use _xy_to_map_idx to convert world → map index before accessing arrays.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import scipy.ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _csgraph_dijkstra


def _xy_to_map_idx(obstacle_map, xy_world: np.ndarray) -> np.ndarray:
    """Convert (N,2) world [x,y] → (N,2) map indices [mc, mr]."""
    px = obstacle_map._xy_to_px(np.atleast_2d(xy_world))
    return px[:, [1, 0]]


def _build_grid_graph_8conn(traversable: np.ndarray) -> csr_matrix:
    """Build 8-connected sparse adjacency for scipy Dijkstra.

    Node index = r * bw + c (row-major). Costs: 1.0 cardinal, sqrt(2) diagonal.
    Edges only between pairs where both endpoints are traversable.
    """
    bh, bw = traversable.shape
    N = bh * bw
    if N == 0:
        return csr_matrix((0, 0), dtype=np.float32)

    idx = np.arange(N, dtype=np.int32).reshape(bh, bw)
    _DIRS = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.4142135), (-1, 1, 1.4142135),
        (1, -1, 1.4142135), (1, 1, 1.4142135),
    )
    src_chunks: list = []
    dst_chunks: list = []
    data_chunks: list = []
    for dr, dc, cost in _DIRS:
        r0 = max(0, -dr); r1 = bh - max(0, dr)
        c0 = max(0, -dc); c1 = bw - max(0, dc)
        if r0 >= r1 or c0 >= c1:
            continue
        valid = traversable[r0:r1, c0:c1] & traversable[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        if not valid.any():
            continue
        src_chunks.append(idx[r0:r1, c0:c1][valid])
        dst_chunks.append(idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc][valid])
        data_chunks.append(np.full(int(valid.sum()), cost, dtype=np.float32))

    if not src_chunks:
        return csr_matrix((N, N), dtype=np.float32)
    src = np.concatenate(src_chunks)
    dst = np.concatenate(dst_chunks)
    data = np.concatenate(data_chunks)
    return csr_matrix((data, (src, dst)), shape=(N, N), dtype=np.float32)


def _multi_source_geodesic_weights(
    centers: List[Tuple[int, int]],
    traversable_mask: np.ndarray,
    cutoff_vox: float,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Per-source bbox-local geodesic distances via scipy Dijkstra.

    Returns list of (dist_grid_local, (r_lo, r_hi, c_lo, c_hi)) per center.
    dist_grid_local shape = (r_hi-r_lo, c_hi-c_lo), indexed by (r-r_lo, c-c_lo).
    """
    h, w = traversable_mask.shape
    if np.isfinite(cutoff_vox):
        cutoff_int = int(np.ceil(cutoff_vox))
        limit = float(cutoff_vox)
    else:
        cutoff_int = max(h, w)
        limit = np.inf

    results: list = []
    for r0, c0 in centers:
        r_lo = max(0, r0 - cutoff_int)
        r_hi = min(h, r0 + cutoff_int + 1)
        c_lo = max(0, c0 - cutoff_int)
        c_hi = min(w, c0 + cutoff_int + 1)
        bh, bw = r_hi - r_lo, c_hi - c_lo
        bbox = (r_lo, r_hi, c_lo, c_hi)

        if bh <= 0 or bw <= 0:
            results.append((np.full((max(bh, 0), max(bw, 0)), np.inf, dtype=np.float32), bbox))
            continue

        lr, lc = r0 - r_lo, c0 - c_lo
        if not (0 <= lr < bh and 0 <= lc < bw):
            results.append((np.full((bh, bw), np.inf, dtype=np.float32), bbox))
            continue

        local_t = traversable_mask[r_lo:r_hi, c_lo:c_hi].copy()
        local_t[max(0, lr - 1):min(bh, lr + 2), max(0, lc - 1):min(bw, lc + 2)] = True

        graph = _build_grid_graph_8conn(local_t)
        src_node = lr * bw + lc
        dist_flat = _csgraph_dijkstra(
            graph, directed=True, indices=src_node, limit=limit, return_predecessors=False
        )
        dist_local = dist_flat.reshape(bh, bw).astype(np.float32, copy=False)
        if np.isfinite(limit):
            dist_local = np.where(dist_local >= limit, np.float32(np.inf), dist_local)
        results.append((dist_local, bbox))

    return results


def compute_semantic_map(
    obstacle_map,
    scored_objects: List[Tuple[float, np.ndarray]],
    semantic_score_baseline: float = 0.20,
    semantic_source_cutoff_m: float = 3.0,
    semantic_geodesic_tau_m: float = 1.0,
    semantic_top_k: int = 5,
    seed_snap_radius_m: float = 1.0,
    seen_mask: np.ndarray = None,
) -> np.ndarray:
    """Compute semantic map by diffusing object CLIP scores through explored free space.

    Args:
        obstacle_map:             vlfm_patch ObstacleMap instance.
        scored_objects:           List of (score, world_xy) where score is cosine
                                  similarity to target text feat and world_xy is (2,).
        semantic_score_baseline:  Scores below this contribute nothing (default 0.20).
        semantic_source_cutoff_m: Max geodesic radius in metres (default 3.0).
        semantic_geodesic_tau_m:  Exponential decay scale in metres (default 1.0).
        semantic_top_k:           Keep only top-k sources per cell (default 5).
        seed_snap_radius_m:       An object's own cell is an obstacle (and is
                                  surrounded by the agent-radius padding), so its
                                  diffusion seed is moved to the nearest explored
                                  traversable cell within this radius; objects
                                  farther than this from explored free space do
                                  not contribute yet (default 1.0).
        seen_mask:                Optional extra (H,W) bool mask of cells known to
                                  be observed (e.g. floor_map.ObservedFloorMap.mask),
                                  OR-ed with obstacle_map.explored_area.

    Returns:
        (H, W) float32 array of semantic values, indexed [mc, mr] (same as
        obstacle_map._navigable_map). Returns zeros if no valid objects.
    """
    shape = obstacle_map._navigable_map.shape
    out_value = np.zeros(shape, dtype=np.float32)
    out_confidence = np.zeros(shape, dtype=np.float32)

    if not scored_objects:
        return out_value

    extra_seen = seen_mask
    seen_mask = obstacle_map.explored_area.astype(bool)
    if extra_seen is not None:
        seen_mask = seen_mask | extra_seen
    traversable_mask = (obstacle_map._navigable_map & seen_mask).astype(bool)
    seen_rows, seen_cols = np.where(seen_mask)
    if len(seen_rows) == 0:
        return out_value

    meters_per_cell = 1.0 / obstacle_map.pixels_per_meter
    if semantic_source_cutoff_m > 0:
        cutoff_vox = float(semantic_source_cutoff_m) * obstacle_map.pixels_per_meter
    else:
        cutoff_vox = np.inf
    tau_vox = max(1e-6, float(semantic_geodesic_tau_m) * obstacle_map.pixels_per_meter)

    raw_scores = np.array([s for s, _ in scored_objects], dtype=np.float32)
    scores = np.maximum(raw_scores - float(semantic_score_baseline), 0.0)
    relevant = scores > 0.0

    xys = np.array([xy for _, xy in scored_objects], dtype=np.float32)
    all_idxs = _xy_to_map_idx(obstacle_map, xys)  # (N,2) [mc, mr]

    if not np.any(relevant):
        logging.info("[SemanticMap] all scores below baseline, returning zero map")
        return out_value

    # Snap each seed onto explored free space (MSGNav's 3x3 unblocking is enough in
    # Habitat's TSDF, not with ObstacleMap's inflated obstacles).
    dist_to_free, nearest = scipy.ndimage.distance_transform_edt(~traversable_mask, return_indices=True)
    snap_cells = float(seed_snap_radius_m) * obstacle_map.pixels_per_meter
    h, w = traversable_mask.shape
    centers, kept = [], []
    for i in np.nonzero(relevant)[0]:
        r0, c0 = int(all_idxs[i, 0]), int(all_idxs[i, 1])
        if not (0 <= r0 < h and 0 <= c0 < w) or dist_to_free[r0, c0] > snap_cells:
            continue
        centers.append((int(nearest[0, r0, c0]), int(nearest[1, r0, c0])))
        kept.append(i)
    if not centers:
        logging.info("[SemanticMap] no relevant object within seed_snap_radius of explored free space")
        return out_value
    scores = np.maximum(raw_scores[kept] - float(semantic_score_baseline), 0.0)

    dist_grids = _multi_source_geodesic_weights(centers, traversable_mask, cutoff_vox)

    M = len(scores)
    N = len(seen_rows)
    weights = np.zeros((M, N), dtype=np.float32)

    for obj_idx in range(M):
        dist_grid_local, (r_lo, r_hi, c_lo, c_hi) = dist_grids[obj_idx]
        in_box = (
            (seen_rows >= r_lo) & (seen_rows < r_hi)
            & (seen_cols >= c_lo) & (seen_cols < c_hi)
        )
        if not np.any(in_box):
            continue
        local_rows = seen_rows[in_box] - r_lo
        local_cols = seen_cols[in_box] - c_lo
        target_dists = dist_grid_local[local_rows, local_cols]
        reachable = np.isfinite(target_dists) & (target_dists < cutoff_vox)
        if not np.any(reachable):
            continue
        in_box_idx = np.nonzero(in_box)[0]
        weights[obj_idx, in_box_idx[reachable]] = np.exp(
            -target_dists[reachable] / tau_vox
        ).astype(np.float32)

    if semantic_top_k > 0 and M > semantic_top_k:
        contribution = weights * scores[:, None]
        top_idx = np.argpartition(contribution, -semantic_top_k, axis=0)[-semantic_top_k:]
        top_weights = np.take_along_axis(weights, top_idx, axis=0)
        top_scores = scores[top_idx]
        value = np.sum(top_weights * top_scores, axis=0)
        confidence = np.sum(top_weights, axis=0)
    else:
        value = np.sum(weights * scores[:, None], axis=0)
        confidence = np.sum(weights, axis=0)

    out_value[seen_rows, seen_cols] = value
    out_confidence[seen_rows, seen_cols] = confidence

    logging.info(
        f"[SemanticMap] M={M} seen_cells={N} max_val={float(out_value.max()):.4f}"
    )
    return out_value
