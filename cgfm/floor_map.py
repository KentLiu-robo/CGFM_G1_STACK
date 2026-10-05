"""Observed-floor grid: cells where the depth camera has actually SEEN floor.

Why: vlfm's ObstacleMap.explored_area comes from reveal_fog_of_war, which ray-
casts the camera cone against the obstacle contours and stops at the first one.
In a cluttered office (chair rows on both sides of a narrow aisle, robot often
inside the agent-radius padding) that reveals only a thin sliver per view, so
the explored free space where CGFM diffuses semantics stays tiny even along the
path the robot walked. MSGNav diffuses over TSDF voxels that were actually
observed free; this is the 2D equivalent: a cell is "seen free" when floor
points from the depth image land in it, plus the robot's own footprint.

Same grid / indexing as obstacle_map._navigable_map ([row, col] = [px[:,1], px[:,0]]).
"""

from __future__ import annotations

import numpy as np

PIX_STRIDE = 4              # subsample the depth image
FLOOR_BELOW_M = 0.15        # accept floor points down to (min_z - 0.10 - this), i.e. a bit below the nominal floor
FOOTPRINT_RADIUS_M = 0.25   # cells around the robot's own position are free by definition


class ObservedFloorMap:
    def __init__(self, obstacle_map, min_z: float, min_depth: float, max_depth: float):
        """min_z: obstacle height threshold (ObstacleMap min_height); points at or
        below it are floor."""
        self._om = obstacle_map
        self._min_z = min_z
        self._min_depth = min_depth
        self._max_depth = max_depth
        self.mask = np.zeros(obstacle_map._map.shape, dtype=bool)

    def update(self, depth_raw: np.ndarray, depth_scale: float, tf_camera_to_episodic: np.ndarray,
               fx: float, fy: float, robot_xy: np.ndarray) -> None:
        H, W = depth_raw.shape[:2]
        vv, uu = np.mgrid[0:H:PIX_STRIDE, 0:W:PIX_STRIDE]
        d = depth_raw[vv, uu].astype(np.float32) * depth_scale
        ok = (d > self._min_depth) & (d < self._max_depth)
        if ok.any():
            d, u, v = d[ok], uu[ok].astype(np.float32), vv[ok].astype(np.float32)
            # Camera convention matches get_point_cloud: (forward, left, up)
            pts_cam = np.stack([d, -(u - W / 2) * d / fx, -(v - H / 2) * d / fy, np.ones_like(d)], axis=1)
            pts = (tf_camera_to_episodic @ pts_cam.T).T[:, :3]
            floor_z_min = self._min_z - 0.10 - FLOOR_BELOW_M
            floor = (pts[:, 2] <= self._min_z) & (pts[:, 2] >= floor_z_min)
            self._mark(pts[floor, :2])

        # robot footprint
        r = FOOTPRINT_RADIUS_M
        g = np.linspace(-r, r, max(3, int(2 * r * self._om.pixels_per_meter) + 1))
        gx, gy = np.meshgrid(g, g)
        disk = np.stack([gx.ravel(), gy.ravel()], axis=1)
        disk = disk[np.hypot(disk[:, 0], disk[:, 1]) <= r] + np.asarray(robot_xy)[None, :2]
        self._mark(disk)

    def _mark(self, xy: np.ndarray) -> None:
        if len(xy) == 0:
            return
        px = self._om._xy_to_px(xy)
        h, w = self.mask.shape
        inb = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        px = px[inb]
        self.mask[px[:, 1], px[:, 0]] = True
