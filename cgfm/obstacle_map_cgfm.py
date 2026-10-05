"""CGFM-only ObstacleMap with a fix to the explored-area update.

vlfm's ObstacleMap.update_map (VLFM_Project/vlfm/vlfm/mapping/obstacle_map.py, same as
VLFM_GO2_STACK/vlfm_patch/obstacle_map.py) has a 2026-09-17 patch that reveals the
whole camera cone with cv2.ellipse when reveal_fog_of_war returns nothing (view clear
out to max_depth). It passes the cone centre as agent_pixel_location[::-1], i.e.
(row, col), but cv2 wants (x=col, y=row) -- the reversal was copied from the
reveal_fog_of_war call above it, which does take (row, col). The cone is therefore
drawn at the robot's position with row/col swapped, metres away from the robot,
creating phantom explored area and phantom frontiers (seen live 2026-09-28: explored
blobs 8-12 m from any robot pose). The cone's angle is correct.

The original VLFM pipeline keeps using vlfm's class unchanged. Here the obstacle part
is delegated to vlfm as-is (super().update_map(..., explore=False)); only the explore
+ frontier part is reproduced, identical except for the cone centre.
"""

from __future__ import annotations

from typing import Any, Union

import cv2
import numpy as np
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from frontier_exploration.utils.general_utils import wrap_heading

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.utils.geometry_utils import extract_yaw


class CGFMObstacleMap(ObstacleMap):
    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        # Obstacles + navigable map: vlfm's code, unchanged.
        super().update_map(
            depth, tf_camera_to_episodic, min_depth, max_depth, fx, fy, topdown_fov,
            explore=False, update_obstacles=update_obstacles,
        )
        if not explore:
            return

        # --- below: vlfm ObstacleMap.update_map's explore part, cone-centre fix only ---
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]

        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )

        # View clear out to max_depth -> reveal the whole cone (vlfm 2026-09-17 patch).
        if not new_explored_area.any():
            curr_pt_cv2 = agent_pixel_location.astype(int)  # FIX: cv2 takes (x=col, y=row); was [::-1]
            angle_cv2 = np.rad2deg(wrap_heading(extract_yaw(tf_camera_to_episodic) + np.pi / 2))
            cone_mask = cv2.ellipse(
                np.zeros_like(self._map, dtype=np.uint8),
                tuple(int(v) for v in curr_pt_cv2),
                (int(max_depth * self.pixels_per_meter), int(max_depth * self.pixels_per_meter)),
                0,
                angle_cv2 - np.rad2deg(topdown_fov) / 2,
                angle_cv2 + np.rad2deg(topdown_fov) / 2,
                1,
                -1,
            )
            new_explored_area = cv2.bitwise_and(cone_mask, self._navigable_map.astype(np.uint8))

        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)

        # Fragment pruning + anti-collapse guard (vlfm 2026-09-17 patch), unchanged.
        prev_explored_area = self.explored_area.copy()
        prev_count = int(prev_explored_area.sum())

        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            areas = [cv2.contourArea(c) for c in contours]
            max_area = max(areas)
            keep_idxs = [i for i, a in enumerate(areas) if a >= max(20.0, 0.05 * max_area)]
            if not keep_idxs:
                keep_idxs = [int(np.argmax(areas))]
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            for idx in keep_idxs:
                cv2.drawContours(new_area, contours, idx, 1, -1)  # type: ignore
            self.explored_area = new_area.astype(bool)

        new_count = int(self.explored_area.sum())
        if prev_count > 200 and new_count < 0.5 * prev_count:
            self.explored_area = prev_explored_area
            self.explored_area[new_explored_area > 0] = 1

        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)
