"""Floor-flatness calibration for the G1 head camera (no motion, read-only).

Projects the floor seen by the D435i into the SLAM map frame using exactly
the transform run_vlfm_pipeline_g1.py uses (G1_CAM_* env vars + the live
/state_estimation pose), fits a plane z = a*fwd + b*left + c to it, and
suggests corrections:
  * a != 0: floor appears to tilt along the viewing direction -> the camera
    pitch is off. Floor rising away from the robot (a > 0) means the camera
    actually looks further DOWN than assumed: increase G1_CAM_PITCH_DOWN_DEG
    by ~atan(a).
  * b != 0: same for roll (iterate G1_CAM_ROLL_DEG until b ~ 0).
  * c: floor height at the robot in the map frame -> G1_FLOOR_Z.
Re-run with the suggested values exported until |a|, |b| are small, the
same way the GO2 camera pitch was re-measured (2026-09-17).

Setup: robot standing still on a flat floor with >= 2 m of free floor in
front of the camera; planner container + pose relay running, e.g.
    docker exec -d autonomy_stack_planner bash -c "source /opt/ros/jazzy/setup.bash && \\
        source /workspace/autonomy_stack/install/setup.bash && export ROS_DOMAIN_ID=2 && \\
        export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \\
        exec python3 /workspace/autonomy_stack/src/vlfm_bridge/scripts/pose_udp_relay.py"
Then, from the vlfm_g1 checkout:
    env -u PYTHONPATH PYTHONPATH=$(pwd) ~/anaconda3/envs/vlfm_g1/bin/python \\
        ~/Taowen/G1_STACK/src/vlfm_bridge/scripts/calibrate_camera_floor.py [--frames 10]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_vlfm_pipeline_g1 as pipe  # noqa: E402  (same transform + clients as the real run)

FLOOR_BAND_M = 0.25       # points within this of the initial floor guess are fitted
MIN_RANGE_M, MAX_RANGE_M = 0.5, 3.0


def camera_points(depth_raw, intr):
    """Depth image -> Nx3 points in vlfm's camera frame (x fwd, y left, z up)."""
    h, w = depth_raw.shape
    z = depth_raw.astype(np.float32) * intr["depth_scale"]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (z > MIN_RANGE_M) & (z < MAX_RANGE_M)
    z, u, v = z[valid], u[valid], v[valid]
    x_right = (u - intr["ppx"]) * z / intr["fx"]
    y_down = (v - intr["ppy"]) * z / intr["fy"]
    return np.stack([z, -x_right, -y_down], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    camera = pipe._RealSenseStreamClient(pipe.JETSON_HOST, pipe.CAMERA_PORT)
    pose_client = pipe.PoseUdpClient(pipe.POSE_UDP_PORT)
    t0 = time.time()
    while camera.intrinsics() is None or pose_client.pose() is None:
        if time.time() - t0 > 20.0:
            sys.exit("no camera (TCP 6000 on the Jetson) or pose (pose_udp_relay) within 20 s")
        time.sleep(0.2)
    intr = camera.intrinsics()
    print(f"intrinsics: {intr}")
    print(f"current: mode={pipe.POSE_MODE} pitch_down={pipe.CAM_PITCH_DOWN_DEG} roll={pipe.CAM_ROLL_DEG} "
          f"t=({pipe.CAM_TX},{pipe.CAM_TY},{pipe.CAM_TZ}) floor_z={pipe.FLOOR_Z} cam_height={pipe.CAMERA_HEIGHT_M}")

    fits = []
    for i in range(args.frames):
        time.sleep(0.5)
        pose = pose_client.pose()
        _, depth_raw = camera.latest()
        if pose is None or depth_raw is None:  # stale camera/pose (see CAMERA/POSE_MAX_AGE_S)
            print(f"[frame {i}] no fresh camera/pose data -- skipped")
            continue
        xy, yaw, _ = pose
        tf = pipe.get_camera_transform(pose)
        pts = camera_points(depth_raw, intr)
        world = (tf[:3, :3] @ pts.T).T + tf[:3, 3]
        # heading-aligned coordinates around the robot
        dx, dy = world[:, 0] - xy[0], world[:, 1] - xy[1]
        fwd = np.cos(yaw) * dx + np.sin(yaw) * dy
        left = -np.sin(yaw) * dx + np.cos(yaw) * dy
        z_guess = pipe._FLOOR_REF
        m = np.abs(world[:, 2] - z_guess) < FLOOR_BAND_M
        if m.sum() < 500:
            print(f"[frame {i}] only {m.sum()} points near z={z_guess:+.2f} -- floor guess too far off? "
                  f"z percentiles 5/50/95: {np.percentile(world[:, 2], [5, 50, 95]).round(2)}")
            continue
        A = np.stack([fwd[m], left[m], np.ones(m.sum())], axis=1)
        (a, b, c), *_ = np.linalg.lstsq(A, world[m, 2], rcond=None)
        resid = world[m, 2] - A @ np.array([a, b, c])
        fits.append((a, b, c))
        print(f"[frame {i}] n={m.sum():6d} slope_fwd={a:+.4f} slope_left={b:+.4f} "
              f"floor_z@robot={c:+.3f} rms={np.sqrt(np.mean(resid ** 2)):.3f}")

    camera.close()
    pose_client.close()
    if not fits:
        sys.exit("no usable frames")
    a, b, c = np.median(np.array(fits), axis=0)
    print("\nmedian fit: slope_fwd={:+.4f} slope_left={:+.4f} floor_z={:+.3f}".format(a, b, c))
    print("suggested next iteration:")
    print(f"  export G1_CAM_PITCH_DOWN_DEG={pipe.CAM_PITCH_DOWN_DEG + np.rad2deg(np.arctan(a)):.1f}")
    print(f"  # slope_left {b:+.4f}: adjust G1_CAM_ROLL_DEG (currently {pipe.CAM_ROLL_DEG}) until it is ~0")
    if pipe.POSE_MODE == "full":
        print(f"  export G1_FLOOR_Z={c:.3f}")
    else:
        print(f"  export G1_CAM_HEIGHT={-c:.3f}")
    print("re-run until |slope| < ~0.01 (1 cm per metre), then export G1_CAM_CALIBRATED=1")


if __name__ == "__main__":
    main()
