#!/usr/bin/env python3
"""Runs on the Jetson: captures RealSense RGBD frames and streams them
over a plain TCP socket to any number of simultaneously connected clients.

v3: supports multiple concurrent clients (a dedicated capture thread reads
from the camera and stores only the latest encoded frame; each client
connection gets its own sender thread that pushes out whatever the latest
frame is). Previous versions only supported one client at a time, which
conflicted with e.g. recording video while also running detection scripts.

Each client still first receives the same one-time intrinsics preamble
(fx, fy, ppx, ppy, width, height, depth_scale), then a stream of
[8-byte header (color_len, depth_len)] + color_bytes + depth_bytes per frame.
"""
import socket
import struct
import threading

import cv2
import numpy as np
import pyrealsense2 as rs

HOST = "0.0.0.0"
PORT = 6000
WIDTH, HEIGHT, FPS = 640, 480, 30

INTRINSICS_FMT = ">4fII f"

frame_cv = threading.Condition()
latest_seq = 0
latest_color_bytes = None
latest_depth_bytes = None
intrinsics_bytes = None


def capture_loop(pipeline):
    global latest_color_bytes, latest_depth_bytes, latest_seq
    # G1 D435i: color (fx~607) and depth (fx~388) have different FOVs, so depth is
    # reprojected into the color frame; detections on color then index depth 1:1.
    align = rs.align(rs.stream.color)
    while True:
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        ok1, color_enc = cv2.imencode(".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        ok2, depth_enc = cv2.imencode(".png", depth_image)
        if not (ok1 and ok2):
            continue

        with frame_cv:
            latest_color_bytes = color_enc.tobytes()
            latest_depth_bytes = depth_enc.tobytes()
            latest_seq += 1
            frame_cv.notify_all()


def client_thread(conn, addr):
    print("Client connected:", addr)
    last_seq = 0
    try:
        conn.sendall(intrinsics_bytes)
        while True:
            with frame_cv:
                got = frame_cv.wait_for(lambda: latest_seq != last_seq, timeout=5.0)
                if not got:
                    break  # no new frame in 5s, assume something's wrong; drop client
                color_bytes, depth_bytes = latest_color_bytes, latest_depth_bytes
                last_seq = latest_seq
            header = struct.pack(">II", len(color_bytes), len(depth_bytes))
            conn.sendall(header)
            conn.sendall(color_bytes)
            conn.sendall(depth_bytes)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        print("Client disconnected:", addr)
        conn.close()


def main():
    global intrinsics_bytes

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    profile = pipeline.start(config)
    print(f"RealSense pipeline started ({WIDTH}x{HEIGHT}@{FPS})")

    # depth is aligned to color in capture_loop -> both images use the color intrinsics
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intrinsics_bytes = struct.pack(
        INTRINSICS_FMT, intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height, depth_scale
    )
    print(f"Color (= aligned depth) intrinsics: fx={intr.fx:.2f} fy={intr.fy:.2f} depth_scale={depth_scale}")

    threading.Thread(target=capture_loop, args=(pipeline,), daemon=True).start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(8)
    print(f"Listening on {HOST}:{PORT} (multi-client) ...")

    try:
        while True:
            conn, addr = server.accept()
            threading.Thread(target=client_thread, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        server.close()


if __name__ == "__main__":
    main()
