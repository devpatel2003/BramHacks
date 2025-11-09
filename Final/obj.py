#!/usr/bin/env python3
# livestream_realsense.py
from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import os

# ========================
# YOLO model setup
# ========================
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.25

print(f"Loading YOLO model from {MODEL_PATH} ...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# ========================
# RealSense initialization
# ========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

align = rs.align(rs.stream.color)
device = profile.get_device()

# Optional: check depth scale
depth_scale = None
for s in device.query_sensors():
    if s.supports(rs.option.depth_units):
        depth_scale = s.get_option(rs.option.depth_units)
        break
if depth_scale:
    print(f"Depth scale: {depth_scale} m/unit")

print("Starting livestream... Press 'q' to quit.")

# ========================
# Live Loop
# ========================
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())

        # Run YOLO detection
        results = model(color, conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        # Optionally show depth for center point of each box
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if 0 <= cx < 640 and 0 <= cy < 480:
                depth = depth_frame.get_distance(cx, cy)
                label = f"Z={depth:.2f}m"
                cv2.putText(annotated, label, (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Display
        cv2.imshow("YOLO + RealSense Livestream", annotated)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Stopped.")
