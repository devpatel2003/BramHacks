#!/usr/bin/env python3
# yolo_realsense_any.py
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# ------------------------------
# Settings (simple defaults)
# ------------------------------
FPS = 30
MODEL_WEIGHTS = "yolov8n.pt"   # use 'yolov8s.pt' if you have GPU headroom
CONF_THRES = 0.35              # detection confidence
IOU_THRES = 0.45               # NMS IoU (ultralytics handles this internally)
SAMPLE_STEP = 5                # depth sampling stride inside bbox (higher = faster, lower = more robust)
NEIGHBOR_R = 1                 # neighborhood (in pixels) to fill zeros via local median
MIN_VALID_SAMPLES = 8          # minimum valid depth samples to accept a 3D centroid

# ------------------------------
# RealSense setup
# ------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
profile = pipeline.start(config)

# Align depth to color so depth[u,v] matches color pixel (u,v)
align = rs.align(rs.stream.color)

# Optional filters (help reduce holes/noise)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole = rs.hole_filling_filter()

# ------------------------------
# YOLO model
# ------------------------------
model = YOLO(MODEL_WEIGHTS)
names = model.model.names if hasattr(model.model, "names") else model.names

# ------------------------------
# Helpers
# ------------------------------
def local_median_depth(depth_frame, u, v, W, H, r=NEIGHBOR_R):
    vals = []
    for dv in range(-r, r+1):
        vv = v + dv
        if vv < 0 or vv >= H: continue
        for du in range(-r, r+1):
            uu = u + du
            if uu < 0 or uu >= W: continue
            d = depth_frame.get_distance(int(uu), int(vv))
            if d > 0:
                vals.append(d)
    return float(np.median(vals)) if vals else 0.0

def bbox_centroid_3d_xyxy(x0, y0, x1, y1, depth_frame, intr, step=SAMPLE_STEP):
    """Return (centroid_xyz (3,), Z_median) or (None, None)."""
    H = depth_frame.get_height()
    W = depth_frame.get_width()
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(W-1, int(x1)); y1 = min(H-1, int(y1))

    points = []
    depths = []
    for v in range(y0, y1, step):
        for u in range(x0, x1, step):
            d = depth_frame.get_distance(u, v)
            if d <= 0.0:
                d = local_median_depth(depth_frame, u, v, W, H, r=NEIGHBOR_R)
            if 0.1 <= d <= 10.0:  # valid range in meters
                X, Y, Z = rs.rs2_deproject_pixel_to_point(
                    intr, [float(u), float(v)], float(d)
                )
                points.append([X, Y, Z])
                depths.append(d)

    if len(points) < MIN_VALID_SAMPLES:
        return None, None

    P = np.asarray(points, dtype=float)  # (N,3)
    c = np.median(P, axis=0).astype(np.float32)  # robust centroid
    z_med = float(np.median(depths))
    if not np.all(np.isfinite(c)):
        return None, None
    return c, z_med

# ------------------------------
# Main
# ------------------------------
def main():
    font = cv2.FONT_HERSHEY_SIMPLEX
    t0 = time.time()
    frame_count = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Filters (optional but recommended)
            df = spatial.process(depth_frame)
            df = temporal.process(df)
            df = hole.process(df)
            depth_frame = df.as_depth_frame()

            color = np.asanyarray(color_frame.get_data())
            H, W, _ = color.shape
            intr = depth_frame.profile.as_video_stream_profile().intrinsics

            # YOLO inference on BGR image
            result = model(color, conf=CONF_THRES, verbose=False)[0]

            # Draw detections
            for box, conf, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy()
            ):
                x0, y0, x1, y1 = box
                cname = names[int(cls)]
                # Get 3D centroid from depth within the box
                p3, z_med = bbox_centroid_3d_xyxy(x0, y0, x1, y1, depth_frame, intr, step=SAMPLE_STEP)
                if p3 is None:
                    # If depth failed, just draw 2D box
                    cv2.rectangle(color, (int(x0), int(y0)), (int(x1), int(y1)), (0, 165, 255), 2)
                    cv2.putText(color, f"{cname} {conf:.2f}", (int(x0), int(y0)-6),
                                font, 0.5, (0,165,255), 2)
                    continue

                X, Y, Z = p3.tolist()
                # Box + label with distance and 3D coords
                cv2.rectangle(color, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                label1 = f"{cname} {conf:.2f}  Z~{z_med:.2f} m"
                label2 = f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f} m"
                cv2.putText(color, label1, (int(x0), max(15, int(y0)-18)), font, 0.5, (0,255,0), 2)
                cv2.putText(color, label2, (int(x0), max(15, int(y0)-2)), font, 0.5, (0,255,0), 2)

            # FPS counter (optional)
            frame_count += 1
            if frame_count % 10 == 0:
                dt = time.time() - t0
                fps = frame_count / max(dt, 1e-6)
                cv2.putText(color, f"FPS: {fps:.1f}", (10, 20), font, 0.6, (255,255,255), 2)

            cv2.imshow("YOLO + RealSense (Any Object)", color)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
