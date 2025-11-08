#!/usr/bin/env python3
# appearance_detector_d435.py
import time
import numpy as np
import cv2
import pyrealsense2 as rs

# ----------------------------
# Tunables
# ----------------------------
# Detection logic
NEAR_THRESH_M = 0.03        # object must be at least this much closer than background (meters)
MIN_BLOB_PIXELS = 300       # ignore tiny blobs in the segmentation mask
MIN_LINEAR_SIZE_M = 0.02    # object must have at least 2 cm physical extent (robust max dimension)
SAMPLE_STEP = 4             # stride when sampling depth inside a bbox (smaller = more robust)
NEIGHBOR_R = 2              # local median hole fill radius (in pixels)
MIN_VALID_3D_SAMPLES = 8    # need at least this many valid 3D points to accept centroid

# Depth/Color stream preferences (auto-pick max FPS)
PREFER_DEPTH_FMT = rs.format.z16
PREFER_COLOR_FMT = rs.format.bgr8

# Morphology
OPEN_KERNEL = (5, 5)        # opening to remove specks
BLUR_KSIZE = 5              # median blur kernel

# ----------------------------
# RealSense setup
# ----------------------------
pipeline = rs.pipeline()
config = rs.config()
# Enable generic 640x480 streams; we’ll let device choose max FPS automatically
config.enable_stream(rs.stream.depth, 640, 480, PREFER_DEPTH_FMT, 0)  # 0 => device picks fps
config.enable_stream(rs.stream.color, 640, 480, PREFER_COLOR_FMT, 0)
profile = pipeline.start(config)

device = profile.get_device()
depth_scale = None
for s in device.query_sensors():
    if s.supports(rs.option.depth_units):
        depth_scale = s.get_option(rs.option.depth_units)
        break

if depth_scale is None:
    raise RuntimeError("Could not find depth_units option; ensure this is a depth-capable device.")

print(f"Depth scale: {depth_scale} meters per unit")
# Align depth -> color so depth[u,v] corresponds to color(u,v)
align = rs.align(rs.stream.color)

# Keep low-latency queues
for s in device.query_sensors():
    if s.supports(rs.option.frames_queue_size):
        s.set_option(rs.option.frames_queue_size, 1)

# Optional: quick auto-exposure settle
time.sleep(0.3)

# ----------------------------
# Helpers
# ----------------------------
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

def bbox_centroid_and_extent_3d_xyxy(x0, y0, x1, y1, depth_frame, intr, step=SAMPLE_STEP):
    """Return (centroid(3,), max_linear_size_m) or (None, None). Uses robust percentiles."""
    H = depth_frame.get_height()
    W = depth_frame.get_width()
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(W-1, int(x1)); y1 = min(H-1, int(y1))
    pts = []
    for v in range(y0, y1, step):
        for u in range(x0, x1, step):
            d = depth_frame.get_distance(u, v)
            if d <= 0.0:
                d = local_median_depth(depth_frame, u, v, W, H, r=NEIGHBOR_R)
            if 0.1 <= d <= 10.0:
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(d))
                pts.append([X, Y, Z])
    if len(pts) < MIN_VALID_3D_SAMPLES:
        return None, None
    P = np.asarray(pts, dtype=float)  # (N,3)
    c = np.median(P, axis=0).astype(np.float32)  # robust centroid
    lo = np.percentile(P, 10, axis=0)
    hi = np.percentile(P, 90, axis=0)
    size_m = float(np.max(hi - lo))  # robust max linear dimension
    if not np.all(np.isfinite(c)) or not np.isfinite(size_m):
        return None, None
    return c, size_m

def recapture_background(num_frames=30):
    """Capture a background z16 median (no foreground objects present)."""
    z_stack = []
    for _ in range(num_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        if not depth_frame:
            continue
        z16 = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        z_stack.append(z16)
    if not z_stack:
        return None
    # Median over time for robustness
    bg = np.median(np.stack(z_stack, axis=0), axis=0).astype(np.uint16)
    return bg

# ----------------------------
# Background capture
# ----------------------------
print("Capturing background... (keep the bristol board clear)")
bg_z16 = recapture_background(num_frames=30)
if bg_z16 is None:
    print("Failed to capture background. Exiting.")
    pipeline.stop()
    raise SystemExit(1)
print("Background captured. Press 'b' to recapture at any time.")

# ----------------------------
# Main loop
# ----------------------------
font = cv2.FONT_HERSHEY_SIMPLEX

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        H, W, _ = color.shape
        intr = depth_frame.profile.as_video_stream_profile().intrinsics

        # Current depth as z16
        z16 = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

        # Compute "things closer than background" mask in meters
        # Note: background invalid (0) should be ignored
        valid_bg = bg_z16 > 0
        valid_cur = z16 > 0
        # Positive where bg - cur > threshold (i.e., current is closer)
        diff_m = (bg_z16.astype(np.int32) - z16.astype(np.int32)) * float(depth_scale)
        mask = (valid_bg & valid_cur & (diff_m > NEAR_THRESH_M)).astype(np.uint8) * 255

        # Clean mask
        if BLUR_KSIZE > 1:
            mask = cv2.medianBlur(mask, BLUR_KSIZE)
        kernel = np.ones(OPEN_KERNEL, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours (new objects)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < MIN_BLOB_PIXELS:
                continue

            # 3D centroid + physical extent check
            cx0, cy0, cx1, cy1 = x, y, x + w, y + h
            p3, size_m = bbox_centroid_and_extent_3d_xyxy(cx0, cy0, cx1, cy1, depth_frame, intr, step=SAMPLE_STEP)
            if p3 is None or size_m is None:
                continue
            if size_m < MIN_LINEAR_SIZE_M:
                continue  # too small physically

            detections.append((x, y, w, h, p3))

        # Draw detections
        for (x, y, w, h, p3) in detections:
            X, Y, Z = p3.tolist()
            cv2.rectangle(color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(color, f"X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f} m",
                        (x, max(15, y - 8)), font, 0.55, (0, 255, 0), 2)

        # UI overlays
        cv2.putText(color, "q: quit, b: recapture background", (10, H - 10),
                    font, 0.55, (255, 255, 255), 2)

        cv2.imshow("Appearance Detector (D435)", color)
        cv2.imshow("Mask (closer-than-background)", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('b'):
            print("Recapturing background... clear the board.")
            bg_z16 = recapture_background(num_frames=30)
            if bg_z16 is not None:
                print("Background updated.")
            else:
                print("Background capture failed; keeping old background.")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
