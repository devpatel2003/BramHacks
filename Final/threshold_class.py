# threshold_detector.py
"""
ThresholdDetector: RealSense-based foreground object finder.
- get_target_with_vis(): returns (target_mm or None, vis_bgr, bbox_xyxy or None)
- recapture_background(): recompute background depth median (press 'b' in main)
Coordinates are in RealSense camera frame: X right, Y down, Z forward (mm).
"""

import time
import numpy as np
import cv2
import pyrealsense2 as rs

NEAR_THRESH_M = 0.03
MIN_BLOB_PIXELS = 300
MIN_LINEAR_SIZE_M = 0.02
SAMPLE_STEP = 4
NEIGHBOR_R = 2
PREFER_DEPTH_FMT = rs.format.z16
PREFER_COLOR_FMT = rs.format.bgr8
OPEN_KERNEL = (5, 5)
BLUR_KSIZE = 5


class ThresholdDetector:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, PREFER_DEPTH_FMT, 0)
        self.config.enable_stream(rs.stream.color, 640, 480, PREFER_COLOR_FMT, 0)
        self.profile = self.pipeline.start(self.config)

        self.device = self.profile.get_device()
        self.depth_scale = None
        for s in self.device.query_sensors():
            if s.supports(rs.option.depth_units):
                self.depth_scale = s.get_option(rs.option.depth_units)
            if s.supports(rs.option.frames_queue_size):
                s.set_option(rs.option.frames_queue_size, 1)
        if self.depth_scale is None:
            raise RuntimeError("RealSense: could not get depth_units.")

        self.align = rs.align(rs.stream.color)
        time.sleep(0.3)
        self.bg_z16 = None
        self.recapture_background()

    # ---------- public ----------
    def recapture_background(self, num_frames=30):
        """Recompute background depth median."""
        z_stack = []
        for _ in range(num_frames):
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            if not depth_frame:
                continue
            z16 = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
            z_stack.append(z16)
        self.bg_z16 = (np.median(np.stack(z_stack, axis=0), axis=0).astype(np.uint16)
                       if z_stack else None)

    def get_target_with_vis(self):
        """
        Returns:
          target_mm: (x_mm, y_mm, z_mm) or None
          vis_bgr: color frame with bbox drawn (np.uint8 HxWx3)
          bbox: (x0,y0,x1,y1) or None (ints, pixel coords in vis_bgr)
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame or self.bg_z16 is None:
            return None, None, None

        z16 = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        color = np.asanyarray(color_frame.get_data()).copy()
        H, W = z16.shape
        intr = depth_frame.profile.as_video_stream_profile().intrinsics

        # Foreground mask from background depth median
        valid_bg = self.bg_z16 > 0
        valid_cur = z16 > 0
        diff_m = (self.bg_z16.astype(np.int32) - z16.astype(np.int32)) * float(self.depth_scale)
        mask = (valid_bg & valid_cur & (diff_m > NEAR_THRESH_M)).astype(np.uint8) * 255

        if BLUR_KSIZE > 1:
            mask = cv2.medianBlur(mask, BLUR_KSIZE)
        kernel = np.ones(OPEN_KERNEL, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best = None
        best_area = 0

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < MIN_BLOB_PIXELS:
                continue
            p3, size_m = self._bbox_centroid_and_extent_3d_xyxy(x, y, x + w, y + h, depth_frame, intr, step=SAMPLE_STEP)
            if p3 is None or size_m is None or size_m < MIN_LINEAR_SIZE_M:
                continue
            area = w * h
            if area > best_area:
                best_area = area
                best = p3
                best_bbox = (x, y, x + w, y + h)

        target_mm = None
        if best is not None:
            X, Y, Z = best  # meters
            target_mm = (float(X * 1000.0), float(Y * 1000.0), float(Z * 1000.0))
            # draw bbox and a little crosshair
            x0, y0, x1, y1 = best_bbox
            cv2.rectangle(color, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cv2.drawMarker(color, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
            depth_text = f"{Z*1000:.0f} mm"
            cv2.putText(color, depth_text, (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        return target_mm, color, best_bbox

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    # ---------- helpers ----------
    def _local_median_depth(self, depth_frame, u, v, W, H, r=NEIGHBOR_R):
        vals = []
        for dv in range(-r, r + 1):
            vv = v + dv
            if vv < 0 or vv >= H: continue
            for du in range(-r, r + 1):
                uu = u + du
                if uu < 0 or uu >= W: continue
                d = depth_frame.get_distance(int(uu), int(vv))
                if d > 0:
                    vals.append(d)
        return float(np.median(vals)) if vals else 0.0

    def _bbox_centroid_and_extent_3d_xyxy(self, x0, y0, x1, y1, depth_frame, intr, step=SAMPLE_STEP):
        H = depth_frame.get_height(); W = depth_frame.get_width()
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(W - 1, int(x1)); y1 = min(H - 1, int(y1))
        pts = []
        for v in range(y0, y1, step):
            for u in range(x0, x1, step):
                d = depth_frame.get_distance(u, v)
                if d <= 0.0:
                    d = self._local_median_depth(depth_frame, u, v, W, H, r=NEIGHBOR_R)
                if 0.1 <= d <= 10.0:
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(d))
                    pts.append([X, Y, Z])
        if len(pts) < 8:
            return None, None
        P = np.asarray(pts, dtype=float)
        c = np.median(P, axis=0).astype(np.float32)
        lo = np.percentile(P, 10, axis=0)
        hi = np.percentile(P, 90, axis=0)
        size_m = float(np.max(hi - lo))
        if not np.all(np.isfinite(c)) or not np.isfinite(size_m):
            return None, None
        return c, size_m
