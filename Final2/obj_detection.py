#!/usr/bin/env python3
"""
YoloDetector: RealSense + YOLO object detector that IGNORES anything whose
class name contains 'satell' (e.g., 'satellite', 'satellites'), and returns
the 3D position of the best non-satellite (debris) target.

"""

from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from typing import Optional, Tuple

# -----------------------
# Tunables
# -----------------------
MODEL_PATH = "best.pt"       # Our trained model for satellite detection
CONF_THRESHOLD = 0.25
FORBIDDEN_SUBSTRINGS = ("satell",)   # filter anything whose class name contains these 
CENTER_WINDOW_R = 2          # radius for median depth around bbox center (pixels)
MIN_VALID_DEPTH_M = 0.10     # drop detections with tiny/invalid depth
MAX_VALID_DEPTH_M = 20.0

class YoloDetector:
    def __init__(self,
                 model_path: str = MODEL_PATH,
                 conf_threshold: float = CONF_THRESHOLD):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = float(conf_threshold)

        # Init camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Stream config
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        self.align = rs.align(rs.stream.color)
        self.device = self.profile.get_device()

        # scale & queue size tuning
        self.depth_scale = None
        for s in self.device.query_sensors():
            if s.supports(rs.option.depth_units):
                self.depth_scale = s.get_option(rs.option.depth_units)
            if s.supports(rs.option.frames_queue_size):
                try:
                    s.set_option(rs.option.frames_queue_size, 1)
                except Exception:
                    pass

        time.sleep(0.2)

        # Names map
        self.names = None
        try:
            self.names = self.model.names
        except Exception:
            self.names = None

    def recapture_background(self):
        return

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    # -------------- helpers --------------
    def _median_depth_m(self, depth_frame: rs.depth_frame, cx: int, cy: int,
                        r: int = CENTER_WINDOW_R) -> float:
        H = depth_frame.get_height()
        W = depth_frame.get_width()
        vals = []
        for dv in range(-r, r + 1):
            y = cy + dv
            if y < 0 or y >= H:
                continue
            for du in range(-r, r + 1):
                x = cx + du
                if x < 0 or x >= W:
                    continue
                d = depth_frame.get_distance(int(x), int(y))
                if d > 0:
                    vals.append(d)
        if not vals:
            return 0.0
        return float(np.median(vals))

    def _is_forbidden(self, clsname: str) -> bool:
        cname = (clsname or "").casefold()
        return any(key in cname for key in FORBIDDEN_SUBSTRINGS)

    # -------------- Gets position from depth camera --------------
    def get_target_with_vis(self):
        """
        Returns:
          target_mm: (x_mm, y_mm, z_mm) or None
          vis_bgr: annotated BGR image
          bbox: (x0,y0,x1,y1) of the chosen target or None
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None

        color = np.asanyarray(color_frame.get_data())
        H, W, _ = color.shape

        # Run YOLO on the color frame (like your obj.py)
        results = self.model(color, conf=self.conf_threshold, verbose=False)
        r0 = results[0]
        annotated = r0.plot()  # baseline visualization

        # Class name map
        names = getattr(r0, "names", None) or self.names or {}
        intr = depth_frame.profile.as_video_stream_profile().intrinsics

        # Choose the best non-satellite detection:
        # First by highest confidence, tie-break by largest area
        best = None
        best_bbox = None
        best_score = -1.0
        best_area = -1.0

        for box in r0.boxes:
            # box fields
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
            cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
            clsname = names.get(cls, str(cls))

            # Ignore forbidden classes (e.g., satellites)
            if self._is_forbidden(clsname):
                # make ignored boxes visually distinct
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red
                cv2.putText(annotated, f"IGNORED:{clsname}",
                            (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                continue

            # center point depth 
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_m = self._median_depth_m(depth_frame, cx, cy, r=CENTER_WINDOW_R)

            if not (MIN_VALID_DEPTH_M <= depth_m <= MAX_VALID_DEPTH_M):
                continue

            # Confidence & area ranking
            area = max(1, (x2 - x1) * (y2 - y1))
            score = conf
            # Keep the highest conf; if tie, prefer larger area
            choose = (score > best_score) or (abs(score - best_score) < 1e-6 and area > best_area)
            if choose:
                # 3D deprojection (meters)
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(cx), float(cy)], float(depth_m))
                best = (X, Y, Z)
                best_bbox = (x1, y1, x2, y2)
                best_score = score
                best_area = area

            # annotate non-forbidden
            cv2.putText(annotated, f"{clsname} {conf:.2f}",
                        (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,50), 2)
            cv2.circle(annotated, (cx, cy), 3, (50,255,50), -1)

        if best is None:
            return None, annotated, None

        # Convert meters → mm 
        X, Y, Z = best
        target_mm = (float(X * 1000.0), float(Y * 1000.0), float(Z * 1000.0))

        # Depth label on the chosen target
        x1, y1, x2, y2 = best_bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
        depth_text = f"Z={Z*1000:.0f}mm"
        cv2.putText(annotated, depth_text, (x1, max(0, y1 - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

        return target_mm, annotated, best_bbox
