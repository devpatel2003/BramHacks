# laser_detector.py (robust)
import time, math
import numpy as np
import cv2
import pyrealsense2 as rs
from typing import Optional, Tuple

# ---------- CAMERA CONTROL ----------
SET_EXPOSURE = True      
EXPOSURE_US  = 5000.0    # 3000..12000 
GAIN         = 32.0      # 16..64 
AUTO_WB_OFF  = True

# ---------- DETECTION TUNABLES ----------
# Small laser dot size (in pixels)
MIN_AREA = 3
MAX_AREA = 400

# Red-detection thresholds
R_DOM_MARGIN   = 40      # R must exceed max(G,B) by at least this
R_ABS_MIN      = 180     # absolute red minimum
V_ABS_MIN      = 150     # value (brightness) minimum

# HSV red bands 
LOW1 = (0,   80,  80)
HIGH1= (10,  255, 255)
LOW2 = (170, 80,  80)
HIGH2= (180, 255, 255)

# Morphology
OPEN_K = (3, 3)
DILATE_ITERS = 1

# Depth sampling neighborhood
DEPTH_R = 2
Z_MIN, Z_MAX = 0.05, 10.0

class LaserDotDetector:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)

        # Align depth to color
        self.align = rs.align(rs.stream.color)
        self.depth_scale = None

        dev = self.profile.get_device()
        for s in dev.query_sensors():
            if s.supports(rs.option.depth_units):
                self.depth_scale = s.get_option(rs.option.depth_units)
            # shrink internal queues to reduce latency
            if s.supports(rs.option.frames_queue_size):
                s.set_option(rs.option.frames_queue_size, 1)

            # Configure color sensor exposure/gain/white balance
            if s.is_color_sensor():
                if SET_EXPOSURE and s.supports(rs.option.enable_auto_exposure):
                    try: s.set_option(rs.option.enable_auto_exposure, 0)
                    except Exception: pass
                if SET_EXPOSURE and s.supports(rs.option.exposure):
                    try: s.set_option(rs.option.exposure, float(EXPOSURE_US))
                    except Exception: pass
                if s.supports(rs.option.gain):
                    try: s.set_option(rs.option.gain, float(GAIN))
                    except Exception: pass
                if AUTO_WB_OFF and s.supports(rs.option.enable_auto_white_balance):
                    try: s.set_option(rs.option.enable_auto_white_balance, 0)
                    except Exception: pass

        if self.depth_scale is None:
            raise RuntimeError("RealSense: depth_units not available.")

        time.sleep(0.3)

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    # --- helpers ---
    def _local_median_depth(self, depth_frame, u, v, r=DEPTH_R) -> float:
        H = depth_frame.get_height()
        W = depth_frame.get_width()
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

    def _depth_at(self, depth_frame, cx, cy) -> float:
        d = self._local_median_depth(depth_frame, cx, cy, r=DEPTH_R)
        return d if Z_MIN <= d <= Z_MAX else 0.0

    def get_target_with_vis(self) -> Tuple[Optional[Tuple[float,float,float]], Optional[np.ndarray], Optional[Tuple[int,int,int,int]]]:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None

        color = np.asanyarray(color_frame.get_data()).copy()
        H, W, _ = color.shape

        # 1) Red dominance + brightness thresholding 
        b, g, r = cv2.split(color)
        r_dom = (r.astype(np.int16) - np.maximum(g, b).astype(np.int16))
        red_dom_mask = (r_dom >= R_DOM_MARGIN) & (r >= R_ABS_MIN)

        v = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[...,2]
        bright_mask = (v >= V_ABS_MIN)

        base_mask = (red_dom_mask & bright_mask).astype(np.uint8) * 255

        # 2) HSV red bands as a soft boost (not required)
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array(LOW1), np.array(HIGH1))
        mask2 = cv2.inRange(hsv, np.array(LOW2), np.array(HIGH2))
        hsv_mask = cv2.bitwise_or(mask1, mask2)

        # Combine (logical OR) so we don't lose saturated “white” dots
        mask = cv2.bitwise_or(base_mask, hsv_mask)

        # 3) Morphological cleanup
        kernel = np.ones(OPEN_K, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if DILATE_ITERS > 0:
            mask = cv2.dilate(mask, kernel, iterations=DILATE_ITERS)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox, best_score = None, -1.0
        # score = area * mean(red dominance + brightness) around the blob
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w*h
            if area < MIN_AREA or area > MAX_AREA:
                continue
            roi_rdom = r_dom[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]
            roi_v    = v[max(0,y):min(H,y+h),   max(0,x):min(W,x+w)]
            if roi_rdom.size == 0: 
                continue
            score = float(area * (np.mean(np.maximum(roi_rdom,0)) + 0.25*np.mean(roi_v)))
            if score > best_score:
                best_score = score
                best_bbox = (x, y, x+w, y+h)

        # 4) Fallback: brightest pixel (top percentile), then small blob around it
        if best_bbox is None:
            # pick the brightest pixel in V channel
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(v)
            if maxVal >= V_ABS_MIN:
                cx, cy = maxLoc
                x0 = max(0, cx-2); y0 = max(0, cy-2)
                x1 = min(W-1, cx+2); y1 = min(H-1, cy+2)
                best_bbox = (x0, y0, x1, y1)

        target_mm = None
        vis = color

        if best_bbox is not None:
            x0, y0, x1, y1 = best_bbox
            cx = int((x0 + x1) / 2)
            cy = int((y0 + y1) / 2)

            d = self._depth_at(depth_frame, cx, cy)
            if d > 0:
                intr = depth_frame.profile.as_video_stream_profile().intrinsics
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(cx), float(cy)], float(d))
                target_mm = (float(X*1000.0), float(Y*1000.0), float(Z*1000.0))

                cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 2)
                cv2.drawMarker(vis, (cx,cy), (0,255,0), cv2.MARKER_CROSS, 14, 2)
                cv2.putText(vis, f"{Z*1000:.0f} mm", (x0, max(0,y0-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        return target_mm, vis, best_bbox
