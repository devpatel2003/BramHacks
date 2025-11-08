#!/usr/bin/env python3
# trajectory_collision_d435.py
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from filterpy.kalman import KalmanFilter

# =========================
# Tunables
# =========================
FPS = 30
DEPTH_DIFF_THRESH_MM = 60         # motion threshold on z16 diff
MIN_BLOB_PIXELS = 150             # ignore tiny blobs
CENTROID_SAMPLE_STEP = 5          # stride when sampling inside bbox
IMPACT_RADIUS_M = 0.10            # camera "bubble" radius around origin
PREDICTION_HORIZON_S = 1.0        # collision lookahead
ASSOC_GATE_M = 0.5                # max 3D distance to associate
MAX_MISSES = 5                    # drop track if too many misses before confirmation
INIT_HITS_TO_CONFIRM = 2          # hits needed to confirm a track
MIN_LINEAR_SIZE_M = 0.02  # 2 cm


# Base-station AABB (in camera frame, meters). Set to None to disable.
BASE_AABB_MIN = np.array([-0.15, -0.10, 0.00], dtype=float)
BASE_AABB_MAX = np.array([ 0.15,  0.10, 0.20], dtype=float)

# =========================
# RealSense setup
# =========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
profile = pipeline.start(config)

# Align depth->color
align = rs.align(rs.stream.color)

# Filters (recommended)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole = rs.hole_filling_filter()

# =========================
# Helpers
# =========================
def depth_median_neighborhood(depth_frame, u, v, w, h, r=2):
    """Median depth (m) around (u,v) ignoring zeros; returns 0.0 if none."""
    vals = []
    for dv in range(-r, r + 1):
        vv = v + dv
        if vv < 0 or vv >= h:
            continue
        for du in range(-r, r + 1):
            uu = u + du
            if uu < 0 or uu >= w:
                continue
            d = depth_frame.get_distance(int(uu), int(vv))
            if d > 0:
                vals.append(d)
    return float(np.median(vals)) if vals else 0.0

def bbox_centroid_3d(x, y, w, h, depth_frame, intr, step=CENTROID_SAMPLE_STEP):
    """Return (centroid(3,), max_linear_size_m) or (None, None).
       Size == max extent across X/Y/Z using robust percentiles (10–90%)."""
    points = []
    H = depth_frame.get_height()
    W = depth_frame.get_width()
    for vv in range(y, y + h, step):
        if vv < 0 or vv >= H: 
            continue
        for uu in range(x, x + w, step):
            if uu < 0 or uu >= W:
                continue
            d = depth_frame.get_distance(int(uu), int(vv))
            if d <= 0.0:
                d = depth_median_neighborhood(depth_frame, uu, vv, W, H, r=1)
            if 0.1 <= d <= 10.0:
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(uu), float(vv)], float(d))
                points.append([X, Y, Z])

    if len(points) < 8:
        return None, None

    P = np.asarray(points, dtype=float)          # shape (N,3)
    if P.ndim != 2 or P.shape[1] != 3:
        return None, None

    # Robust centroid and extent
    c = np.median(P, axis=0).astype(np.float32)  # (3,)
    lo = np.percentile(P, 10, axis=0)            # 10th percentile per axis
    hi = np.percentile(P, 90, axis=0)            # 90th percentile per axis
    extents = hi - lo                             # (3,)
    size_m = float(np.max(extents))              # max linear dimension in meters

    if not np.all(np.isfinite(c)) or not np.isfinite(size_m):
        return None, None
    return c, size_m


def to_vec3(p):
    """Normalize to (3,) float or return None if invalid."""
    if p is None:
        return None
    a = np.asarray(p, dtype=float)
    if a.ndim == 1 and a.size == 3:
        return a
    if a.ndim == 2 and 3 in a.shape and a.size == 3:
        return a.reshape(3,)
    return None

def ttc_and_dmin_to_origin(p, v):
    """TTC and closest approach to origin for constant-velocity motion."""
    v2 = float(np.dot(v, v))
    if v2 < 1e-8:
        return np.inf, float(np.linalg.norm(p))
    t_star = -float(np.dot(p, v)) / v2
    if t_star <= 0.0:
        return np.inf, float(np.linalg.norm(p))
    d_min = float(np.linalg.norm(p + t_star * v))
    return t_star, d_min

def point_to_aabb_distance(p, aabb_min, aabb_max):
    """Euclidean distance from point to AABB (0 if inside)."""
    d = np.maximum(np.maximum(aabb_min - p, 0.0), p - aabb_max)
    return float(np.linalg.norm(d))

# =========================
# Kalman tracking
# =========================
def make_kf(dt=1.0/FPS):
    kf = KalmanFilter(dim_x=6, dim_z=3)
    # State: [x y z vx vy vz]^T
    kf.F = np.array([[1,0,0,dt,0,0],
                     [0,1,0,0,dt,0],
                     [0,0,1,0,0,dt],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1]], dtype=float)
    kf.H = np.array([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0]], dtype=float)
    # Process noise
    q = 2.0
    G = np.array([[0.5*dt**2],[0.5*dt**2],[0.5*dt**2],[dt],[dt],[dt]], dtype=float)
    kf.Q = (q**2) * (G @ G.T)
    # Measurement noise (≈2 cm)
    r = 0.02
    kf.R = np.diag([r*r, r*r, r*r])
    # Covariance
    kf.P = np.diag([0.1,0.1,0.1, 1.0,1.0,1.0])
    return kf

class Track:
    _next_id = 1
    def __init__(self, p3, dt=1.0/FPS):
        p3 = to_vec3(p3)
        if p3 is None:
            raise ValueError("Track init got invalid p3")

        self.id = Track._next_id; Track._next_id += 1
        self.kf = make_kf(dt)
        self.kf.x[:3, 0] = p3
        self.kf.x[3:, 0] = 0.0
        self.hits = 1
        self.misses = 0
        self.age = 0
        self.valid = False

    def predict(self):
        self.kf.predict()
        self.age += 1

    def update(self, p3):
        z = to_vec3(p3)
        if z is None:
            self.misses += 1
            return
        self.kf.update(z.reshape(3,1))
        self.hits += 1
        self.misses = 0
        if not self.valid and self.hits >= INIT_HITS_TO_CONFIRM:
            self.valid = True

    def state(self):
        x = self.kf.x.reshape(-1)
        return x[:3].copy(), x[3:].copy()

def associate(tracks, detections, gate=ASSOC_GATE_M):
    """Greedy NN association on 3D Euclidean distance."""
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    T, D = len(tracks), len(detections)
    cost = np.full((T, D), np.inf, dtype=float)
    for i, tr in enumerate(tracks):
        p_pred, _ = tr.state()
        for j, p_meas in enumerate(detections):
            cost[i, j] = float(np.linalg.norm(p_pred - p_meas))
    matches, used_t, used_d = [], set(), set()
    while True:
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        if not np.isfinite(cost[i, j]) or cost[i, j] > gate:
            break
        if i in used_t or j in used_d:
            cost[i, j] = np.inf
            continue
        matches.append((i, j))
        used_t.add(i); used_d.add(j)
        cost[i, :], cost[:, j] = np.inf, np.inf
    unmatched_t = [i for i in range(T) if i not in used_t]
    unmatched_d = [j for j in range(D) if j not in used_d]
    return matches, unmatched_t, unmatched_d

# =========================
# Main loop
# =========================
def main():
    depth_prev_z16 = None
    tracks = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Filters
            df = spatial.process(depth_frame)
            df = temporal.process(df)
            df = hole.process(df)
            depth_frame = df.as_depth_frame()

            color = np.asanyarray(color_frame.get_data())
            H, W, _ = color.shape
            intr = depth_frame.profile.as_video_stream_profile().intrinsics

            # ---- Depth motion mask (on z16)
            z16 = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
            if depth_prev_z16 is None:
                depth_prev_z16 = z16.copy()
                cv2.imshow("Trajectory & Collision", color)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

            diff = cv2.absdiff(z16, depth_prev_z16)
            mask = (diff > DEPTH_DIFF_THRESH_MM).astype(np.uint8) * 255
            mask = cv2.medianBlur(mask, 5)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

            # ---- Contours -> 3D detections
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections, bboxes = [], []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w * h < MIN_BLOB_PIXELS:
                    continue
                p3_raw, size_m = bbox_centroid_3d(x, y, w, h, depth_frame, intr, step=CENTROID_SAMPLE_STEP)
                if p3_raw is None or size_m is None:
                    continue
                if size_m < MIN_LINEAR_SIZE_M:
                    # Too small physically (< 2 cm), ignore as dust/noise
                    continue

                p3 = to_vec3(p3_raw)
                if p3 is None:
                    continue

                detections.append(p3)
                bboxes.append((x, y, w, h))

            # ---- Predict all tracks
            for t in tracks:
                t.predict()

            # ---- Associate & update
            matches, um_t, um_d = associate(tracks, detections)
            for i, j in matches:
                tracks[i].update(detections[j])
            for i in um_t:
                tracks[i].misses += 1
            for j in um_d:
                try:
                    tracks.append(Track(detections[j]))
                except ValueError:
                    pass  # skip bad init (shouldn't happen due to to_vec3)

            # Prune unconfirmed stale
            tracks = [t for t in tracks if not (t.misses >= MAX_MISSES and t.hits < INIT_HITS_TO_CONFIRM)]

            # ---- Draw bboxes
            for (x, y, w, h) in bboxes:
                cv2.rectangle(color, (x, y), (x + w, y + h), (80, 80, 80), 1)

            # ---- Collision checks & overlay
            for t in tracks:
                p, v = t.state()
                # TTC & closest approach to camera origin
                ttc, dmin = ttc_and_dmin_to_origin(p, v)
                camera_hit = (np.isfinite(ttc) and ttc <= PREDICTION_HORIZON_S and dmin <= IMPACT_RADIUS_M)

                # Base AABB check
                base_hit = False
                if BASE_AABB_MIN is not None and BASE_AABB_MAX is not None:
                    if np.linalg.norm(v) > 1e-3:
                        for alpha in np.linspace(0.0, PREDICTION_HORIZON_S, 10):
                            p_pred = p + v * alpha
                            if point_to_aabb_distance(p_pred, BASE_AABB_MIN, BASE_AABB_MAX) == 0.0:
                                base_hit = True
                                break

                # Project current p to pixel for a dot (approx pinhole)
                if p[2] > 0.05:
                    u = int(round((p[0] * intr.fx / p[2]) + intr.ppx))
                    v_img = int(round((p[1] * intr.fy / p[2]) + intr.ppy))
                    if 0 <= u < W and 0 <= v_img < H:
                        color_code = (0, 0, 255) if (camera_hit or base_hit) else (0, 255, 0)
                        cv2.circle(color, (u, v_img), 6, color_code, -1)

                label = f"ID{t.id} TTC={ttc if np.isfinite(ttc) else float('inf'):.2f}s dmin={dmin*100:.0f}cm"
                cv2.putText(color, label, (10, 22 + 18*(t.id % 20)), font, 0.55,
                            (0, 0, 255) if (camera_hit or base_hit) else (50, 220, 50), 2)

            cv2.imshow("Trajectory & Collision", color)
            depth_prev_z16 = z16.copy()
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# =========================
# Entry
# =========================
if __name__ == "__main__":
    main()
