#!/usr/bin/env python3
# hand_trajectory_d435.py
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from filterpy.kalman import KalmanFilter
import mediapipe as mp

# =========================
# Tunables
# =========================
FPS = 30
IMPACT_RADIUS_M = 0.10          # "lens bubble" around camera origin
PREDICTION_HORIZON_S = 1.0
INIT_HITS_TO_CONFIRM = 2
MAX_MISSES = 10
NEIGHBOR_R = 2                  # pixel radius for depth hole filling
MIN_VALID_SAMPLES = 6           # min valid 3D points per hand centroid
SAMPLE_LANDMARKS = [            # landmarks to form a robust palm/hand centroid
    0, 5, 9, 13, 17, 8          # wrist + MCPs + index tip
]
SHOW_LANDMARKS = True

# Optional base-station AABB (meters, camera frame); set to None to disable
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

align = rs.align(rs.stream.color)

# Filters to stabilize depth
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole = rs.hole_filling_filter()

# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Helpers
# =========================
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

def to_vec3(p):
    if p is None: return None
    a = np.asarray(p, dtype=float)
    if a.ndim == 1 and a.size == 3: return a
    if a.ndim == 2 and 3 in a.shape and a.size == 3: return a.reshape(3,)
    return None

def ttc_and_dmin_to_origin(p, v):
    v2 = float(np.dot(v, v))
    if v2 < 1e-8:
        return np.inf, float(np.linalg.norm(p))
    t_star = -float(np.dot(p, v)) / v2
    if t_star <= 0.0:
        return np.inf, float(np.linalg.norm(p))
    d_min = float(np.linalg.norm(p + t_star * v))
    return t_star, d_min

def point_to_aabb_distance(p, aabb_min, aabb_max):
    d = np.maximum(np.maximum(aabb_min - p, 0.0), p - aabb_max)
    return float(np.linalg.norm(d))

def make_kf(dt=1.0/FPS):
    kf = KalmanFilter(dim_x=6, dim_z=3)
    # State: [x y z vx vy vz]
    kf.F = np.array([[1,0,0,dt,0,0],
                     [0,1,0,0,dt,0],
                     [0,0,1,0,0,dt],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1]], dtype=float)
    kf.H = np.array([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0]], dtype=float)
    # Process/measurement noise (tune if needed)
    q = 2.0
    G = np.array([[0.5*dt**2],[0.5*dt**2],[0.5*dt**2],[dt],[dt],[dt]], dtype=float)
    kf.Q = (q**2) * (G @ G.T)
    r = 0.02  # ~2 cm depth accuracy zone
    kf.R = np.diag([r*r, r*r, r*r])
    kf.P = np.diag([0.05,0.05,0.05, 1.0,1.0,1.0])
    return kf

class Track:
    _next_id = 1
    def __init__(self, p3, dt=1.0/FPS):
        p3 = to_vec3(p3)
        if p3 is None: raise ValueError("Track init got invalid p3")
        self.id = Track._next_id; Track._next_id += 1
        self.kf = make_kf(dt)
        self.kf.x[:3, 0] = p3
        self.kf.x[3:, 0] = 0.0
        self.hits = 1
        self.misses = 0
        self.valid = False

    def predict(self):
        self.kf.predict()

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

def associate(tracks, detections, gate=0.25):
    """Greedy nearest-neighbor in 3D; small gate since it's 1–2 hands."""
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

def hand_centroid_3d(hand_landmarks, depth_frame, intr, W, H):
    """Compute robust 3D centroid from selected landmarks using depth."""
    pts = []
    for idx in SAMPLE_LANDMARKS:
        lm = hand_landmarks.landmark[idx]
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            continue
        u = int(round(lm.x * (W - 1)))
        v = int(round(lm.y * (H - 1)))
        # depth with hole-filling
        d = depth_frame.get_distance(u, v)
        if d <= 0.0:
            d = local_median_depth(depth_frame, u, v, W, H, r=NEIGHBOR_R)
        if 0.1 <= d <= 10.0:
            X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(d))
            pts.append([X, Y, Z])

    if len(pts) < MIN_VALID_SAMPLES:
        return None
    P = np.asarray(pts, dtype=float)  # (N,3)
    c = np.median(P, axis=0).astype(np.float32)
    if not np.all(np.isfinite(c)):
        return None
    return c  # (3,)

# =========================
# Main
# =========================
def main():
    tracks = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    t0 = time.time()
    n = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Filter depth
            df = spatial.process(depth_frame)
            df = temporal.process(df)
            df = hole.process(df)
            depth_frame = df.as_depth_frame()

            color = np.asanyarray(color_frame.get_data())
            H, W, _ = color.shape
            intr = depth_frame.profile.as_video_stream_profile().intrinsics

            # MediaPipe hands
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            results = hands.process(color_rgb)

            # Collect detections (3D hand centroids)
            detections, hand_pixels = [], []  # hand_pixels holds (u,v) to draw
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    p3 = hand_centroid_3d(hand_landmarks, depth_frame, intr, W, H)
                    p3 = to_vec3(p3)
                    if p3 is not None:
                        detections.append(p3)
                        # project for quick 2D drawing
                        if p3[2] > 0.05:
                            u = int(round((p3[0] * intr.fx / p3[2]) + intr.ppx))
                            v = int(round((p3[1] * intr.fy / p3[2]) + intr.ppy))
                            hand_pixels.append((u, v))
                    if SHOW_LANDMARKS:
                        mp_draw.draw_landmarks(
                            color,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

            # Predict all tracks
            for t in tracks:
                t.predict()

            # Associate & update
            matches, um_t, um_d = associate(tracks, detections, gate=0.25)
            for i, j in matches:
                tracks[i].update(detections[j])
            for i in um_t:
                tracks[i].misses += 1
            for j in um_d:
                try:
                    tracks.append(Track(detections[j]))
                except ValueError:
                    pass

            # Prune stale unconfirmed tracks
            tracks = [t for t in tracks if not (t.misses >= MAX_MISSES and not t.valid)]

            # Draw and compute collision metrics
            # Draw detected hand pixels
            for (u, v) in hand_pixels:
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(color, (u, v), 6, (0, 255, 255), -1)

            # For each track: TTC & closest approach
            y0 = 20
            for t in tracks:
                p, v = t.state()
                ttc, dmin = ttc_and_dmin_to_origin(p, v)
                camera_hit = (np.isfinite(ttc) and ttc <= PREDICTION_HORIZON_S and dmin <= IMPACT_RADIUS_M)

                base_hit = False
                if BASE_AABB_MIN is not None and BASE_AABB_MAX is not None:
                    if np.linalg.norm(v) > 1e-3:
                        for alpha in np.linspace(0.0, PREDICTION_HORIZON_S, 10):
                            p_pred = p + v * alpha
                            if point_to_aabb_distance(p_pred, BASE_AABB_MIN, BASE_AABB_MAX) == 0.0:
                                base_hit = True
                                break

                # project current state for visualization
                if p[2] > 0.05:
                    u = int(round((p[0] * intr.fx / p[2]) + intr.ppx))
                    vv = int(round((p[1] * intr.fy / p[2]) + intr.ppy))
                    if 0 <= u < W and 0 <= vv < H:
                        col = (0,0,255) if (camera_hit or base_hit) else (0,255,0)
                        cv2.circle(color, (u, vv), 7, col, -1)

                label = f"HandID{t.id} TTC={ttc if np.isfinite(ttc) else float('inf'):.2f}s dmin={dmin*100:.0f}cm"
                cv2.putText(color, label, (10, y0), font, 0.6,
                            (0, 0, 255) if (camera_hit or base_hit) else (50, 220, 50), 2)
                y0 += 22

            # FPS display
            n += 1
            if n % 10 == 0:
                dt = time.time() - t0
                fps = n / max(dt, 1e-6)
                cv2.putText(color, f"FPS: {fps:.1f}", (10, H-12), font, 0.6, (255,255,255), 2)

            cv2.imshow("Hand trajectory + TTC (RealSense + MediaPipe)", color)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        hands.close()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
