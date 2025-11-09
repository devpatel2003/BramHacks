#!/usr/bin/env python3
"""
Calibrate camera->pivot offset (px,py,pz) in mm and servo zero angles (pan_zero, tilt_zero) in deg.

Workflow:
1) The script moves the gimbal through a small grid of pan/tilt commands around your current “forward”.
2) For each pose it shows the RealSense preview. When your detector finds the target (bbox turns green),
   press SPACE to record that sample (3D point in camera frame + the commanded servo angles).
3) After N samples, the script runs a small coordinate-descent least-squares to fit:
      theta = [px, py, pz, pan_zero, tilt_zero]
   minimizing the sum of squared perpendicular distances from measured points to the rays defined by (pivot, direction(pan,tilt)).
4) Results are written to calibration.json and a suggested AimerConfig is printed.

Assumptions:
- pan_sign and tilt_sign are known and fixed (flip if needed).
- RealSense gives camera-frame XYZ in millimeters (X right, Y down, Z forward) — "y_down".
- Your Arduino listens for "PAN=...,TILT=..." at 115200 baud like your current sketch.

Requires: numpy, opencv-python, pyserial, your existing threshold_detector.py and trig_class.py
"""

import json, time, math, sys
import numpy as np
import cv2
import serial

from laser_detector import LaserDotDetector as ThresholdDetector
#from trig_class import AimerConfig  # for printing a suggestion, not strictly needed

SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200
WINDOW = "Calib view"

# Known (don’t solve in this script). Flip if your axes go the wrong way.
PAN_SIGN  = 1.0
TILT_SIGN = -1.0
AXES_CONV = "y_down"  # RealSense

# Initial guesses (use your current config as a starting point)
INIT_PX, INIT_PY, INIT_PZ = (0.0, 50.0, -60.0)   # mm
INIT_PAN_ZERO, INIT_TILT_ZERO = (98.0, 43.0)   # deg

# Sampling poses (deg) relative to the current belief of zero (small grid around forward)
REL_PANS  = [-20, -10, 0, +10, +20]
REL_TILTS = [-20, -10, 0, +10, +20]

# Slew wait between commands
SETTLE_SEC = 0.25

# ----------------------------------------------------------------------------------

def send_angles(ser, pan, tilt):
    cmd = f"PAN={pan:.2f},TILT={tilt:.2f}\n"
    ser.write(cmd.encode("ascii"))

def dir_from_angles(pan_deg_rel, tilt_deg_rel):
    """
    Convert relative angles (0 = forward) into a unit direction vector in camera frame.
    pan: +right around Y axis (atan2(x,z))
    tilt: up/down. With "y_down", up is negative tilt in math.
    """
    pan = math.radians(pan_deg_rel)
    tilt = math.radians(tilt_deg_rel)
    # Forward along +Z, pan rotates around Y, tilt around X' (assuming small gimbal model)
    # Camera frame: X right, Y down, Z forward
    # Start with forward vector (0,0,1), apply pan then tilt
    # After pan: (sin pan, 0, cos pan)
    xz = np.array([math.sin(pan), math.cos(pan)], dtype=float)
    x, z = xz[0], xz[1]
    # Apply tilt: positive tilt (in math) is +Y up; with y_down, "up" is -Y.
    # Vector before tilt: (x, 0, z). Tilt around +X:
    # y' = -sin(tilt)*z  (negative for y_down convention)
    # z' =  cos(tilt)*z
    y = -math.sin(tilt) * z
    z =  math.cos(tilt) * z
    v = np.array([x, y, z], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def point_line_distance_sq(P, R0, v):
    """
    Squared distance from point P to the infinite line L(t)=R0 + t*v (||v||=1).
    dist^2 = ||(P - R0) - ((P - R0)·v) v||^2
    """
    w = P - R0
    t = np.dot(w, v)
    perp = w - t * v
    return float(np.dot(perp, perp))

def objective(theta, samples):
    """
    theta = [px,py,pz, pan_zero, tilt_zero]
    samples = list of dicts with keys:
      - P (np.array([x,y,z]) in mm)
      - pan_cmd (deg), tilt_cmd (deg)
    Returns mean squared distance (mm^2)
    """
    px, py, pz, pan0, tilt0 = theta
    R0 = np.array([px, py, pz], dtype=float)
    s = 0.0
    for smp in samples:
        pan_rel  = PAN_SIGN  * (smp["pan_cmd"]  - pan0)
        tilt_rel = TILT_SIGN * (smp["tilt_cmd"] - tilt0)
        v = dir_from_angles(pan_rel, tilt_rel)
        s += point_line_distance_sq(smp["P"], R0, v)
    return s / max(1, len(samples))

def coord_descent(theta0, samples, steps, iters=12, shrink=0.5):
    """
    Very simple coordinate descent with step shrinking.
    steps = initial step sizes for each parameter.
    """
    th = np.array(theta0, dtype=float)
    st = np.array(steps, dtype=float)
    best = objective(th, samples)

    for k in range(iters):
        improved = False
        for i in range(len(th)):
            for d in (+1.0, -1.0):
                trial = th.copy()
                trial[i] += d * st[i]
                val = objective(trial, samples)
                if val < best:
                    best = val
                    th = trial
                    improved = True
        if not improved:
            st *= shrink
    return th, best

def main():
    # Connect
    det = ThresholdDetector()
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.2)
    time.sleep(2.0)

    # Build absolute command grid around current guesses
    base_pan  = INIT_PAN_ZERO
    base_tilt = INIT_TILT_ZERO
    poses = [(base_pan + dp, base_tilt + dt) for dp in REL_PANS for dt in REL_TILTS]

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("Calibration:")
    print(" - The script will cycle through a set of pan/tilt commands.")
    print(" - When the detector shows the bbox on your target, press SPACE to record.")
    print(" - Press 's' to skip a pose, 'q' to quit early.")
    print()

    samples = []

    try:
        for (pan_cmd, tilt_cmd) in poses:
            print(f"Pose: PAN={pan_cmd:.1f}, TILT={tilt_cmd:.1f}  -> sending")
            send_angles(ser, pan_cmd, tilt_cmd)
            time.sleep(SETTLE_SEC)

            recorded = False
            while True:
                target, frame, bbox = det.get_target_with_vis()
                if frame is None:
                    frame = np.zeros((240,320,3), np.uint8)
                    cv2.putText(frame, "No frame", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200),2)

                txt = f"Pose pan={pan_cmd:.1f} tilt={tilt_cmd:.1f} | SPACE=record  s=skip  q=finish"
                cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60,220,255), 1)
                if target is None:
                    cv2.putText(frame, "NO TARGET", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                else:
                    cv2.putText(frame, f"Target mm: {target[0]:.0f},{target[1]:.0f},{target[2]:.0f}", (8, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.imshow(WINDOW, frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(' '):  # space = record
                    if target is not None:
                        samples.append({
                            "P": np.array(target, dtype=float),
                            "pan_cmd": float(pan_cmd),
                            "tilt_cmd": float(tilt_cmd),
                        })
                        print(f"  recorded: P={target}")
                        recorded = True
                        break
                    else:
                        print("  no target to record.")
                elif k == ord('s'):
                    print("  skipped.")
                    break
                elif k == ord('q'):
                    print("  finishing early.")
                    recorded = True
                    break

            if not recorded and k == ord('q'):
                break

        if len(samples) < 6:
            print(f"Not enough samples ({len(samples)}). Take at least ~9–15.")
            return

        # Optimize theta = [px,py,pz, pan_zero, tilt_zero]
        theta0 = np.array([INIT_PX, INIT_PY, INIT_PZ, INIT_PAN_ZERO, INIT_TILT_ZERO], dtype=float)
        steps0 = np.array([30.0, 30.0, 30.0, 5.0, 5.0], dtype=float)  # mm, mm, mm, deg, deg

        print("\nOptimizing (this is quick)...")
        theta, loss = coord_descent(theta0, samples, steps0, iters=18, shrink=0.6)
        px, py, pz, pan0, tilt0 = theta.tolist()

        print("\n=== Calibration Result ===")
        print(f"pivot_mm  ≈ ({px:.1f}, {py:.1f}, {pz:.1f})  [mm]")
        print(f"pan_zero  ≈ {pan0:.2f}  [deg]")
        print(f"tilt_zero ≈ {tilt0:.2f}  [deg]")
        print(f"mean squared distance to rays: {loss:.2f} mm^2 (RMS ≈ {math.sqrt(loss):.2f} mm)")

        data = {
            "pivot_mm": [px, py, pz],
            "pan_zero": pan0,
            "tilt_zero": tilt0,
            "pan_sign": PAN_SIGN,
            "tilt_sign": TILT_SIGN,
            "axes": AXES_CONV,
            "samples": [
                {"P": smp["P"].tolist(), "pan_cmd": smp["pan_cmd"], "tilt_cmd": smp["tilt_cmd"]}
                for smp in samples
            ],
        }
        with open("calibration.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Saved -> calibration.json")

        print("\nPaste this into your AimerConfig:")
        print(f"""CFG = AimerConfig(
    pivot_mm=({px:.1f}, {py:.1f}, {pz:.1f}),
    camera_axes_convention="{AXES_CONV}",
    pan_zero={pan0:.2f},
    tilt_zero={tilt0:.2f},
    pan_sign={PAN_SIGN},
    tilt_sign={TILT_SIGN},
    pan_limits=(0,180),
    tilt_limits=(0,180),
)""")

    finally:
        try:
            ser.write(b"PAN=90,TILT=90\n")
        except Exception:
            pass
        ser.close()
        det.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
