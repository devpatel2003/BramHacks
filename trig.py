import numpy as np
from math import atan2, degrees, sqrt

def aim_to_target(target, pivot=(0.0,0.0,0.0),
                  camera_axes_convention="y_up",  # "y_up" or "y_down"
                  pan_zero=0.0, tilt_zero=0.0,
                  pan_sign=1.0, tilt_sign=1.0,
                  pan_limits=(-90.0, 90.0), tilt_limits=(-90.0,90.0)):
    """
    Compute pan & tilt servo angles (degrees) to point a 2-DOF gimbal at target.

    Args:
      target: (x,y,z) in mm in camera frame.
      pivot:  (px,py,pz) in mm in same frame = gimbal rotation center.
      camera_axes_convention: "y_up" (default) or "y_down".
      pan_zero, tilt_zero: servo angle (deg) that corresponds to pointing straight ahead (+z).
      pan_sign, tilt_sign: 1 or -1 to flip direction of servo angle mapping.
      pan_limits, tilt_limits: (min_deg, max_deg) to clamp outputs.
    Returns:
      dict with keys: pan_deg, tilt_deg, dist_mm, Rx,Ry,Rz
    """
    x,y,z = target
    px,py,pz = pivot

    Rx = x - px
    Ry = y - py
    Rz = z - pz

    dist = sqrt(Rx*Rx + Ry*Ry + Rz*Rz)
    horiz = sqrt(Rx*Rx + Rz*Rz) if (Rx != 0 or Rz != 0) else 0.0

    pan_rad = atan2(Rx, Rz)   # pan: left/right
    if camera_axes_convention == "y_down":
        tilt_rad = atan2(-Ry, horiz)  # flip sign if +y is down
    else:
        tilt_rad = atan2(Ry, horiz)

    pan_deg  = degrees(pan_rad)
    tilt_deg = degrees(tilt_rad)

    # map to servo frame
    servo_pan  = pan_zero  + pan_sign  * pan_deg
    servo_tilt = tilt_zero + tilt_sign * tilt_deg

    # clamp
    servo_pan  = max(min(servo_pan, pan_limits[1]), pan_limits[0])
    servo_tilt = max(min(servo_tilt, tilt_limits[1]), tilt_limits[0])

    return {
        "pan_deg_raw": pan_deg,
        "tilt_deg_raw": tilt_deg,
        "servo_pan_deg": servo_pan,
        "servo_tilt_deg": servo_tilt,
        "dist_mm": dist,
        "Rx": Rx, "Ry": Ry, "Rz": Rz
    }

# Example usage:
if __name__ == "__main__":
    target = (100.0, -20.0, 1000.0)   # mm (x right, y up, z forward)
    pivot  = (0.0, 90.0, -70.0)         # e.g., gimbal 50 mm above camera center
    result = aim_to_target(target, pivot=pivot, camera_axes_convention="y_up",
                           pan_zero=90.0, tilt_zero=90.0, # example servo midpoints
                           pan_sign=1.0, tilt_sign=-1.0,  # depends on hardware
                           pan_limits=(0,180), tilt_limits=(0,180))
    for k,v in result.items():
        print(f"{k}: {v}")
