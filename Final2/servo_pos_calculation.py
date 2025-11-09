# trig_class.py
"""
Pan/tilt aiming utilities.

- AimerConfig stores calibration, signs, and limits.
- TrigAimer.aim_to_target((X_mm, Y_mm, Z_mm)) computes servo angles.

Camera frame convention (matches RealSense deprojection used in your code):
  X: right (+), Y: down (+), Z: forward from camera (+)
"""

from math import atan2, degrees, sqrt
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class AimerConfig:
    # Gimbal rotation pivot position in CAMERA frame (mm)
    pivot_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Axis convention hint (kept for clarity; calculations assume Y down, Z forward)
    camera_axes_convention: str = "y_down"

    # Servo neutral angles (deg) when target is straight ahead
    pan_zero: float = 90.0
    tilt_zero: float = 90.0

    # Sign flips (use -1.0 to invert if your mechanism moves opposite)
    pan_sign: float = 1.0
    tilt_sign: float = 1.0

    # Mechanical limits (deg)
    pan_limits: Tuple[float, float] = (0.0, 180.0)
    tilt_limits: Tuple[float, float] = (0.0, 180.0)


class TrigAimer:
    def __init__(self, cfg: AimerConfig):
        self.cfg = cfg

    def aim_to_target(self, target_mm: Tuple[float, float, float]) -> Dict[str, float]:
        """
        Convert a 3D point (in CAMERA frame, mm) into servo pan/tilt angles (deg).


        """
        X, Y, Z = target_mm
        px, py, pz = self.cfg.pivot_mm

        # Vector from pivot to target (camera frame)
        Rx = X - px
        Ry = Y - py
        Rz = Z - pz

        # Distance for info
        dist = sqrt(Rx * Rx + Ry * Ry + Rz * Rz)

        # --- Geometry ---
        pan_deg = degrees(atan2(Rx, Rz))
        forward = sqrt(Rx * Rx + Rz * Rz)
        tilt_deg = degrees(atan2(Ry, forward))

        # Apply signs and zero offsets to get servo commands
        servo_pan = self.cfg.pan_zero + self.cfg.pan_sign * pan_deg
        servo_tilt = self.cfg.tilt_zero + self.cfg.tilt_sign * tilt_deg

        # Clamp to limits
        lo, hi = self.cfg.pan_limits
        servo_pan = max(min(servo_pan, hi), lo)
        lo, hi = self.cfg.tilt_limits
        servo_tilt = max(min(servo_tilt, hi), lo)

        return {
            "pan_deg_raw": pan_deg,
            "tilt_deg_raw": tilt_deg,
            "servo_pan_deg": servo_pan,
            "servo_tilt_deg": servo_tilt,
            "dist_mm": dist,
            "Rx": Rx, "Ry": Ry, "Rz": Rz,
        }
