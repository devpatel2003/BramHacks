# trig_class.py
"""
A thin class wrapper around your pan/tilt solver so we can hold calibration
(pivot offset, servo zero/sign, limits) in one place.
"""

from math import atan2, degrees, sqrt
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class AimerConfig:
    # Gimbal rotation pivot position in CAMERA frame (mm)
    pivot_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Camera axes convention: RealSense deprojection gives Y down, Z forward
    camera_axes_convention: str = "y_down"   # "y_up" or "y_down"
    # Servo mapping
    pan_zero: float = 80.0
    tilt_zero: float = 43.0
    pan_sign: float = 1.0
    tilt_sign: float = -1.0
    pan_limits: Tuple[float, float] = (0.0, 180.0)
    tilt_limits: Tuple[float, float] = (0.0, 180.0)


class TrigAimer:
    def __init__(self, cfg: AimerConfig):
        self.cfg = cfg

    def aim_to_target(self, target_mm: Tuple[float, float, float]) -> Dict[str, float]:
        x, y, z = target_mm
        px, py, pz = self.cfg.pivot_mm

        Rx = x - px
        Ry = y - py
        Rz = z - pz

        dist = (Rx*Rx + Ry*Ry + Rz*Rz) ** 0.5
        horiz = (Rx*Rx + Rz*Rz) ** 0.5 if (Rx != 0 or Rz != 0) else 0.0

        pan_rad = atan2(Rx, Rz)
        if self.cfg.camera_axes_convention == "y_down":
            tilt_rad = atan2(-Ry, horiz)
        else:
            tilt_rad = atan2(Ry, horiz)

        pan_deg = degrees(pan_rad)
        tilt_deg = degrees(tilt_rad)

        servo_pan = self.cfg.pan_zero + self.cfg.pan_sign * pan_deg
        servo_tilt = self.cfg.tilt_zero + self.cfg.tilt_sign * tilt_deg

        # Clamp
        servo_pan = max(min(servo_pan, self.cfg.pan_limits[1]), self.cfg.pan_limits[0])
        servo_tilt = max(min(servo_tilt, self.cfg.tilt_limits[1]), self.cfg.tilt_limits[0])

        return {
            "pan_deg_raw": pan_deg,
            "tilt_deg_raw": tilt_deg,
            "servo_pan_deg": servo_pan,
            "servo_tilt_deg": servo_tilt,
            "dist_mm": dist,
            "Rx": Rx, "Ry": Ry, "Rz": Rz
        }

# 