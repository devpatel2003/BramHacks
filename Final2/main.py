#!/usr/bin/env python3
"""
Drop-in replacement main that uses YoloDetector (ignores satellites) and
otherwise keeps your serial/homing/TrigAimer behavior.

Keys:
  q = quit
  b = (no-op for YOLO, kept for compatibility)
"""

import time
import cv2
import serial

from servo_pos_calculation import TrigAimer, AimerConfig
from obj_detection import YoloDetector

SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200

# HS-430BH step approximation and deadband
SERVO_STEP_DEG = 0.5
DEADBAND_DEG   = 0.25

# HOME behavior
HOME_PAN = 85.0
HOME_TILT = 43.0
NO_TARGET_TIMEOUT_S = 0.6

CFG = AimerConfig(
    pivot_mm=(40.0, 77.6, -29.6),    # your calibrated pivot
    camera_axes_convention="y_down",
    pan_zero=HOME_PAN,
    tilt_zero=HOME_TILT,
    pan_sign=1.0,
    tilt_sign=-1.0,
    pan_limits=(0, 180),
    tilt_limits=(0, 180),
)

def quantize(angle_deg: float, step_deg: float) -> float:
    step = max(step_deg, 1e-9)
    return round(angle_deg / step) * step

def send_angles(ser, pan, tilt):
    cmd = f"PAN={pan:.2f},TILT={tilt:.2f}\n"
    ser.write(cmd.encode("ascii"))

def main():
    det = YoloDetector()
    aimer = TrigAimer(CFG)
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.2)
    time.sleep(2.0)

    last_cmd_pan = None
    last_cmd_tilt = None
    last_seen_ts = time.time()

    print("Running... keys: [b]=noop, [q]=quit")
    try:
        while True:
            target, frame, bbox = det.get_target_with_vis()

            now = time.time()
            if target is not None:
                res = aimer.aim_to_target(target)
                pan = quantize(res["servo_pan_deg"], SERVO_STEP_DEG)
                tilt = quantize(res["servo_tilt_deg"], SERVO_STEP_DEG)

                if (last_cmd_pan is None or abs(pan - last_cmd_pan) >= DEADBAND_DEG) or \
                   (last_cmd_tilt is None or abs(tilt - last_cmd_tilt) >= DEADBAND_DEG):
                    send_angles(ser, pan, tilt)
                    last_cmd_pan, last_cmd_tilt = pan, tilt

                last_seen_ts = now

                if frame is not None:
                    info = f"Target: x={target[0]:.0f} y={target[1]:.0f} z={target[2]:.0f} mm | pan={pan:.1f} tilt={tilt:.1f}"
                    cv2.putText(frame, info, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 255), 1)
            else:
                # No non-satellite target -> home after timeout
                if now - last_seen_ts > NO_TARGET_TIMEOUT_S:
                    if last_cmd_pan != HOME_PAN or last_cmd_tilt != HOME_TILT:
                        send_angles(ser, HOME_PAN, HOME_TILT)
                        last_cmd_pan, last_cmd_tilt = HOME_PAN, HOME_TILT
                if frame is not None:
                    cv2.putText(frame, "NO TARGET (or only satellites) - homing after timeout",
                                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

            if frame is not None:
                cv2.imshow("YOLO (non-satellite targeting)", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('b'):
                    # compatibility: do nothing, but reset timer to avoid immediate home
                    det.recapture_background()
                    last_seen_ts = time.time()

            # Optional: read echo from Arduino (non-blocking)
            try:
                line = ser.readline().decode("ascii", errors="ignore").strip()
                if line:
                    print("ARD:", line)
            except Exception:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        # Park on exit (optional)
        try:
            send_angles(ser, HOME_PAN, HOME_TILT)
        except Exception:
            pass
        ser.close()
        det.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
