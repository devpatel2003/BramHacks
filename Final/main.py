# main.py
import time
import math
import cv2
import serial

from threshold_class import ThresholdDetector

from trig_class import TrigAimer, AimerConfig

SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200

# HS-430BH step approximation and deadband
SERVO_STEP_DEG = 0.5
DEADBAND_DEG   = 0.25

# HOME behavior
HOME_PAN = 85.0   # or set to your preferred parked angles
HOME_TILT = 43.0
NO_TARGET_TIMEOUT_S = 0.6   # if no target seen for this long, go home

CFG = AimerConfig(
    pivot_mm=(40.0, 77.6, -29.6),  # <- use YOUR calibrated pivot
    camera_axes_convention="y_down",
    pan_zero=HOME_PAN,          # <- use YOUR calibrated zeros
    tilt_zero=HOME_TILT,        # <- use YOUR calibrated zeros
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
    det = ThresholdDetector()
    aimer = TrigAimer(CFG)
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.2)
    time.sleep(2.0)

    last_cmd_pan = None
    last_cmd_tilt = None
    last_seen_ts = time.time()

    print("Running... keys: [b]=recapture bg, [q]=quit")
    try:
        while True:
            target, frame, bbox = det.get_target_with_vis()

            now = time.time()
            if target is not None:
                # Got a target → compute aim and send
                res = aimer.aim_to_target(target)
                pan = quantize(res["servo_pan_deg"], SERVO_STEP_DEG)
                tilt = quantize(res["servo_tilt_deg"], SERVO_STEP_DEG)

                # Deadband vs last command to avoid chatter
                if (last_cmd_pan is None or abs(pan - last_cmd_pan) >= DEADBAND_DEG) or \
                   (last_cmd_tilt is None or abs(tilt - last_cmd_tilt) >= DEADBAND_DEG):
                    send_angles(ser, pan, tilt)
                    last_cmd_pan, last_cmd_tilt = pan, tilt

                last_seen_ts = now

                # On-screen text
                if frame is not None:
                    info = f"Target: x={target[0]:.0f} y={target[1]:.0f} z={target[2]:.0f} mm | pan={pan:.1f} tilt={tilt:.1f}"
                    cv2.putText(frame, info, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 255), 1)
            else:
                # No target → if timeout elapsed, send HOME once
                if now - last_seen_ts > NO_TARGET_TIMEOUT_S:
                    if last_cmd_pan != HOME_PAN or last_cmd_tilt != HOME_TILT:
                        send_angles(ser, HOME_PAN, HOME_TILT)
                        last_cmd_pan, last_cmd_tilt = HOME_PAN, HOME_TILT
                    # keep last_seen_ts old so we don’t spam (but no harm if we do)
                if frame is not None:
                    cv2.putText(frame, "NO TARGET - homing after timeout", (6, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

            # Show preview window
            if frame is not None:
                cv2.imshow("RealSense preview", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('b'):
                    print("Re-capturing background...")
                    det.recapture_background()
                    last_seen_ts = time.time()  # reset timer after BG change

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
