"""
PyBullet Pan/Tilt Laser Aiming Simulator
----------------------------------------
- Minimal single-file simulation of a 2-DOF gimbal (pan/tilt) that aims a laser at a 3D target.
- Uses the inverse-kinematics math we discussed.
- Simulates HS-430BH servo resolution via angle quantization.
- Shows the ideal (raw) and quantized beam, the target, and the hit point on a wall at 1 m.

Controls
--------
- SPACE: toggle pause
- R: randomize target near the 1 m wall
- ARROWS/PageUp/PageDown: nudge target (±5 mm in X/Z, ±10 mm in Y)
- S: cycle servo step (0.8°, 0.5°, 0.1°)
- G: cycle gear ratio (1x, 3x, 5x)
- ESC/Q: quit

World/Frame Conventions
-----------------------
PyBullet uses X-right, Y-forward, Z-up (meters). We point “forward” along +Y toward the wall at Y=1.0 m.
The pan joint yaws about Z. The tilt joint pitches about +X. The laser “points” along +Y after rotations.

Requirements: pybullet, numpy (optional but used for convenience)
Run: `python pybullet_gimbal_laser_sim.py`
"""

import math
import random
from typing import Tuple

import pybullet as p
import pybullet_data
import numpy as np

# =====================
# Tunable parameters
# =====================
WALL_Y_M = 1.0  # meters (the target plane is Y=1m)

# Initial target (meters)
TARGET = np.array([0.05, 1.0, 0.03], dtype=float)  # 5 cm right, 1 m forward, 3 cm up

# Servo behavior
SERVO_STEP_DEG_OPTIONS = [0.8, 0.5, 0.1]  # emulate HS-430BH ~0.5–0.8°, plus a finer option
GEAR_OPTIONS = [1.0, 3.0, 5.0]

# Position control stiffness/damping
KP = 1.0
KD = 0.2

# Visuals
BEAM_LEN_M = 2.0
DT = 1.0 / 240.0


# =====================
# Math helpers
# =====================
def aim_angles_from_target_world(target_xyz: np.ndarray, pivot_xyz: np.ndarray) -> Tuple[float, float]:
    """Compute raw pan/tilt (deg) for world frame where forward is +Y, up is +Z.
    pan: yaw about +Z; tilt: pitch about +X.
    """
    Rx, Ry, Rz = (target_xyz - pivot_xyz)
    horiz = math.hypot(Rx, Ry)  # sqrt(x^2 + y^2)
    pan = math.degrees(math.atan2(Rx, Ry))  # yaw around Z to face target in X-Y plane
    tilt = math.degrees(math.atan2(Rz, horiz))  # pitch up/down
    return pan, tilt


def quantize_angle(angle_deg: float, step_deg: float, gear_ratio: float) -> float:
    step = max(step_deg / max(gear_ratio, 1.0), 1e-9)
    return round(angle_deg / step) * step


def rot_yaw_pitch_to_dir(pan_deg: float, tilt_deg: float) -> np.ndarray:
    """Return a unit direction vector for forward=+Y using yaw(Z)=pan, pitch(X)=tilt.
    dir = R_x(tilt) * R_z(pan) * [0,1,0]
    """
    pan = math.radians(pan_deg)
    tilt = math.radians(tilt_deg)

    cz, sz = math.cos(pan), math.sin(pan)
    cx, sx = math.cos(tilt), math.sin(tilt)

    # Rz(pan)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    # Rx(tilt)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)

    v = np.array([0.0, 1.0, 0.0], dtype=float)
    d = Rx @ (Rz @ v)
    return d / (np.linalg.norm(d) + 1e-12)


def ray_plane_hit(origin, direction, plane_y=WALL_Y_M):
    """Intersection of a ray (origin + t*dir, t>=0) with plane Y=plane_y.
    Returns: point (np.array) or None if parallel or behind.
    """
    if abs(direction[1]) < 1e-9:
        return None
    t = (plane_y - origin[1]) / direction[1]
    if t < 0:
        return None
    return origin + direction * t


# =====================
# PyBullet model creation
# =====================

def create_gimbal(pivot_xyz=(0, 0, 0)):
    """Create a simple 2-DOF gimbal with two revolute joints and a laser link.
    J0: pan (yaw) about +Z
    J1: tilt (pitch) about +X
    Returns body unique ID.
    """
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Base (fixed) visual
    base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.01], rgbaColor=[0.4, 0.4, 0.4, 1])
    base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.01])

    # Pan frame (link 0)
    pan_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.02, rgbaColor=[0.2, 0.6, 1.0, 1],
                                  visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]))
    pan_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.015, height=0.02)

    # Tilt frame (link 1)
    tilt_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.03, 0.015], rgbaColor=[0.2, 1.0, 0.6, 1])
    tilt_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.03, 0.015])

    # Laser tube (link 2) aligned along +Y
    laser_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.003, length=0.12, rgbaColor=[1.0, 0.2, 0.2, 1],
                                    visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
                                    visualFramePosition=[0, 0.06, 0])
    laser_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.003, height=0.12)

    mass = 0.2

    # Build multibody with two revolute joints
    body_uid = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=base_col,
        baseVisualShapeIndex=base_vis,
        basePosition=pivot_xyz,
        linkMasses=[mass, mass, 0.05],
        linkCollisionShapeIndices=[pan_col, tilt_col, laser_col],
        linkVisualShapeIndices=[pan_vis, tilt_vis, laser_vis],
        linkPositions=[(0, 0, 0.03), (0, 0.04, 0), (0, 0.08, 0)],  # relative to parent
        linkOrientations=[p.getQuaternionFromEuler((0, 0, 0))] * 3,
        linkInertialFramePositions=[(0, 0, 0)] * 3,
        linkInertialFrameOrientations=[p.getQuaternionFromEuler((0, 0, 0))] * 3,
        linkParentIndices=[-1, 0, 1],
        linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_FIXED],
        linkJointAxis=[(0, 0, 1), (1, 0, 0), (0, 0, 0)],
    )

    # Disable default motors; we’ll drive with POSITION_CONTROL
    for j in range(p.getNumJoints(body_uid)):
        p.setJointMotorControl2(body_uid, j, p.VELOCITY_CONTROL, force=0)

    return body_uid


def create_wall_and_floor():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Wall at Y=1.0 m
    wall_thickness = 0.02
    wall_half = [0.5, wall_thickness / 2, 0.5]
    wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half)
    wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=wall_half, rgbaColor=[0.9, 0.7, 0.4, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_col, baseVisualShapeIndex=wall_vis,
                      basePosition=[0, WALL_Y_M, 0.5])


# =====================
# Main loop
# =====================

def main():
    # Try GUI; fall back to DIRECT if OpenGL context creation fails
    try:
        physics_client = p.connect(p.GUI)
        headless = False
    except Exception:
        physics_client = p.connect(p.DIRECT)
        headless = True
    p.setTimeStep(DT)
    p.setGravity(0, 0, -9.81)

    # Prepare optional headless preview variables
    imfig = None
    imax = None
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0, 0.5, 0.3])

    create_wall_and_floor()

    pivot = np.array([0.0, 0.0, 0.1])  # raise the gimbal a bit above the floor
    body = create_gimbal(tuple(pivot))

    # Visual markers
    target_marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.015, rgbaColor=[0.1, 0.9, 0.1, 1])
    hit_marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1.0, 0.2, 0.2, 1])
    tgt_id = p.createMultiBody(0, baseVisualShapeIndex=target_marker, basePosition=TARGET)
    hit_id = p.createMultiBody(0, baseVisualShapeIndex=hit_marker, basePosition=[0, -10, 0])  # offscreen initially

    # State
    paused = False
    step_idx = 1  # start at 0.5°
    gear_idx = 0  # 1x
    target = TARGET.copy()

    beam_raw_uid = None
    beam_q_uid = None

    def draw_beams(pan_raw, tilt_raw, pan_q, tilt_q):

        # If in headless (DIRECT) mode, render a preview with TinyRenderer -> Matplotlib
        if headless:
            cam_pos = [0.6, -0.6, 0.6]
            cam_target = [0.0, 0.6, 0.4]
            cam_up = [0, 0, 1]
            view = p.computeViewMatrix(cam_pos, cam_target, cam_up)
            proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.4, nearVal=0.01, farVal=5.0)
            w, h, rgb, _, _ = p.getCameraImage(700, 500, view, proj, renderer=p.ER_TINY_RENDERER)
            import numpy as _np
            import matplotlib.pyplot as _plt
            img = _np.reshape(rgb, (h, w, 4))[:, :, :3]
            if imfig is None:
                imfig = _plt.figure("PyBullet Headless Preview")
                imax = _plt.imshow(img)
                _plt.axis('off')
            else:
                imax.set_data(img)
                _plt.pause(0.001)
        nonlocal beam_raw_uid, beam_q_uid
        # Remove old
        if beam_raw_uid is not None:
            p.removeUserDebugItem(beam_raw_uid)
        if beam_q_uid is not None:
            p.removeUserDebugItem(beam_q_uid)

        o = pivot
        d_raw = rot_yaw_pitch_to_dir(pan_raw, tilt_raw)
        d_q = rot_yaw_pitch_to_dir(pan_q, tilt_q)

        beam_raw_uid = p.addUserDebugLine(o, (o + d_raw * BEAM_LEN_M), [0, 0, 1], lineWidth=1.5)
        beam_q_uid = p.addUserDebugLine(o, (o + d_q * BEAM_LEN_M), [1, 0, 0], lineWidth=2.0)

        # Update hit marker at plane Y=1m for quantized beam
        hit = ray_plane_hit(o, d_q, WALL_Y_M)
        if hit is not None:
            p.resetBasePositionAndOrientation(hit_id, hit, [0, 0, 0, 1])
        else:
            p.resetBasePositionAndOrientation(hit_id, [0, -10, 0], [0, 0, 0, 1])

    while True:
        # Keyboard handling
        keys = p.getKeyboardEvents()
        if keys:
            if p.B3G_ESCAPE in keys or ord('q') in keys or ord('Q') in keys:
                break
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                paused = not paused
            if ord('r') in keys or ord('R') in keys:
                if keys.get(ord('r'), 0) & p.KEY_WAS_TRIGGERED or keys.get(ord('R'), 0) & p.KEY_WAS_TRIGGERED:
                    # Randomize target around the wall area
                    target = np.array([
                        random.uniform(-0.2, 0.2),
                        WALL_Y_M,
                        random.uniform(-0.1, 0.5),
                    ])
                    p.resetBasePositionAndOrientation(tgt_id, target, [0, 0, 0, 1])
            if ord('s') in keys or ord('S') in keys:
                if keys.get(ord('s'), 0) & p.KEY_WAS_TRIGGERED or keys.get(ord('S'), 0) & p.KEY_WAS_TRIGGERED:
                    step_idx = (step_idx + 1) % len(SERVO_STEP_DEG_OPTIONS)
            if ord('g') in keys or ord('G') in keys:
                if keys.get(ord('g'), 0) & p.KEY_WAS_TRIGGERED or keys.get(ord('G'), 0) & p.KEY_WAS_TRIGGERED:
                    gear_idx = (gear_idx + 1) % len(GEAR_OPTIONS)
            # Nudge target
            nudge = 0.005
            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                target[0] -= nudge
            if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                target[0] += nudge
            if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                target[2] += nudge
            if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                target[2] -= nudge
            if p.B3G_PAGE_UP in keys and keys[p.B3G_PAGE_UP] & p.KEY_IS_DOWN:
                target[1] += 0.01
            if p.B3G_PAGE_DOWN in keys and keys[p.B3G_PAGE_DOWN] & p.KEY_IS_DOWN:
                target[1] -= 0.01
            # Keep target near the wall and above floor
            target[1] = max(0.2, min(1.8, target[1]))
            target[2] = max(0.0, target[2])
            p.resetBasePositionAndOrientation(tgt_id, target, [0, 0, 0, 1])

        # Compute raw and quantized aiming
        pan_raw, tilt_raw = aim_angles_from_target_world(target, pivot)
        step_deg = SERVO_STEP_DEG_OPTIONS[step_idx]
        gear = GEAR_OPTIONS[gear_idx]
        pan_q = quantize_angle(pan_raw, step_deg, gear)
        tilt_q = quantize_angle(tilt_raw, step_deg, gear)

        # Drive joints with quantized commands (simulate servo)
        # Joint indices: 0=pan(Z), 1=tilt(X)
        p.setJointMotorControl2(body, 0, p.POSITION_CONTROL, targetPosition=math.radians(pan_q), positionGain=KP, velocityGain=KD)
        p.setJointMotorControl2(body, 1, p.POSITION_CONTROL, targetPosition=math.radians(tilt_q), positionGain=KP, velocityGain=KD)

        # Draw beams and update hit marker
        draw_beams(pan_raw, tilt_raw, pan_q, tilt_q)

        # On-screen text
        txt = f"Target(m): x={target[0]:.3f}, y={target[1]:.3f}, z={target[2]:.3f} | Raw pan/tilt(deg): {pan_raw:.2f}/{tilt_raw:.2f} | Quantized: {pan_q:.2f}/{tilt_q:.2f} | step={step_deg:.3f}°, gear={gear:.1f}x"
        p.addUserDebugText(txt, [0, 0.1, 1.0], textColorRGB=[1, 1, 1], textSize=1.2, lifeTime=0.1)

        if not paused:
            p.stepSimulation()
        p.sleep(DT)

    p.disconnect()


if __name__ == "__main__":
    main()
