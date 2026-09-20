"""Small joint-space trajectory helpers for direct-SDK calibration motion."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .ik import solve_pose_ik


def minimum_jerk_samples(start, target, duration_s: float, rate_hz: float,
                         max_joint_step_rad: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if start.shape != target.shape:
        raise ValueError("Trajectory endpoints must have matching shapes")
    if duration_s <= 0 or rate_hz <= 0 or max_joint_step_rad <= 0:
        raise ValueError("Trajectory duration, rate and max step must be positive")
    delta = target - start
    timed_steps = int(np.ceil(duration_s * rate_hz))
    # The derivative of 10s^3-15s^4+6s^5 peaks at 1.875, so account for it
    # when turning the configured maximum step into a sample count.
    limited_steps = int(np.ceil(1.875 * np.max(np.abs(delta)) / max_joint_step_rad))
    steps = max(1, timed_steps, limited_steps)
    phase = np.linspace(1.0 / steps, 1.0, steps)
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    return start + blend[:, None] * delta


def cartesian_geodesic_samples(start: np.ndarray, target: np.ndarray,
                               duration_s: float, rate_hz: float) -> tuple[np.ndarray, ...]:
    """Minimum-jerk translation plus SO(3) geodesic rotation."""
    start = np.asarray(start, dtype=np.float64).reshape(4, 4)
    target = np.asarray(target, dtype=np.float64).reshape(4, 4)
    if duration_s <= 0 or rate_hz <= 0:
        raise ValueError("Cartesian trajectory duration and rate must be positive")
    steps = max(2, int(np.ceil(float(duration_s) * float(rate_hz))))
    phase = np.linspace(1.0 / steps, 1.0, steps)
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    delta_rotation = Rotation.from_matrix(start[:3, :3].T @ target[:3, :3]).as_rotvec()
    result = []
    for value in blend:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = start[:3, 3] + value * (target[:3, 3] - start[:3, 3])
        transform[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(
            value * delta_rotation
        ).as_matrix()
        result.append(transform)
    return tuple(result)


def track_cartesian_trajectory(
    kinematics, start_joints, target: np.ndarray, *, duration_s: float,
    rate_hz: float, max_joint_step_rad: float, joint_margin_rad: float,
    position_tolerance_m: float, rotation_tolerance_deg: float,
    null_gain: float = 0.1,
) -> np.ndarray:
    """Track an SE(3) path with local sequential IK and reject branch jumps."""
    current = np.asarray(start_joints, dtype=np.float64).reshape(6).copy()
    if null_gain < 0:
        raise ValueError("Cartesian null gain must be non-negative")
    poses = cartesian_geodesic_samples(
        kinematics.forward(current), target, duration_s, rate_hz
    )
    joints = []
    for index, pose in enumerate(poses, start=1):
        solution = solve_pose_ik(
            kinematics, pose, current,
            joint_margin_rad=joint_margin_rad, random_starts=0,
            position_tolerance_m=position_tolerance_m,
            rotation_tolerance_deg=rotation_tolerance_deg,
            joint_regularization=null_gain,
        )
        if not solution.success:
            raise RuntimeError(
                f"Cartesian trajectory IK failed at {index}/{len(poses)}: "
                f"position={solution.position_error_m * 1000:.2f} mm, "
                f"rotation={solution.rotation_error_deg:.2f} deg"
            )
        step = float(np.max(np.abs(solution.joints - current)))
        if step > float(max_joint_step_rad) + 1e-9:
            raise RuntimeError(
                f"Cartesian trajectory branch jump at {index}/{len(poses)}: "
                f"{step:.4f} rad > {max_joint_step_rad:.4f} rad"
            )
        current = solution.joints.copy()
        joints.append(current)
    return np.asarray(joints, dtype=np.float64)


def calibration_joint_targets(seed, lower, upper, amplitudes, count: int,
                              kinematics, *, margin_rad: float,
                              min_tcp_z_m: float,
                              max_tcp_translation_m: float) -> list[np.ndarray]:
    seed = np.asarray(seed, dtype=np.float64).reshape(6)
    lower = np.asarray(lower, dtype=np.float64).reshape(6) + float(margin_rad)
    upper = np.asarray(upper, dtype=np.float64).reshape(6) - float(margin_rad)
    amplitudes = np.asarray(amplitudes, dtype=np.float64).reshape(6)
    if np.any(seed < lower) or np.any(seed > upper):
        raise ValueError("Current seed pose is too close to a configured URDF joint limit")
    seed_xyz = kinematics.forward(seed)[:3, 3]
    targets: list[np.ndarray] = []
    bases = (2, 3, 5, 7, 11, 13)
    candidate_index = 1
    while len(targets) < int(count) and candidate_index <= max(100, int(count) * 30):
        unit = np.array([_radical_inverse(candidate_index, base) for base in bases])
        target = seed + amplitudes * (2.0 * unit - 1.0)
        candidate_index += 1
        if np.any(target < lower) or np.any(target > upper):
            continue
        xyz = kinematics.forward(target)[:3, 3]
        if xyz[2] < float(min_tcp_z_m):
            continue
        if np.linalg.norm(xyz - seed_xyz) > float(max_tcp_translation_m):
            continue
        if any(np.max(np.abs(target - previous)) < 0.03 for previous in targets):
            continue
        targets.append(target)
    if len(targets) < int(count):
        raise RuntimeError(
            f"Could generate only {len(targets)}/{count} calibration targets; "
            "reduce count/margins or amplitudes"
        )
    return targets


def _radical_inverse(index: int, base: int) -> float:
    result, fraction = 0.0, 1.0 / base
    while index:
        result += (index % base) * fraction
        index //= base
        fraction /= base
    return result
