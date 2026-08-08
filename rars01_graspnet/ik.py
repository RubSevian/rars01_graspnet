"""Deterministic offline 6D IK helpers; this module has no robot transport."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class IKSolution:
    joints: np.ndarray
    position_error_m: float
    rotation_error_deg: float
    successful_starts: int
    success: bool


def pregrasp_transform(T_grasp_base: np.ndarray, distance_m: float) -> np.ndarray:
    """Back away opposite GraspNet's +X approach axis."""
    transform = _rigid_transform(T_grasp_base).copy()
    if distance_m <= 0:
        raise ValueError("Pre-grasp distance must be positive")
    transform[:3, 3] -= float(distance_m) * transform[:3, 0]
    return transform


def tcp_target_from_grasp(T_grasp_base: np.ndarray, T_grasp_tcp: np.ndarray) -> np.ndarray:
    """Solve T_grasp_base = T_tcp_base @ T_grasp_tcp for T_tcp_base."""
    return _rigid_transform(T_grasp_base) @ np.linalg.inv(_rigid_transform(T_grasp_tcp))


def solve_pose_ik(kinematics, target: np.ndarray, reference, *,
                  joint_margin_rad: float = 0.05, random_starts: int = 32,
                  position_tolerance_m: float = 0.002,
                  rotation_tolerance_deg: float = 2.0,
                  joint_regularization: float = 0.0,
                  random_seed: int = 7) -> IKSolution:
    """Multi-start bounded least-squares IK using the project's URDF FK."""
    target = _rigid_transform(target)
    reference = np.asarray(reference, dtype=np.float64).reshape(6)
    lower = np.asarray(kinematics.lower_limits, dtype=np.float64) + joint_margin_rad
    upper = np.asarray(kinematics.upper_limits, dtype=np.float64) - joint_margin_rad
    if np.any(lower >= upper):
        raise ValueError("Joint margin removes the available joint range")
    if joint_regularization < 0:
        raise ValueError("Joint regularization must be non-negative")
    reference = np.clip(reference, lower, upper)
    rng = np.random.default_rng(random_seed)
    starts = [reference]
    starts.extend(rng.uniform(lower, upper) for _ in range(max(0, int(random_starts))))
    accepted: list[tuple[float, np.ndarray, float, float]] = []
    best_failure = None

    def residual(q: np.ndarray) -> np.ndarray:
        actual = kinematics.forward(q)
        position = (actual[:3, 3] - target[:3, 3]) / 0.005
        rotation = Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec() / 0.05
        posture = float(joint_regularization) * (q - reference) / (upper - lower)
        return np.concatenate((position, rotation, posture))

    for start in starts:
        result = least_squares(
            residual, start, bounds=(lower, upper), max_nfev=1200,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        actual = kinematics.forward(result.x)
        position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        rotation_error = float(np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()
        )))
        normalized_distance = float(np.linalg.norm((result.x - reference) / (upper - lower)))
        candidate = (normalized_distance, result.x.copy(), position_error, rotation_error)
        if best_failure is None or (position_error + np.deg2rad(rotation_error)) < (
            best_failure[2] + np.deg2rad(best_failure[3])
        ):
            best_failure = candidate
        if position_error <= position_tolerance_m and rotation_error <= rotation_tolerance_deg:
            if not any(np.max(np.abs(result.x - item[1])) < 1e-4 for item in accepted):
                accepted.append(candidate)

    selected = min(accepted, key=lambda item: item[0]) if accepted else best_failure
    if selected is None:
        raise RuntimeError("IK optimizer did not produce a candidate")
    _, joints, position_error, rotation_error = selected
    return IKSolution(
        joints=joints,
        position_error_m=position_error,
        rotation_error_deg=rotation_error,
        successful_starts=len(accepted),
        success=bool(accepted),
    )


def _rigid_transform(value: np.ndarray) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64).reshape(4, 4).copy()
    if not np.all(np.isfinite(transform)):
        raise ValueError("Transform contains NaN or infinity")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous transform bottom row")
    u, _, vt = np.linalg.svd(transform[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    transform[:3, :3] = rotation
    return transform
