"""Локальная FK, DLS IK и SE(3) minimum-jerk математика RARS01.

Единицы: метры, радианы, секунды. Модель задаёт вызывающий код; здесь нет
загрузки чужого hardware YAML, обращения к моторам или выбора оборудования.
Эти функции проверяют кинематику, но не геометрические коллизии.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass
class IKParams:
    max_iter: int = 200
    tolerance: float = 1e-4
    step_size: float = 0.5
    damping: float = 1e-6


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    error: float
    iterations: int


def get_end_effector_frame_id(model: pin.Model) -> int:
    """Find the adapter's TCP alias; never read configuration outside RARS01."""
    frame = model.getFrameId("end_link")
    if frame >= model.nframes:
        frame = model.getFrameId("End_link")
    if frame >= model.nframes:
        raise ValueError("RARS01 model has no End_link TCP frame")
    return frame


def pad_q_for_model(model, q, controlled_joints=None):
    values = np.asarray(q, dtype=np.float64).reshape(-1)
    count = min(values.size, model.nq,
                model.nq if controlled_joints is None else controlled_joints)
    padded = np.zeros(model.nq)
    padded[:count] = values[:count]
    return padded


def compute_fk(model, q, frame_name=None):
    """Return TCP position, rotation and homogeneous transform in model root."""
    if np.shape(q) != (model.nq,):
        raise ValueError(f"Expected q shape ({model.nq},), got {np.shape(q)}")
    frame = (get_end_effector_frame_id(model) if frame_name is None
             else model.getFrameId(frame_name))
    if frame >= model.nframes:
        raise ValueError(f"Unknown frame: {frame_name}")
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pose = data.oMf[frame]
    return pose.translation.copy(), pose.rotation.copy(), pose.homogeneous.copy()


def pos_rot_to_se3(pos, rot=None, roll=0.0, pitch=0.0, yaw=0.0):
    if rot is None:
        rot = pin.rpy.rpyToMatrix(roll, pitch, yaw)
    return pin.SE3(rot, np.asarray(pos, dtype=np.float64))


def _clamp(model, q):
    lo = np.where(np.isfinite(model.lowerPositionLimit), model.lowerPositionLimit, 0.0)
    hi = np.where(np.isfinite(model.upperPositionLimit), model.upperPositionLimit, 0.0)
    return np.minimum(np.maximum(q, lo), hi)


def _error(model, data, frame, q, target):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    error = pin.log6(data.oMf[frame].inverse() * target).vector
    return float(np.linalg.norm(error)), error


def solve_ik(model, data, end_frame_id, target, q_init,
             params=None, controlled_joints=None):
    """Damped local-frame IK with four-step backtracking.

    Keep the baseline iteration policy (including refinement after reaching
    tolerance) so candidate ranking and the selected solution do not change.
    """
    params = params or IKParams()
    n = model.nq if controlled_joints is None else controlled_joints
    q = pad_q_for_model(model, q_init, n)
    norm, error = _error(model, data, end_frame_id, q, target)
    if norm < params.tolerance:
        return IKResult(q[:n], True, norm, 0)
    for _ in range(params.max_iter):
        pin.computeJointJacobians(model, data, q)
        jacobian = pin.getFrameJacobian(model, data, end_frame_id, pin.LOCAL)
        matrix = jacobian @ jacobian.T
        matrix[np.diag_indices_from(matrix)] += params.damping * max(1.0, norm * 10.0)
        step = params.step_size * jacobian.T @ np.linalg.solve(matrix, error)
        for alpha in (1.0, 0.5, 0.25, 0.125):
            candidate = _clamp(model, pin.integrate(model, q, alpha * step))
            new_norm, new_error = _error(model, data, end_frame_id, candidate, target)
            if new_norm < norm:
                q, norm, error = candidate, new_norm, new_error
                break
    return IKResult(q[:n], norm < params.tolerance, norm, params.max_iter)


def minimum_jerk(phase):
    """Normalized blend with zero first and second derivatives at endpoints."""
    s = np.clip(phase, 0.0, 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def cartesian_geodesic(start, end, duration, dt):
    """Sample the baseline SE(3) geodesic; this is not a collision planner."""
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive")
    start, end = pin.SE3(start), pin.SE3(end)
    count = max(2, int(np.ceil(duration / dt)) + 1)
    delta = pin.log6(start.inverse() * end)
    return [start * pin.exp6(delta * minimum_jerk(i / (count - 1)))
            for i in range(count)]


def _limit_gradient(model, q):
    lo, hi = model.lowerPositionLimit, model.upperPositionLimit
    dl, dh = q - lo, hi - q
    mask = np.isfinite(lo) & np.isfinite(hi) & (dl > 1e-6) & (dh > 1e-6)
    result = np.zeros(model.nv)
    result[mask] = (dh[mask] - dl[mask]) / (dl[mask] * dh[mask])
    return result


def track_poses(model, end_frame_id, poses, q_init, params=None, null_gain=0.1):
    """Track Cartesian poses with DLS; return configurations and convergence flags."""
    params = params or IKParams(step_size=0.8)
    q = np.asarray(q_init, dtype=np.float64).copy()
    data = model.createData()
    points, converged = [], []
    for pose in poses:
        success = False
        for _ in range(params.max_iter):
            pin.computeJointJacobians(model, data, q)
            pin.updateFramePlacements(model, data)
            error = pin.log6(data.oMf[end_frame_id].inverse() * pose).vector
            norm = np.linalg.norm(error)
            if norm < params.tolerance:
                success = True
                break
            jacobian = pin.getFrameJacobian(model, data, end_frame_id, pin.LOCAL)
            matrix = jacobian @ jacobian.T
            matrix[np.diag_indices_from(matrix)] += params.damping * max(1.0, norm * 10.0)
            step = params.step_size * jacobian.T @ np.linalg.solve(matrix, error)
            if null_gain > 0.0:
                gradient = _limit_gradient(model, q)
                step += null_gain * (gradient - jacobian.T @ np.linalg.solve(matrix, jacobian @ gradient))
            q = _clamp(model, pin.integrate(model, q, step))
        points.append(q.copy())
        converged.append(success)
    return points, converged
