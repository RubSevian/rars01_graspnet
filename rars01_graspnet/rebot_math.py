"""Local Pinocchio motion mathematics adapted from reBot-DevArm-Grasp.

This module contains no reBot transport, motor configuration, or actuator
imports.  It is used by the RARS01 adapter directly.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

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


def pad_q_for_model(model: pin.Model, q: np.ndarray, controlled_joints: int) -> np.ndarray:
    result = np.zeros(model.nq, dtype=np.float64)
    values = np.asarray(q, dtype=np.float64).reshape(-1)
    count = min(values.size, int(controlled_joints), model.nq)
    result[:count] = values[:count]
    return result


def pos_rot_to_se3(position: np.ndarray, *, roll: float = 0.0, pitch: float = 0.0,
                   yaw: float = 0.0) -> pin.SE3:
    return pin.SE3(pin.rpy.rpyToMatrix(roll, pitch, yaw), np.asarray(position, dtype=np.float64))


def compute_fk(model: pin.Model, q: np.ndarray, frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pose = data.oMf[frame_id]
    return pose.translation.copy(), pose.rotation.copy(), pose.homogeneous.copy()


def _clamp(model: pin.Model, q: np.ndarray) -> np.ndarray:
    lo = np.where(np.isfinite(model.lowerPositionLimit), model.lowerPositionLimit, 0.0)
    hi = np.where(np.isfinite(model.upperPositionLimit), model.upperPositionLimit, 0.0)
    return np.clip(q, lo, hi)


def solve_ik(model: pin.Model, data: pin.Data, frame_id: int, target: pin.SE3,
             q_initial: np.ndarray, params: IKParams, controlled_joints: int) -> IKResult:
    """Damped local-frame IK with reBot's adaptive damping and line search."""
    q = pad_q_for_model(model, q_initial, controlled_joints)
    previous = float("inf")
    error = np.zeros(6)
    for _ in range(params.max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        error = pin.log6(data.oMf[frame_id].inverse() * target).vector
        norm = float(np.linalg.norm(error))
        if norm < params.tolerance:
            return IKResult(q[:controlled_joints], True, norm)
        pin.computeJointJacobians(model, data, q)
        jacobian = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL)
        jjt = jacobian @ jacobian.T
        jjt[np.diag_indices_from(jjt)] += params.damping * max(1.0, norm * 10.0)
        dq = params.step_size * jacobian.T @ np.linalg.solve(jjt, error)
        previous = norm
        for alpha in (1.0, 0.5, 0.25, 0.125):
            candidate = _clamp(model, pin.integrate(model, q, alpha * dq))
            pin.forwardKinematics(model, data, candidate)
            pin.updateFramePlacements(model, data)
            candidate_norm = float(np.linalg.norm(pin.log6(data.oMf[frame_id].inverse() * target).vector))
            if candidate_norm < previous:
                q = candidate
                break
    return IKResult(q[:controlled_joints], False, previous)


def _minimum_jerk(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _cartesian_geodesic(start: pin.SE3, end: pin.SE3, duration: float,
                         dt: float) -> list[pin.SE3]:
    count = max(2, int(np.ceil(duration / dt)) + 1)
    delta = pin.log6(start.inverse() * end)
    return [start * pin.exp6(delta * _minimum_jerk(index / (count - 1))) for index in range(count)]


def _limit_gradient(model: pin.Model, q: np.ndarray) -> np.ndarray:
    lower, upper = model.lowerPositionLimit, model.upperPositionLimit
    distance_lower, distance_upper = q - lower, upper - q
    valid = (np.isfinite(lower) & np.isfinite(upper)
             & (distance_lower > 1e-6) & (distance_upper > 1e-6))
    result = np.zeros(model.nv)
    result[valid] = ((distance_upper[valid] - distance_lower[valid])
                     / (distance_lower[valid] * distance_upper[valid]))
    return result


def _track(model: pin.Model, frame_id: int, poses: list[pin.SE3], q_start: np.ndarray,
           params: IKParams) -> list[np.ndarray]:
    q = q_start.copy()
    data = model.createData()
    points: list[np.ndarray] = []
    for target in poses:
        for _ in range(params.max_iter):
            pin.computeJointJacobians(model, data, q)
            pin.updateFramePlacements(model, data)
            error = pin.log6(data.oMf[frame_id].inverse() * target).vector
            if np.linalg.norm(error) < params.tolerance:
                break
            jacobian = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL)
            jjt = jacobian @ jacobian.T
            jjt[np.diag_indices_from(jjt)] += params.damping * max(1.0, np.linalg.norm(error) * 10.0)
            solve = np.linalg.solve(jjt, error)
            gradient = _limit_gradient(model, q)
            delta = params.step_size * jacobian.T @ solve
            delta += 0.1 * (gradient - jacobian.T @ np.linalg.solve(jjt, jacobian @ gradient))
            q = _clamp(model, pin.integrate(model, q, delta))
        points.append(q.copy())
    return points


class RarsEndPoseController:
    """Local replacement for reBot's end-pose controller in the RARS01 path."""

    def __init__(self, arm: Any, *, dt: float, arm_control_mode: str = "posvel") -> None:
        if arm_control_mode != "posvel":
            raise ValueError("RARS01 execution uses POS/VEL for arm joints")
        self.rebotarm = arm  # Compatibility with the grasp workflow's wait helper.
        self._arm_group = arm.groups["arm"]
        self._n = self._arm_group.num_joints
        self._dt = float(dt)
        self._arm_control_mode = arm_control_mode
        self._has_gripper = True
        self._model = arm.load_kinematic_model()
        self._data = self._model.createData()
        self._end_frame_id = self._model.getFrameId("end_link")
        self._ik_params = IKParams()
        self._track_params = IKParams(step_size=0.8)
        self._q_target = np.zeros(self._n)
        self._qd_target = np.zeros(self._n)
        self._running = False
        self._send_thread: threading.Thread | None = None
        self._stop_send = threading.Event()

    def _loop_cb(self, _arm: Any, _dt: float) -> None:
        self._arm_group.send_pos_vel(self._q_target, vlim=self._arm_group._pv_vlim)

    def move_to_traj(self, x: float, y: float, z: float, roll: float = 0.0,
                     pitch: float = 0.0, yaw: float = 0.0, duration: float = 2.0) -> bool:
        if not self._running:
            return False
        q_start = pad_q_for_model(self._model, self.rebotarm.get_state()[0], self._n)
        target = pos_rot_to_se3(np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw)
        solution = solve_ik(self._model, self._data, self._end_frame_id, target,
                            q_start, self._ik_params, self._n)
        if not solution.success:
            print(f"[RARS01/Traj] IK failed: error={solution.error:.4f}")
            return False
        q_end = pad_q_for_model(self._model, solution.q, self._n)
        start = pin.SE3(compute_fk(self._model, q_start, self._end_frame_id)[2])
        end = pin.SE3(compute_fk(self._model, q_end, self._end_frame_id)[2])
        points = _track(self._model, self._end_frame_id,
                        _cartesian_geodesic(start, end, duration, self._dt),
                        q_start, self._track_params)
        self._stop_send.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=5.0)
        self._stop_send.clear()
        self._send_thread = threading.Thread(target=self._send, args=(points, duration), daemon=True)
        self._send_thread.start()
        return True

    def _send(self, points: list[np.ndarray], duration: float) -> None:
        interval = duration / len(points)
        for point in points:
            if self._stop_send.is_set():
                return
            self._q_target[:] = point[:self._n]
            time.sleep(interval)

    def safe_home(
        self,
        max_vel: float = 0.5,
        send_freq: float = 50.0,
        settle_thresh: float = 0.01,
        timeout: float = 15.0,
    ) -> None:
        """Return home and wait for the physical joints to settle before disable.

        This keeps the original reBot ``safe_home`` behaviour: the target follows
        a minimum-jerk joint trajectory and remains enabled at the final target
        until feedback confirms that the arm has actually arrived.
        """
        if not self._running:
            return

        # A trajectory sender left alive after an interrupted grasp could
        # overwrite the home targets while cleanup is running.
        self._stop_send.set()
        if self._send_thread is not None and self._send_thread.is_alive():
            self._send_thread.join(timeout=5.0)
        self._send_thread = None

        start = self.rebotarm.get_state()[0][:self._n]
        configured_home = np.asarray(
            getattr(self.rebotarm, "_home", np.zeros(self._n)), dtype=np.float64
        ).reshape(-1)
        if configured_home.size != self._n:
            raise ValueError(
                f"RARS01 home pose must contain {self._n} joints, "
                f"got {configured_home.size}"
            )
        maximum = float(np.max(np.abs(configured_home - start)))
        if maximum < settle_thresh:
            return

        if max_vel <= 0.0 or send_freq <= 0.0:
            raise ValueError("safe_home max_vel and send_freq must be positive")
        duration = min(float(timeout), max(1.0, 2.0 * maximum / max_vel))
        steps = max(2, int(duration * send_freq))
        for index in range(steps):
            progress = _minimum_jerk(index / (steps - 1))
            self._q_target[:] = start + (configured_home - start) * progress
            time.sleep(duration / steps)
        self._q_target[:] = configured_home

        # Do not disable immediately after publishing the final sample.  In
        # POS/VEL mode the motor still needs time to reach it; disabling here
        # is perceived as the arm dropping at the end of the return.
        settle_deadline = time.monotonic() + 3.0
        while time.monotonic() < settle_deadline:
            q_now = self.rebotarm.get_state(request_feedback=False)[0][:self._n]
            if float(np.max(np.abs(q_now - configured_home))) < settle_thresh:
                return
            time.sleep(self._dt)
        remaining = float(np.max(np.abs(q_now - configured_home)))
        print(f"[RARS01/Home] settle timeout; max error={remaining:.4f} rad")

    def end(self) -> None:
        self.safe_home()
        self.rebotarm.disconnect()
        self._running = False
