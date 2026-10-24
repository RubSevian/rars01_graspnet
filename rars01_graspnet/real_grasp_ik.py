"""Pure real-grasp IK and constant-orientation Cartesian waypoint logic.

Mechanically extracted from scripts/grasp.py; no SDK or camera imports.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
from utils.transforms import pose6d_to_mat4

@dataclass(frozen=True)
class IkSolution:
    success: bool
    error: float
    joints: np.ndarray
    position_error_m: float = float("inf")
    orientation_error_rad: float = float("inf")

class IkChecker:
    def __init__(
        self,
        arm: Any,
        retry_count: int = 3,
        position_tolerance_m: float = 0.010,
        orientation_tolerance_rad: float = np.deg2rad(5.0),
    ) -> None:
        from rars01_graspnet.pinocchio_math import (
            compute_fk,
            get_end_effector_frame_id,
            pad_q_for_model,
            pos_rot_to_se3,
            solve_ik,
        )
        from rars01_graspnet.pinocchio_math import IKParams

        self.cancel_check = None
        self.diagnostics = None
        self.context = {}
        self._cache = OrderedDict()
        self._arm = arm
        self._arm_group = arm.groups.get("arm")
        if self._arm_group is None:
            raise ValueError("Hardware config missing groups.arm")
        self._n = self._arm_group.num_joints
        self._pad_q_for_model = pad_q_for_model
        self._compute_fk = compute_fk
        self._pos_rot_to_se3 = pos_rot_to_se3
        self._solve_ik = solve_ik
        self._model = arm.load_kinematic_model()
        self._end_frame_id = get_end_effector_frame_id(self._model)
        self._params = IKParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)
        self._retry_count = max(0, int(retry_count))
        self._position_tolerance_m = float(position_tolerance_m)
        self._orientation_tolerance_rad = float(orientation_tolerance_rad)
        if self._position_tolerance_m <= 0.0 or self._orientation_tolerance_rad <= 0.0:
            raise ValueError("IK FK tolerances must be positive")
        self._lower = np.asarray(self._model.lowerPositionLimit[:self._n], dtype=np.float64)
        self._upper = np.asarray(self._model.upperPositionLimit[:self._n], dtype=np.float64)
        self._frame_name = self._model.frames[self._end_frame_id].name

    def current_joints(self) -> np.ndarray:
        return np.asarray(
            self._arm.get_state(request_feedback=False)[0][:self._n], dtype=np.float64
        ).copy()

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower.copy(), self._upper.copy()

    def fk_errors(
        self, pose6d: tuple[float, ...], joints: np.ndarray
    ) -> tuple[float, float]:
        """Return independent FK position and orientation errors for one target."""
        q = self._pad_q_for_model(
            self._model, np.asarray(joints, dtype=np.float64).reshape(self._n), self._n
        )
        position, rotation, _ = self._compute_fk(
            self._model, q, frame_name=self._frame_name
        )
        target = pose6d_to_mat4(*pose6d)
        position_error = float(np.linalg.norm(position - target[:3, 3]))
        R_error = target[:3, :3].T @ rotation
        cosine = float(np.clip((np.trace(R_error) - 1.0) * 0.5, -1.0, 1.0))
        orientation_error = float(np.arccos(cosine))
        return position_error, orientation_error

    def begin_operation(self, diagnostics, cancel_check=None):
        """One worker owns this checker. Never reuse a cache across operations."""
        self._cache.clear()
        self.diagnostics = diagnostics
        self.cancel_check = cancel_check
        self.context = {}

    def solve(self, x, y, z, roll, pitch, yaw, reference_joints=None):
        started = time.monotonic()
        cpu_started = time.thread_time()
        if self.cancel_check is not None:
            self.cancel_check()
        reference = (self.current_joints() if reference_joints is None else
                     np.asarray(reference_joints, dtype=float).reshape(self._n))
        pose = (x, y, z, roll, pitch, yaw)
        key = (np.asarray(pose, dtype=float).tobytes(), reference.tobytes())
        record = dict(self.context, target_pose6d=list(map(float, pose)),
                      reference_joints=reference.tolist(), attempts=[], cache_hit=False)
        try:
            if key in self._cache:
                record['cache_hit'] = True
                result = self._cache[key]
            else:
                result = self._solve_uncached(*pose, reference_joints=reference, record=record)
                if len(self._cache) >= 512:
                    self._cache.popitem(last=False)
                self._cache[key] = result
            record.update(success=bool(result.success), error=float(result.error),
                          joints=result.joints.tolist(),
                          position_error_m=float(result.position_error_m),
                          orientation_error_rad=float(result.orientation_error_rad),
                          reason='OK' if result.success else 'IK_OR_FK_TOLERANCE_OR_LIMIT')
            record.update(joint_lower_rad=self._lower.tolist(),joint_upper_rad=self._upper.tolist(),
                          lower_margin_rad=(result.joints-self._lower).tolist(),
                          upper_margin_rad=(self._upper-result.joints).tolist(),
                          position_tolerance_m=self._position_tolerance_m,
                          orientation_tolerance_rad=self._orientation_tolerance_rad)
            return IkSolution(result.success, result.error, result.joints.copy(),
                              result.position_error_m, result.orientation_error_rad)
        except Exception as error:
            record.update(success=False, reason=str(error))
            raise
        finally:
            record['elapsed_s'] = time.monotonic() - started
            record['thread_cpu_s'] = time.thread_time() - cpu_started
            record['attempt_count'] = len(record['attempts'])
            if self.diagnostics is not None:
                self.diagnostics.append(record)

    def _solve_uncached(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        reference_joints: Optional[np.ndarray] = None,
        record=None,
    ) -> IkSolution:
        reference = (
            self.current_joints()
            if reference_joints is None
            else np.asarray(reference_joints, dtype=np.float64).reshape(self._n)
        )
        target = self._pos_rot_to_se3(
            np.array([x, y, z], dtype=np.float64), roll=roll, pitch=pitch, yaw=yaw
        )
        midpoint = 0.5 * (self._lower + self._upper)
        seeds = [reference]
        for fraction in (0.25, 0.50, 0.75)[:self._retry_count]:
            seeds.append((1.0 - fraction) * reference + fraction * midpoint)

        best_result = None
        best_errors = (float("inf"), float("inf"))
        seen_seeds = set()
        data = self._model.createData()
        for seed in seeds:
            seed_key = seed.tobytes()
            if seed_key in seen_seeds:
                continue
            seen_seeds.add(seed_key)
            started = time.monotonic()
            try:
                result = self._solve_ik(
                    self._model,
                    data,
                    self._end_frame_id,
                    target,
                    self._pad_q_for_model(self._model, seed, self._n),
                    self._params,
                    controlled_joints=self._n,
                    cancel_check=self.cancel_check,
                )
            except Exception as error:
                if record is not None:
                    record['attempts'].append(dict(seed=seed.tolist(), elapsed_s=time.monotonic()-started,
                        termination='interrupted', reason=str(error), iterations=None))
                raise
            q = np.asarray(result.q[:self._n], dtype=np.float64)
            position_error, orientation_error = self.fk_errors(
                (x, y, z, roll, pitch, yaw), q
            )
            if record is not None:
                reason = ('JOINT_LIMIT' if np.any(q < self._lower) or np.any(q > self._upper)
                          else 'FK_POSITION' if position_error > self._position_tolerance_m
                          else 'FK_ORIENTATION' if orientation_error > self._orientation_tolerance_rad
                          else 'SOLVER_NOT_CONVERGED' if not result.success else 'OK')
                record['attempts'].append(dict(seed=seed.tolist(), elapsed_s=time.monotonic()-started,
                    iterations=result.iterations, termination=result.termination, reason=reason,
                    error=float(result.error), position_error_m=position_error,
                    orientation_error_rad=orientation_error,joints=q.tolist(),
                    lower_margin_rad=(q-self._lower).tolist(),
                    upper_margin_rad=(self._upper-q).tolist()))
            if best_result is None or float(result.error) < float(best_result.error):
                best_result = result
                best_errors = (position_error, orientation_error)
            if result.success:
                # Seeds are ordered from the previous chain point toward the
                # joint-range midpoint, so the first success is the most local.
                if (
                    np.all(q >= self._lower)
                    and np.all(q <= self._upper)
                    and position_error <= self._position_tolerance_m
                    and orientation_error <= self._orientation_tolerance_rad
                ):
                    return IkSolution(
                        True, float(result.error), q.copy(), position_error, orientation_error
                    )
        assert best_result is not None
        q = np.asarray(best_result.q[:self._n], dtype=np.float64).copy()
        return IkSolution(
            False,
            float(best_result.error),
            q,
            *best_errors,
        )


def _cartesian_waypoint_pose(
    start6d: tuple[float, ...], end6d: tuple[float, ...], alpha: float
) -> tuple[float, ...]:
    """Linearly interpolate TCP position while keeping the end orientation."""
    start = np.asarray(start6d, dtype=np.float64)
    end = np.asarray(end6d, dtype=np.float64)
    T = pose6d_to_mat4(*end6d)
    T[:3, 3] = (1.0 - alpha) * start[:3] + alpha * end[:3]
    from utils.transforms import mat4_to_pose6d

    return mat4_to_pose6d(T)


def _plan_cartesian_segment(
    ik_checker: IkChecker,
    start6d: tuple[float, ...],
    end6d: tuple[float, ...],
    start_joints: np.ndarray,
    cartesian_cfg: dict[str, Any],
    label: str,
    verbose: bool = True,
) -> Optional[tuple[np.ndarray, ...]]:
    """Solve a constant-orientation Cartesian segment as continuous IK points."""
    def report(*args):
        if verbose:print(*args)
    waypoint_count = int(cartesian_cfg["waypoint_count"])
    warn_step_rad = np.deg2rad(float(cartesian_cfg["joint_step_warn_deg"]))
    reject_step_rad = np.deg2rad(float(cartesian_cfg["max_joint_step_deg"]))
    previous = np.asarray(start_joints, dtype=np.float64).copy()
    points: list[np.ndarray] = []

    for waypoint_index in range(1, waypoint_count + 1):
        alpha = waypoint_index / waypoint_count
        target = _cartesian_waypoint_pose(start6d, end6d, alpha)
        if hasattr(ik_checker, "context"):
            ik_checker.context.update(stage=label, waypoint=waypoint_index)
        solution = ik_checker.solve(*target, reference_joints=previous)
        delta = np.abs(solution.joints - previous)
        max_joint_index = int(np.argmax(delta))
        max_delta = float(delta[max_joint_index])
        status = "OK" if solution.success else "FAIL"
        report(
            f"[Cartesian/{label}] waypoint={waypoint_index}/{waypoint_count} "
            f"target_xyz={np.round(np.asarray(target[:3]), 5).tolist()} "
            f"target_rpy={np.round(np.asarray(target[3:]), 5).tolist()} "
            f"IK={status} fk_pos_mm={solution.position_error_m * 1000.0:.2f} "
            f"fk_rot_deg={np.rad2deg(solution.orientation_error_rad):.2f} "
            f"max_delta_q={max_delta:.4f} joint={max_joint_index + 1}"
        )
        if not solution.success:
            if solution.position_error_m > float(cartesian_cfg["ik_position_tolerance_m"]):
                reason = "FK position error exceeds tolerance"
            elif solution.orientation_error_rad > np.deg2rad(
                float(cartesian_cfg["ik_orientation_tolerance_deg"])
            ):
                reason = "FK orientation error exceeds tolerance"
            else:
                reason = "IK did not converge or target violates joint limits"
            report(f"[Cartesian/{label}] reject: {reason}")
            return None
        if max_delta > reject_step_rad:
            report(
                f"[Cartesian/{label}] reject: joint {max_joint_index + 1} step "
                f"{np.rad2deg(max_delta):.2f} deg exceeds "
                f"{np.rad2deg(reject_step_rad):.2f} deg"
            )
            return None
        if max_delta > warn_step_rad:
            report(
                f"[Cartesian/{label}] warning: joint {max_joint_index + 1} step "
                f"{np.rad2deg(max_delta):.2f} deg exceeds "
                f"{np.rad2deg(warn_step_rad):.2f} deg"
            )
        points.append(solution.joints)
        previous = solution.joints

    return tuple(points)
