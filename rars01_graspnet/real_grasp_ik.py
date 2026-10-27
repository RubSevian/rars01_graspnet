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
        bounded_fallback: bool = False,
    ) -> None:
        from rars01_graspnet.pinocchio_math import (
            compute_fk,
            get_end_effector_frame_id,
            pad_q_for_model,
            pos_rot_to_se3,
            solve_ik,
        )
        from rars01_graspnet.pinocchio_math import IKParams

        if type(bounded_fallback) is not bool:
            raise ValueError("bounded_fallback must be boolean")
        self._bounded_fallback = bounded_fallback
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
        if (not np.isfinite([self._position_tolerance_m, self._orientation_tolerance_rad]).all()
                or self._position_tolerance_m <= 0.0 or self._orientation_tolerance_rad <= 0.0):
            raise ValueError("IK FK tolerances must be positive")
        self._lower = np.asarray(self._model.lowerPositionLimit[:self._n], dtype=np.float64)
        self._upper = np.asarray(self._model.upperPositionLimit[:self._n], dtype=np.float64)
        self._frame_name = self._model.frames[self._end_frame_id].name
        self._least_squares = None
        self.solver_preparation = dict(enabled=bounded_fallback, ready=False, elapsed_s=0.)
        if bounded_fallback:
            self.prepare_bounded_solver()

    def prepare_bounded_solver(self):
        """Load and exercise SciPy before mission admission, without robot IO."""
        if self._least_squares is not None:
            return dict(self.solver_preparation)
        started = time.monotonic()
        from scipy.optimize import least_squares
        # Exercise numerical Jacobian, TRF and LAPACK with a small bounded
        # six-joint problem. This is initialization, not reachability approval.
        center = .5 * (self._lower + self._upper)
        seed = center + .01 * (self._upper - self._lower)
        fit = least_squares(lambda q: q - center, seed,
                            bounds=(self._lower, self._upper), method='trf', max_nfev=20)
        if not np.isfinite(fit.x).all() or np.linalg.norm(fit.x-center) > 1e-5:
            raise RuntimeError('BOUNDED_IK_WARMUP_FAILED')
        self._least_squares = least_squares
        self.solver_preparation = dict(enabled=True, ready=True,
                                       elapsed_s=time.monotonic()-started)
        return dict(self.solver_preparation)

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
        best_score = float("inf")
        best_errors = (float("inf"), float("inf"))
        seen_seeds = set()
        data = self._model.createData()
        for seed in seeds:
            seed_key = seed.tobytes()
            if seed_key in seen_seeds:
                continue
            seen_seeds.add(seed_key)
            started = time.monotonic()
            feasible = None
            feasible_score = float("inf")

            def observe(q, error, position, rotation):
                nonlocal feasible, feasible_score
                position_error = float(np.linalg.norm(position - target.translation))
                cosine = np.clip((np.trace(target.rotation.T @ rotation) - 1.0) * 0.5, -1., 1.)
                orientation_error = float(np.arccos(cosine))
                if (np.isfinite(q).all() and np.all(q >= self._lower)
                        and np.all(q <= self._upper)
                        and np.isfinite([position_error, orientation_error]).all()
                        and position_error <= self._position_tolerance_m
                        and orientation_error <= self._orientation_tolerance_rad):
                    score = np.hypot(position_error / self._position_tolerance_m,
                                     orientation_error / self._orientation_tolerance_rad)
                    if score < feasible_score:
                        feasible_score = score
                        feasible = IkSolution(True, float(error), q.copy(),
                                              position_error, orientation_error)

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
                    iterate_callback=observe,
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
                reason = ('NONFINITE' if not np.isfinite(q).all() or not np.isfinite([position_error, orientation_error]).all()
                          else 'JOINT_LIMIT' if np.any(q < self._lower) or np.any(q > self._upper)
                          else 'FK_POSITION' if position_error > self._position_tolerance_m
                          else 'FK_ORIENTATION' if orientation_error > self._orientation_tolerance_rad
                          else 'OK')
                record['attempts'].append(dict(seed=seed.tolist(), elapsed_s=time.monotonic()-started,
                    method="dls", iterations=result.iterations, termination=result.termination,
                    reason="OK" if feasible is not None else reason, final_iterate_reason=reason,
                    solver_success=bool(result.success), accepted=feasible is not None,
                    accepted_joints=None if feasible is None else feasible.joints.tolist(),
                    error=float(result.error), position_error_m=position_error,
                    orientation_error_rad=orientation_error,joints=q.tolist(),
                    lower_margin_rad=(q-self._lower).tolist(),
                    upper_margin_rad=(self._upper-q).tolist()))
            score = float(np.hypot(position_error / self._position_tolerance_m,
                                   orientation_error / self._orientation_tolerance_rad))
            if not np.isfinite(score):
                score = float("inf")
            if best_result is None or score < best_score:
                best_result = result
                best_score = score
                best_errors = (position_error, orientation_error)
            # Keep refinement, but do not lose an admissible intermediate pose
            # when reducing the solver's mixed SE(3) norm violates an FK budget.
            if feasible is not None:
                return feasible
        assert best_result is not None
        if self._bounded_fallback:
            fallback_seeds = [reference, best_result.q[:self._n], *seeds[1:]]
            fallback = self._solve_bounded(target, fallback_seeds, record)
            if fallback.success or np.hypot(
                    fallback.position_error_m / self._position_tolerance_m,
                    fallback.orientation_error_rad / self._orientation_tolerance_rad) < best_score:
                return fallback
        q = np.asarray(best_result.q[:self._n], dtype=np.float64).copy()
        return IkSolution(
            False,
            float(best_result.error),
            q,
            *best_errors,
        )


    def _solve_bounded(self, target, seeds, record):
        """Bounded, tolerance-normalized fallback; cancellation runs on every FK.

        Numerical optimizer success is diagnostic only. Admission always uses
        independent FK budgets and joint limits, including intermediate results.
        """
        import pinocchio as pin
        if self._least_squares is None:
            self.prepare_bounded_solver()
        least_squares = self._least_squares
        data = self._model.createData()
        scale = np.r_[np.full(3, self._position_tolerance_m),
                      np.full(3, self._orientation_tolerance_rad)]
        best = None
        best_score = float("inf")
        seen = set()
        for seed in seeds:
            seed = np.clip(np.asarray(seed, dtype=float), self._lower, self._upper)
            if not np.isfinite(seed).all() or seed.tobytes() in seen:
                continue
            seen.add(seed.tobytes())
            feasible = None
            feasible_score = float("inf")
            evaluations = 0
            started = time.monotonic()

            def residual(q):
                nonlocal best, best_score, feasible, feasible_score, evaluations
                if self.cancel_check is not None:
                    self.cancel_check()
                evaluations += 1
                full = self._pad_q_for_model(self._model, q, self._n)
                pin.forwardKinematics(self._model, data, full)
                pin.updateFramePlacements(self._model, data)
                pose = data.oMf[self._end_frame_id]
                error = np.r_[pose.translation - target.translation,
                              pin.log3(target.rotation.T @ pose.rotation)]
                pe, oe = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
                valid = (np.isfinite(q).all() and np.isfinite(error).all()
                         and np.all(q >= self._lower) and np.all(q <= self._upper)
                         and pe <= self._position_tolerance_m
                         and oe <= self._orientation_tolerance_rad)
                score = float(np.linalg.norm(error / scale))
                solution = IkSolution(bool(valid), float(np.linalg.norm(
                    pin.log6(pose.inverse() * target).vector)), q.copy(), pe, oe)
                if score < best_score:
                    best, best_score = solution, score
                if valid and score < feasible_score:
                    feasible, feasible_score = solution, score
                return error / scale

            try:
                fit = least_squares(residual, seed, bounds=(self._lower, self._upper),
                                    method="trf", max_nfev=500,
                                    ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception as error:
                if record is not None:
                    record['attempts'].append(dict(method='bounded_least_squares',
                        seed=seed.tolist(), elapsed_s=time.monotonic()-started,
                        termination='interrupted', reason=str(error), evaluations=evaluations))
                raise
            chosen = feasible if feasible is not None else best
            if record is not None:
                record['attempts'].append(dict(method='bounded_least_squares',
                    seed=seed.tolist(), elapsed_s=time.monotonic()-started,
                    iterations=None, evaluations=evaluations, nfev=int(fit.nfev),
                    termination=str(fit.message), solver_success=bool(fit.success),
                    accepted=feasible is not None, reason='OK' if feasible is not None else 'FK_TOLERANCE',
                    error=chosen.error, position_error_m=chosen.position_error_m,
                    orientation_error_rad=chosen.orientation_error_rad,
                    joints=chosen.joints.tolist()))
            if feasible is not None:
                return feasible
        if best is None:
            raise ValueError('NO_FINITE_IK_SEED')
        return best


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


def reverse_approach_retreat(ik_checker, pre6d, grasp6d, retreat6d,
                             pre_joints, approach_waypoints, cartesian_cfg):
    """Reuse a validated IK branch only for the same Cartesian segment.

    Returns None to request ordinary IK if reuse is inapplicable or invalid.
    Collision and time-parameterized velocity checks remain with the caller.
    """
    if not np.allclose(pre6d, retreat6d, rtol=0., atol=1e-9):
        return None
    if len(approach_waypoints) != int(cartesian_cfg['waypoint_count']):
        return None
    chain = np.vstack([pre_joints, approach_waypoints]).astype(float)
    lower, upper = ik_checker.joint_limits()
    if (not np.isfinite(chain).all() or np.any(chain < lower)
            or np.any(chain > upper)):
        return None
    if np.max(np.abs(np.diff(chain, axis=0))) > np.deg2rad(cartesian_cfg['max_joint_step_deg']):
        return None
    reverse = chain[-2::-1].copy()
    # Include the grasp start: the actual execution still has its feedback gate.
    for index, joints in enumerate(np.vstack([chain[-1], reverse])):
        if ik_checker.cancel_check is not None:
            ik_checker.cancel_check()
        pose = _cartesian_waypoint_pose(grasp6d, retreat6d, index / len(reverse))
        position, rotation = ik_checker.fk_errors(pose, joints)
        if (not np.isfinite([position, rotation]).all()
                or position > cartesian_cfg['ik_position_tolerance_m']
                or rotation > np.deg2rad(cartesian_cfg['ik_orientation_tolerance_deg'])):
            return None
    return tuple(reverse)


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
