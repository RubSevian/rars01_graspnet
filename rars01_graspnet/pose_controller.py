"""Контроллер позы RARS01: локальная кинематика и поток целей POS/VEL.

GraspDriver запускает transport и управляет гриппером в MIT. Здесь находятся
совместимые с рабочим baseline Cartesian-движения, sender и возврат Home.
Коллизии этот контроллер пока не проверяет — см. GO2_COLLISION_PLAN.md.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .pinocchio_math import (
    IKParams, cartesian_geodesic, compute_fk, get_end_effector_frame_id,
    minimum_jerk, pad_q_for_model, pos_rot_to_se3, solve_ik, track_poses,
)


class RarsPoseController:
    def __init__(self, arm, dt=0.01, arm_control_mode="posvel"):
        if arm_control_mode != "posvel":
            raise ValueError("RARS01 pose controller supports POS/VEL arm control only")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Controller dt must be positive and finite")
        self.arm = arm
        self._arm_group = arm.groups["arm"]
        self._n = self._arm_group.num_joints
        self._dt = float(dt)
        self._arm_control_mode = arm_control_mode
        self._has_gripper = False
        self._model = arm.load_kinematic_model()
        self._data = self._model.createData()
        self._end_frame_id = get_end_effector_frame_id(self._model)
        self._ik_solver_params = IKParams()
        self._clik_params = IKParams(step_size=0.8)
        self._q_target = np.zeros(self._n)
        self._qd_target = np.zeros(self._n)
        self._running = False
        self._traj = []
        self._moving = False
        self._send_thread = None
        self._stop_send = threading.Event()
        self._vlim_override = None

    def _loop_cb(self, _arm, _dt):
        """Called by the 100 Hz transport loop; gripper is owned by GraspDriver."""
        vlim = self._vlim_override
        if vlim is None:
            vlim = self._arm_group._pv_vlim
        self._arm_group.send_pos_vel(self._q_target, vlim=vlim)

    def move_to_traj(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0):
        if not self._running:
            return False
        q_start = pad_q_for_model(self._model, self.arm.get_state()[0], self._n)
        target = pos_rot_to_se3(np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw)
        result = solve_ik(self._model, self._data, self._end_frame_id, target,
                          q_start, self._ik_solver_params, self._n)
        if not result.success:
            print(f"[RARS01/Trajectory] IK failed: error={result.error:.4f}")
            return False
        start = compute_fk(self._model, q_start)[2]
        end = compute_fk(self._model, pad_q_for_model(self._model, result.q, self._n))[2]
        if duration <= 0:
            duration = max(1.0, float(np.linalg.norm(target.translation - start[:3, 3])) / 0.1)
        points, _ = track_poses(
            self._model, self._end_frame_id,
            cartesian_geodesic(start, end, duration, self._dt), q_start, self._clik_params,
        )
        if not points:
            print("[RARS01/Trajectory] Empty Cartesian trajectory")
            return False
        self._stop_sender()
        self._traj = [q[:self._n].copy() for q in points]
        self._moving = True
        self._stop_send.clear()
        self._send_thread = threading.Thread(target=self._send_loop, args=(duration,), daemon=True)
        self._send_thread.start()
        return True

    def _stop_sender(self):
        """Ensure the old sender cannot overwrite targets of another motion."""
        self._stop_send.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=5.0)
            if self._send_thread.is_alive():
                raise RuntimeError("Trajectory sender did not stop")
        self._moving = False

    def _send_loop(self, duration):
        interval = duration / len(self._traj) if self._traj else self._dt
        try:
            for point in self._traj:
                if self._stop_send.is_set():
                    return
                self._q_target[:] = point
                time.sleep(interval)
        finally:
            self._moving = False

    def safe_home(self, max_vel=0.5, send_freq=50.0, settle_thresh=0.01, timeout=15.0):
        """Baseline joint minimum-jerk return plus up to 3 s feedback settling.

        This is a kinematic return, not a collision-checked recovery route.
        A timeout is reported before the caller's disconnect policy runs.
        """
        if not self._running:
            return
        self._stop_sender()
        start = self.arm.get_state()[0][:self._n].copy()
        target = np.asarray(self.arm._home, dtype=np.float64).reshape(self._n)
        maximum = float(np.max(np.abs(target - start)))
        if maximum < 0.01:
            return
        duration = 2.0 * maximum / max_vel
        count = max(2, int(duration * send_freq))
        phases = np.linspace(0, duration, count) / duration
        trajectory = start + (target - start) * minimum_jerk(phases[:, None])
        deadline = time.monotonic() + timeout
        self._vlim_override = np.full(self._n, max_vel)
        try:
            for point in trajectory:
                if time.monotonic() > deadline:
                    print("[RARS01/Home] trajectory timeout")
                    break
                self._q_target[:] = point
                time.sleep(duration / count)
            self._q_target[:] = target
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                q_now = self.arm.get_state()[0][:self._n]
                if np.max(np.abs(q_now - target)) < settle_thresh:
                    return
                time.sleep(self._dt)
            print("[RARS01/Home] feedback settling timed out")
        finally:
            self._vlim_override = None

    def end(self):
        if not self._running:
            return
        self.safe_home()
        self.arm.disconnect()
        self._running = False
