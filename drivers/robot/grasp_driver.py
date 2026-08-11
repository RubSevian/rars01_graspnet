"""Small grasp-side helper for reBotArm visual grasping.

The SDK owns arm connection, mode switching, Cartesian planning, gravity
compensation, and the control loop. This module provides only the extra
gripper and pose helpers used by the vision workflows.

GraspDriver:
  start(): start SDK control and attach gripper tick handling.
  open_gripper(): open to a requested jaw distance.
  grasp(): close with force control and report object contact.
  release_gripper(): open and return the gripper to closed rest.
  get_gripper_state(): return cached position, velocity, and torque.
  get_tcp_pose(): return the current TCP pose as a 4x4 matrix.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np


GRIPPER_MAX_DISTANCE_M = 0.09


def _motor_array(value: Any, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size != 7:
        raise ValueError(f"robot.rars01.{name} must contain seven values")
    return values


class _RarsMotor:
    def __init__(self, owner: "RarsRebotArm", index: int) -> None:
        self._owner = owner
        self._index = index

    def get_state(self) -> Any:
        state = self._owner._state
        if state is None:
            return None
        return SimpleNamespace(
            pos=float(state[0][self._index]),
            vel=float(state[1][self._index]),
            torq=float(state[2][self._index]),
        )


class _RarsGroup:
    """RARS motor group used by the local RARS01 motion controller."""

    def __init__(self, owner: "RarsRebotArm", name: str, indices: list[int]) -> None:
        self._owner = owner
        self.name = name
        self._indices = indices
        self._jcfgs = [SimpleNamespace(name=f"joint{i + 1}") for i in indices]
        self._mm = {
            cfg.name: _RarsMotor(owner, index)
            for cfg, index in zip(self._jcfgs, indices)
        }
        self._mit_kp = owner._kp[indices].copy()
        self._mit_kd = owner._kd[indices].copy()
        self._pv_vlim = np.full(len(indices), owner._velocity_limit, dtype=np.float64)
        self._mode = "mit"

    @property
    def num_joints(self) -> int:
        return len(self._indices)

    def enable(self) -> None:
        self._owner._enable()

    def disable(self) -> None:
        self._owner._disable()

    def mode_mit(self, kp: Any = None, kd: Any = None) -> bool:
        self._mode = "mit"
        if kp is not None:
            self._mit_kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        if kd is not None:
            self._mit_kd = np.asarray(kd, dtype=np.float64).reshape(-1)
        self._owner._set_group_mode(self._indices, "mit")
        return True

    def mode_pos_vel(self, vlim: Any = None) -> bool:
        self._mode = "pos_vel"
        if vlim is not None:
            self._pv_vlim = np.asarray(vlim, dtype=np.float64).reshape(-1)
        self._owner._set_group_mode(self._indices, "pos_vel", self._pv_vlim)
        return True

    def send_mit(self, pos: Any, vel: Any = None, kp: Any = None,
                 kd: Any = None, tau: Any = None) -> None:
        n = self.num_joints
        self._owner._update_command(
            self._indices,
            pos=np.asarray(pos, dtype=np.float64).reshape(n),
            vel=np.zeros(n) if vel is None else np.asarray(vel, dtype=np.float64).reshape(n),
            kp=self._mit_kp if kp is None else np.asarray(kp, dtype=np.float64).reshape(n),
            kd=self._mit_kd if kd is None else np.asarray(kd, dtype=np.float64).reshape(n),
            tau=np.zeros(n) if tau is None else np.asarray(tau, dtype=np.float64).reshape(n),
            send=self.name == "arm",
        )

    def send_pos_vel(self, pos: Any, vlim: Any = None) -> None:
        del vlim
        values = np.asarray(pos, dtype=np.float64).reshape(self.num_joints)
        self._owner._update_command(
            self._indices, values, np.zeros(self.num_joints),
            self._mit_kp, self._mit_kd, np.zeros(self.num_joints),
            send=self.name == "arm",
        )

    def _request_feedback(self) -> None:
        self._owner._poll_feedback()

    def get_positions(self, request_feedback: bool = True) -> np.ndarray:
        state = self._owner.get_state(request_feedback=request_feedback)[0]
        return state[self._indices]


class RarsRebotArm:
    """Transport adapter: reBot IK/control unchanged, RARS SDK underneath."""

    backend = "rars01"

    def __init__(self, config: dict[str, Any], project_root: str | Path) -> None:
        hardware = config.get("rars01", config)
        module_path = Path(hardware.get("sdk_python_path", "../rars_arm_sdk/build-python"))
        if not module_path.is_absolute():
            module_path = (Path(project_root) / module_path).resolve()
        if str(module_path) not in sys.path:
            sys.path.insert(0, str(module_path))
        try:
            import rars_arm_py as sdk
        except ImportError as exc:
            raise RuntimeError(f"rars_arm_py not found in {module_path}") from exc
        self._sdk = sdk

        sdk_cfg = sdk.ArmConfiguration()
        sdk_cfg.port_name = str(hardware.get("port", "/dev/ttyACM0"))
        sdk_cfg.baud_rate = int(hardware.get("baud_rate", 921600))
        sdk_cfg.feedback_watchdog_enabled = bool(hardware.get("feedback_watchdog_enabled", True))
        sdk_cfg.feedback_timeout_ms = int(hardware.get("feedback_watchdog_timeout_ms", 1000))
        sdk_cfg.initial_feedback_grace_ms = int(hardware.get("initial_feedback_grace_ms", 1500))
        self._kp = _motor_array(hardware.get("position_kp", [70, 120, 120, 50, 20, 20, 20]), "position_kp")
        self._kd = _motor_array(hardware.get("position_kd", [1, 1, 1, 1, 2, 1, 1]), "position_kd")
        sdk_cfg.default_kp = self._kp.tolist()
        sdk_cfg.default_kd = self._kd.tolist()
        self._rate = float(hardware.get("command_rate_hz", 100.0))
        self._velocity_limit = float(hardware.get("velocity_limit_rad_s", 0.5))
        sdk_cfg.command_rate_hz = self._rate
        mode_names = hardware.get("control_modes", ["pos_vel"] * 6 + ["mit"])
        if len(mode_names) != 7 or any(name not in ("pos_vel", "mit") for name in mode_names):
            raise ValueError("robot.rars01.control_modes must contain seven pos_vel/mit values")
        sdk_cfg.control_modes = [
            sdk.ArmControlMode.POSITION_VELOCITY
            if name == "pos_vel" else sdk.ArmControlMode.MIT
            for name in mode_names
        ]
        pv_limits = hardware.get(
            "position_velocity_limits_rad_s", [self._velocity_limit] * 6 + [2.0]
        )
        sdk_cfg.position_velocity_limits = _motor_array(
            pv_limits, "position_velocity_limits_rad_s"
        ).tolist()

        directions = _motor_array(
            hardware.get("joint_directions", [1, 1, 1, 1, 1, 1, 1]),
            "joint_directions",
        )
        if not np.all(np.isin(directions, (-1.0, 1.0))):
            raise ValueError("robot.rars01.joint_directions values must be +1 or -1")
        lower = np.empty(7, dtype=np.float64)
        upper = np.empty(7, dtype=np.float64)
        for index, requested_direction in enumerate(directions):
            motor = sdk_cfg.motor(index)
            old_direction = float(motor.direction)
            old_lower = float(motor.joint_position_min)
            old_upper = float(motor.joint_position_max)
            coordinate_sign = float(requested_direction) * old_direction
            transformed = (coordinate_sign * old_lower, coordinate_sign * old_upper)
            motor.direction = float(requested_direction)
            motor.joint_position_min = min(transformed)
            motor.joint_position_max = max(transformed)
            lower[index] = float(motor.joint_position_min)
            upper[index] = float(motor.joint_position_max)

        self._sdk_arm = sdk.RarsArm(sdk_cfg)
        urdf_path = Path(hardware.get("urdf_path", "../rars01_description/urdf/rars01.urdf"))
        if not urdf_path.is_absolute():
            urdf_path = (Path(project_root) / urdf_path).resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"RARS01 URDF not found: {urdf_path}")
        self._urdf_path = urdf_path
        self._end_effector_frame = str(hardware.get("end_effector_frame", "End_link"))
        self._feedback_timeout = float(hardware.get("feedback_timeout_s", 2.0))
        self.max_grasp_width_m = float(hardware.get("max_grasp_width_m", 0.111))
        self._joint_lower = lower
        self._joint_upper = upper
        self._limit_epsilon = float(hardware.get("command_limit_epsilon_rad", 0.01))
        if self._limit_epsilon < 0.0:
            raise ValueError("robot.rars01.command_limit_epsilon_rad must be non-negative")
        self._home = np.asarray(hardware.get("home_joints_rad", [0] * 6), dtype=np.float64)
        self._start_tolerance = float(hardware.get("start_tolerance_rad", 0.15))
        self._state: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._raw_position: Optional[np.ndarray] = None
        self._pos = np.zeros(7, dtype=np.float64)
        self._vel = np.zeros(7, dtype=np.float64)
        self._cmd_kp = self._kp.copy()
        self._cmd_kd = self._kd.copy()
        self._tau = np.zeros(7, dtype=np.float64)
        self._lock = threading.RLock()
        self._connected = False
        self._enabled = False
        self._running = False
        self._ctrl_thread: Optional[threading.Thread] = None
        self._failure: Optional[Exception] = None
        self._start_validated = False
        self.groups = {
            "arm": _RarsGroup(self, "arm", list(range(6))),
            "gripper": _RarsGroup(self, "gripper", [6]),
        }
        self._motor_map = {
            name: motor
            for group in self.groups.values()
            for name, motor in group._mm.items()
        }

    @property
    def has_gripper(self) -> bool:
        return True

    @property
    def rate(self) -> float:
        return self._rate

    def __getattr__(self, name: str) -> Any:
        if name in ("arm", "gripper") and "groups" in self.__dict__:
            return self.groups[name]
        raise AttributeError(name)

    def enable_all(self) -> None:
        self._enable()

    def disable_all(self) -> None:
        self._disable()

    def constrain_model(self, model: Any) -> None:
        """Intersect the unchanged reBot URDF limits with RARS hardware limits."""
        n = min(6, int(model.nq))
        model.lowerPositionLimit[:n] = np.maximum(
            model.lowerPositionLimit[:n], self._joint_lower[:n]
        )
        model.upperPositionLimit[:n] = np.minimum(
            model.upperPositionLimit[:n], self._joint_upper[:n]
        )
        if np.any(model.lowerPositionLimit[:n] >= model.upperPositionLimit[:n]):
            raise ValueError("reBot URDF and RARS01 hardware joint limits do not overlap")

    def load_kinematic_model(self) -> Any:
        """Load the unchanged RARS URDF while retaining the reBot algorithms."""
        import pinocchio as pin

        model = pin.buildModelFromUrdf(str(self._urdf_path))
        frame_id = model.getFrameId(self._end_effector_frame)
        if frame_id >= model.nframes:
            raise ValueError(
                f"RARS01 end-effector frame not found: {self._end_effector_frame}"
            )
        # reBot helper functions request the configured name `end_link`.
        # Add a Pinocchio-only alias; the URDF file itself remains unchanged.
        if model.getFrameId("end_link") >= model.nframes:
            source = model.frames[frame_id]
            alias = pin.Frame(
                "end_link",
                source.parentJoint,
                source.parentFrame,
                source.placement,
                pin.FrameType.OP_FRAME,
            )
            model.addFrame(alias, False)
        self.constrain_model(model)
        return model

    def configure_controller_kinematics(self, controller: Any) -> None:
        model = self.load_kinematic_model()
        controller._model = model
        controller._data = model.createData()
        controller._end_frame_id = model.getFrameId("end_link")

    def connect(self) -> None:
        if not self._connected:
            if not self._sdk_arm.connect():
                raise RuntimeError(f"RARS01 connect failed: {self._sdk_arm.last_error}")
            self._connected = True

    def _enable(self) -> None:
        if not self._enabled:
            if not self._sdk_arm.enable():
                raise RuntimeError(f"RARS01 enable failed: {self._sdk_arm.last_error}")
            self._enabled = True

    def _set_group_mode(self, indices: list[int], mode: str,
                        velocity_limits: Optional[np.ndarray] = None) -> None:
        del velocity_limits
        modes = list(self._sdk_arm.control_modes)
        selected = (self._sdk.ArmControlMode.POSITION_VELOCITY
                    if mode == "pos_vel" else self._sdk.ArmControlMode.MIT)
        if all(modes[index] == selected for index in indices):
            return
        if self._enabled:
            raise RuntimeError("Control modes can only be changed while motors are disabled")
        for index in indices:
            modes[index] = selected
        if not self._sdk_arm.set_control_modes(modes):
            raise RuntimeError(f"RARS01 mode change failed: {self._sdk_arm.last_error}")

    def _disable(self) -> None:
        if self._enabled:
            if not self._sdk_arm.disable():
                raise RuntimeError(f"RARS01 disable failed: {self._sdk_arm.last_error}")
            self._enabled = False

    def _poll_feedback(self) -> bool:
        raw = self._sdk_arm.try_read_joint_state()
        if raw is None or not all(raw.valid):
            return False
        measured_position = np.asarray(raw.position, dtype=np.float64)
        control_position = measured_position.copy()
        near_lower = (
            (control_position < self._joint_lower)
            & ((self._joint_lower - control_position) <= self._limit_epsilon)
        )
        near_upper = (
            (control_position > self._joint_upper)
            & ((control_position - self._joint_upper) <= self._limit_epsilon)
        )
        control_position[near_lower] = self._joint_lower[near_lower]
        control_position[near_upper] = self._joint_upper[near_upper]
        state = (
            control_position,
            np.asarray(raw.velocity, dtype=np.float64),
            np.asarray(raw.torque, dtype=np.float64),
        )
        with self._lock:
            self._raw_position = measured_position
            self._state = state
        return True

    def get_state(self, request_feedback: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._failure is not None:
            raise RuntimeError(f"RARS01 control loop failed: {self._failure}") from self._failure
        deadline = time.monotonic() + self._feedback_timeout
        while self._state is None and time.monotonic() < deadline:
            self._poll_feedback()
            if self._state is None:
                time.sleep(0.01)
        if request_feedback and self._state is not None and not self._running:
            self._poll_feedback()
        if self._state is None:
            raise RuntimeError("No valid feedback from all seven RARS01 motors")
        with self._lock:
            return tuple(values.copy() for values in self._state)  # type: ignore[return-value]

    def _update_command(self, indices: list[int], pos: np.ndarray, vel: np.ndarray,
                        kp: np.ndarray, kd: np.ndarray, tau: np.ndarray, send: bool) -> None:
        with self._lock:
            self._pos[indices] = pos
            self._vel[indices] = vel
            self._cmd_kp[indices] = kp
            self._cmd_kd[indices] = kd
            self._tau[indices] = tau
            if not send:
                return
            self._pos[:] = self._positions_for_command(self._pos)
            ok = self._sdk_arm.send_configured(
                self._pos.tolist(), self._vel.tolist(), self._cmd_kp.tolist(),
                self._cmd_kd.tolist(), self._tau.tolist(),
            )
        if not ok:
            raise RuntimeError(f"RARS01 command failed: {self._sdk_arm.last_error}")
        self._poll_feedback()

    def _positions_for_command(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64).reshape(7)
        below = self._joint_lower - values
        above = values - self._joint_upper
        violation = np.maximum(below, above)
        if float(np.max(violation)) > self._limit_epsilon:
            index = int(np.argmax(violation))
            raise RuntimeError(
                f"RARS01 logical joint {index + 1} target {values[index]:.6f} rad "
                f"is outside [{self._joint_lower[index]:.6f}, "
                f"{self._joint_upper[index]:.6f}]"
            )
        return np.clip(values, self._joint_lower, self._joint_upper)

    def start_control_loop(self, control_fn: Any, rate: Optional[float] = None) -> None:
        initial = self.get_state()
        if not self._start_validated:
            error = float(np.max(np.abs(initial[0][:6] - self._home)))
            if error > self._start_tolerance:
                raise RuntimeError(
                    f"RARS01 must start in home; max error={error:.3f} rad, "
                    f"limit={self._start_tolerance:.3f} rad"
                )
            self._start_validated = True
        with self._lock:
            initial_command = self._positions_for_command(initial[0])
            self._pos[:] = initial_command
        measured = self._raw_position if self._raw_position is not None else initial[0]
        print(f"[RARS01] initial feedback [rad]: {np.round(measured, 6).tolist()}")
        if not np.array_equal(measured, initial_command):
            print(
                "[RARS01] boundary-corrected first command [rad]: "
                f"{np.round(initial_command, 6).tolist()}"
            )
        self._running = True
        self._failure = None
        loop_rate = float(rate or self._rate)

        def loop() -> None:
            dt = 1.0 / loop_rate
            while self._running:
                started = time.perf_counter()
                try:
                    control_fn(self, dt)
                except Exception as exc:
                    self._failure = exc
                    self._running = False
                    return
                time.sleep(max(0.0, dt - (time.perf_counter() - started)))

        self._ctrl_thread = threading.Thread(target=loop, name="rars-rebot-control", daemon=True)
        self._ctrl_thread.start()

    def stop_control_loop(self) -> None:
        self._running = False
        if self._ctrl_thread is not None:
            self._ctrl_thread.join(timeout=2.0)
            self._ctrl_thread = None

    def disconnect(self) -> None:
        self.stop_control_loop()
        try:
            self._disable()
        finally:
            if self._connected:
                self._sdk_arm.disconnect()
                self._connected = False


class GraspDriver:
    MAX_DISTANCE_M = GRIPPER_MAX_DISTANCE_M
    _STATE_IDLE = "idle"
    _STATE_POSITION = "position"
    _STATE_CLOSING = "closing"
    _STATE_HOLDING = "holding"

    def __init__(
        self,
        arm: Any,
        controller: Any,
        gripper_config: Optional[dict] = None,
        repo_root: Optional[str] = None,
    ) -> None:
        self._arm = arm
        self._controller = controller
        self._arm_group = arm.groups.get("arm")
        self._gripper_group = arm.groups.get("gripper")
        if self._arm_group is None:
            raise ValueError("Hardware config missing groups.arm")
        if self._gripper_group is None or not arm.has_gripper:
            raise ValueError("Hardware config missing groups.gripper")
        gripper_jcfgs = getattr(self._gripper_group, "_jcfgs", [])
        if not gripper_jcfgs:
            raise ValueError("groups.gripper has no joints")
        self._gripper_name = gripper_jcfgs[0].name
        self._gripper_motor: Any = None

        from rars01_graspnet.rebot_math import compute_fk, pad_q_for_model

        self._compute_fk = compute_fk
        self._pad_q_for_model = pad_q_for_model
        self._model = arm.load_kinematic_model()
        self._end_frame_id = self._model.getFrameId("end_link")
        self._n = self._arm_group.num_joints

        # RARS01 does not read a reBot hardware YAML.  The external reBot
        # project is not a runtime dependency of this driver.
        del repo_root
        backend = str(getattr(arm, "backend", "rars01"))
        defaults = {
            "dm": {"angle_open": 5.0, "counterclockwise": True, "tau_max": 1.5, "close_torque": 1.0, "default_force": 0.30},
            "rs": {"angle_open": 5.0, "counterclockwise": False, "tau_max": 1.5, "close_torque": 1.0, "default_force": 0.30},
            "rars01": {"angle_open": 1.0, "counterclockwise": False, "tau_max": 1.5, "close_torque": 1.0, "default_force": 0.30},
        }[backend]
        gcfg = {**defaults, **((gripper_config or {}).get(backend) or {})}
        self.MAX_DISTANCE_M = float(getattr(arm, "max_grasp_width_m", GRIPPER_MAX_DISTANCE_M))
        motion_sign = 1.0 if bool(gcfg.get("counterclockwise")) else -1.0
        self._angle_open = -motion_sign * abs(float(gcfg["angle_open"]))
        self._tau_max = abs(float(gcfg["tau_max"]))
        self._open_sign = 1.0 if self._angle_open >= 0.0 else -1.0
        self._close_sign = motion_sign
        self._close_torque = self._close_sign * abs(float(gcfg["close_torque"]))
        self._default_force = self._close_sign * abs(float(gcfg["default_force"]))
        self._open_soft_limit = 0.98 * self._angle_open
        self._open_lo = min(self._open_soft_limit, 0.0)
        self._open_hi = max(self._open_soft_limit, 0.0)
        self._hard_stop_angle = self._open_sign * 0.05
        self._arrive_tol = 0.12
        self._kp_move = 5.0
        self._kd_move = 1.0
        self._kd_close = 0.5
        # A single moving jaw transmits a noticeable impulse to the wrist if
        # position holding is enabled in one control tick after contact.
        self._hold_ramp_s = max(0.0, float(gcfg.get("hold_ramp_s", 0.30)))
        self._hold_ramp_elapsed = 0.0
        self._stall_vel = 0.05
        self._startup_dist = 0.30
        self._state_lock = threading.Lock()
        self._state = self._STATE_IDLE
        self._target_pos = 0.0
        self._start_pos = 0.0
        self._contact_pos = 0.0
        self._hold_torque = self._default_force
        self._position_reached = True
        self._grasp_result: Optional[bool] = None
        self._last_gripper_state: Optional[tuple[float, float, float]] = None
        self._backend = backend
        if backend == "rars01":
            self._gripper_pivot = np.asarray(
                gcfg.get("pivot_End_link_m", [-0.130, -0.0255, 0.0]), dtype=np.float64
            )
            self._moving_inner_tip = np.asarray(
                gcfg.get("moving_inner_tip_m", [0.130, 0.0255, -0.0055]), dtype=np.float64
            )
            self._fixed_inner_tip = np.asarray(
                gcfg.get("fixed_inner_tip_End_link_m", [0.0, 0.0, -0.0055]), dtype=np.float64
            )
            self._R_grasp_tcp = np.asarray(
                gcfg.get(
                    "R_grasp_End_link",
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                ),
                dtype=np.float64,
            ).reshape(3, 3)
            geometric_max = self._rars_opening_at_angle(abs(self._angle_open))[0]
            self.MAX_DISTANCE_M = min(self.MAX_DISTANCE_M, geometric_max)

    def start(self) -> None:
        """Start the SDK arm controller and let this driver own the gripper."""
        if getattr(self._controller, "_running", False):
            return

        self._controller._has_gripper = False
        self._arm.connect()
        self._gripper_motor = self._gripper_group._mm[self._gripper_name]
        if self._arm_group:
            if self._controller._arm_control_mode == "mit":
                self._arm_group.mode_mit(
                    kp=self._arm_group._mit_kp,
                    kd=self._arm_group._mit_kd,
                )
            else:
                self._arm_group.mode_pos_vel()
            self._arm_group.enable()

        self._gripper_group.mode_mit()
        self._gripper_group.enable()
        self._prime_arm_target()
        self._prime_gripper_state()
        self._arm.start_control_loop(self._loop_cb)
        self._controller._running = True

    def _loop_cb(self, r: Any, dt: float) -> None:
        self._controller._loop_cb(r, dt)
        self.gripper_tick(dt)

    def _ensure_running(self) -> None:
        if not getattr(self._controller, "_running", False):
            raise RuntimeError("GraspDriver is not started; call grasp_driver.start() first")

    def _send_gripper_mit(
        self,
        pos: float,
        vel: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        tau: float = 0.0,
    ) -> None:
        pos_cmd = float(np.clip(pos, self._open_lo, self._open_hi))
        tau_cmd = float(np.clip(tau, -self._tau_max, self._tau_max))
        self._gripper_group.send_mit(
            np.array([pos_cmd], dtype=np.float64),
            vel=np.array([vel], dtype=np.float64),
            kp=np.array([kp], dtype=np.float64),
            kd=np.array([kd], dtype=np.float64),
            tau=np.array([tau_cmd], dtype=np.float64),
        )

    def _prime_arm_target(self) -> None:
        q_now = self._arm.get_state()[0][: self._n]
        self._controller._q_target[:] = q_now
        self._controller._qd_target[:] = 0.0

    def _prime_gripper_state(self, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._gripper_group._request_feedback()
            state = self._read_gripper_state_cached()
            if state is not None:
                with self._state_lock:
                    self._target_pos = state[0]
                    self._contact_pos = state[0]
                    self._state = self._STATE_IDLE
                    self._position_reached = True
                    self._grasp_result = None
                return
            time.sleep(0.02)

    def _read_gripper_state_cached(self) -> Optional[tuple[float, float, float]]:
        if self._gripper_motor is None:
            return self._last_gripper_state
        st = self._gripper_motor.get_state()
        if st is None:
            return self._last_gripper_state
        self._last_gripper_state = (float(st.pos), float(st.vel), float(st.torq))
        return self._last_gripper_state

    def _wait_gripper_state(self, timeout: float = 1.0) -> tuple[float, float, float]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._read_gripper_state_cached()
            if state is not None:
                return state
            time.sleep(0.02)
        raise RuntimeError("Gripper feedback is not ready")

    def get_gripper_state(self) -> tuple[float, float, float]:
        state = self._read_gripper_state_cached()
        if state is None:
            raise RuntimeError("Gripper feedback is not ready")
        return state

    def grasp_tcp_transform(self, width_m: float) -> np.ndarray:
        """Return grasp-center -> TCP transform for the current gripper."""
        T = np.eye(4, dtype=np.float64)
        if self._backend != "rars01":
            return T
        _, center = self._rars_geometry_for_width(width_m)
        T[:3, :3] = self._R_grasp_tcp
        T[:3, 3] = center
        return T

    def minimum_jaw_height(self, tcp_pose: tuple[float, ...], width_m: float,
                           include_max_open: bool = True) -> float:
        """Lowest modeled inner jaw point in base coordinates."""
        if self._backend != "rars01":
            return float(tcp_pose[2])
        from utils.transforms import pose6d_to_mat4

        _, _, moving = self._rars_geometry_for_width(width_m, return_moving=True)
        points = [self._fixed_inner_tip, moving]
        if include_max_open:
            points.append(self._rars_opening_at_angle(abs(self._angle_open))[2])
        T = pose6d_to_mat4(*tcp_pose)
        return min(float((T @ np.r_[point, 1.0])[2]) for point in points)

    def _rars_geometry_for_width(
        self, width_m: float, return_moving: bool = False
    ) -> tuple[Any, ...]:
        width = float(np.clip(width_m, 0.0, self.MAX_DISTANCE_M))
        lo, hi = 0.0, abs(self._angle_open)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self._rars_opening_at_angle(mid)[0] < width:
                lo = mid
            else:
                hi = mid
        angle = 0.5 * (lo + hi)
        _, center, moving = self._rars_opening_at_angle(angle)
        return (angle, center, moving) if return_moving else (angle, center)

    def _rars_opening_at_angle(
        self, angle_rad: float
    ) -> tuple[float, np.ndarray, np.ndarray]:
        c, s = np.cos(float(angle_rad)), np.sin(float(angle_rad))
        rotation = np.array(
            [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]],
            dtype=np.float64,
        )
        moving = self._gripper_pivot + rotation @ self._moving_inner_tip
        center = 0.5 * (self._fixed_inner_tip + moving)
        width = float(moving[2] - self._fixed_inner_tip[2])
        return width, center, moving

    def _set_position_target(self, target: float) -> None:
        with self._state_lock:
            self._target_pos = float(target)
            self._state = self._STATE_POSITION
            self._position_reached = False
            self._grasp_result = None

    def _wait_until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def _position_done(self) -> bool:
        with self._state_lock:
            return self._position_reached

    def _grasp_done(self) -> bool:
        with self._state_lock:
            return self._grasp_result is not None

    def gripper_tick(self, dt: float = 0.0) -> None:
        pos_vel_torq = self._read_gripper_state_cached()
        with self._state_lock:
            state = self._state
            target = self._target_pos
            command: Optional[tuple[float, float, float, float, float]] = None

            if state == self._STATE_POSITION:
                command = (target, 0.0, self._kp_move, self._kd_move, 0.0)
                if pos_vel_torq is not None and abs(pos_vel_torq[0] - target) < self._arrive_tol:
                    self._position_reached = True

            elif state == self._STATE_CLOSING:
                command = (0.0, 0.0, 0.0, self._kd_close, self._close_torque)
                if pos_vel_torq is not None:
                    pos, vel, _ = pos_vel_torq
                    self._contact_pos = pos
                    moved = abs(pos - self._start_pos) >= self._startup_dist
                    at_hard_stop = self._open_sign * pos < self._open_sign * self._hard_stop_angle
                    if moved and at_hard_stop:
                        self._target_pos = 0.0
                        self._state = self._STATE_POSITION
                        self._position_reached = False
                        self._grasp_result = False
                        command = (0.0, 0.0, self._kp_move, self._kd_move, 0.0)
                    elif moved and abs(vel) < self._stall_vel:
                        self._target_pos = pos
                        self._state = self._STATE_HOLDING
                        self._hold_ramp_elapsed = 0.0
                        self._grasp_result = True
                        # Keep the contact command during this first tick;
                        # Kp and holding torque are blended in below.
                        command = (pos, 0.0, 0.0, self._kd_close, self._close_torque)

            elif state == self._STATE_HOLDING:
                self._hold_ramp_elapsed += max(0.0, float(dt))
                alpha = (
                    1.0
                    if self._hold_ramp_s <= 0.0
                    else min(1.0, self._hold_ramp_elapsed / self._hold_ramp_s)
                )
                kp = alpha * self._kp_move
                kd = self._kd_close + alpha * (self._kd_move - self._kd_close)
                tau = self._close_torque + alpha * (self._hold_torque - self._close_torque)
                command = (self._target_pos, 0.0, kp, kd, tau)

        if command is not None:
            pos, vel, kp, kd, tau = command
            self._send_gripper_mit(pos, vel=vel, kp=kp, kd=kd, tau=tau)

    def open_gripper(self, distance_m: Optional[float] = None, timeout: float = 3.0) -> None:
        self._ensure_running()
        if distance_m is None:
            distance_m = self.MAX_DISTANCE_M
        d = float(np.clip(distance_m, 0.0, self.MAX_DISTANCE_M))
        raw_target = (d / self.MAX_DISTANCE_M) * self._angle_open
        target = float(np.clip(raw_target, self._open_lo, self._open_hi))

        self._set_position_target(target)
        self._wait_until(self._position_done, timeout)

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        self._ensure_running()
        start_pos, _, _ = self._wait_gripper_state()
        hold_torque = self._close_sign * float(
            np.clip(abs(force if force is not None else self._default_force), 0.05, self._tau_max)
        )
        with self._state_lock:
            self._start_pos = start_pos
            self._contact_pos = start_pos
            self._target_pos = 0.0
            self._hold_torque = hold_torque
            self._state = self._STATE_CLOSING
            self._position_reached = False
            self._grasp_result = None

        if not self._wait_until(self._grasp_done, timeout):
            with self._state_lock:
                if self._grasp_result is None:
                    self._target_pos = self._contact_pos
                    self._state = self._STATE_POSITION
                    self._position_reached = False
                    self._grasp_result = False

        with self._state_lock:
            return bool(self._grasp_result)

    def release_gripper(self, timeout: float = 4.0) -> None:
        self._ensure_running()
        self.open_gripper(timeout=min(2.0, timeout))
        self._set_position_target(0.0)
        self._wait_until(self._position_done, timeout)

    def get_tcp_pose(self) -> np.ndarray:
        q_arm = self._arm.get_state(request_feedback=False)[0][: self._n]
        q = self._pad_q_for_model(self._model, q_arm, self._n)
        pos, rot, _ = self._compute_fk(self._model, q, self._end_frame_id)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T
