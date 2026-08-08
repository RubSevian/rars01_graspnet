from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time

import numpy as np

from .contracts import Header, RobotState
from .trajectory import minimum_jerk_samples


@dataclass(frozen=True)
class GripperCloseResult:
    commanded_angle_rad: float
    measured_angle_rad: float
    measured_torque_nm: float
    elapsed_s: float


class RarsRobot:
    """Thin wrapper around the existing rars_arm_py hardware SDK."""

    def __init__(self, port: str, baud_rate: int, feedback_timeout_s: float = 2.0,
                 position_kp=None, position_kd=None, *,
                 feedback_watchdog_enabled: bool = True,
                 feedback_watchdog_timeout_ms: int = 1000,
                 initial_feedback_grace_ms: int = 1500,
                 target_tolerance_rad: float = 0.08,
                 target_settle_timeout_s: float = 1.5,
                 target_stable_samples: int = 3):
        try:
            import rars_arm_py as sdk
        except ImportError as exc:
            raise RuntimeError(
                "rars_arm_py is not importable. Build ../rars_arm_sdk with "
                "-DRARS_ARM_BUILD_PYTHON=ON and add its build directory to PYTHONPATH."
            ) from exc
        config = sdk.ArmConfiguration()
        config.port_name = port
        config.baud_rate = int(baud_rate)
        config.feedback_watchdog_enabled = bool(feedback_watchdog_enabled)
        config.feedback_timeout_ms = int(feedback_watchdog_timeout_ms)
        config.initial_feedback_grace_ms = int(initial_feedback_grace_ms)
        if config.feedback_timeout_ms <= 0 or config.initial_feedback_grace_ms <= 0:
            raise ValueError("SDK feedback watchdog timeouts must be positive")
        if position_kp is not None:
            config.default_kp = _motor_values(position_kp, "robot.control.position_kp")
        if position_kd is not None:
            config.default_kd = _motor_values(position_kd, "robot.control.position_kd")
        self.sdk = sdk
        self.arm = sdk.RarsArm(config)
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.target_tolerance_rad = float(target_tolerance_rad)
        self.target_settle_timeout_s = float(target_settle_timeout_s)
        self.target_stable_samples = int(target_stable_samples)
        if (self.target_tolerance_rad <= 0 or self.target_settle_timeout_s <= 0
                or self.target_stable_samples < 1):
            raise ValueError("Motion completion tolerance, timeout and stable samples must be positive")
        self._cleanup_return = None
        self._last_command = None

    def connect(self) -> None:
        if not self.arm.connect():
            raise RuntimeError(self.arm.last_error)

    def enable(self) -> None:
        """Explicitly enable every motor. Never called implicitly."""
        if not self.arm.enable():
            raise RuntimeError(self.arm.last_error)

    def disable(self) -> None:
        if self.arm.is_enabled() and not self.arm.disable():
            raise RuntimeError(self.arm.last_error)

    def current_joints(self) -> np.ndarray:
        return self.read_state(request_position=self._last_command).position.copy()

    def arm_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = [], []
        configuration = self.arm.configuration
        for index in range(6):
            motor = configuration.motor(index)
            lower.append(float(motor.joint_position_min))
            upper.append(float(motor.joint_position_max))
        return np.asarray(lower), np.asarray(upper)

    def position_gains(self) -> tuple[np.ndarray, np.ndarray]:
        configuration = self.arm.configuration
        return (
            np.asarray(configuration.default_kp, dtype=np.float64),
            np.asarray(configuration.default_kd, dtype=np.float64),
        )

    def move_joints(self, target, *, duration_s: float, rate_hz: float = 50.0,
                    max_joint_step_rad: float = 0.025) -> np.ndarray:
        """Stream a smooth six-joint target while preserving gripper position."""
        if not self.arm.is_enabled():
            raise RuntimeError("Cannot move while motors are disabled")
        state = self.read_state(request_position=self._last_command)
        start = state.position.copy()
        # When an object prevents the gripper from reaching its force-producing
        # target, preserve that commanded target instead of the measured angle.
        if self._last_command is not None:
            start[6] = self._last_command[6]
        target_arm = np.asarray(target, dtype=np.float64).reshape(-1)
        if target_arm.size != 6:
            raise ValueError("Expected six arm joint targets")
        target_all = start.copy()
        target_all[:6] = target_arm
        samples = minimum_jerk_samples(
            start, target_all, duration_s, rate_hz, max_joint_step_rad
        )
        period = 1.0 / float(rate_hz)
        next_tick = time.monotonic()
        for sample in samples:
            if not self.arm.send_position_targets(sample.tolist()):
                raise self._command_error("Arm trajectory command failed")
            self._last_command = sample.copy()
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return self._settle_joint_target(target_all, target_arm, rate_hz).position.copy()

    def follow_joint_trajectory(self, arm_points, *, duration_s: float,
                                max_joint_step_rad: float = 0.02) -> np.ndarray:
        """Execute an already planned continuous six-joint trajectory."""
        if not self.arm.is_enabled():
            raise RuntimeError("Cannot move while motors are disabled")
        points = np.asarray(arm_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 6 or len(points) < 2:
            raise ValueError("Expected at least two six-joint trajectory points")
        if duration_s <= 0 or max_joint_step_rad <= 0:
            raise ValueError("Trajectory duration and maximum joint step must be positive")
        state = self.read_state(request_position=self._last_command)
        gripper = (float(self._last_command[6]) if self._last_command is not None
                   else float(state.position[6]))
        # Validate continuity against the last commanded endpoint, not against
        # lagging feedback. The latter may legally be inside the wider settling
        # tolerance and would otherwise create a false branch-jump failure
        # between two geometrically continuous Cartesian segments.
        command_start = (self._last_command[:6] if self._last_command is not None
                         else state.position[:6])
        all_arm = np.vstack((command_start, points))
        largest_step = float(np.max(np.abs(np.diff(all_arm, axis=0))))
        if largest_step > float(max_joint_step_rad) + 1e-9:
            raise RuntimeError(
                f"Precomputed trajectory step {largest_step:.4f} rad exceeds "
                f"{max_joint_step_rad:.4f} rad"
            )
        period = float(duration_s) / len(points)
        next_tick = time.monotonic()
        target_all = np.empty(7, dtype=np.float64)
        target_all[6] = gripper
        for point in points:
            target_all[:6] = point
            if not self.arm.send_position_targets(target_all.tolist()):
                raise self._command_error("Cartesian trajectory command failed")
            self._last_command = target_all.copy()
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        rate_hz = len(points) / float(duration_s)
        return self._settle_joint_target(target_all, points[-1], rate_hz).position.copy()

    def _settle_joint_target(self, target_all: np.ndarray, target_arm: np.ndarray,
                             rate_hz: float) -> RobotState:
        """Hold the endpoint until several fresh feedback samples are in tolerance."""
        deadline = time.monotonic() + self.target_settle_timeout_s
        period = 1.0 / float(rate_hz)
        stable = 0
        last_state = None
        while time.monotonic() < deadline:
            if not self.arm.send_position_targets(target_all.tolist()):
                raise self._command_error("Final joint hold command failed")
            self._last_command = target_all.copy()
            raw = self.arm.try_read_joint_state()
            if raw is not None and all(raw.valid[:6]):
                last_state = _robot_state(raw)
                error = np.abs(last_state.position[:6] - target_arm)
                stable = stable + 1 if float(np.max(error)) <= self.target_tolerance_rad else 0
                if stable >= self.target_stable_samples:
                    return last_state
            time.sleep(period)
        if last_state is None:
            raise RuntimeError(
                f"Joint target settling received no complete feedback for "
                f"{self.target_settle_timeout_s:.2f}s"
            )
        errors = last_state.position[:6] - target_arm
        worst = int(np.argmax(np.abs(errors)))
        raise RuntimeError(
            f"Joint target did not settle within {self.target_settle_timeout_s:.2f}s; "
            f"joint{worst + 1} error={errors[worst]:+.3f} rad "
            f"({np.rad2deg(errors[worst]):+.2f} deg), per_joint_rad="
            f"{np.round(errors, 4).tolist()}, tolerance={self.target_tolerance_rad:.3f} rad"
        )

    def close_gripper_until_torque(
        self, *, torque_limit_nm: float, kp: float, kd: float,
        close_rate_rad_s: float, minimum_angle_rad: float,
        timeout_s: float, stable_samples: int = 2, rate_hz: float = 50.0,
    ) -> GripperCloseResult:
        """Close motor 7 slowly and stop at an absolute feedback torque limit."""
        if not self.arm.is_enabled():
            raise RuntimeError("Cannot close gripper while motors are disabled")
        if torque_limit_nm <= 0 or close_rate_rad_s <= 0 or timeout_s <= 0 or rate_hz <= 0:
            raise ValueError("Gripper torque, rate, timeout and command rate must be positive")
        if stable_samples < 1:
            raise ValueError("stable_samples must be at least one")
        motor = self.arm.configuration.motor(6)
        minimum = float(minimum_angle_rad)
        if minimum < float(motor.joint_position_min) or minimum > float(motor.joint_position_max):
            raise ValueError("Gripper minimum angle is outside SDK joint limits")
        if torque_limit_nm > float(motor.joint_torque_max):
            raise ValueError("Gripper torque limit exceeds the SDK joint torque limit")

        initial = self.read_state(request_position=self._last_command)
        command = initial.position.copy()
        if self._last_command is not None:
            command[:6] = self._last_command[:6]
            command[6] = self._last_command[6]
        gains_kp, gains_kd = self.position_gains()
        gains_kp[6], gains_kd[6] = float(kp), float(kd)
        zeros = np.zeros(7, dtype=np.float64)
        period = 1.0 / float(rate_hz)
        started = time.monotonic()
        last_feedback = started
        consecutive = 0

        while time.monotonic() - started < float(timeout_s):
            command[6] = max(minimum, command[6] - float(close_rate_rad_s) * period)
            send = getattr(self.arm, "send_configured", self.arm.send_mit)
            if not send(command.tolist(), zeros.tolist(), gains_kp.tolist(),
                        gains_kd.tolist(), zeros.tolist()):
                raise self._command_error("Gripper MIT command failed")
            self._last_command = command.copy()
            time.sleep(period)
            raw = self.arm.try_read_joint_state()
            if raw is None:
                if time.monotonic() - last_feedback > self.feedback_timeout_s:
                    raise RuntimeError("No feedback while closing gripper")
                continue
            if not all(raw.valid):
                continue
            last_feedback = time.monotonic()
            state = _robot_state(raw)
            torque = abs(float(state.effort[6]))
            consecutive = consecutive + 1 if torque >= float(torque_limit_nm) else 0
            if consecutive >= int(stable_samples):
                return GripperCloseResult(
                    commanded_angle_rad=float(command[6]),
                    measured_angle_rad=float(state.position[6]),
                    measured_torque_nm=torque,
                    elapsed_s=time.monotonic() - started,
                )
            if command[6] <= minimum + 1e-9:
                raise RuntimeError(
                    f"Gripper reached minimum angle {minimum:.4f} rad without reaching "
                    f"{torque_limit_nm:.3f} Nm (last={torque:.3f} Nm)"
                )
        raise RuntimeError(
            f"Gripper close timeout after {timeout_s:.1f}s before reaching {torque_limit_nm:.3f} Nm"
        )

    def move_gripper(self, target_angle_rad: float, *, duration_s: float,
                     rate_hz: float = 50.0, max_step_rad: float = 0.01) -> np.ndarray:
        """Move only motor 7 while holding the six measured arm joints."""
        if not self.arm.is_enabled():
            raise RuntimeError("Cannot move gripper while motors are disabled")
        motor = self.arm.configuration.motor(6)
        target = float(target_angle_rad)
        if target < float(motor.joint_position_min) or target > float(motor.joint_position_max):
            raise ValueError("Gripper target is outside SDK joint limits")
        state = self.read_state(request_position=self._last_command)
        start = state.position.copy()
        if self._last_command is not None:
            start[:6] = self._last_command[:6]
        target_all = start.copy()
        target_all[6] = target
        samples = minimum_jerk_samples(start, target_all, duration_s, rate_hz, max_step_rad)
        period = 1.0 / float(rate_hz)
        next_tick = time.monotonic()
        for sample in samples:
            if not self.arm.send_position_targets(sample.tolist()):
                raise self._command_error("Gripper position command failed")
            self._last_command = sample.copy()
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        final_state = self.read_state(request_position=target_all)
        error = abs(float(final_state.position[6]) - target)
        if error > 0.08:
            raise RuntimeError(f"Gripper target was not reached; error={error:.3f} rad")
        return final_state.position.copy()

    def hold_positions(self, positions) -> None:
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if values.size != 7:
            raise ValueError("Expected all seven motor positions for hold")
        if not self.arm.send_position_targets(values.tolist()):
            raise self._command_error("Position hold command failed")
        self._last_command = values.copy()

    def _command_error(self, context: str) -> RuntimeError:
        """Attach SDK communication counters to a failed command."""
        message = f"{context}: {self.arm.last_error}"
        try:
            status = self.arm.communication_status()
            message += (
                f" (connected={status.connected}, enabled={status.enabled}, "
                f"watchdog_tripped={status.watchdog_tripped}, "
                f"feedback_age_ms={status.feedback_age_ms}, "
                f"valid_frames={status.valid_frames}, invalid_frames={status.invalid_frames}, "
                f"read_timeouts={status.read_timeouts}, read_errors={status.read_errors})"
            )
        except Exception:
            pass
        return RuntimeError(message)

    @contextmanager
    def continuous_hold(self, positions, *, rate_hz: float = 50.0):
        """Keep a stationary arm commanded while camera/CUDA work blocks."""
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if values.size != 7:
            raise ValueError("Expected all seven motor positions for continuous hold")
        if rate_hz <= 0:
            raise ValueError("Continuous hold rate must be positive")
        stop = threading.Event()
        failure: list[Exception] = []

        def run() -> None:
            period = 1.0 / float(rate_hz)
            while not stop.is_set():
                started = time.monotonic()
                try:
                    self.hold_positions(values)
                    # During CUDA inference this is the only thread touching the
                    # robot. Consume the latest feedback as part of the same
                    # command cycle, matching the SDK's intended write/read use.
                    self.arm.try_read_joint_state()
                except Exception as exc:
                    failure.append(exc)
                    stop.set()
                    return
                stop.wait(max(0.0, period - (time.monotonic() - started)))

        thread = threading.Thread(target=run, name="rars01-position-hold", daemon=True)
        thread.start()

        def check() -> None:
            if failure:
                raise RuntimeError(f"Continuous position hold failed: {failure[0]}") from failure[0]

        try:
            yield check
            check()
        finally:
            stop.set()
            thread.join(timeout=max(1.0, 2.0 / float(rate_hz)))

    def set_cleanup_return(self, *, home, via, via_duration_s: float,
                           home_duration_s: float, rate_hz: float,
                           max_joint_step_rad: float) -> None:
        """Register a best-effort via->home return executed before disable()."""
        self.set_cleanup_route(
            waypoints=(via, home),
            durations_s=(via_duration_s, home_duration_s),
            rate_hz=rate_hz,
            max_joint_step_rad=max_joint_step_rad,
        )

    def set_cleanup_route(self, *, waypoints, durations_s, rate_hz: float,
                          max_joint_step_rad: float) -> None:
        """Replace the best-effort route followed before the motors are disabled."""
        route = tuple(np.asarray(point, dtype=np.float64).reshape(6) for point in waypoints)
        durations = tuple(float(value) for value in durations_s)
        if len(route) != len(durations):
            raise ValueError("Cleanup waypoints and durations must have equal length")
        if any(value <= 0 for value in durations):
            raise ValueError("Cleanup durations must be positive")
        self._cleanup_return = {
            "waypoints": route,
            "durations_s": durations,
            "rate_hz": float(rate_hz),
            "max_joint_step_rad": float(max_joint_step_rad),
        }

    def clear_cleanup_route(self) -> None:
        """Clear cleanup only after an explicit successful home motion."""
        self._cleanup_return = None

    def read_state(self, request_position=None) -> RobotState:
        if not self.arm.is_enabled():
            raise RuntimeError(
                "Motor feedback is unavailable while RARS01 motors are disabled. "
                "Call enable() explicitly only after making the robot safe."
            )
        request = None
        if request_position is not None:
            request = np.asarray(request_position, dtype=np.float64).reshape(-1)
            if request.size != 7:
                raise ValueError("Feedback request position must contain seven values")
        deadline = time.monotonic() + self.feedback_timeout_s
        next_request = 0.0
        last_invalid = None
        while time.monotonic() < deadline:
            now = time.monotonic()
            if request is not None and now >= next_request:
                if not self.arm.send_position_targets(request.tolist()):
                    raise RuntimeError(
                        f"Failed while requesting joint feedback: {self.arm.last_error}"
                    )
                self._last_command = request.copy()
                next_request = now + 0.02
            state = self.arm.try_read_joint_state()
            if state is not None and all(state.valid[:6]):
                return _robot_state(state)
            if state is not None:
                last_invalid = state
            time.sleep(0.01)
        status = self.arm.communication_status()
        details = (
            f"connected={status.connected}, enabled={status.enabled}, "
            f"watchdog_tripped={status.watchdog_tripped}, "
            f"feedback_age_ms={status.feedback_age_ms}, valid_frames={status.valid_frames}, "
            f"invalid_frames={status.invalid_frames}, read_timeouts={status.read_timeouts}, "
            f"read_errors={status.read_errors}"
        )
        if last_invalid is not None:
            details += (
                f", motor_ids={list(last_invalid.motor_id)}, "
                f"valid={list(last_invalid.valid)}, errors={list(last_invalid.error)}"
            )
        raise RuntimeError(f"No valid feedback from all six arm joints ({details})")

    def close(self) -> None:
        cleanup = self._cleanup_return
        self._cleanup_return = None
        if cleanup is not None and self.arm.is_enabled():
            try:
                for waypoint, duration_s in zip(
                    cleanup["waypoints"], cleanup["durations_s"], strict=True
                ):
                    self.move_joints(
                        waypoint, duration_s=duration_s,
                        rate_hz=cleanup["rate_hz"],
                        max_joint_step_rad=cleanup["max_joint_step_rad"],
                    )
                print(f"Cleanup return route completed ({len(cleanup['waypoints'])} waypoints)")
            except Exception as exc:
                print(f"[warn] cleanup return to home failed: {exc}")
        try:
            self.disable()
        except Exception:
            # Best-effort shutdown must still close the serial transport.
            pass
        self.arm.disconnect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()


def robot_from_config(config: dict) -> RarsRobot:
    robot = config["robot"]
    control = robot.get("control", {})
    completion = control.get("motion_completion", {})
    return RarsRobot(
        robot["port"], robot["baud_rate"], robot.get("feedback_timeout_s", 2.0),
        position_kp=control.get("position_kp"), position_kd=control.get("position_kd"),
        feedback_watchdog_enabled=robot.get("feedback_watchdog_enabled", True),
        feedback_watchdog_timeout_ms=robot.get("feedback_watchdog_timeout_ms", 1000),
        initial_feedback_grace_ms=robot.get("initial_feedback_grace_ms", 1500),
        target_tolerance_rad=completion.get("target_tolerance_rad", 0.08),
        target_settle_timeout_s=completion.get("settle_timeout_s", 1.5),
        target_stable_samples=completion.get("stable_samples", 3),
    )


def _motor_values(values, name: str) -> list[float]:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size != 7:
        raise ValueError(f"{name} must contain seven values, got {result.size}")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return result.tolist()


def _robot_state(state) -> RobotState:
    return RobotState(
        header=Header(time.time_ns(), "base_link"),
        names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"),
        position=np.asarray(state.position, dtype=np.float64),
        velocity=np.asarray(state.velocity, dtype=np.float64),
        effort=np.asarray(state.torque, dtype=np.float64),
        valid=np.asarray(state.valid, dtype=bool),
        error=np.asarray(state.error, dtype=np.uint8),
        mos_temperature=np.asarray(state.mos_temperature, dtype=np.float64),
        rotor_temperature=np.asarray(state.rotor_temperature, dtype=np.float64),
    )
