from types import SimpleNamespace

import numpy as np
import pytest

from rars01_graspnet.robot import RarsRobot


class _FeedbackArm:
    def __init__(self):
        self.sent = []
        self.read_count = 0

    def is_enabled(self):
        return True

    def send_position_targets(self, positions):
        self.sent.append(list(positions))
        return True

    def try_read_joint_state(self):
        self.read_count += 1
        if self.read_count == 1:
            return None
        return SimpleNamespace(
            position=[0.0] * 7,
            velocity=[0.0] * 7,
            torque=[0.0] * 7,
            valid=[True] * 7,
            error=[0] * 7,
            motor_id=list(range(1, 8)),
            mos_temperature=[25.0] * 7,
            rotor_temperature=[25.0] * 7,
        )


def test_read_state_repeats_hold_to_request_missing_feedback():
    robot = RarsRobot.__new__(RarsRobot)
    robot.arm = _FeedbackArm()
    robot.feedback_timeout_s = 0.2
    robot._last_command = None

    state = robot.read_state(request_position=[0.1] * 7)

    assert state.position.tolist() == [0.0] * 7
    assert robot.arm.sent
    assert robot._last_command.tolist() == [0.1] * 7


class _SettlingArm:
    def __init__(self, arm_positions):
        self.positions = iter(arm_positions)
        self.last = np.asarray(arm_positions[-1], dtype=float)

    def send_position_targets(self, _positions):
        return True

    def try_read_joint_state(self):
        try:
            self.last = np.asarray(next(self.positions), dtype=float)
        except StopIteration:
            pass
        return SimpleNamespace(
            position=np.concatenate((self.last, [0.0])).tolist(),
            velocity=[0.0] * 7, torque=[0.0] * 7, valid=[True] * 7,
            error=[0] * 7, motor_id=list(range(1, 8)),
            mos_temperature=[25.0] * 7, rotor_temperature=[25.0] * 7,
        )


def _settling_robot(positions):
    robot = RarsRobot.__new__(RarsRobot)
    robot.arm = _SettlingArm(positions)
    robot._last_command = None
    robot.target_tolerance_rad = 0.08
    robot.target_settle_timeout_s = 0.05
    robot.target_stable_samples = 2
    return robot


def test_joint_target_waits_for_stable_feedback_instead_of_first_sample():
    robot = _settling_robot([[0.10] * 6, [0.04] * 6, [0.03] * 6])

    state = robot._settle_joint_target(np.zeros(7), np.zeros(6), rate_hz=1000)

    np.testing.assert_allclose(state.position[:6], [0.03] * 6)


def test_joint_target_reports_persistent_error_per_joint():
    robot = _settling_robot([[0.0, 0.0, 0.102, 0.0, 0.0, 0.0]])

    with pytest.raises(RuntimeError, match=r"joint3 error=\+0.102 rad"):
        robot._settle_joint_target(np.zeros(7), np.zeros(6), rate_hz=1000)
