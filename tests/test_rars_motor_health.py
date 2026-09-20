from types import SimpleNamespace
import threading

import numpy as np
import pytest

from drivers.robot.grasp_driver import RarsArmAdapter


class _SdkArm:
    def __init__(self, statuses):
        self._statuses = statuses

    def try_read_joint_state(self):
        return SimpleNamespace(
            position=[0.0] * 7,
            velocity=[0.0] * 7,
            torque=[0.0] * 7,
            valid=[True] * 7,
            error=self._statuses,
        )

    def communication_status(self):
        return SimpleNamespace(watchdog_tripped=False, stm32_watchdog_tripped=False)


def _adapter(statuses):
    arm = RarsArmAdapter.__new__(RarsArmAdapter)
    arm._sdk_arm = _SdkArm(statuses)
    arm._lock = threading.RLock()
    arm._joint_lower = np.full(7, -2.0)
    arm._joint_upper = np.full(7, 2.0)
    arm._limit_epsilon = 0.01
    arm._enabled = True
    arm._state = None
    arm._raw_position = None
    arm._feedback_sequence = 0
    arm._feedback_monotonic = None
    arm._last_feedback_poll_monotonic = None
    arm._last_feedback_valid = False
    arm._motor_enabled_seen = np.zeros(7, dtype=bool)
    return arm


def test_motor_fault_status_stops_feedback_processing():
    arm = _adapter([1, 1, 10, 1, 1, 1, 1])

    with pytest.raises(RuntimeError, match="motor 3=status 10"):
        arm._poll_feedback()


def test_motor_becoming_disabled_after_green_is_detected():
    arm = _adapter([1] * 7)
    assert arm._poll_feedback()
    arm._sdk_arm._statuses = [1, 1, 1, 0, 1, 1, 1]

    with pytest.raises(RuntimeError, match="disabled unexpectedly: 4"):
        arm._poll_feedback()
