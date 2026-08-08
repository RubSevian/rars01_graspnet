import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rars01_graspnet.robot import RarsRobot


class _ArmConfiguration:
    def __init__(self):
        self.port_name = ""
        self.baud_rate = 0
        self.default_kp = [60.0] * 7
        self.default_kd = [1.0] * 7
        self.feedback_watchdog_enabled = True
        self.feedback_timeout_ms = 200
        self.initial_feedback_grace_ms = 1000


class _Arm:
    def __init__(self, configuration):
        self.configuration = configuration


def _sdk():
    return SimpleNamespace(ArmConfiguration=_ArmConfiguration, RarsArm=_Arm)


def test_robot_uses_explicit_seven_motor_gains(monkeypatch):
    monkeypatch.setitem(sys.modules, "rars_arm_py", _sdk())
    robot = RarsRobot("/dev/test", 123, position_kp=[1, 2, 3, 4, 5, 6, 7],
                      position_kd=[0.1] * 7)
    kp, kd = robot.position_gains()
    np.testing.assert_allclose(kp, [1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(kd, [0.1] * 7)


def test_robot_uses_configured_feedback_watchdog(monkeypatch):
    monkeypatch.setitem(sys.modules, "rars_arm_py", _sdk())
    robot = RarsRobot(
        "/dev/test", 123, feedback_watchdog_enabled=True,
        feedback_watchdog_timeout_ms=1000, initial_feedback_grace_ms=1500,
    )
    configuration = robot.arm.configuration
    assert configuration.feedback_watchdog_enabled is True
    assert configuration.feedback_timeout_ms == 1000
    assert configuration.initial_feedback_grace_ms == 1500


def test_robot_rejects_wrong_gain_count(monkeypatch):
    monkeypatch.setitem(sys.modules, "rars_arm_py", _sdk())
    with pytest.raises(ValueError, match="seven values"):
        RarsRobot("/dev/test", 123, position_kp=[1, 2, 3])
