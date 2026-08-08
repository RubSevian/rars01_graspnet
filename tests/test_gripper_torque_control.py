from types import SimpleNamespace

import numpy as np

from rars01_graspnet.robot import RarsRobot


def _state(angle, torque):
    values = [0.0] * 7
    values[6] = angle
    efforts = [0.0] * 7
    efforts[6] = torque
    return SimpleNamespace(
        position=values, velocity=[0.0] * 7, torque=efforts,
        valid=[True] * 7, error=[0] * 7, motor_id=list(range(1, 8)),
        mos_temperature=[25.0] * 7, rotor_temperature=[25.0] * 7,
    )


class _Configuration:
    default_kp = [70, 120, 120, 50, 20, 20, 20]
    default_kd = [1, 1, 1, 1, 2, 1, 1]

    @staticmethod
    def motor(index):
        return SimpleNamespace(
            joint_position_min=0.0 if index == 6 else -2.0,
            joint_position_max=1.0 if index == 6 else 2.0,
            joint_torque_max=5.0,
        )


class _Arm:
    def __init__(self):
        self.configuration = _Configuration()
        self.states = iter([
            _state(0.70, 0.0),  # Initial state.
            _state(0.698, 0.4),
            _state(0.696, 1.1),
            _state(0.694, 1.2),
        ])
        self.commands = []
        self.last_error = ""

    @staticmethod
    def is_enabled():
        return True

    def try_read_joint_state(self):
        return next(self.states, None)

    def send_mit(self, position, velocity, kp, kd, torque):
        self.commands.append((position, velocity, kp, kd, torque))
        return True

    def send_position_targets(self, position):
        self.commands.append(list(position))
        return True


def test_gripper_stops_after_stable_absolute_torque_limit():
    robot = RarsRobot.__new__(RarsRobot)
    robot.arm = _Arm()
    robot.feedback_timeout_s = 0.1
    robot._last_command = None

    result = robot.close_gripper_until_torque(
        torque_limit_nm=1.0, kp=20.0, kd=1.0,
        close_rate_rad_s=0.1, minimum_angle_rad=0.0,
        timeout_s=1.0, stable_samples=2, rate_hz=50.0,
    )

    assert np.isclose(result.measured_torque_nm, 1.2)
    assert result.commanded_angle_rad < 0.70
    assert len(robot.arm.commands) == 3
    assert robot.arm.commands[-1][2][6] == 20.0
    assert robot.arm.commands[-1][3][6] == 1.0
    assert np.isclose(robot._last_command[6], result.commanded_angle_rad)


def test_move_gripper_preserves_six_arm_targets():
    robot = RarsRobot.__new__(RarsRobot)
    robot.arm = _Arm()
    robot.feedback_timeout_s = 0.1
    robot._last_command = None

    robot.move_gripper(0.65, duration_s=0.001, rate_hz=1000.0, max_step_rad=0.1)

    motion_command = robot.arm.commands[0]
    assert motion_command[:6] == [0.0] * 6
    assert np.isclose(motion_command[6], 0.65)
