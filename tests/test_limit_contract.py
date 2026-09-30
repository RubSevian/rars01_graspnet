"""Offline contract tests; no RarsArm instance or serial transport."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rars01_graspnet.limit_contract import arm_limit_contract, command_positions_within_limits


class FakeSdkConfig:
    def __init__(self):
        self.motors = [SimpleNamespace(direction=1.0, joint_position_min=lo,
                                       joint_position_max=hi)
                       for lo, hi in ((-2.8, 2.8), (0, 3.5), (0, 3.14),
                                      (-2, 1.4), (-1.57, 1.57), (-3.14, 3.14))]

    def motor(self, index):
        return self.motors[index]


def test_sdk_urdf_intersection_and_original_command_epsilon():
    model = SimpleNamespace(nq=6,
        lowerPositionLimit=np.array([-2.8, 0, 0, -2, -1.57, -2.0]),
        upperPositionLimit=np.array([2.8, 3.6, 3.14, 1.4, 1.57, 2.0]))
    config = {"robot": {"rars01": {"joint_directions": [1] * 7,
                                  "command_limit_epsilon_rad": 0.01}}}
    with patch("rars01_graspnet.limit_contract._sdk_configuration",
               return_value=(FakeSdkConfig(), "fake/no-serial")):
        lower, upper, epsilon, source = arm_limit_contract(config, model)
    assert upper[1] == 3.5  # SDK stricter than URDF joint2=3.6.
    assert upper[5] == 2.0  # URDF stricter than SDK joint6=3.14.
    assert epsilon == 0.01
    assert source["sdk_python_path"] == "fake/no-serial"
    measured = np.array([0, -0.000181, -0.002955, 0, 0, 0], dtype=float)
    command = command_positions_within_limits(measured, lower, upper, epsilon)
    np.testing.assert_array_equal(command[1:3], [0.0, 0.0])
    assert measured[1] < 0  # Raw feedback must remain available for diagnostics.
    try:
        command_positions_within_limits(np.array([0, -0.011, 0, 0, 0, 0]),
                                        lower, upper, epsilon)
    except RuntimeError as error:
        assert "joint2" in str(error)
    else:
        raise AssertionError("A command beyond epsilon must be rejected")


if __name__ == "__main__":
    test_sdk_urdf_intersection_and_original_command_epsilon()
    print("limit contract: PASS")
