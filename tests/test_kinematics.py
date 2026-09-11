from pathlib import Path

import numpy as np

from rars01_graspnet.kinematics import RarsKinematics


URDF = Path(__file__).resolve().parents[2] / "rars01_description" / "urdf" / "rars01.urdf"


def test_rars01_chain_and_zero_pose():
    fk = RarsKinematics(URDF, "base_link", "End_link")
    assert fk.joint_names == ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    transform = fk.forward(np.zeros(6))
    np.testing.assert_allclose(transform[:3, 3], [0.301261207, 0.00015, 0.197371074], atol=1e-8)
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)


def test_joint_limits_match_urdf():
    fk = RarsKinematics(URDF, "base_link", "End_link")
    np.testing.assert_allclose(fk.lower_limits, [-2.8, 0.0, 0.0, -2.0, -1.57, -2.0])
    np.testing.assert_allclose(fk.upper_limits, [2.8, 3.6, 3.14, 1.4, 1.57, 2.0])
