from pathlib import Path

import numpy as np

from rars01_graspnet.kinematics import RarsKinematics


DESCRIPTION = Path(__file__).resolve().parents[2] / "rars01_description" / "urdf"
URDF = (
    DESCRIPTION
    / "rars01_control.urdf"
)
FULL_URDF = DESCRIPTION / "rars01.urdf"


def test_rars01_chain_and_zero_pose():
    fk = RarsKinematics(URDF, "base_link", "End_link")
    assert fk.joint_names == ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    transform = fk.forward(np.zeros(6))
    # The final CAD export raises joint1 by 2 mm relative to the previous model.
    np.testing.assert_allclose(transform[:3, 3], [0.301261, 0.00015, 0.199371], atol=1e-8)
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)


def test_joint_limits_match_urdf():
    fk = RarsKinematics(URDF, "base_link", "End_link")
    np.testing.assert_allclose(fk.lower_limits, [-2.8, 0.0, 0.0, -2.0, -1.57, -2.0])
    np.testing.assert_allclose(fk.upper_limits, [2.8, 3.6, 3.14, 1.4, 1.57, 2.0])


def test_full_mount_model_preserves_base_link_arm_kinematics():
    q = np.array([0.2, 1.0, 0.8, -0.3, 0.1, -0.4])
    control = RarsKinematics(URDF, "base_link", "End_link").forward(q)
    mounted = RarsKinematics(FULL_URDF, "arm_mount_link", "End_link").forward(q)
    T_mount_base = np.eye(4)
    T_mount_base[:3, 3] = [0.074304, 0.0, 0.0145]

    np.testing.assert_allclose(mounted, T_mount_base @ control, atol=1e-10)
