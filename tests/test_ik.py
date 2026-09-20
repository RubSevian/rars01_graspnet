from pathlib import Path

import numpy as np

from rars01_graspnet.ik import pregrasp_transform, solve_pose_ik, tcp_target_from_grasp
from rars01_graspnet.kinematics import RarsKinematics


ROOT = Path(__file__).resolve().parents[2]


def _kinematics():
    return RarsKinematics(
        ROOT / "rars01_description/urdf/rars01_control.urdf",
        "base_link",
        "End_link",
    )


def test_pregrasp_backs_away_along_negative_grasp_x():
    grasp = np.eye(4)
    grasp[:3, 3] = [0.4, 0.1, 0.2]

    pregrasp = pregrasp_transform(grasp, 0.08)

    np.testing.assert_allclose(pregrasp[:3, 3], [0.32, 0.1, 0.2])


def test_tcp_target_accounts_for_external_tool_offset():
    grasp_base = np.eye(4)
    grasp_base[0, 3] = 0.5
    grasp_tcp = np.eye(4)
    grasp_tcp[0, 3] = 0.1

    tcp_base = tcp_target_from_grasp(grasp_base, grasp_tcp)

    np.testing.assert_allclose(tcp_base[:3, 3], [0.4, 0.0, 0.0])


def test_pose_ik_reproduces_fk_target_without_hardware():
    kinematics = _kinematics()
    expected = np.array([0.1, 1.4, 1.1, -0.5, 0.3, 0.4])
    target = kinematics.forward(expected)

    solution = solve_pose_ik(kinematics, target, expected, random_starts=0)

    assert solution.success
    assert solution.position_error_m < 1e-8
    assert solution.rotation_error_deg < 1e-6
    np.testing.assert_allclose(solution.joints, expected, atol=1e-8)
