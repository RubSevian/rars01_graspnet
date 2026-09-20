from pathlib import Path

import numpy as np

from rars01_graspnet.kinematics import RarsKinematics
from rars01_graspnet.trajectory import (
    calibration_joint_targets,
    cartesian_geodesic_samples,
    minimum_jerk_samples,
    track_cartesian_trajectory,
)


URDF = (
    Path(__file__).resolve().parents[2]
    / "rars01_description"
    / "urdf"
    / "rars01_control.urdf"
)


def test_minimum_jerk_reaches_target_and_limits_steps():
    samples = minimum_jerk_samples(np.zeros(2), np.array([0.5, -0.2]), 1.0, 10.0, 0.02)
    np.testing.assert_allclose(samples[-1], [0.5, -0.2])
    assert np.max(np.abs(np.diff(np.vstack((np.zeros(2), samples)), axis=0))) <= 0.0201


def test_calibration_targets_are_deterministic_and_within_limits():
    fk = RarsKinematics(URDF, "base_link", "End_link")
    seed = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    kwargs = dict(margin_rad=0.05, min_tcp_z_m=-1.0, max_tcp_translation_m=1.0)
    targets = calibration_joint_targets(
        seed, fk.lower_limits, fk.upper_limits, [0.2] * 6, 20, fk, **kwargs
    )
    repeated = calibration_joint_targets(
        seed, fk.lower_limits, fk.upper_limits, [0.2] * 6, 20, fk, **kwargs
    )
    assert len(targets) == 20
    np.testing.assert_allclose(targets, repeated)
    assert np.all(np.asarray(targets) > fk.lower_limits)
    assert np.all(np.asarray(targets) < fk.upper_limits)


def test_cartesian_geodesic_ends_at_target_and_moves_on_a_line():
    start = np.eye(4)
    target = np.eye(4)
    target[:3, 3] = [0.2, -0.1, 0.3]
    target[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    poses = cartesian_geodesic_samples(start, target, 1.0, 20.0)

    np.testing.assert_allclose(poses[-1], target, atol=1e-9)
    for pose in poses:
        assert np.linalg.norm(np.cross(pose[:3, 3], target[:3, 3])) < 1e-9


def test_cartesian_tracking_reaches_known_fk_without_joint_jump():
    fk = _kinematics_for_test()
    start = np.array([0.0, 1.0, 1.0, -0.5, 0.0, 0.0])
    target_joints = start + np.array([0.02, 0.02, -0.02, 0.02, 0.01, -0.01])

    points = track_cartesian_trajectory(
        fk, start, fk.forward(target_joints), duration_s=0.5, rate_hz=10,
        max_joint_step_rad=0.03, joint_margin_rad=0.05,
        position_tolerance_m=0.002, rotation_tolerance_deg=2.0,
    )

    np.testing.assert_allclose(fk.forward(points[-1]), fk.forward(target_joints), atol=1e-6)


def _kinematics_for_test():
    return RarsKinematics(URDF, "base_link", "End_link")
