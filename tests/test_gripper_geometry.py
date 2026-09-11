import numpy as np

from rars01_graspnet.gripper_geometry import RarsGripperGeometry


def _geometry():
    return RarsGripperGeometry(
        linkage_radius_m=0.0375,
        connecting_rod_length_m=0.040,
        carriage_width_m=0.030,
        maximum_width_m=0.100,
        jaw_center_End_link_m=[-0.040, 0.0, 0.0],
        jaw_depth_m=0.080,
        jaw_height_m=0.046,
        R_grasp_End_link=np.eye(3),
        motor_angle_limit_rad=1.0,
    )


def test_linkage_formula_maps_full_opening_to_relative_motor_angle():
    result = _geometry().solve_opening(0.100)

    assert np.isclose(result.motor_angle_rad, 0.9458322263, atol=1e-9)
    assert np.isclose(result.actual_width_m, 0.100, atol=1e-9)
    np.testing.assert_allclose(result.center_End_link_m, [-0.040, 0.0, 0.0])


def test_width_angle_round_trip():
    geometry = _geometry()
    for width in np.linspace(0.0, 0.100, 11):
        angle = geometry.motor_angle_for_width(width)
        assert np.isclose(geometry.opening_width(angle), width, atol=1e-9)


def test_parallel_grasp_axes_match_updated_urdf():
    mapping = _geometry().solve_opening(0.050).T_grasp_End_link[:3, :3]

    np.testing.assert_allclose(mapping, np.eye(3))
    assert np.isclose(np.linalg.det(mapping), 1.0)


def test_linkage_rejects_width_above_physical_limit():
    with np.testing.assert_raises(ValueError):
        _geometry().solve_opening(0.101)
