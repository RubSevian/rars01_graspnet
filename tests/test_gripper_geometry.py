import numpy as np

from rars01_graspnet.gripper_geometry import RarsGripperGeometry


def _geometry():
    return RarsGripperGeometry(
        pivot_End_link_m=[-0.130, -0.0255, 0.0],
        moving_inner_tip_m=[0.130, 0.0255, -0.0055],
        fixed_inner_tip_End_link_m=[0.0, 0.0, -0.0055],
        maximum_angle_rad=1.0,
        R_grasp_End_link=[[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    )


def test_required_banana_width_maps_to_urdf_motor_angle_and_center():
    result = _geometry().solve_opening(0.0651)

    assert np.isclose(result.motor_angle_rad, 0.518084776, atol=1e-8)
    assert np.isclose(result.actual_width_m, 0.0651, atol=1e-9)
    np.testing.assert_allclose(result.center_End_link_m,
                               [-0.00716815, 0.0, 0.02705], atol=1e-8)


def test_urdf_maximum_opening_is_about_112_mm():
    assert np.isclose(_geometry().maximum_width_m, 0.111919565, atol=1e-8)


def test_grasp_axes_map_opening_y_to_rars_end_link_z():
    mapping = _geometry().opening_at_angle(0.0).T_grasp_End_link[:3, :3]

    np.testing.assert_allclose(mapping @ [1, 0, 0], [1, 0, 0])
    np.testing.assert_allclose(mapping @ [0, 1, 0], [0, 0, 1])
    np.testing.assert_allclose(mapping @ [0, 0, 1], [0, -1, 0])
    assert np.isclose(np.linalg.det(mapping), 1.0)
