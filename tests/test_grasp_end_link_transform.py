import numpy as np

from utils.transforms import transform_graspnet_grasp_to_end_link_base_with_retreat


def test_graspnet_depth_places_end_link_at_virtual_finger_front_edge():
    grasp_translation = np.array([0.10, -0.02, 0.30])
    # The jaw centre is intentionally behind End_link and must not shift TCP.
    T_grasp_End_link = np.eye(4)
    T_grasp_End_link[:3, 3] = [-0.04, 0.0, 0.0]

    grasp, pregrasp, retreat = transform_graspnet_grasp_to_end_link_base_with_retreat(
        grasp_translation,
        np.eye(3),
        0.03,
        np.eye(4),
        T_grasp_End_link,
        pregrasp_offset_m=0.08,
        retreat_offset_m=0.05,
        allow_parallel_flip=False,
    )

    np.testing.assert_allclose(grasp[:3], [0.13, -0.02, 0.30])
    np.testing.assert_allclose(pregrasp[:3], [0.05, -0.02, 0.30])
    np.testing.assert_allclose(retreat[:3], [0.08, -0.02, 0.30])


def test_end_link_positive_x_is_graspnet_approach_axis_after_rotation():
    angle = np.pi / 2.0
    R_grasp = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    T_cam2base = np.eye(4)
    T_cam2base[:3, 3] = [1.0, 2.0, 3.0]

    grasp, _, _ = transform_graspnet_grasp_to_end_link_base_with_retreat(
        np.zeros(3), R_grasp, 0.04, T_cam2base, np.eye(4), 0.08, 0.08,
        allow_parallel_flip=False,
    )

    # GraspNet +X and End_link +X both point along base +Y.
    np.testing.assert_allclose(grasp[:3], [1.0, 2.04, 3.0], atol=1e-12)
    np.testing.assert_allclose(grasp[3:], [0.0, 0.0, np.pi / 2.0], atol=1e-12)


def test_end_link_axis_convention_must_preserve_graspnet_x():
    T_grasp_End_link = np.eye(4)
    T_grasp_End_link[:3, :3] = np.diag([-1.0, 1.0, -1.0])

    with np.testing.assert_raises_regex(ValueError, r"End_link \+X"):
        transform_graspnet_grasp_to_end_link_base_with_retreat(
            np.zeros(3), np.eye(3), 0.03, np.eye(4), T_grasp_End_link, 0.08, 0.08,
            allow_parallel_flip=False,
        )
