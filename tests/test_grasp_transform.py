import numpy as np

from rars01_graspnet.hand_eye import grasp_to_base, pose_transform


def test_pose_transform_places_rotation_and_translation():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = pose_transform([0.1, 0.2, 0.3], rotation)

    np.testing.assert_allclose(transform[:3, :3], rotation)
    np.testing.assert_allclose(transform[:3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])


def test_grasp_to_base_composes_tcp_camera_and_grasp():
    T_tcp_base = np.eye(4)
    T_tcp_base[:3, 3] = [1.0, 0.0, 0.0]
    T_camera_tcp = np.eye(4)
    T_camera_tcp[:3, 3] = [0.0, 2.0, 0.0]

    result = grasp_to_base(T_tcp_base, T_camera_tcp, [0.0, 0.0, 3.0], np.eye(3))

    np.testing.assert_allclose(result[:3, 3], [1.0, 2.0, 3.0])
