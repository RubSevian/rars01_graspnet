import numpy as np

from rars01_graspnet.hand_eye_solver import solve_eye_in_hand


def _rotation(vector):
    angle = np.linalg.norm(vector)
    axis = vector / angle
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _transform(rotation_vector, translation):
    result = np.eye(4)
    result[:3, :3] = _rotation(np.asarray(rotation_vector, dtype=float))
    result[:3, 3] = translation
    return result


def test_numpy_hand_eye_solver_recovers_known_camera_to_gripper_transform():
    camera_gripper = _transform([0.35, -0.2, 0.5], [0.032, -0.041, 0.078])
    marker_base = _transform([-0.2, 0.3, 0.1], [0.45, -0.12, 0.31])
    gripper_base = [
        _transform([0.2, -0.1, 0.3], [0.20, -0.10, 0.25]),
        _transform([-0.4, 0.2, 0.15], [0.24, 0.08, 0.30]),
        _transform([0.1, 0.55, -0.25], [0.31, -0.12, 0.22]),
        _transform([-0.3, -0.35, 0.4], [0.16, 0.11, 0.35]),
        _transform([0.45, 0.15, -0.5], [0.38, 0.03, 0.28]),
        _transform([-0.15, 0.4, 0.35], [0.27, -0.18, 0.39]),
    ]
    marker_camera = [
        np.linalg.inv(camera_gripper) @ np.linalg.inv(gripper) @ marker_base
        for gripper in gripper_base
    ]

    solved = solve_eye_in_hand(gripper_base, marker_camera)
    assert np.allclose(solved, camera_gripper, atol=1e-8)
