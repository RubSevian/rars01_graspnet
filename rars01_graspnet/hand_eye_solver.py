"""Small NumPy fallback for eye-in-hand calibration.

Some Jetson OpenCV builds expose the hand-eye enum values but omit
``cv2.calibrateHandEye``.  This module solves the same AX = XB problem without
depending on that optional OpenCV binding.
"""
from __future__ import annotations

import numpy as np


def _inverse_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle vector for a proper 3x3 rotation matrix."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    sine = np.sin(angle)
    if abs(sine) > 1e-6:
        axis = np.array(
            [rotation[2, 1] - rotation[1, 2],
             rotation[0, 2] - rotation[2, 0],
             rotation[1, 0] - rotation[0, 1]],
            dtype=np.float64,
        ) / (2.0 * sine)
        return axis * angle

    # For a rotation close to pi the anti-symmetric part is near zero.
    values, vectors = np.linalg.eigh((rotation + np.eye(3)) * 0.5)
    axis = vectors[:, int(np.argmax(values))]
    skew = np.array(
        [rotation[2, 1] - rotation[1, 2],
         rotation[0, 2] - rotation[2, 0],
         rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    )
    if np.dot(axis, skew) < 0.0:
        axis = -axis
    return axis * angle


def solve_eye_in_hand(
    T_gripper_base: list[np.ndarray] | tuple[np.ndarray, ...],
    T_target_camera: list[np.ndarray] | tuple[np.ndarray, ...],
) -> np.ndarray:
    """Solve for ``T_camera_gripper`` from paired eye-in-hand samples.

    For every pair of samples this forms ``A X = X B`` and applies the
    Park--Martin least-squares rotation solution followed by least-squares
    translation.  Inputs and output use homogeneous transforms in metres.
    """
    if len(T_gripper_base) != len(T_target_camera) or len(T_gripper_base) < 5:
        raise ValueError("Hand-eye calibration needs at least five paired samples")

    gripper = [np.asarray(item, dtype=np.float64) for item in T_gripper_base]
    target = [np.asarray(item, dtype=np.float64) for item in T_target_camera]
    if any(item.shape != (4, 4) for item in (*gripper, *target)):
        raise ValueError("Hand-eye transforms must be 4x4 matrices")

    rotation_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for first in range(len(gripper) - 1):
        for second in range(first + 1, len(gripper)):
            A = _inverse_transform(gripper[second]) @ gripper[first]
            B = target[second] @ _inverse_transform(target[first])
            if np.linalg.norm(_rotation_vector(A[:3, :3])) > 1e-5:
                rotation_pairs.append((A, B))
    if len(rotation_pairs) < 3:
        raise ValueError("Hand-eye poses have insufficient rotational motion")

    correlation = np.zeros((3, 3), dtype=np.float64)
    for A, B in rotation_pairs:
        correlation += np.outer(_rotation_vector(A[:3, :3]), _rotation_vector(B[:3, :3]))
    left, singular, right_t = np.linalg.svd(correlation)
    if singular[-1] < 1e-8:
        raise ValueError("Hand-eye poses do not span three independent rotations")
    rotation = left @ np.diag([1.0, 1.0, np.linalg.det(left @ right_t)]) @ right_t

    lhs, rhs = [], []
    for A, B in rotation_pairs:
        lhs.append(A[:3, :3] - np.eye(3))
        rhs.append(rotation @ B[:3, 3] - A[:3, 3])
    translation, _, rank, _ = np.linalg.lstsq(np.vstack(lhs), np.hstack(rhs), rcond=None)
    if rank < 3 or not np.all(np.isfinite(translation)):
        raise ValueError("Hand-eye poses have insufficient translational motion")

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result
