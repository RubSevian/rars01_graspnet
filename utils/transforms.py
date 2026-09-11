"""Coordinate transform utilities."""
import numpy as np


_ROT_X_PI = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _nearest_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"rotation matrix must be (3, 3), got {R.shape}")

    if not np.all(np.isfinite(R)):
        raise ValueError("rotation matrix contains non-finite values")

    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0.0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho.astype(np.float64)


def pose6d_to_mat4(x, y, z, rx, ry, rz, degrees=False) -> np.ndarray:
    """
    Convert a 6D pose to a 4x4 homogeneous transform.

    Args:
        x, y, z: translation in meters.
        rx, ry, rz: intrinsic ZYX Euler angles in roll, pitch, yaw order.
        degrees: True when angles are in degrees; False for radians.

    Returns:
        T: (4, 4) numpy array
    """
    if degrees:
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    # Rotation around X.
    Rx = np.array([
        [1,          0,           0],
        [0,  np.cos(rx), -np.sin(rx)],
        [0,  np.sin(rx),  np.cos(rx)],
    ])
    # Rotation around Y.
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    # Rotation around Z.
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [         0,           0, 1],
    ])

    # Intrinsic ZYX rotation: R = Rz @ Ry @ Rx.
    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def quat_to_mat4(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """
    Convert translation and quaternion to a 4x4 homogeneous transform.

    Args:
        x, y, z: translation in meters.
        qx, qy, qz, qw: Hamilton quaternion.

    Returns:
        T: (4, 4) numpy array
    """
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def mat4_to_pose6d(T: np.ndarray) -> tuple:
    """
    Convert a 4x4 transform to (x, y, z, rx, ry, rz) in radians.
    """
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    rpy = rotation_matrix_to_euler_zyx(T[:3, :3])
    return float(x), float(y), float(z), float(rpy[0]), float(rpy[1]), float(rpy[2])


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to intrinsic ZYX Euler angles."""
    R = _nearest_rotation_matrix(R)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def canonicalize_parallel_gripper_tcp_rotation(R: np.ndarray) -> np.ndarray:
    """Pick a stable equivalent TCP rotation for a parallel gripper.

    For a symmetric parallel gripper, a 180-degree twist around the tool X axis
    is usually grasp-equivalent. Choose the branch between ``R`` and
    ``R @ Rx(pi)`` that has the smaller absolute roll, making the output RPY
    easier to inspect and debug.
    """
    R = _nearest_rotation_matrix(R)
    alt = R @ _ROT_X_PI

    roll = float(rotation_matrix_to_euler_zyx(R)[0])
    alt_roll = float(rotation_matrix_to_euler_zyx(alt)[0])
    return alt if abs(alt_roll) < abs(roll) else R


def grasp_axes_to_rebot_tcp_rotation(
    grip_axis: np.ndarray,
    open_axis: np.ndarray,
    approach_axis: np.ndarray,
) -> np.ndarray:
    """Map grasp-frame axes to the reBotArm TCP frame.

    Vision grasp convention:
      - X = grip_axis
      - Y = open_axis
      - Z = approach_axis

    reBotArm TCP convention:
      - X = tool-forward / approach direction
      - Y = gripper opening direction
      - Z = right-handed completion
    """
    grip = np.asarray(grip_axis, dtype=np.float64)
    open_vec = np.asarray(open_axis, dtype=np.float64)
    approach = np.asarray(approach_axis, dtype=np.float64)

    grip /= max(np.linalg.norm(grip), 1e-8)
    open_vec /= max(np.linalg.norm(open_vec), 1e-8)
    approach /= max(np.linalg.norm(approach), 1e-8)

    # tcp_x = tool-forward = approach direction pointing INTO the object (downward in base).
    # plane.normal points toward the camera (upward), so negate it here.
    tcp_x = -approach
    tcp_y = open_vec - float(np.dot(open_vec, tcp_x)) * tcp_x
    tcp_y /= max(np.linalg.norm(tcp_y), 1e-8)
    tcp_z = np.cross(tcp_x, tcp_y)
    tcp_z /= max(np.linalg.norm(tcp_z), 1e-8)

    # Keep tcp_z aligned with the grip axis after the approach-axis flip.
    if float(np.dot(tcp_z, grip)) < 0.0:
        tcp_y = -tcp_y
        tcp_z = -tcp_z

    R = np.column_stack([tcp_x, tcp_y, tcp_z]).astype(np.float64)
    if np.linalg.det(R) < 0.0:
        R[:, 2] *= -1.0
    return R


def grasp_rotation_to_rebot_tcp_rotation(grasp_rotation: np.ndarray) -> np.ndarray:
    """Convert a [grip, open, approach] rotation matrix to reBotArm TCP rotation."""
    R = np.asarray(grasp_rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"grasp_rotation must be (3, 3), got {R.shape}")
    return grasp_axes_to_rebot_tcp_rotation(R[:, 0], R[:, 1], R[:, 2])


def _make_grasp_base_transform(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
) -> np.ndarray:
    T_grasp_cam = np.eye(4, dtype=np.float64)
    T_grasp_cam[:3, :3] = np.asarray(tcp_rotation_cam, dtype=np.float64)
    T_grasp_cam[:3, 3] = np.asarray(position_cam, dtype=np.float64)

    T_grasp_base = np.asarray(T_cam2base, dtype=np.float64) @ T_grasp_cam
    T_grasp_base[:3, :3] = canonicalize_parallel_gripper_tcp_rotation(T_grasp_base[:3, :3])
    return T_grasp_base


def _offset_along_tool_x(T: np.ndarray, offset_m: float) -> np.ndarray:
    T_offset = T.copy()
    T_offset[:3, 3] = T[:3, 3] - T[:3, 0] * float(offset_m)
    return T_offset


def transform_grasp_pose_to_base(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert a camera-frame grasp pose to base-frame grasp/pregrasp poses."""
    T_grasp_base = _make_grasp_base_transform(position_cam, tcp_rotation_cam, T_cam2base)
    T_grasp_base = _offset_along_tool_x(T_grasp_base, -insertion_depth_m)
    T_pregrasp_base = _offset_along_tool_x(T_grasp_base, pregrasp_offset_m)
    return mat4_to_pose6d(T_grasp_base), mat4_to_pose6d(T_pregrasp_base)


def transform_grasp_pose_to_base_with_retreat(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Convert a camera-frame grasp pose to base-frame grasp, pregrasp, and retreat poses."""
    T_grasp_base = _make_grasp_base_transform(position_cam, tcp_rotation_cam, T_cam2base)
    T_grasp_base = _offset_along_tool_x(T_grasp_base, -insertion_depth_m)
    T_pregrasp_base = _offset_along_tool_x(T_grasp_base, pregrasp_offset_m)
    T_retreat_base = _offset_along_tool_x(T_grasp_base, retreat_offset_m)
    return mat4_to_pose6d(T_grasp_base), mat4_to_pose6d(T_pregrasp_base), mat4_to_pose6d(T_retreat_base)


def transform_grasp_frame_to_tcp_base_with_retreat(
    position_cam: np.ndarray,
    grasp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
    T_grasp_tcp: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float = 0.0,
    allow_parallel_flip: bool = True,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Transform a grasp-center pose when the TCP is offset from jaw center."""
    T_grasp_tcp = np.asarray(T_grasp_tcp, dtype=np.float64).reshape(4, 4)
    R_raw = _nearest_rotation_matrix(grasp_rotation_cam)
    alternatives = []
    branches = (np.eye(3), _ROT_X_PI) if allow_parallel_flip else (np.eye(3),)
    for branch in branches:
        R_grasp = R_raw @ branch
        R_tcp = R_grasp @ T_grasp_tcp[:3, :3].T
        roll = abs(float(rotation_matrix_to_euler_zyx(R_tcp)[0]))
        alternatives.append((roll, R_grasp))
    R_grasp = min(alternatives, key=lambda item: item[0])[1]

    T_grasp_cam = np.eye(4, dtype=np.float64)
    T_grasp_cam[:3, :3] = R_grasp
    T_grasp_cam[:3, 3] = np.asarray(position_cam, dtype=np.float64)
    T_grasp_base = np.asarray(T_cam2base, dtype=np.float64) @ T_grasp_cam
    T_grasp_base = _offset_along_tool_x(T_grasp_base, -insertion_depth_m)
    T_pregrasp_base = _offset_along_tool_x(T_grasp_base, pregrasp_offset_m)
    T_retreat_base = _offset_along_tool_x(T_grasp_base, retreat_offset_m)

    T_tcp_grasp = np.linalg.inv(T_grasp_tcp)
    return (
        mat4_to_pose6d(T_grasp_base @ T_tcp_grasp),
        mat4_to_pose6d(T_pregrasp_base @ T_tcp_grasp),
        mat4_to_pose6d(T_retreat_base @ T_tcp_grasp),
    )


def graspnet_rotation_to_rebot_tcp_rotation(grasp_rotation: np.ndarray) -> np.ndarray:
    """Convert a GraspNet rotation_matrix to reBotArm TCP rotation."""
    R = np.asarray(grasp_rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"grasp_rotation must be (3, 3), got {R.shape}")

    return _nearest_rotation_matrix(np.column_stack([R[:, 0], R[:, 1], np.cross(R[:, 0], R[:, 1])]))


def graspnet_rotation_to_rars_tcp_rotation(grasp_rotation: np.ndarray) -> np.ndarray:
    """Map GraspNet axes to parallel RARS End_link: X=approach, Y=opening."""
    R = np.asarray(grasp_rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"grasp_rotation must be (3, 3), got {R.shape}")
    return _nearest_rotation_matrix(
        np.column_stack([R[:, 0], R[:, 1], np.cross(R[:, 0], R[:, 1])])
    )
