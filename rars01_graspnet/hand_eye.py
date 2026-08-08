from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass(frozen=True)
class HandEyeResult:
    # OpenCV returns camera -> gripper. Here gripper is the URDF End_link.
    T_camera_tcp: np.ndarray
    translation_rms_m: float
    rotation_rms_deg: float
    method: str


@dataclass(frozen=True)
class HandEyeSamples:
    T_tcp_base: tuple[np.ndarray, ...]
    T_marker_camera: tuple[np.ndarray, ...]
    joints: tuple[np.ndarray, ...]

    def __len__(self) -> int:
        return len(self.T_tcp_base)


def solve(T_tcp_base: list[np.ndarray], T_marker_camera: list[np.ndarray], method="TSAI") -> HandEyeResult:
    if len(T_tcp_base) != len(T_marker_camera) or len(T_tcp_base) < 5:
        raise ValueError("Hand-eye calibration needs at least five paired samples")
    rotations_tcp_base = [T[:3, :3] for T in T_tcp_base]
    translations_tcp_base = [T[:3, 3] for T in T_tcp_base]
    rotations_marker_camera = [T[:3, :3] for T in T_marker_camera]
    translations_marker_camera = [T[:3, 3] for T in T_marker_camera]
    requested_method = method.upper()
    if requested_method == "AUTO":
        candidates = []
        for candidate_method in ("PARK", "HORAUD", "ANDREFF", "TSAI", "DANIILIDIS"):
            try:
                candidate = solve(T_tcp_base, T_marker_camera, candidate_method)
                if np.all(np.isfinite(candidate.T_camera_tcp)):
                    candidates.append(candidate)
            except (cv2.error, ValueError, np.linalg.LinAlgError):
                continue
        if not candidates:
            raise RuntimeError("All OpenCV hand-eye solvers failed")
        return min(candidates, key=lambda item: (item.translation_rms_m, item.rotation_rms_deg))
    if requested_method not in METHODS:
        raise ValueError(f"Unknown hand-eye method: {method}")

    rotation, translation = cv2.calibrateHandEye(
        rotations_tcp_base, translations_tcp_base,
        rotations_marker_camera, translations_marker_camera,
        method=METHODS[requested_method],
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation.reshape(3)

    # A stationary marker must have the same pose in base for every sample.
    marker_base = [a @ result @ b for a, b in zip(T_tcp_base, T_marker_camera)]
    mean_t = np.mean([T[:3, 3] for T in marker_base], axis=0)
    translation_rms = float(np.sqrt(np.mean([np.sum((T[:3, 3] - mean_t) ** 2) for T in marker_base])))
    reference_rotation = marker_base[0][:3, :3]
    angle_errors = []
    for transform in marker_base:
        delta = reference_rotation.T @ transform[:3, :3]
        cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        angle_errors.append(np.degrees(np.arccos(cosine)))
    return HandEyeResult(
        result, translation_rms,
        float(np.sqrt(np.mean(np.square(angle_errors)))), requested_method,
    )


def save(result: HandEyeResult, path: str | Path, sample_count: int, method: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output, T_camera_tcp=result.T_camera_tcp, sample_count=sample_count,
        method=result.method, requested_method=method,
        translation_rms_m=result.translation_rms_m,
        rotation_rms_deg=result.rotation_rms_deg,
    )


def load_samples(path: str | Path) -> HandEyeSamples:
    source = Path(path)
    if not source.exists():
        return HandEyeSamples((), (), ())
    with np.load(source, allow_pickle=False) as data:
        tcp = np.asarray(data["T_tcp_base"], dtype=np.float64)
        marker = np.asarray(data["T_marker_camera"], dtype=np.float64)
        joints = np.asarray(data["joints"], dtype=np.float64)
    if tcp.ndim != 3 or tcp.shape[1:] != (4, 4) or marker.shape != tcp.shape:
        raise ValueError(f"Invalid hand-eye sample transforms in {source}")
    if joints.ndim != 2 or joints.shape[0] != tcp.shape[0] or joints.shape[1] < 6:
        raise ValueError(f"Invalid hand-eye joint samples in {source}")
    return HandEyeSamples(tuple(tcp), tuple(marker), tuple(joints))


def append_sample(path: str | Path, T_tcp_base: np.ndarray,
                  T_marker_camera: np.ndarray, joints: np.ndarray) -> int:
    samples = load_samples(path)
    tcp = np.asarray((*samples.T_tcp_base, np.asarray(T_tcp_base, dtype=np.float64)))
    marker = np.asarray((*samples.T_marker_camera, np.asarray(T_marker_camera, dtype=np.float64)))
    joint_values = np.asarray((*samples.joints, np.asarray(joints, dtype=np.float64)[:6]))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez(temporary, T_tcp_base=tcp, T_marker_camera=marker, joints=joint_values)
    temporary.replace(output)
    return len(tcp)


def load(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["T_camera_tcp"], dtype=np.float64)


def camera_to_base(T_tcp_base: np.ndarray, T_camera_tcp: np.ndarray) -> np.ndarray:
    """T_camera_base = T_tcp_base @ T_camera_tcp."""
    return np.asarray(T_tcp_base) @ np.asarray(T_camera_tcp)


def pose_transform(position_m: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from a 3D position and rotation matrix."""
    position = np.asarray(position_m, dtype=np.float64).reshape(3)
    orientation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
        raise ValueError("Pose contains NaN or infinity")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = orientation
    transform[:3, 3] = position
    return transform


def grasp_to_base(T_tcp_base: np.ndarray, T_camera_tcp: np.ndarray,
                  position_camera_m: np.ndarray, rotation_camera: np.ndarray) -> np.ndarray:
    """Transform a GraspNet pose from the camera optical frame to base_link."""
    return camera_to_base(T_tcp_base, T_camera_tcp) @ pose_transform(
        position_camera_m, rotation_camera
    )
