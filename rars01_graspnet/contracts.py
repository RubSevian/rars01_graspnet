"""Transport-neutral data contracts designed to map cleanly to ROS 2 messages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Header:
    stamp_ns: int
    frame_id: str
    sequence: int = 0


@dataclass(frozen=True)
class CameraIntrinsics:
    K: np.ndarray
    D: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class RgbdFrame:
    header: Header
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class RobotState:
    header: Header
    names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    valid: np.ndarray
    error: np.ndarray
    mos_temperature: np.ndarray
    rotor_temperature: np.ndarray


@dataclass(frozen=True)
class Detection2D:
    header: Header
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: Optional[np.ndarray] = None
    center_xyz_m: Optional[np.ndarray] = None


@dataclass(frozen=True)
class Pose:
    position_m: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class GraspCandidate:
    header: Header
    pose: Pose
    score: float
    width_m: float

