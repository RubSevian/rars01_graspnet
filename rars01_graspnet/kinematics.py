from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    T_origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def _values(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    result = np.fromstring(text, sep=" ", dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"Expected three values, got {text!r}")
    return result


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if element is not None:
        transform[:3, :3] = _rpy_matrix(_values(element.get("rpy"), (0, 0, 0)))
        transform[:3, 3] = _values(element.get("xyz"), (0, 0, 0))
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("Revolute joint has a zero axis")
    x, y, z = axis / norm
    c, s, d = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    rotation = np.array(
        [[c + x*x*d, x*y*d - z*s, x*z*d + y*s],
         [y*x*d + z*s, c + y*y*d, y*z*d - x*s],
         [z*x*d - y*s, z*y*d + x*s, c + z*z*d]], dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


class RarsKinematics:
    """NumPy-only FK read from the unchanged RARS01 URDF."""

    def __init__(self, urdf: str | Path, base_frame: str, tcp_frame: str):
        root = ET.parse(str(urdf)).getroot()
        by_child: dict[str, _Joint] = {}
        for element in root.findall("joint"):
            parent_element, child_element = element.find("parent"), element.find("child")
            if parent_element is None or child_element is None:
                continue
            axis_element = element.find("axis")
            limit_element = element.find("limit")
            joint = _Joint(
                name=str(element.get("name")), kind=str(element.get("type")),
                parent=str(parent_element.get("link")), child=str(child_element.get("link")),
                T_origin=_origin_transform(element.find("origin")),
                axis=_values(axis_element.get("xyz") if axis_element is not None else None, (1, 0, 0)),
                lower=float(limit_element.get("lower", "-inf")) if limit_element is not None else -np.inf,
                upper=float(limit_element.get("upper", "inf")) if limit_element is not None else np.inf,
            )
            by_child[joint.child] = joint

        reverse_chain: list[_Joint] = []
        link, visited = tcp_frame, set()
        while link != base_frame:
            if link in visited or link not in by_child:
                raise ValueError(f"No URDF chain from {base_frame!r} to {tcp_frame!r}")
            visited.add(link)
            joint = by_child[link]
            reverse_chain.append(joint)
            link = joint.parent
        self.chain = list(reversed(reverse_chain))
        self.movable = [j for j in self.chain if j.kind in ("revolute", "continuous")]
        if len(self.movable) != 6:
            raise ValueError(f"Expected six movable joints, found {[j.name for j in self.movable]}")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.movable)

    @property
    def lower_limits(self) -> np.ndarray:
        return np.asarray([joint.lower for joint in self.movable], dtype=np.float64)

    @property
    def upper_limits(self) -> np.ndarray:
        return np.asarray([joint.upper for joint in self.movable], dtype=np.float64)

    def forward(self, joint_positions) -> np.ndarray:
        q = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if q.size < len(self.movable):
            raise ValueError(f"Expected at least {len(self.movable)} joints, got {q.size}")
        transform, q_index = np.eye(4, dtype=np.float64), 0
        for joint in self.chain:
            transform = transform @ joint.T_origin
            if joint.kind in ("revolute", "continuous"):
                transform = transform @ _axis_rotation(joint.axis, float(q[q_index]))
                q_index += 1
            elif joint.kind != "fixed":
                raise ValueError(f"Unsupported joint type {joint.kind!r} for {joint.name}")
        return transform
