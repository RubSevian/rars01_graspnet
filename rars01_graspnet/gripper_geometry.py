"""RARS01 single-pivot gripper geometry derived from the unchanged URDF/STLs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class GripperOpening:
    requested_width_m: float
    motor_angle_rad: float
    actual_width_m: float
    center_End_link_m: np.ndarray
    T_grasp_End_link: np.ndarray


class RarsGripperGeometry:
    def __init__(self, *, pivot_End_link_m, moving_inner_tip_m,
                 fixed_inner_tip_End_link_m, maximum_angle_rad: float,
                 R_grasp_End_link):
        self.pivot = np.asarray(pivot_End_link_m, dtype=np.float64).reshape(3)
        self.moving_tip = np.asarray(moving_inner_tip_m, dtype=np.float64).reshape(3)
        self.fixed_tip = np.asarray(fixed_inner_tip_End_link_m, dtype=np.float64).reshape(3)
        self.maximum_angle_rad = float(maximum_angle_rad)
        self.R_grasp_End_link = np.asarray(R_grasp_End_link, dtype=np.float64).reshape(3, 3)
        if self.maximum_angle_rad <= 0:
            raise ValueError("Gripper maximum angle must be positive")

    @classmethod
    def from_config(cls, config: dict) -> "RarsGripperGeometry":
        return cls(**config)

    def opening_width(self, angle_rad: float) -> float:
        moving = self._moving_inner_tip(angle_rad)
        # End_link +Z is the physical opening direction for the current URDF.
        return float(moving[2] - self.fixed_tip[2])

    @property
    def maximum_width_m(self) -> float:
        return self.opening_width(self.maximum_angle_rad)

    def solve_opening(self, width_m: float) -> GripperOpening:
        width = float(width_m)
        if width < 0:
            raise ValueError("Gripper width cannot be negative")
        maximum = self.maximum_width_m
        if width > maximum + 1e-9:
            raise ValueError(
                f"Requested gripper width {width:.4f} m exceeds geometric maximum {maximum:.4f} m"
            )
        angle = 0.0 if width == 0 else float(brentq(
            lambda value: self.opening_width(value) - width,
            0.0, self.maximum_angle_rad,
        ))
        return self.opening_at_angle(angle, requested_width_m=width)

    def opening_at_angle(self, angle_rad: float, *,
                         requested_width_m: float | None = None) -> GripperOpening:
        """Return jaw-center geometry for an already known motor angle."""
        angle = float(angle_rad)
        moving = self._moving_inner_tip(angle)
        center = 0.5 * (self.fixed_tip + moving)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.R_grasp_End_link
        transform[:3, 3] = center
        actual_width = self.opening_width(angle)
        requested_width = actual_width if requested_width_m is None else float(requested_width_m)
        return GripperOpening(
            requested_width, angle, actual_width, center, transform
        )

    def _moving_inner_tip(self, angle_rad: float) -> np.ndarray:
        angle = float(angle_rad)
        if not 0.0 <= angle <= self.maximum_angle_rad:
            raise ValueError("Gripper angle is outside configured limits")
        c, s = np.cos(angle), np.sin(angle)
        # URDF axis is (0, -1, 0): rotation R_y(-angle).
        rotation = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
        return self.pivot + rotation @ self.moving_tip
