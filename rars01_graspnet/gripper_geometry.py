"""Geometry of the symmetric crank-linkage gripper installed on RARS01."""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, sin, sqrt

import numpy as np


@dataclass(frozen=True)
class GripperOpening:
    requested_width_m: float
    motor_angle_rad: float
    actual_width_m: float
    center_End_link_m: np.ndarray
    T_grasp_End_link: np.ndarray


class RarsGripperGeometry:
    """Map inner-jaw width to motor travel using the physical linkage."""

    def __init__(self, *, linkage_radius_m: float, connecting_rod_length_m: float,
                 carriage_width_m: float, maximum_width_m: float,
                 jaw_center_End_link_m, jaw_depth_m: float, jaw_height_m: float,
                 R_grasp_End_link, motor_angle_limit_rad: float):
        self.radius_m = float(linkage_radius_m)
        self.rod_length_m = float(connecting_rod_length_m)
        self.carriage_width_m = float(carriage_width_m)
        self._maximum_width_m = float(maximum_width_m)
        self.jaw_center = np.asarray(jaw_center_End_link_m, dtype=np.float64).reshape(3)
        self.jaw_depth_m = float(jaw_depth_m)
        self.jaw_height_m = float(jaw_height_m)
        self.R_grasp_End_link = np.asarray(R_grasp_End_link, dtype=np.float64).reshape(3, 3)
        self.motor_angle_limit_rad = abs(float(motor_angle_limit_rad))
        dimensions = (self.radius_m, self.rod_length_m, self.carriage_width_m,
                      self._maximum_width_m, self.jaw_depth_m, self.jaw_height_m,
                      self.motor_angle_limit_rad)
        if any(value <= 0.0 for value in dimensions):
            raise ValueError("All gripper dimensions and limits must be positive")
        if not np.allclose(self.R_grasp_End_link.T @ self.R_grasp_End_link,
                           np.eye(3), atol=1e-6):
            raise ValueError("R_grasp_End_link must be an orthonormal rotation")

        self._closed_linkage_angle_rad = self.linkage_angle_for_width(0.0)
        # Kept as a public attribute for the older diagnostic scripts.
        self.maximum_angle_rad = self.motor_angle_for_width(self._maximum_width_m)
        if self.maximum_angle_rad > self.motor_angle_limit_rad + 1e-9:
            raise ValueError(
                f"Linkage needs {self.maximum_angle_rad:.4f} rad for maximum width, "
                f"above motor limit {self.motor_angle_limit_rad:.4f} rad"
            )

    @classmethod
    def from_config(cls, config: dict) -> "RarsGripperGeometry":
        return cls(**config)

    @property
    def maximum_width_m(self) -> float:
        return self._maximum_width_m

    def linkage_angle_for_width(self, width_m: float) -> float:
        width = float(width_m)
        if width < 0.0:
            raise ValueError("Gripper width cannot be negative")
        x = 0.5 * (width + self.carriage_width_m)
        denominator = 2.0 * x * self.radius_m
        cosine = (x * x + self.radius_m**2 - self.rod_length_m**2) / denominator
        if cosine < -1.0 - 1e-9 or cosine > 1.0 + 1e-9:
            raise ValueError(f"Width {width:.4f} m is outside the linkage workspace")
        return acos(float(np.clip(cosine, -1.0, 1.0)))

    def motor_angle_for_width(self, width_m: float) -> float:
        width = float(width_m)
        if width > self._maximum_width_m + 1e-9:
            raise ValueError(
                f"Requested gripper width {width:.4f} m exceeds maximum "
                f"{self._maximum_width_m:.4f} m"
            )
        return self._closed_linkage_angle_rad - self.linkage_angle_for_width(width)

    def opening_width(self, motor_angle_rad: float) -> float:
        motor_angle = float(motor_angle_rad)
        if not 0.0 <= motor_angle <= self.motor_angle_limit_rad + 1e-9:
            raise ValueError("Gripper motor angle is outside configured limits")
        theta = self._closed_linkage_angle_rad - motor_angle
        root = max(0.0, self.rod_length_m**2 - self.radius_m**2 * sin(theta) ** 2)
        x = self.radius_m * cos(theta) + sqrt(root)
        return 2.0 * x - self.carriage_width_m

    def solve_opening(self, width_m: float) -> GripperOpening:
        width = float(width_m)
        if width < 0:
            raise ValueError("Gripper width cannot be negative")
        angle = self.motor_angle_for_width(width)
        return self.opening_at_angle(angle, requested_width_m=width)

    def opening_at_angle(self, angle_rad: float, *,
                         requested_width_m: float | None = None) -> GripperOpening:
        """Return jaw-center geometry for an already known motor angle."""
        angle = float(angle_rad)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.R_grasp_End_link
        transform[:3, 3] = self.jaw_center
        actual_width = self.opening_width(angle)
        requested_width = actual_width if requested_width_m is None else float(requested_width_m)
        return GripperOpening(
            requested_width, angle, actual_width, self.jaw_center.copy(), transform
        )

    def jaw_collision_points(self, width_m: float,
                             include_max_open: bool = True) -> np.ndarray:
        """Return conservative corners of both carriages in End_link."""
        widths = [float(np.clip(width_m, 0.0, self._maximum_width_m))]
        if include_max_open and widths[0] != self._maximum_width_m:
            widths.append(self._maximum_width_m)
        points = []
        for width in widths:
            for side in (-1.0, 1.0):
                y_inner = side * width / 2.0
                y_outer = side * (width / 2.0 + self.carriage_width_m)
                for x in (-self.jaw_depth_m / 2.0, self.jaw_depth_m / 2.0):
                    for y in (y_inner, y_outer):
                        for z in (-self.jaw_height_m / 2.0, self.jaw_height_m / 2.0):
                            points.append(self.jaw_center + np.array([x, y, z]))
        return np.asarray(points, dtype=np.float64)
