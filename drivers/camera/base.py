"""Shared RGB-D camera interface.

CameraDriver:
  open() / close(): start or stop camera streams.
  get_frame(): return one BGR color frame and one depth-mm frame.
  K / D: camera matrix and distortion coefficients.
  warm_up(): drop early frames before inference or calibration.
  setup_aruco() / detect_aruco() / draw_aruco(): optional ArUco helpers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class CameraFrameError(RuntimeError):
    """Raised after repeated camera frame acquisition failures."""


class CameraDriver(ABC):
    """Common color and depth camera interface."""

    _FRAME_FAIL_WARN = 10
    _FRAME_FAIL_LIMIT = 60

    def _reset_frame_failures(self) -> None:
        self._frame_failures = 0

    def _record_frame_failure(self, reason: str) -> None:
        count = int(getattr(self, "_frame_failures", 0)) + 1
        self._frame_failures = count
        if count == self._FRAME_FAIL_WARN:
            print(f"[Camera] {count} consecutive frame failures: {reason}")
        if count >= self._FRAME_FAIL_LIMIT:
            raise CameraFrameError(
                f"Camera failed for {count} consecutive frames: {reason}\n"
                "  Check connection, USB permissions, or other camera clients."
            )

    # Lifecycle

    @abstractmethod
    def open(self) -> None:
        """Open camera streams."""

    @abstractmethod
    def close(self) -> None:
        """Stop streams and release resources."""

    # Frames

    @abstractmethod
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return one color/depth frame pair.

        Returns:
            color_bgr: uint8 BGR image, or None.
            depth_mm: uint16 depth image in millimeters, or None.
        """

    # Intrinsics

    @property
    @abstractmethod
    def K(self) -> np.ndarray:
        """Camera matrix, shape (3, 3), float64."""

    @property
    @abstractmethod
    def D(self) -> np.ndarray:
        """Distortion coefficients, shape (1, N), float64."""

    # Helpers

    def warm_up(self, n_frames: int = 20) -> None:
        """Drop early frames while exposure settles."""
        for _ in range(n_frames):
            self.get_frame()

    def setup_aruco(
        self,
        marker_length_m: float,
        dict_id: int = 0,
        target_marker_id: Optional[int] = None,
    ) -> None:
        """Create an ArUco detector using this camera calibration.

        Args:
            marker_length_m: marker side length in meters.
            dict_id: cv2.aruco dictionary id.
            target_marker_id: detect only this id; None selects the nearest one.
        """
        from calibration.aruco_pose import ArUcoDetector
        self._aruco = ArUcoDetector(marker_length_m, dict_id, target_marker_id)

    def detect_aruco(self, bgr: np.ndarray):
        """Detect an ArUco marker. Call setup_aruco first."""
        return self._aruco.detect(bgr, self.K, self.D)

    def draw_aruco(self, bgr: np.ndarray) -> np.ndarray:
        """Draw detected ArUco markers. Call setup_aruco first."""
        return self._aruco.draw_detected(bgr, self.K, self.D)

    # Context manager

    def __enter__(self) -> "CameraDriver":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()
