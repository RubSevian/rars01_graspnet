from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerObservation:
    marker_id: int
    T_marker_camera: np.ndarray
    corners: np.ndarray


class ArucoPoseEstimator:
    def __init__(self, dictionary_name: str, marker_id: int, marker_size_m: float):
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.marker_id = int(marker_id)
        half = float(marker_size_m) / 2.0
        self.object_points = np.array(
            [[-half, half, 0.0], [half, half, 0.0],
             [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float64
        )

    def detect(self, image: np.ndarray, K: np.ndarray, D: np.ndarray) -> MarkerObservation | None:
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if ids is None:
            return None
        for marker_corners, found_id in zip(corners, ids.flatten()):
            if int(found_id) != self.marker_id:
                continue
            points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(
                self.object_points, points, K, D.reshape(-1, 1),
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                return None
            rotation, _ = cv2.Rodrigues(rvec)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transform[:3, 3] = tvec.reshape(3)
            return MarkerObservation(int(found_id), transform, points)
        return None

    @staticmethod
    def draw(image: np.ndarray, observation: MarkerObservation | None, K, D) -> np.ndarray:
        output = image.copy()
        if observation is None:
            cv2.putText(output, "ArUco: not found", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            return output
        corners = observation.corners.reshape(1, 4, 2).astype(np.float32)
        cv2.aruco.drawDetectedMarkers(output, [corners], np.array([[observation.marker_id]]))
        rotation = observation.T_marker_camera[:3, :3]
        rvec, _ = cv2.Rodrigues(rotation)
        cv2.drawFrameAxes(output, K, D, rvec, observation.T_marker_camera[:3, 3], 0.04)
        cv2.putText(output, f"ArUco {observation.marker_id}: SPACE=capture Q=solve",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        return output

