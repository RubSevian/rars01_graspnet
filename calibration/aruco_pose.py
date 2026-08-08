"""ArUco 标记检测与位姿估计。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MarkerPose:
    """单个 ArUco 标记的检测结果。"""
    id: int
    T_marker2cam: np.ndarray   # (4, 4) 标记到相机坐标系的变换


class ArUcoDetector:
    """ArUco 标记检测器，返回最接近相机的目标标记的位姿。

    Args:
        marker_length_m: 标记实际边长（米，黑色边框外沿）
        aruco_dict_id:   cv2.aruco 字典 ID，默认 DICT_4X4_50 = 0
        target_marker_id: 指定只检测该 ID 的标记；None = 使用检测到的第一个
    """

    def __init__(
        self,
        marker_length_m: float = 0.05,
        aruco_dict_id: int = 0,
        target_marker_id: Optional[int] = None,
    ) -> None:
        self._length = marker_length_m
        self._tid = target_marker_id
        self._dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self._params = self._create_detector_params()
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params) if hasattr(cv2.aruco, "ArucoDetector") else None
        half = self._length / 2.0
        self._object_points = np.array(
            [
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _create_detector_params():
        if hasattr(cv2.aruco, "DetectorParameters"):
            return cv2.aruco.DetectorParameters()
        return cv2.aruco.DetectorParameters_create()

    def _detect_markers(self, gray: np.ndarray):
        if self._detector is not None:
            return self._detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self._dict, parameters=self._params)

    def _estimate_pose(self, corner: np.ndarray, K: np.ndarray, D: np.ndarray):
        image_points = np.asarray(corner, dtype=np.float64).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(
            self._object_points,
            image_points,
            np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64).reshape(-1, 1),
            flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
        )
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                self._object_points,
                image_points,
                np.asarray(K, dtype=np.float64),
                np.asarray(D, dtype=np.float64).reshape(-1, 1),
            )
        if not ok:
            return None, None
        return rvec.reshape(3), tvec.reshape(3)

    def detect(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> Optional[MarkerPose]:
        """
        在图像中检测 ArUco 标记并返回位姿。

        Args:
            bgr: BGR 图像 (H, W, 3)
            K:   相机内参矩阵 (3, 3)
            D:   畸变系数 (1, N) 或 (N,)

        Returns:
            MarkerPose 或 None（未检测到）
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect_markers(gray)

        if ids is None or len(ids) == 0:
            return None

        ids_flat = ids.flatten()

        # 筛选目标 ID
        if self._tid is not None:
            mask = ids_flat == self._tid
            if not np.any(mask):
                return None
            corners = [corners[i] for i in np.where(mask)[0]]
            ids_flat = ids_flat[mask]

        # 多标记时取最近（Z 最小）的
        best: Optional[MarkerPose] = None
        best_z = float("inf")

        for corner, mid in zip(corners, ids_flat):
            rvec, tvec = self._estimate_pose(corner, K, D)
            if rvec is None:
                continue
            z = float(tvec[2])
            if z < best_z:
                best_z = z
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3,  3] = tvec
                best = MarkerPose(id=int(mid), T_marker2cam=T)

        return best

    def draw_detected(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        axis_length: float = 0.03,
    ) -> np.ndarray:
        """
        在图像上绘制检测到的所有 ArUco 标记（框 + 坐标轴）。

        Returns:
            带标注的 BGR 图像副本
        """
        vis = bgr.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect_markers(gray)

        if ids is None or len(ids) == 0:
            return vis

        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        for corner in corners:
            rvec, tvec = self._estimate_pose(corner, K, D)
            if rvec is not None:
                cv2.drawFrameAxes(vis, K, D, rvec, tvec, axis_length)

        return vis
