"""手眼标定 — 基于 OpenCV calibrateHandEye。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np


class CalibMode(Enum):
    EYE_IN_HAND = "eye_in_hand"   # 相机在末端，随末端运动


_METHOD_MAP = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class CalibResult:
    T_result: np.ndarray    # (4, 4) 手眼变换矩阵
    mode: str               # CalibMode.value
    n_samples: int
    method: str


@dataclass
class _Sample:
    T_gripper2base: np.ndarray   # (4, 4)
    T_marker2cam:   np.ndarray   # (4, 4)


class HandEyeCalibrator:
    """
    手眼标定器。

    Eye-in-Hand 模式：
        求解 T_cam2gripper，使得
            T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam
        在所有姿态下恒成立。

    使用方法：
        calib = HandEyeCalibrator(CalibMode.EYE_IN_HAND)
        calib.add_sample(T_gripper2base, T_marker2cam)
        ...
        result = calib.calibrate()
        HandEyeCalibrator.save(result, "hand_eye.npz")
    """

    def __init__(
        self,
        mode: CalibMode = CalibMode.EYE_IN_HAND,
        method: str = "TSAI",
    ) -> None:
        if mode != CalibMode.EYE_IN_HAND:
            raise ValueError("Only eye-in-hand calibration is supported")
        self._mode = mode
        self._method = method.upper()
        self._samples: List[_Sample] = []

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def add_sample(
        self,
        T_gripper2base: np.ndarray,
        T_marker2cam: np.ndarray,
    ) -> None:
        """
        添加一个标定样本。

        Args:
            T_gripper2base: (4,4) 末端到基座的变换（正运动学 FK 输出）
            T_marker2cam:   (4,4) 标记到相机的变换（ArUco 检测输出）
        """
        self._samples.append(_Sample(
            T_gripper2base=np.asarray(T_gripper2base, dtype=np.float64),
            T_marker2cam=np.asarray(T_marker2cam, dtype=np.float64),
        ))

    def calibrate(self, min_samples: int = 5) -> CalibResult:
        """
        计算手眼变换。

        Args:
            min_samples: 最少样本数（< 此值会抛出异常）

        Returns:
            CalibResult，T_result 即手眼变换矩阵
        """
        if self.n_samples < min_samples:
            raise ValueError(
                f"样本不足：{self.n_samples} < {min_samples}，请继续采集"
            )

        cv_method = _METHOD_MAP.get(self._method, cv2.CALIB_HAND_EYE_TSAI)

        # OpenCV 接口：R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
        R_g2b = [s.T_gripper2base[:3, :3] for s in self._samples]
        t_g2b = [s.T_gripper2base[:3,  3].reshape(3, 1) for s in self._samples]
        R_t2c = [s.T_marker2cam[:3, :3] for s in self._samples]
        t_t2c = [s.T_marker2cam[:3,  3].reshape(3, 1) for s in self._samples]

        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_t2c, t_t2c, method=cv_method
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_c2g
        T[:3,  3] = t_c2g.flatten()

        return CalibResult(
            T_result=T,
            mode=self._mode.value,
            n_samples=self.n_samples,
            method=self._method,
        )

    @staticmethod
    def save(result: CalibResult, path: Union[str, Path]) -> None:
        """保存标定结果为 .npz 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            T_result=result.T_result,
            mode=np.array([result.mode]),
            n_samples=np.array([result.n_samples]),
            method=np.array([result.method]),
        )

    @staticmethod
    def load(path: Union[str, Path]) -> CalibResult:
        """从 .npz 文件加载标定结果。"""
        data = np.load(str(path), allow_pickle=False)
        return CalibResult(
            T_result=data["T_result"],
            mode=str(data["mode"][0]),
            n_samples=int(data["n_samples"][0]),
            method=str(data["method"][0]) if "method" in data else "TSAI",
        )
