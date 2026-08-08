from __future__ import annotations

from pathlib import Path

from .base import CameraDriver, CameraFrameError
from .orbbec_gemini2 import OrbbecGemini2
from .realsense import RealsenseCamera

__all__ = ["CameraDriver", "CameraFrameError", "OrbbecGemini2", "RealsenseCamera", "make_camera"]


def make_camera(cfg: dict) -> CameraDriver:
    """Create a camera driver from config/default.yaml."""
    cam_cfg  = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "").lower()
    w   = cam_cfg.get("color_width",  1280)
    h   = cam_cfg.get("color_height", 720)
    fps = cam_cfg.get("fps", 30)
    alignment = cam_cfg.get("alignment", "auto")

    _root     = Path(__file__).resolve().parent.parent.parent
    calib_dir = str(_root / "config" / "calibration" / cam_type)

    if "orbbec" in cam_type:
        return OrbbecGemini2(w, h, fps, calib_dir=calib_dir, alignment=alignment)
    elif "realsense" in cam_type:
        return RealsenseCamera(w, h, fps, calib_dir=calib_dir)
    else:
        raise ValueError(
            f"Unsupported camera type: {cam_type!r}\n"
            f"Set camera.type in config/default.yaml to:\n"
            f"  orbbec_gemini2 | realsense_d435i | realsense_d405"
        )
