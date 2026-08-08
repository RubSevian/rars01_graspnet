"""Orbbec Gemini 2 camera driver.

OrbbecGemini2:
  open() / close(): manage the pyorbbecsdk pipeline.
  get_frame(): return aligned BGR and depth-mm frames.
  K / D: camera matrix and distortion coefficients.
"""
from __future__ import annotations

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

from .base import CameraDriver, CameraFrameError


class OrbbecGemini2(CameraDriver):
    """Orbbec Gemini 2 RGB-D camera driver."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        calib_dir: Optional[str] = None,
        alignment: str = "auto",
    ) -> None:
        self._w = width
        self._h = height
        self._fps = fps
        self._calib_dir = Path(calib_dir) if calib_dir else None
        self._alignment = str(alignment).lower()
        if self._alignment not in ("auto", "hardware", "software"):
            raise ValueError("Orbbec alignment must be auto, hardware, or software")

        self._pipeline = None
        self._align_filter = None
        self._pending_frames = None
        self.active_alignment: Optional[str] = None
        self._depth_scale_mm: float = 1.0
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._aruco = None
        self._reset_frame_failures()

    # Lifecycle

    def open(self) -> None:
        """Open the camera pipeline."""
        # Import first so native load errors stay visible.
        try:
            from pyorbbecsdk import (
                AlignFilter, Pipeline, Config,
                OBSensorType, OBFormat, OBAlignMode,
                OBFrameAggregateOutputMode, OBStreamType,
                Context,
            )
        except ImportError as e:
            raise RuntimeError(f"pyorbbecsdk is not installed: {e}") from e

        # Silence noisy native logs during SDK initialization.
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)

        try:
            try:
                from pyorbbecsdk import OBLogSeverity
                Context().set_logger_severity(OBLogSeverity.FATAL)
            except Exception:
                pass

            try:
                self._pipeline = Pipeline()
            except Exception as e:
                raise RuntimeError(
                    f"Orbbec camera not found: {e}\n"
                    "  Check USB connection and udev permissions.\n"
                    "  Permission quick fix: sudo chmod a+rw /dev/bus/usb/*/*"
                ) from e

            cfg = None
            if self._alignment in ("auto", "hardware"):
                cfg = self._hardware_d2c_config(Config, OBAlignMode, OBFormat, OBSensorType)
                if cfg is None and self._alignment == "hardware":
                    raise RuntimeError(
                        "Gemini 336 has no hardware-D2C profile for the requested stream; "
                        "set camera.alignment to auto or software"
                    )

            use_software = cfg is None
            if cfg is not None:
                try:
                    self._pipeline.enable_frame_sync()
                except Exception as exc:
                    print(f"[OrbbecGemini2] Frame sync warning: {exc}")
                self._pipeline.start(cfg)
                self.active_alignment = "hardware"
                for _ in range(3):
                    self._pending_frames = self._pipeline.wait_for_frames(1000)
                    if self._pending_frames is not None:
                        break
                if self._pending_frames is None:
                    if self._alignment == "hardware":
                        self._pipeline.stop()
                        raise RuntimeError("Hardware D2C started but produced no RGB-D frames")
                    print("[OrbbecGemini2] Hardware D2C produced no frames; using software")
                    self._pipeline.stop()
                    self._pipeline = Pipeline()
                    use_software = True

            if use_software:
                cfg = Config()
                color_profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                color_profile = self._profile(
                    color_profiles, (OBFormat.MJPG, OBFormat.RGB)
                )
                depth_profiles = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = self._profile(depth_profiles, (OBFormat.Y16,))
                cfg.enable_stream(color_profile)
                cfg.enable_stream(depth_profile)
                try:
                    cfg.set_frame_aggregate_output_mode(
                        OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                    )
                except Exception:
                    pass
                self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                self._pipeline.start(cfg)
                self.active_alignment = "software"

            print(f"[OrbbecGemini2] D2C alignment: {self.active_alignment}")
            self._reset_frame_failures()

            # Intrinsics from SDK
            intr = self._pipeline.get_camera_param().rgb_intrinsic
            self._K = np.array([
                [intr.fx, 0,       intr.cx],
                [0,       intr.fy, intr.cy],
                [0,       0,       1      ],
            ], dtype=np.float64)

            # Distortion
            self._D = self._load_distortion()

        finally:
            os.dup2(saved, 2)
            os.close(saved)

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._align_filter = None
        self._pending_frames = None
        self.active_alignment = None

    # Frames

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._pipeline is None:
            return None, None
        try:
            from pyorbbecsdk import OBFormat
            frames = self._pending_frames
            self._pending_frames = None
            if frames is None:
                frames = self._pipeline.wait_for_frames(500)
            if frames is None:
                self._record_frame_failure("wait_for_frames timeout")
                return None, None
            if self._align_filter is not None:
                frames = self._align_filter.process(frames)
                if frames is None:
                    self._record_frame_failure("software D2C returned no frames")
                    return None, None

            color_bgr = None
            cf = frames.get_color_frame()
            if cf is not None:
                w, h = cf.get_width(), cf.get_height()
                raw = np.asanyarray(cf.get_data(), dtype=np.uint8)
                fmt = cf.get_format()
                try:
                    if fmt == OBFormat.MJPG:
                        color_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    elif fmt == OBFormat.RGB:
                        color_bgr = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
                    else:
                        color_bgr = raw.reshape(h, w, 3)
                except Exception:
                    pass

            depth_mm = None
            df = frames.get_depth_frame()
            if df is not None:
                dw, dh = df.get_width(), df.get_height()
                depth_raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(dh, dw)
                depth_scale = self._depth_scale_mm
                try:
                    depth_scale = float(df.get_depth_scale())
                    self._depth_scale_mm = depth_scale
                except Exception:
                    pass
                depth_mm = np.clip(
                    np.rint(depth_raw.astype(np.float32) * depth_scale),
                    0,
                    np.iinfo(np.uint16).max,
                ).astype(np.uint16)

            if color_bgr is None or depth_mm is None:
                self._record_frame_failure("missing color or depth frame")
            elif color_bgr.shape[:2] != depth_mm.shape:
                self._record_frame_failure(
                    f"RGB/depth alignment mismatch: RGB={color_bgr.shape[:2]}, "
                    f"depth={depth_mm.shape}"
                )
                return None, None
            else:
                self._reset_frame_failures()
            return color_bgr, depth_mm
        except CameraFrameError:
            raise
        except Exception as exc:
            self._record_frame_failure(str(exc))
            return None, None

    # Intrinsics

    @property
    def K(self) -> np.ndarray:
        if self._K is None:
            raise RuntimeError("Camera is not open")
        return self._K

    @property
    def D(self) -> np.ndarray:
        if self._D is None:
            raise RuntimeError("Camera is not open")
        return self._D

    # Internals

    def _hardware_d2c_config(self, Config, OBAlignMode, OBFormat, OBSensorType):
        """Build a config only from a device-approved HW D2C profile pair."""
        try:
            color_profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                color_profile = color_profiles.get_video_stream_profile(
                    self._w, self._h, OBFormat.RGB, self._fps
                )
            except Exception:
                return None
            depth_profiles = self._pipeline.get_d2c_depth_profile_list(
                color_profile, OBAlignMode.HW_MODE
            )
            if len(depth_profiles) == 0:
                return None
            candidates = [depth_profiles[index] for index in range(len(depth_profiles))]
            candidates.sort(key=lambda profile: (
                profile.get_fps() != color_profile.get_fps(),
                profile.get_width() != color_profile.get_width(),
                profile.get_height() != color_profile.get_height(),
            ))
            depth_profile = candidates[0]
            cfg = Config()
            cfg.enable_stream(depth_profile)
            cfg.enable_stream(color_profile)
            cfg.set_align_mode(OBAlignMode.HW_MODE)
            return cfg
        except Exception as exc:
            print(f"[OrbbecGemini2] Hardware D2C unavailable: {exc}")
            return None

    def _profile(self, profiles, formats):
        for image_format in formats:
            try:
                return profiles.get_video_stream_profile(
                    self._w, self._h, image_format, self._fps
                )
            except Exception:
                continue
        return profiles.get_default_video_stream_profile()

    def _load_distortion(self) -> np.ndarray:
        """Load distortion; fall back to zeros for invalid calibration."""
        if self._calib_dir is not None:
            npz_path = self._calib_dir / "intrinsics.npz"
            if npz_path.exists():
                try:
                    data = np.load(str(npz_path))
                    D = data["dist_coeffs"].flatten()
                    if abs(D[0]) > 5.0:
                        print(f"[OrbbecGemini2] Invalid k1={D[0]:.2f}; using zero distortion")
                        return np.zeros((1, 5), dtype=np.float64)
                    return D.reshape(1, -1)
                except Exception:
                    pass
        return np.zeros((1, 5), dtype=np.float64)
