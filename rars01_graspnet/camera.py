from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .contracts import CameraIntrinsics, Header, RgbdFrame


class OrbbecCamera:
    """Small Orbbec SDK v2 adapter returning depth aligned to the RGB image."""

    def __init__(self, width: int, height: int, fps: int, timeout_ms: int = 1000,
                 frame_id: str = "camera_color_optical_frame", alignment: str = "auto"):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.timeout_ms = int(timeout_ms)
        self.frame_id = str(frame_id)
        self.alignment = str(alignment).lower()
        if self.alignment not in ("auto", "hardware", "software"):
            raise ValueError("camera.alignment must be auto, hardware, or software")
        self.sequence = 0
        self.pipeline = None
        self._align_filter = None
        self._pending_frames = None
        self.active_alignment: Optional[str] = None
        self.K: Optional[np.ndarray] = None
        self.D = np.zeros((1, 5), dtype=np.float64)

    def open(self) -> None:
        try:
            from pyorbbecsdk import (
                AlignFilter, Config, OBAlignMode, OBFormat,
                OBFrameAggregateOutputMode, OBSensorType, OBStreamType, Pipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Orbbec Python SDK is missing. Install pyorbbecsdk/pyorbbecsdk2 "
                "and verify that `import pyorbbecsdk` works."
            ) from exc

        self.pipeline = Pipeline()
        config = None

        # The earlier camera driver selected arbitrary color/depth profiles and then
        # forced HW_MODE. Gemini 336 rejects such a pair with:
        # "Current stream profile is not support hardware d2c process".
        # The official Orbbec sample first asks the device for depth profiles
        # compatible with a particular color profile.
        if self.alignment in ("auto", "hardware"):
            config = self._hardware_d2c_config(
                Config, OBAlignMode, OBFormat, OBSensorType
            )
            if config is None and self.alignment == "hardware":
                raise RuntimeError(
                    "Gemini 336 has no hardware-D2C-compatible profile for the "
                    "requested stream. Set camera.alignment: auto or software."
                )

        use_software = config is None
        if config is not None:
            try:
                self.pipeline.enable_frame_sync()
            except Exception as exc:
                print(f"[Orbbec] Frame sync warning: {exc}")
            self.pipeline.start(config)
            self.active_alignment = "hardware"
            # Some firmware accepts a HW profile but never produces a complete
            # frameset. AUTO must detect that and reopen with AlignFilter.
            for _ in range(3):
                self._pending_frames = self.pipeline.wait_for_frames(self.timeout_ms)
                if self._pending_frames is not None:
                    break
            if self._pending_frames is None:
                if self.alignment == "hardware":
                    self.pipeline.stop()
                    raise RuntimeError("Hardware D2C started but produced no RGB-D frames")
                print("[Orbbec] Hardware D2C produced no frames; falling back to software")
                self.pipeline.stop()
                self.pipeline = Pipeline()
                use_software = True

        if use_software:
            # Official fallback: start ordinary RGB + depth streams and apply
            # AlignFilter(D2C) to every frameset. Config.SW_MODE is deliberately
            # not used because the upstream example uses the software filter.
            config = Config()
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = self._profile(color_profiles, (OBFormat.MJPG, OBFormat.RGB))
            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = self._profile(depth_profiles, (OBFormat.Y16,))
            config.enable_stream(color_profile)
            config.enable_stream(depth_profile)
            try:
                config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            except Exception:
                pass
            self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            self.pipeline.start(config)
            self.active_alignment = "software"

        print(f"[Orbbec] D2C alignment: {self.active_alignment}")

        intrinsic = self.pipeline.get_camera_param().rgb_intrinsic
        self.K = np.array(
            [[intrinsic.fx, 0.0, intrinsic.cx],
             [0.0, intrinsic.fy, intrinsic.cy],
             [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def _hardware_d2c_config(self, Config, OBAlignMode, OBFormat, OBSensorType):
        """Return an official device-approved HW D2C config, or None."""
        try:
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            # Do not silently downgrade 1280x720 to 640x480 just to obtain HW
            # alignment: intrinsics, detector resolution and user config would
            # unexpectedly change. Query only the requested RGB profile.
            try:
                color_profile = color_profiles.get_video_stream_profile(
                    self.width, self.height, OBFormat.RGB, self.fps
                )
            except Exception:
                return None
            depth_profiles = self.pipeline.get_d2c_depth_profile_list(
                color_profile, OBAlignMode.HW_MODE
            )
            if len(depth_profiles) == 0:
                return None
            # Frame synchronization needs matching FPS. Prefer matching output
            # dimensions as a secondary criterion.
            candidates = [depth_profiles[index] for index in range(len(depth_profiles))]
            candidates.sort(key=lambda profile: (
                profile.get_fps() != color_profile.get_fps(),
                profile.get_width() != color_profile.get_width(),
                profile.get_height() != color_profile.get_height(),
            ))
            depth_profile = candidates[0]
            config = Config()
            config.enable_stream(depth_profile)
            config.enable_stream(color_profile)
            config.set_align_mode(OBAlignMode.HW_MODE)
            print(
                "[Orbbec] Device-approved HW D2C profiles: "
                f"color={self._profile_text(color_profile)}, "
                f"depth={self._profile_text(depth_profile)}"
            )
            return config
        except Exception as exc:
            print(f"[Orbbec] Hardware D2C unavailable: {exc}")
        return None

    @staticmethod
    def _profile_text(profile) -> str:
        try:
            return (
                f"{profile.get_width()}x{profile.get_height()}@{profile.get_fps()} "
                f"{profile.get_format()}"
            )
        except Exception:
            return repr(profile)

    def _profile(self, profiles, formats):
        for image_format in formats:
            try:
                return profiles.get_video_stream_profile(
                    self.width, self.height, image_format, self.fps
                )
            except Exception:
                continue
        return profiles.get_default_video_stream_profile()

    def read(self) -> Optional[RgbdFrame]:
        if self.pipeline is None:
            raise RuntimeError("Camera is not open")
        from pyorbbecsdk import OBFormat

        frames = self._pending_frames
        self._pending_frames = None
        if frames is None:
            frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None
        if self._align_filter is not None:
            frames = self._align_filter.process(frames)
            if frames is None:
                return None
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        width, height = color_frame.get_width(), color_frame.get_height()
        raw_color = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        if color_frame.get_format() == OBFormat.MJPG:
            color = cv2.imdecode(raw_color, cv2.IMREAD_COLOR)
        elif color_frame.get_format() == OBFormat.RGB:
            color = cv2.cvtColor(raw_color.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        else:
            color = raw_color.reshape(height, width, 3)

        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width()
        )
        scale = float(depth_frame.get_depth_scale())
        depth_mm = np.rint(depth.astype(np.float32) * scale).clip(0, 65535).astype(np.uint16)
        if color is None or color.shape[:2] != depth_mm.shape:
            raise RuntimeError(
                f"RGB/depth alignment failed: RGB={None if color is None else color.shape[:2]}, "
                f"depth={depth_mm.shape}"
            )
        self.sequence += 1
        intrinsics = CameraIntrinsics(
            K=self.K.copy(), D=self.D.copy(), width=color.shape[1], height=color.shape[0]
        )
        return RgbdFrame(
            header=Header(time.time_ns(), self.frame_id, self.sequence),
            color_bgr=color, depth_mm=depth_mm, intrinsics=intrinsics,
        )

    def warm_up(self, frames: int = 20) -> None:
        for _ in range(frames):
            self.read()

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self._align_filter = None
        self._pending_frames = None
        self.active_alignment = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


def camera_from_config(config: dict) -> OrbbecCamera:
    camera = config["camera"]
    # The project-wide camera schema uses color_width/color_height.  Keep the
    # short names as a compatibility fallback for older standalone configs.
    width = camera.get("color_width", camera.get("width"))
    height = camera.get("color_height", camera.get("height"))
    if width is None or height is None:
        raise KeyError(
            "camera.color_width and camera.color_height are required "
            "(legacy camera.width/camera.height are also accepted)"
        )
    return OrbbecCamera(
        width, height, camera["fps"], camera.get("timeout_ms", 1000),
        camera.get("frame_id", "camera_color_optical_frame"),
        camera.get("alignment", "auto"),
    )
