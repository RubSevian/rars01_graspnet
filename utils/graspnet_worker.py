"""Isolated GraspNet CUDA worker.

GraspNet inference runs in a spawned process so CUDA/Python work cannot starve
the 100 Hz RARS01 command thread in the mission process.  The worker never
opens the camera or serial port and returns only plain NumPy data over a pipe.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np


def _worker_main(connection: Any, checkpoint: str, num_view: int) -> None:
    try:
        from utils import graspnet_utils

        net = graspnet_utils.build_net(Path(checkpoint), num_view)
        connection.send({"type": "ready"})
        while True:
            request = connection.recv()
            if request.get("type") == "stop":
                return
            if request.get("type") != "infer":
                continue

            started = time.monotonic()
            end_points, _, raw_cloud = graspnet_utils.build_end_points(
                request["color_bgr"], request["depth_mm"], request["K"],
                request["num_point"], request["min_depth"], request["max_depth"],
            )
            grasps, counts = graspnet_utils.infer_grasps(
                net, end_points, raw_cloud, request["collision_thresh"],
                request["voxel_size"],
            )
            pre_bbox = grasps.grasp_group_array.copy()
            bbox = request.get("bbox_xyxy")
            if bbox is not None:
                grasps = graspnet_utils.filter_grasps_by_bbox(
                    grasps, tuple(bbox), request["K"],
                    margin_px=request["target_margin_px"],
                    expand_ratio=request["target_expand_ratio"],
                    image_shape=request["color_bgr"].shape[:2],
                )
            bbox_grasps = grasps.grasp_group_array.copy()
            grasps = graspnet_utils.filter_grasps_by_width(
                grasps, request.get("max_grasp_width_m")
            )
            grasps = graspnet_utils.filter_grasps_by_depth(
                grasps, request.get("max_grasp_depth_m")
            )
            connection.send({
                "type": "result",
                "grasps": grasps.grasp_group_array,
                "pre_bbox_grasps": pre_bbox,
                "bbox_grasps": bbox_grasps,
                "counts": counts,
                "elapsed_s": time.monotonic() - started,
            })
    except BaseException as exc:
        try:
            connection.send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class GraspNetWorker:
    """Persistent spawned process that owns the GraspNet model and CUDA work."""

    def __init__(self, checkpoint: Path, num_view: int, startup_timeout_s: float = 120.0):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_worker_main,
            args=(child, str(checkpoint), int(num_view)),
            name="graspnet-cuda-worker",
            daemon=True,
        )
        self._process.start()
        child.close()
        message = self._receive(startup_timeout_s, "GraspNet worker startup")
        if message.get("type") != "ready":
            self.close(force=True)
            raise RuntimeError(message.get("error", "GraspNet worker failed to start"))

    def infer(
        self,
        color_bgr: np.ndarray,
        depth_mm: np.ndarray,
        K: np.ndarray,
        *,
        num_point: int,
        min_depth: float,
        max_depth: float,
        collision_thresh: float,
        voxel_size: float,
        bbox_xyxy: Optional[tuple[int, int, int, int]],
        target_margin_px: int,
        target_expand_ratio: float,
        max_grasp_width_m: Optional[float],
        max_grasp_depth_m: Optional[float],
        timeout_s: float,
    ) -> dict[str, Any]:
        if not self._process.is_alive():
            raise RuntimeError("GraspNet worker is not running")
        self._connection.send({
            "type": "infer",
            "color_bgr": np.ascontiguousarray(color_bgr),
            "depth_mm": np.ascontiguousarray(depth_mm),
            "K": np.asarray(K, dtype=np.float64),
            "num_point": int(num_point),
            "min_depth": float(min_depth),
            "max_depth": float(max_depth),
            "collision_thresh": float(collision_thresh),
            "voxel_size": float(voxel_size),
            "bbox_xyxy": bbox_xyxy,
            "target_margin_px": int(target_margin_px),
            "target_expand_ratio": float(target_expand_ratio),
            "max_grasp_width_m": max_grasp_width_m,
            "max_grasp_depth_m": max_grasp_depth_m,
        })
        message = self._receive(timeout_s, "GraspNet inference")
        if message.get("type") == "error":
            raise RuntimeError(message["error"])
        if message.get("type") != "result":
            raise RuntimeError(f"Unexpected GraspNet worker response: {message.get('type')}")
        return message

    def _receive(self, timeout_s: float, operation: str) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._connection.poll(0.05):
                return self._connection.recv()
            if not self._process.is_alive():
                raise RuntimeError(f"{operation} process exited with code {self._process.exitcode}")
        raise TimeoutError(f"{operation} exceeded {timeout_s:.1f}s")

    def close(self, force: bool = False) -> None:
        if getattr(self, "_process", None) is None:
            return
        if self._process.is_alive() and not force:
            try:
                self._connection.send({"type": "stop"})
                self._process.join(timeout=5.0)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._connection.close()
