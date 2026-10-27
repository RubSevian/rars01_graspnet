"""Isolated GraspNet CUDA worker.

GraspNet inference runs in a spawned process so CUDA/Python work cannot starve
the 100 Hz RARS01 command thread in the mission process.  The worker never
opens the camera or serial port and returns only plain NumPy data over a pipe.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import time
import threading
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
            if request.get("type") == "warmup":
                torch = graspnet_utils.torch
                # Exercise actual CUDA pointnet2 + decode without requiring an
                # object/depth in the startup HOME camera view.
                rng = np.random.default_rng(0)
                points = rng.uniform([-.1, -.1, .2], [.1, .1, .4],
                                     (1, request['num_point'], 3)).astype(np.float32)
                with torch.no_grad():
                    outputs = net({'point_clouds': torch.from_numpy(points).cuda(),
                                   'cloud_colors': np.zeros((request['num_point'], 3))})
                    decoded = graspnet_utils.pred_decode(outputs)
                    if len(decoded)!=1 or not all(bool(torch.isfinite(x).all()) for x in decoded):
                        raise RuntimeError('GraspNet warmup nonfinite output')
                    torch.cuda.synchronize()
                del outputs, decoded, points
                # Startup is followed by idle NAV: return scratch allocations
                # to the shared Jetson memory pool, keep model weights resident.
                torch.cuda.empty_cache()
                connection.send({'type': 'warmup_ready',
                                 'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                                 'cuda_reserved_bytes': torch.cuda.memory_reserved()})
                continue
            if request.get("type") != "infer":
                continue

            started = time.monotonic()
            sampling = {}
            end_points, _, raw_cloud = graspnet_utils.build_end_points(
                request["color_bgr"], request["depth_mm"], request["K"],
                request["num_point"], request["min_depth"], request["max_depth"],
                visualize=False,
                target_mask=request.get('target_mask'), mask_source=request.get('mask_source','unknown'),
                target_roi=request.get('target_roi'), sampling_diagnostics=sampling,
            )
            stage_arrays={} if request.get("include_diagnostics",True) else None
            grasps, counts = graspnet_utils.infer_grasps(
                net, end_points, raw_cloud, request["collision_thresh"],
                request["voxel_size"], diagnostics=stage_arrays,
                minimum_grasp_depth_m=request.get("minimum_grasp_depth_m",0.),
                max_grasp_depth_m=request.get("max_grasp_depth_m"),
                cloud_collision=request.get("cloud_collision"),
            )
            filter_counts = dict(counts, after_cloud_collision=len(grasps.grasp_group_array))
            diagnostics=request.get("include_diagnostics",True)
            pre_bbox = grasps.grasp_group_array.copy() if diagnostics else None
            bbox = request.get("bbox_xyxy")
            if bbox is not None:
                grasps = graspnet_utils.filter_grasps_by_bbox(
                    grasps, tuple(bbox), request["K"],
                    margin_px=request["target_margin_px"],
                    expand_ratio=request["target_expand_ratio"],
                    image_shape=request["color_bgr"].shape[:2],
                )
            bbox_grasps = grasps.grasp_group_array.copy() if diagnostics else None
            filter_counts['after_bbox'] = len(grasps.grasp_group_array)
            grasps = graspnet_utils.filter_grasps_by_width(
                grasps, request.get("max_grasp_width_m")
            )
            filter_counts['after_width'] = len(grasps.grasp_group_array)
            width_grasps = grasps.grasp_group_array.copy() if diagnostics else None
            grasps = graspnet_utils.filter_grasps_by_depth(
                grasps, request.get("max_grasp_depth_m")
            )
            filter_counts['after_depth'] = len(grasps.grasp_group_array)
            response={
                "type": "result",
                "grasps": grasps.grasp_group_array,
                "counts": counts,
                "filter_counts": filter_counts,
                "filter_parameters": dict(bbox_xyxy=bbox, target_roi=sampling,
                    target_margin_px=request['target_margin_px'],
                    target_expand_ratio=request['target_expand_ratio'],
                    max_grasp_width_m=request.get('max_grasp_width_m'),
                    max_grasp_depth_m=request.get('max_grasp_depth_m'),
                    minimum_grasp_depth_m=request.get('minimum_grasp_depth_m',0.),
                    cloud_collision=request.get('cloud_collision'),
                    collision_thresh=request['collision_thresh'],voxel_size=request['voxel_size']),
                "elapsed_s": time.monotonic() - started,
            }
            if diagnostics:response.update(pre_bbox_grasps=pre_bbox,bbox_grasps=bbox_grasps,
                width_grasps=width_grasps,**stage_arrays)
            connection.send(response)
            # recv() may block throughout NAV. Do not retain the last RGB-D
            # request or CUDA feature tensors while waiting for the next job.
            del request, end_points, raw_cloud, grasps, pre_bbox, bbox_grasps, width_grasps, stage_arrays, counts, response, _
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
        self._close_lock = threading.Lock()
        self._closed = False
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
        try:
            message = self._receive(startup_timeout_s, "GraspNet worker startup")
        except BaseException:
            self.close(force=True)
            raise
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
        include_diagnostics: bool = True,
        target_mask=None,
        mask_source: str = 'unknown',
        target_roi=None,
        minimum_grasp_depth_m: float = 0.0,
        cloud_collision: Optional[dict] = None,
    ) -> dict[str, Any]:
        if not self._process.is_alive():
            raise RuntimeError("GraspNet worker is not running")
        self._connection.send({
            "type": "infer",
            "include_diagnostics": bool(include_diagnostics),
            "target_mask": None if target_mask is None else np.ascontiguousarray(target_mask, dtype=bool),
            "mask_source": mask_source,
            "target_roi": target_roi,
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
            "minimum_grasp_depth_m": float(minimum_grasp_depth_m),
            "cloud_collision": cloud_collision,
        })
        message = self._receive(timeout_s, "GraspNet inference")
        if message.get("type") == "error":
            raise RuntimeError(message["error"])
        if message.get("type") != "result":
            raise RuntimeError(f"Unexpected GraspNet worker response: {message.get('type')}")
        return message

    def is_alive(self) -> bool:
        return not self._closed and self._process.is_alive()

    def warmup(self, num_point: int, timeout_s: float) -> None:
        if not self.is_alive():raise RuntimeError('GraspNet worker is not running')
        self._connection.send({'type': 'warmup', 'num_point': int(num_point)})
        message = self._receive(timeout_s, 'GraspNet warmup')
        if message.get('type') != 'warmup_ready':
            raise RuntimeError(message.get('error', 'GraspNet warmup failed'))
        self.last_warmup_memory={key:message[key] for key in ('cuda_allocated_bytes','cuda_reserved_bytes')}

    def _receive(self, timeout_s: float, operation: str) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._connection.poll(0.05):
                return self._connection.recv()
            if not self._process.is_alive():
                raise RuntimeError(f"{operation} process exited with code {self._process.exitcode}")
        raise TimeoutError(f"{operation} exceeded {timeout_s:.1f}s")

    def close(self, force: bool = False) -> None:
        with self._close_lock:
            if self._closed:return
            self._closed=True
            self._close(force)

    def _close(self, force: bool) -> None:
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
        if self._process.is_alive():
            self._process.kill() # Own inference-only subprocess; no robot owner.
            self._process.join(timeout=2.0)
        self._connection.close()
        if self._process.is_alive():raise RuntimeError('GraspNet worker did not exit')
