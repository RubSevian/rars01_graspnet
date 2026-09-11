"""GraspNet-baseline adapter. No camera, robot or ROS imports live here."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import Detection2D, GraspCandidate, Pose, RgbdFrame


@dataclass(frozen=True)
class GraspResult:
    candidates: tuple[GraspCandidate, ...]
    raw_cloud_xyz_m: np.ndarray
    raw_cloud_rgb: np.ndarray
    decoded_count: int
    collision_removed: int
    target: Detection2D | None


def prepare_graspnet_imports(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    required = (root / "models", root / "utils", root / "pointnet2", root / "graspnetAPI")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete graspnet-baseline checkout; missing: " + ", ".join(missing))
    for path in (root, *required):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


class GraspNetEstimator:
    """CUDA GraspNet inference with collision and YOLO target filtering."""

    def __init__(self, root: str | Path, checkpoint: str | Path, *, num_view: int = 300,
                 num_point: int = 20000, collision_thresh: float = 0.01,
                 voxel_size_m: float = 0.01, min_depth_m: float = 0.08,
                 max_depth_m: float = 1.0, top_k: int = 30,
                 target_margin_px: int = 12, target_expand_ratio: float = 1.1,
                 max_grasp_width_m: float | None = None,
                 max_grasp_depth_m: float | None = None):
        self.root = prepare_graspnet_imports(root)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"GraspNet checkpoint not found: {self.checkpoint}\n"
                "Download official checkpoint-rs.tar and place it at this path."
            )
        try:
            import torch
            from collision_detector import ModelFreeCollisionDetector
            from graspnet import GraspNet, pred_decode
            from graspnetAPI import GraspGroup
        except ImportError as exc:
            raise RuntimeError(
                "GraspNet runtime is incomplete. Run: bash scripts/install_graspnet.sh"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("GraspNet pointnet2 operators require an available CUDA GPU")

        self.torch = torch
        self.GraspGroup = GraspGroup
        self.pred_decode = pred_decode
        self.collision_detector = ModelFreeCollisionDetector
        self.device = torch.device("cuda:0")
        self.num_point = int(num_point)
        self.collision_thresh = float(collision_thresh)
        self.voxel_size_m = float(voxel_size_m)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.top_k = int(top_k)
        self.target_margin_px = int(target_margin_px)
        self.target_expand_ratio = float(target_expand_ratio)
        self.max_grasp_width_m = max_grasp_width_m
        self.max_grasp_depth_m = max_grasp_depth_m

        self.net = GraspNet(
            input_feature_dim=0, num_view=int(num_view), num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False,
        ).to(self.device)
        checkpoint_data = torch.load(str(self.checkpoint), map_location=self.device,
                                     weights_only=False)
        self.net.load_state_dict(checkpoint_data["model_state_dict"])
        self.net.eval()
        print(f"GraspNet: {self.checkpoint.name}, epoch={checkpoint_data.get('epoch', '?')}, device={self.device}")

    def infer(self, frame: RgbdFrame, detections: list[Detection2D],
              target_class: str | None = None) -> GraspResult:
        target = select_target(detections, target_class)
        if detections and target is None:
            return GraspResult((), np.empty((0, 3)), np.empty((0, 3)), 0, 0, None)
        points, colors, sampled, sampled_colors = _frame_cloud(
            frame, self.num_point, self.min_depth_m, self.max_depth_m
        )
        end_points = {
            "point_clouds": self.torch.from_numpy(sampled[np.newaxis]).to(
                self.device, non_blocking=True
            ),
            "cloud_colors": sampled_colors,
        }
        with self.torch.no_grad():
            decoded = self.pred_decode(self.net(end_points))[0].detach().cpu().numpy()
        grasps = self.GraspGroup(decoded)
        decoded_count = len(grasps)
        collision_removed = 0
        if len(grasps) and self.collision_thresh > 0:
            detector = self.collision_detector(points, voxel_size=self.voxel_size_m)
            collision = detector.detect(
                grasps, approach_dist=0.05, collision_thresh=self.collision_thresh
            )
            collision_removed = int(np.count_nonzero(collision))
            grasps = grasps[~collision]
        if target is not None:
            grasps = _filter_target(
                grasps, target, frame, self.target_margin_px, self.target_expand_ratio
            )
        if self.max_grasp_width_m is not None and len(grasps):
            grasps = grasps[np.asarray(grasps.widths) <= float(self.max_grasp_width_m)]
        if self.max_grasp_depth_m is not None and len(grasps):
            grasps = grasps[np.asarray(grasps.depths) <= float(self.max_grasp_depth_m)]
        try:
            grasps = grasps.nms()
        except (ImportError, ModuleNotFoundError):
            pass
        grasps.sort_by_score()
        grasps = grasps[:self.top_k]
        candidates = tuple(
            GraspCandidate(
                header=frame.header,
                pose=Pose(np.asarray(grasp.translation, dtype=np.float64).copy(),
                          np.asarray(grasp.rotation_matrix, dtype=np.float64).copy()),
                score=float(grasp.score), width_m=float(grasp.width),
            )
            for grasp in grasps
        )
        return GraspResult(candidates, points, colors, decoded_count, collision_removed, target)


def estimator_from_config(config: dict[str, Any], resolve_path) -> GraspNetEstimator:
    gc = config["graspnet"]
    hardware = config.get("robot", {}).get("rars01", {})
    return GraspNetEstimator(
        resolve_path(config, gc["root"]), resolve_path(config, gc["checkpoint"]),
        num_view=int(gc.get("num_view", 300)), num_point=int(gc.get("num_point", 20000)),
        collision_thresh=float(gc.get("collision_thresh", 0.01)),
        voxel_size_m=float(gc.get("voxel_size_m", 0.01)),
        min_depth_m=float(gc.get("min_depth_m", 0.08)),
        max_depth_m=float(gc.get("max_depth_m", 1.0)), top_k=int(gc.get("top_k", 30)),
        target_margin_px=int(gc.get("target_margin_px", 12)),
        target_expand_ratio=float(gc.get("target_expand_ratio", 1.1)),
        max_grasp_width_m=gc.get("max_grasp_width_m", hardware.get("max_grasp_width_m")),
        max_grasp_depth_m=gc.get("max_grasp_depth_m", hardware.get("max_grasp_depth_m")),
    )


def select_target(detections: list[Detection2D], target_class: str | None) -> Detection2D | None:
    if not detections:
        return None
    candidates = detections
    if target_class:
        wanted = target_class.casefold()
        candidates = [item for item in detections if item.class_name.casefold() == wanted]
        if not candidates:
            candidates = [item for item in detections if wanted in item.class_name.casefold()]
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def _frame_cloud(frame: RgbdFrame, num_point: int, min_depth_m: float,
                 max_depth_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth_m = frame.depth_mm.astype(np.float32) / 1000.0
    valid = (depth_m > min_depth_m) & (depth_m < max_depth_m)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        raise RuntimeError("No valid depth pixels in configured GraspNet range")
    z = depth_m[rows, cols]
    K = frame.intrinsics.K
    points = np.column_stack(((cols - K[0, 2]) * z / K[0, 0],
                              (rows - K[1, 2]) * z / K[1, 1], z)).astype(np.float32)
    rgb = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    colors = rgb[rows, cols]
    rng = np.random.default_rng()
    indices = rng.choice(len(points), size=num_point, replace=len(points) < num_point)
    return points, colors, points[indices], colors[indices]


def _filter_target(grasps, target: Detection2D, frame: RgbdFrame,
                   margin_px: int, expand_ratio: float):
    xyz = np.asarray(grasps.translations, dtype=np.float64)
    z = xyz[:, 2]
    K = frame.intrinsics.K
    u = K[0, 0] * xyz[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * xyz[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
    h, w = frame.depth_mm.shape
    ui = np.rint(u).astype(int)
    vi = np.rint(v).astype(int)
    inside = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if target.mask is not None:
        mask = target.mask.astype(np.uint8)
        if margin_px > 0:
            size = 2 * int(margin_px) + 1
            mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
        keep = inside.copy()
        keep[inside] &= mask[vi[inside], ui[inside]].astype(bool)
        return grasps[keep]
    x1, y1, x2, y2 = target.bbox_xyxy
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    half_w = 0.5 * (x2 - x1) * max(1.0, expand_ratio) + margin_px
    half_h = 0.5 * (y2 - y1) * max(1.0, expand_ratio) + margin_px
    x1, x2, y1, y2 = cx - half_w, cx + half_w, cy - half_h, cy + half_h
    return grasps[inside & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)]
