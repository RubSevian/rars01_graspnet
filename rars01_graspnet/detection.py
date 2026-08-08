from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import Detection2D, RgbdFrame


def select_torch_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class YoloDetector:
    """Ultralytics adapter with transport-neutral Detection2D output."""

    def __init__(self, model_path: str | Path, *, device: str = "auto", confidence: float = 0.5,
                 iou: float = 0.45, image_size: int = 640,
                 classes: list[str] | None = None, use_open_vocabulary: bool = False):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is missing: uv sync --extra vision") from exc
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self.device = select_torch_device(device)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.model = YOLO(str(self.model_path))
        if use_open_vocabulary and classes:
            if not hasattr(self.model, "set_classes"):
                raise RuntimeError(f"Model {self.model_path.name} does not support open-vocabulary classes")
            self.model.set_classes(classes)
        print(f"YOLO: {self.model_path.name}, device={self.device}, imgsz={self.image_size}")

    def detect(self, frame: RgbdFrame) -> list[Detection2D]:
        results = self.model.predict(
            frame.color_bgr, verbose=False, device=self.device,
            conf=self.confidence, iou=self.iou, imgsz=self.image_size,
        )
        detections: list[Detection2D] = []
        height, width = frame.color_bgr.shape[:2]
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for index, box in enumerate(boxes):
                xyxy = np.asarray(box.xyxy[0].detach().cpu(), dtype=np.float64)
                x1, y1, x2, y2 = _clip_bbox(xyxy, width, height)
                class_id = int(np.asarray(box.cls[0].detach().cpu()).item())
                confidence = float(np.asarray(box.conf[0].detach().cpu()).item())
                names: Any = getattr(result, "names", getattr(self.model, "names", {}))
                name = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
                mask = _result_mask(result, index, (height, width), (x1, y1, x2, y2))
                center = _object_center_3d(frame, mask)
                detections.append(Detection2D(
                    header=frame.header, class_name=name, confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2), mask=mask, center_xyz_m=center,
                ))
        return detections


def detector_from_config(config: dict, resolve_path) -> YoloDetector:
    yc = config["yolo"]
    return YoloDetector(
        resolve_path(config, yc["model"]), device=str(yc.get("device", "auto")),
        confidence=float(yc.get("confidence", 0.5)), iou=float(yc.get("iou", 0.45)),
        image_size=int(yc.get("image_size", 640)), classes=list(yc.get("classes", [])),
        use_open_vocabulary=bool(yc.get("use_open_vocabulary", False)),
    )


def _clip_bbox(values: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = values.reshape(-1)[:4]
    return (
        int(np.clip(round(x1), 0, width - 1)), int(np.clip(round(y1), 0, height - 1)),
        int(np.clip(round(x2), 0, width - 1)), int(np.clip(round(y2), 0, height - 1)),
    )


def _result_mask(result, index: int, shape: tuple[int, int], bbox) -> np.ndarray:
    masks = getattr(getattr(result, "masks", None), "data", None)
    if masks is not None and len(masks) > index:
        mask = np.asarray(masks[index].detach().cpu(), dtype=np.float32)
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask > 0.5
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = bbox
    mask[y1:y2 + 1, x1:x2 + 1] = True
    return mask


def _object_center_3d(frame: RgbdFrame, mask: np.ndarray) -> np.ndarray | None:
    valid = mask & (frame.depth_mm > 0)
    ys, xs = np.nonzero(valid)
    if xs.size < 10:
        return None
    depths = frame.depth_mm[ys, xs]
    # Median depth rejects holes and background outliers inside a bbox.
    z_mm = float(np.median(depths))
    near = np.abs(depths.astype(np.float64) - z_mm) < max(15.0, z_mm * 0.03)
    if np.count_nonzero(near) >= 5:
        u, v, z_mm = float(np.median(xs[near])), float(np.median(ys[near])), float(np.median(depths[near]))
    else:
        u, v = float(np.median(xs)), float(np.median(ys))
    z = z_mm / 1000.0
    K = frame.intrinsics.K
    return np.array([(u - K[0, 2]) * z / K[0, 0],
                     (v - K[1, 2]) * z / K[1, 1], z], dtype=np.float64)


def draw_detections(image: np.ndarray, detections: list[Detection2D]) -> np.ndarray:
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 80), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        if detection.center_xyz_m is not None:
            x, y, z = detection.center_xyz_m
            label += f" | {x:+.3f} {y:+.3f} {z:.3f}m"
        cv2.putText(output, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 120), 1, cv2.LINE_AA)
    return output

