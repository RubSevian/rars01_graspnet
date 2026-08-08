#!/usr/bin/env python3
"""YOLO + RGB-D preview. Does not connect to or move the robot."""
from __future__ import annotations

import argparse
import time

import cv2

from rars01_graspnet.camera import camera_from_config
from rars01_graspnet.config import load_config, resolve_path
from rars01_graspnet.detection import detector_from_config, draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="RARS01 YOLO RGB-D camera demo")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    detector = detector_from_config(config, resolve_path)
    infer_every = max(1, int(config["yolo"].get("inference_every_n_frames", 3)))
    detections, frame_count = [], 0
    last_inference_ms = 0.0

    with camera_from_config(config) as camera:
        camera.warm_up()
        print("Camera frame:", config["camera"].get("frame_id"))
        print("Q/Esc: quit. This process never connects to the robot.")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            frame_count += 1
            if frame_count == 1 or frame_count % infer_every == 0:
                started = time.perf_counter()
                detections = detector.detect(frame)
                last_inference_ms = (time.perf_counter() - started) * 1000.0
            output = draw_detections(frame.color_bgr, detections)
            status = f"{detector.device} | detections={len(detections)} | inference={last_inference_ms:.0f} ms"
            cv2.putText(output, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("RARS01 YOLO RGB-D", output)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
