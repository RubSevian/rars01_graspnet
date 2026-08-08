#!/usr/bin/env python3
from __future__ import annotations

import argparse

import cv2
import numpy as np

from rars01_graspnet.camera import camera_from_config
from rars01_graspnet.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Gemini 336 RGB/depth/alignment")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)

    with camera_from_config(config) as camera:
        camera.warm_up()
        print("RGB intrinsics K:\n", camera.K)
        print("Q/Esc closes the window")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(frame.depth_mm, alpha=255.0 / 1500.0), cv2.COLORMAP_TURBO
            )
            valid = frame.depth_mm > 0
            median = int(np.median(frame.depth_mm[valid])) if np.any(valid) else 0
            output = np.hstack((frame.color_bgr, depth_vis))
            cv2.putText(output, f"median depth: {median} mm", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Gemini 336: RGB | aligned depth", output)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

