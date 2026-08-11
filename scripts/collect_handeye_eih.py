#!/usr/bin/env python3
"""Compatibility entry point for RARS01 eye-in-hand calibration.

Automatic collection is implemented by ``calibrate_hand_eye_auto.py``.  The
old reBot gravity-compensation implementation is intentionally not retained.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    if "--manual" in sys.argv:
        print("Manual gravity-compensation mode was reBot-specific and was removed.")
        print("Use scripts/calibrate_hand_eye.py to capture manual ArUco samples.")
        raise SystemExit(2)
    target = Path(__file__).with_name("calibrate_hand_eye_auto.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
