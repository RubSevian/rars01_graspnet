#!/usr/bin/env python3
"""Reserved entry point for a future RARS01 pick-and-place workflow.

The previous hardware-specific placement sequence is not a validated RARS01
workflow. Exit before importing a camera or enabling any motor.
"""
if __name__ == "__main__":
    raise SystemExit(
        "Pick-and-place is not implemented for RARS01. "
        "Use scripts/grasp.py for the supported grasp workflow."
    )
