#!/usr/bin/env python3
"""Compatibility entry point for the maintained RARS01 grasp workflow."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("grasp.py")), run_name="__main__")
