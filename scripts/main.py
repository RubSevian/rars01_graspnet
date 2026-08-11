#!/usr/bin/env python3
"""Compatibility entry point for the RARS01 visual grasp workflow.

Use ``scripts/grasp.py`` directly in new commands.  This wrapper keeps the
old ``scripts/main.py`` invocation working without importing reBot code.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("grasp.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
