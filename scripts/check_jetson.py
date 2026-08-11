#!/usr/bin/env python3
"""Fail-fast preflight for the Jetson Orin Nano grasping profile."""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"{'OK' if ok else 'FAIL'}  {label}: {detail}")
    return ok


def main() -> int:
    ok = True
    machine = platform.machine()
    ok &= check("architecture", machine == "aarch64", machine)
    ok &= check("python", (3, 10) <= sys.version_info[:2] < (3, 13), sys.version.split()[0])
    nvcc = shutil.which("nvcc")
    if not nvcc:
        toolkits = sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"))
        nvcc = str(toolkits[-1]) if toolkits else None
    ok &= check("nvcc", nvcc is not None, nvcc or "not found")
    if nvcc:
        version = subprocess.run([nvcc, "--version"], text=True, capture_output=True, check=False)
        print((version.stdout or version.stderr).strip().splitlines()[-1])

    try:
        import torch
    except ImportError:
        print("FAIL  PyTorch: not installed; install the NVIDIA Jetson wheel first")
        return 1

    cuda_ok = torch.cuda.is_available()
    ok &= check("PyTorch CUDA", cuda_ok, f"torch={torch.__version__}, cuda={torch.version.cuda}")
    if cuda_ok:
        capability = torch.cuda.get_device_capability(0)
        ok &= check("GPU capability", capability == (8, 7), f"{capability[0]}.{capability[1]} ({torch.cuda.get_device_name(0)})")

    for module in ("pyorbbecsdk", "rars_arm_py", "pinocchio"):
        try:
            imported = importlib.import_module(module)
        except ImportError as error:
            ok = False
            print(f"FAIL  {module}: {error}")
        else:
            print(f"OK    {module}: {getattr(imported, '__file__', 'built-in')}")
    print("Next: uv run python scripts/check_camera.py --config config/jetson_orin_nano.yaml")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
