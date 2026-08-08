#!/usr/bin/env python3
"""Report the PyTorch device which YOLO/GraspNet will use."""
from __future__ import annotations

import os
import re
import shutil
import subprocess


def main() -> None:
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is not installed. Run: uv sync --extra vision")

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("Selected GPU:", torch.cuda.get_device_name(0))
        print("Pipeline device: cuda:0")
    else:
        print("Pipeline device: cpu")
        print("YOLO can run, but live YOLO + GraspNet will be substantially slower.")

    nvcc = shutil.which("nvcc")
    print("nvcc:", nvcc or "not found")
    nvcc_cuda = None
    if nvcc:
        completed = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=False
        )
        output = completed.stdout + completed.stderr
        match = re.search(r"release\s+(\d+\.\d+)", output)
        nvcc_cuda = match.group(1) if match else None
        print("nvcc CUDA:", nvcc_cuda or "unknown")
    print("CUDA_HOME:", os.environ.get("CUDA_HOME", "not set"))

    torch_cuda = torch.version.cuda
    if torch_cuda and nvcc_cuda and torch_cuda.split(".")[:2] != nvcc_cuda.split(".")[:2]:
        print(
            "ERROR: PyTorch CUDA and nvcc do not match. YOLO inference can work, "
            "but GraspNet CUDA extensions must not be compiled in this state."
        )
        raise SystemExit(2)
    if torch_cuda and not nvcc:
        print("ERROR: CUDA PyTorch is installed, but a CUDA compiler toolkit is missing.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
