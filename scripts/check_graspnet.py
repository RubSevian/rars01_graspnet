#!/usr/bin/env python3
"""Validate GraspNet sources, checkpoint and compiled CUDA operators."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"


def prepare_graspnet_imports() -> None:
    for subdir in ("models", "dataset", "utils", "pointnet2", "knn", "graspnetAPI"):
        path = str(GRASPNET_ROOT / subdir)
        if path not in sys.path:
            sys.path.insert(0, path)
    root = str(GRASPNET_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--allow-missing-checkpoint", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser() if args.config else PROJECT_ROOT / "config" / "default.yaml"
    with config_path.resolve().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    checkpoint = Path(config["graspnet"]["checkpoint"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    prepare_graspnet_imports()
    print("GraspNet root:", GRASPNET_ROOT)
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, available={torch.cuda.is_available()}")
    for module in ("pointnet2._ext", "knn_pytorch.knn_pytorch", "graspnetAPI", "graspnet"):
        imported = importlib.import_module(module)
        print(f"Import {module}: OK ({getattr(imported, '__file__', 'built-in')})")
    if checkpoint.is_file():
        print(f"Checkpoint: OK ({checkpoint}, {checkpoint.stat().st_size / 1024**2:.1f} MiB)")
    elif args.allow_missing_checkpoint:
        print(f"Checkpoint: MISSING ({checkpoint})")
    else:
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")


if __name__ == "__main__":
    main()
