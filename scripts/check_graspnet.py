#!/usr/bin/env python3
"""Validate GraspNet sources, checkpoint and compiled CUDA operators."""
from __future__ import annotations

import argparse
import os
import importlib
import sys
from pathlib import Path

import torch

from rars01_graspnet.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = Path(os.environ.get("GRASPNET_ROOT", str(PROJECT_ROOT / "sdk" / "graspnet-baseline"))).expanduser().absolute()


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
    parser.add_argument("--cuda-smoke", action="store_true", help="Execute pointnet2, KNN and model decode on synthetic input")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser() if args.config else PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)
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

    if args.cuda_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke requires an available GPU")
        if not checkpoint.is_file():
            raise FileNotFoundError("CUDA model smoke requires the checkpoint")
        import numpy as np
        from utils.graspnet_utils import build_net, pred_decode
        from pointnet2_utils import furthest_point_sample, gather_operation
        from knn_modules import knn
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], device="cuda")
        indices = furthest_point_sample(points, 2)
        gathered = gather_operation(points.transpose(1, 2).contiguous(), indices)
        if indices.shape != (1, 2) or not bool(torch.isfinite(gathered).all()):
            raise RuntimeError("Invalid pointnet2 result")
        nearest = knn(points.transpose(1, 2).contiguous(), points.transpose(1, 2).contiguous(), 1)
        # Upstream KNN returns one-based reference indices.
        expected = torch.tensor([[[1, 2, 3]]], device="cuda")
        if not torch.equal(nearest, expected):
            raise RuntimeError(f"KNN self-query mismatch: {nearest}")
        rng = np.random.default_rng(0)
        cloud = rng.uniform([-.1, -.1, .2], [.1, .1, .4], (1, 20000, 3)).astype(np.float32)
        net = build_net(checkpoint, int(config["graspnet"].get("num_view", 300)))
        with torch.no_grad():
            decoded = pred_decode(net({"point_clouds": torch.from_numpy(cloud).cuda(),
                                       "cloud_colors": np.zeros((20000, 3))}))
        torch.cuda.synchronize()
        if len(decoded) != 1 or not decoded[0].numel() or not bool(torch.isfinite(decoded[0]).all()):
            raise RuntimeError("Invalid model decode")
        print("CUDA pointnet2, KNN and model decode: OK", tuple(decoded[0].shape))


if __name__ == "__main__":
    main()
