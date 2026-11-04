"""Resolve GraspNet source paths without importing its inference runtime."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"
_SHARED_GRASPNET_ROOT = PROJECT_ROOT.parent / "rars01_graspnet" / "sdk" / "graspnet-baseline"
GRASPNET_ROOT = Path(os.environ.get(
    "GRASPNET_ROOT",
    str(_LOCAL_GRASPNET_ROOT if _LOCAL_GRASPNET_ROOT.is_dir() else _SHARED_GRASPNET_ROOT),
)).expanduser().resolve()

def prepare_graspnet_api_imports(graspnet_root: Path = GRASPNET_ROOT) -> None:
    """Expose the grasp containers without adding inference/model directories."""
    path = str(graspnet_root / "graspnetAPI")
    if path not in sys.path:
        sys.path.insert(0, path)


def prepare_graspnet_imports(graspnet_root: Path = GRASPNET_ROOT) -> None:
    for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI"):
        path = str(graspnet_root / subdir)
        if path not in sys.path:
            sys.path.insert(0, path)
    root = str(graspnet_root)
    if root not in sys.path:
        sys.path.insert(0, root)
