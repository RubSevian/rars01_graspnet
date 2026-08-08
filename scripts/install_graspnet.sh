#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRASPNET_DIR="$PROJECT_DIR/sdk/graspnet-baseline"
API_DIR="$GRASPNET_DIR/graspnetAPI"

if [[ ! -d "$GRASPNET_DIR/.git" ]]; then
  echo "Missing $GRASPNET_DIR"
  echo "Run: git clone --depth 1 https://github.com/graspnet/graspnet-baseline.git $GRASPNET_DIR"
  exit 1
fi
if [[ ! -d "$API_DIR/.git" ]]; then
  echo "Missing $API_DIR"
  echo "Run: git clone --depth 1 https://github.com/graspnet/graspnetAPI.git $API_DIR"
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

cd "$PROJECT_DIR"
uv run python scripts/check_compute.py
install -m 0644 "$PROJECT_DIR/patches/graspnetAPI/__init__.py" \
  "$API_DIR/graspnetAPI/__init__.py"
uv pip install --reinstall "$API_DIR" --no-deps
uv pip install --reinstall "$GRASPNET_DIR/pointnet2" --no-build-isolation
uv pip install --reinstall "$GRASPNET_DIR/knn" --no-build-isolation

uv run python scripts/check_graspnet.py --allow-missing-checkpoint
