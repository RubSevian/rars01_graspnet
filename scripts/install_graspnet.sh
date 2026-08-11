#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRASPNET_DIR="$PROJECT_DIR/sdk/graspnet-baseline"
API_DIR="$GRASPNET_DIR/graspnetAPI"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

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
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment: $PYTHON_BIN"
  echo "Create it first with: uv venv --python 3.12"
  exit 1
fi

MACHINE_ARCH="$(uname -m)"
if [[ "$MACHINE_ARCH" == "aarch64" ]]; then
  DEFAULT_TORCH_ARCH="8.7"
else
  DEFAULT_TORCH_ARCH="8.9"
fi

# Prefer the JetPack-managed CUDA symlink, but accept an installation where
# only a versioned toolkit directory exists (for example cuda-13.2 on JP 7.2).
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
  else
    CUDA_CANDIDATE="$(find /usr/local -maxdepth 1 -type d -name 'cuda-*' -print | sort -V | tail -n 1)"
    CUDA_HOME="${CUDA_CANDIDATE:-/usr/local/cuda}"
  fi
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$DEFAULT_TORCH_ARCH}"

echo "Building GraspNet CUDA extensions for $MACHINE_ARCH (SM $TORCH_CUDA_ARCH_LIST)"

cd "$PROJECT_DIR"
"$PYTHON_BIN" scripts/check_compute.py
install -m 0644 "$PROJECT_DIR/patches/graspnetAPI/__init__.py" \
  "$API_DIR/graspnetAPI/__init__.py"
if grep -q '^import open3d as o3d$' "$API_DIR/graspnetAPI/grasp.py"; then
  git -C "$API_DIR" apply "$PROJECT_DIR/patches/graspnetAPI/optional_open3d.patch"
fi
if grep -q '^import open3d as o3d$' "$GRASPNET_DIR/utils/collision_detector.py"; then
  git -C "$GRASPNET_DIR" apply "$PROJECT_DIR/patches/graspnet-baseline/optional_open3d.patch"
fi
uv pip install --python "$PYTHON_BIN" --reinstall "$API_DIR" --no-deps
uv pip install --python "$PYTHON_BIN" --reinstall "$GRASPNET_DIR/pointnet2" --no-build-isolation
uv pip install --python "$PYTHON_BIN" --reinstall "$GRASPNET_DIR/knn" --no-build-isolation

"$PYTHON_BIN" scripts/check_graspnet.py --allow-missing-checkpoint
