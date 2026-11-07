#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRASPNET_DIR="${GRASPNET_ROOT:-$PROJECT_DIR/sdk/graspnet-baseline}"
API_DIR="$GRASPNET_DIR/graspnetAPI"
WORKSPACE="$(cd -- "$PROJECT_DIR/../.." && pwd)"
export MAX_JOBS="${MAX_JOBS:-1}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

if [[ ! -d "$GRASPNET_DIR/.git" ]]; then
  echo "Missing $GRASPNET_DIR"
  echo "Run: python3 $WORKSPACE/tools/environment/prepare_grasp_sources.py --destination $GRASPNET_DIR"
  exit 1
fi
if [[ ! -d "$API_DIR/.git" ]]; then
  echo "Missing $API_DIR"
  echo "Prepare baseline and API together with tools/environment/prepare_grasp_sources.py"
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
"$PYTHON_BIN" scripts/check_compute.py --require-cuda
"$PYTHON_BIN" "$WORKSPACE/tools/environment/prepare_grasp_sources.py" --destination "$GRASPNET_DIR" --verify
uv pip install --python "$PYTHON_BIN" --reinstall "$API_DIR" --no-deps
uv pip install --python "$PYTHON_BIN" --reinstall "$GRASPNET_DIR/pointnet2" --no-build-isolation --no-deps
uv pip install --python "$PYTHON_BIN" --reinstall "$GRASPNET_DIR/knn" --no-build-isolation --no-deps

"$PYTHON_BIN" scripts/check_graspnet.py --config "$PROJECT_DIR/config/jetson_orin_nano.yaml" --cuda-smoke
