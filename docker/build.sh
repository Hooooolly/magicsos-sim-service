#!/bin/bash
# Build sim-interactive image with pre-compiled curobo CUDA kernels.
#
# 4-step process:
#   1. docker build -base image (no GPU needed)
#   2. Run curobo JIT compilation on GPU (produces .so files)
#   3. Build final image with baked-in .so
#   4. Import to containerd for K8s
#
# Usage:
#   ./docker/build.sh 5.1.0          # build for isaac-sim 5.1.0
#   ./docker/build.sh 4.5.0          # build for isaac-sim 4.5.0
#   GPU=2 ./docker/build.sh 5.1.0    # use GPU 2
#
# Prerequisites:
#   - Docker with GPU support (nvidia-docker)
#   - third_party/curobo/src/curobo/ exists (git submodule)
#   - Run from repo root: sim-service/
#
# Version matrix:
#   4.5.0: BASE=nvcr.io/nvidia/isaac-sim:4.5.0, CUDA_APT=11-8, ARCH=7.5
#   5.1.0: BASE=nvcr.io/nvidia/isaac-sim:5.1.0, CUDA_APT=12-8, ARCH=7.5

set -euo pipefail

VERSION="${1:?Usage: $0 <version> (e.g. 5.1.0 or 4.5.0)}"
GPU="${GPU:-0}"
IMAGE_NAME="sim-interactive"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Version-specific settings
case "$VERSION" in
  4.5.0)
    BASE_IMAGE="nvcr.io/nvidia/isaac-sim:4.5.0"
    CUDA_APT="11-8"
    ;;
  5.1.0)
    BASE_IMAGE="nvcr.io/nvidia/isaac-sim:5.1.0"
    CUDA_APT="12-8"
    ;;
  *)
    echo "ERROR: Unknown version $VERSION. Supported: 4.5.0, 5.1.0"
    exit 1
    ;;
esac

TORCH_ARCH="${TORCH_ARCH:-7.5}"
BUILD_CTX=$(mktemp -d)
CACHE_DIR="${REPO_DIR}/docker/curobo-cache-${VERSION}"

echo "[$(date)] Building ${IMAGE_NAME}:${VERSION}"
echo "  BASE_IMAGE=$BASE_IMAGE"
echo "  CUDA_APT=$CUDA_APT"
echo "  TORCH_ARCH=$TORCH_ARCH"
echo "  GPU=$GPU"
echo "  BUILD_CTX=$BUILD_CTX"

# --- Step 1: Build base image ---
echo ""
echo "[$(date)] Step 1: Docker build ${IMAGE_NAME}:${VERSION}-base"

cp "$SCRIPT_DIR/Dockerfile.sim-interactive" "$BUILD_CTX/Dockerfile"
cp -r "$REPO_DIR/third_party/curobo/src/curobo/" "$BUILD_CTX/curobo/"

docker build -t "${IMAGE_NAME}:${VERSION}-base" \
  -f "$BUILD_CTX/Dockerfile" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg CUDA_APT_VERSION="$CUDA_APT" \
  --build-arg TORCH_CUDA_ARCH_LIST="$TORCH_ARCH" \
  "$BUILD_CTX"

echo "[$(date)] Step 1 DONE"

# --- Step 2: Curobo JIT compilation on GPU ---
echo ""
echo "[$(date)] Step 2: Curobo JIT compilation on GPU $GPU"

mkdir -p "$CACHE_DIR"

docker run --gpus "device=${GPU}" --rm \
  --name "curobo-compile-${VERSION//\./-}" \
  --entrypoint bash \
  -v "$CACHE_DIR:/opt/curobo-cache" \
  -v "$SCRIPT_DIR/compile_curobo.py:/tmp/compile_curobo.py:ro" \
  "${IMAGE_NAME}:${VERSION}-base" \
  -c "
ln -sf /usr/local/cuda-active /usr/local/cuda-active 2>/dev/null || true
for d in /usr/local/cuda-12.8 /usr/local/cuda-11.8 /usr/local/cuda; do
  if [ -x \"\$d/bin/nvcc\" ]; then
    export CUDA_HOME=\$d
    export PATH=\$d/bin:\$PATH
    ln -sf \$d /usr/local/cuda-active
    break
  fi
done
source /isaac-sim/setup_python_env.sh
export PYTHONPATH=/opt/curobo-src:\$PYTHONPATH
export TORCH_CUDA_ARCH_LIST=${TORCH_ARCH}
echo \"nvcc:\" && nvcc --version | tail -1
/isaac-sim/kit/python/bin/python3 /tmp/compile_curobo.py
"

echo "[$(date)] Step 2 DONE. Cached .so files:"
find "$CACHE_DIR" -name "*.so" -ls

# --- Step 3: Build final image with cached .so ---
echo ""
echo "[$(date)] Step 3: Build final image ${IMAGE_NAME}:${VERSION}"

cat > "$BUILD_CTX/Dockerfile.final" << 'DEOF'
ARG BASE
FROM ${BASE}
RUN for d in /usr/local/cuda-12.8 /usr/local/cuda-11.8 /usr/local/cuda; do \
      if [ -x "$d/bin/nvcc" ]; then ln -sf "$d" /usr/local/cuda-active; break; fi; \
    done
COPY curobo-cache/ /opt/curobo-cache/
DEOF

cp -r "$CACHE_DIR" "$BUILD_CTX/curobo-cache"

docker build -t "${IMAGE_NAME}:${VERSION}" \
  -f "$BUILD_CTX/Dockerfile.final" \
  --build-arg "BASE=${IMAGE_NAME}:${VERSION}-base" \
  "$BUILD_CTX"

echo "[$(date)] Step 3 DONE"

# --- Step 4: Import to containerd ---
echo ""
echo "[$(date)] Step 4: Import to containerd"
echo "Run manually:"
echo "  docker save ${IMAGE_NAME}:${VERSION} -o /tmp/${IMAGE_NAME}-${VERSION}.tar"
echo "  sudo ctr -n k8s.io images import /tmp/${IMAGE_NAME}-${VERSION}.tar"
echo "  rm /tmp/${IMAGE_NAME}-${VERSION}.tar"

# Cleanup
rm -rf "$BUILD_CTX"

echo ""
echo "[$(date)] === BUILD COMPLETE ==="
echo "Docker image: ${IMAGE_NAME}:${VERSION}"
docker images | grep "${IMAGE_NAME}" | grep "${VERSION}"
