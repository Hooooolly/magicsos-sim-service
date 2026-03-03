# Build sim-interactive Docker Image

## Overview

sim-interactive packages Isaac Sim + curobo (motion planning) + sim-service dependencies into a single image. We pre-compile curobo's CUDA kernels during build so pods start in ~1 second instead of 7+ minutes.

## Version Matrix

| Isaac Sim | Python | PyTorch | CUDA | curobo cache tag |
|-----------|--------|---------|------|------------------|
| 4.5.0 | 3.10 | 2.5+cu118 | 11.8 | `py310_cu118` |
| 5.1.0 | 3.11 | 2.7+cu128 | 12.8 | `py311_cu128` |

Both versions target `sm_75` (RTX 2080 Ti). Change `TORCH_ARCH` for other GPUs.

## Prerequisites

- Docker with nvidia-docker (GPU access)
- `third_party/curobo/` submodule checked out
- Node 108: 8x RTX 2080 Ti, Docker + containerd

## Quick Start

```bash
cd sim-service

# Build for 5.1.0 on GPU 2
GPU=2 ./docker/build.sh 5.1.0

# Build for 4.5.0 on GPU 0
GPU=0 ./docker/build.sh 4.5.0
```

## 4-Step Build Process

### Step 1: Base Image (no GPU)

```bash
docker build -t sim-interactive:5.1.0-base \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0 \
  --build-arg CUDA_APT_VERSION=12-8 \
  --build-arg TORCH_CUDA_ARCH_LIST=7.5 \
  -f docker/Dockerfile.sim-interactive <build-context>
```

Installs: g++, nvcc, ninja, flask, pandas, pyarrow, tqdm, curobo source.

The build script creates a temporary build context with `curobo/` copied from `third_party/curobo/src/curobo/`. This works around `.dockerignore` restrictions.

### Step 2: Curobo JIT on GPU

```bash
docker run --gpus "device=2" --rm \
  -v ./docker/curobo-cache-5.1.0:/opt/curobo-cache \
  -v ./docker/compile_curobo.py:/tmp/compile_curobo.py:ro \
  --entrypoint bash \
  sim-interactive:5.1.0-base \
  -c "source /isaac-sim/setup_python_env.sh && \
      PYTHONPATH=/opt/curobo-src:\$PYTHONPATH \
      /isaac-sim/kit/python/bin/python3 /tmp/compile_curobo.py"
```

Compiles 5 CUDA kernels (~7 min):
- `kinematics_fused_cu.so`
- `tensor_step_cu.so`
- `lbfgs_step_cu.so`
- `line_search_cu.so`
- `geom_cu.so`

Output lands in `docker/curobo-cache-5.1.0/py311_cu128/`.

### Step 3: Final Image

Bakes the compiled `.so` files into the image:

```bash
docker build -t sim-interactive:5.1.0 \
  --build-arg BASE=sim-interactive:5.1.0-base \
  -f Dockerfile.final <context-with-cache>
```

Also fixes the `/usr/local/cuda-active` symlink.

### Step 4: Import to containerd

K8s uses containerd, not Docker. We must import manually:

```bash
docker save sim-interactive:5.1.0 -o /tmp/sim510.tar
sudo ctr -n k8s.io images import /tmp/sim510.tar
rm /tmp/sim510.tar
```

Verify:
```bash
sudo ctr -n k8s.io images list | grep sim-interactive
```

## Gotchas

### 5.1.0: python.sh launches full Kit app

`/isaac-sim/python.sh` in 5.1.0 starts the entire Kit/Isaac Sim application (takes ~150s) and does NOT execute script arguments. This breaks `-c "code"` and `-m pip install`.

**Fix**: Use the bare Python interpreter with environment setup:
```bash
source /isaac-sim/setup_python_env.sh
/isaac-sim/kit/python/bin/python3 your_script.py
```

### cuda-active symlink

Isaac Sim sets `CUDA_HOME=/usr/local/cuda-active`, but this symlink does not exist in the base image. The Dockerfile and final image both create it:
```bash
ln -sf /usr/local/cuda-12.8 /usr/local/cuda-active  # 5.1.0
ln -sf /usr/local/cuda-11.8 /usr/local/cuda-active  # 4.5.0
```

Without this symlink, PyTorch's `cpp_extension` fails with `nvcc: not found`.

### Build context and .dockerignore

If the repo has a restrictive `.dockerignore` (e.g. `*`), `COPY third_party/...` fails silently. The build script avoids this by creating a clean temp directory as build context and copying only the needed files.

### containerd import pipe

`docker save | sudo ctr import -` does NOT work when combined with password piping (`echo pw | sudo -S`). Always save to a tar file first, then import separately.

### Pre-compiled .so are version-locked

The `.so` files are compiled against a specific Python + PyTorch + CUDA combination. `py311_cu128` kernels will NOT load in a `py310_cu118` environment. Each Isaac Sim version needs its own JIT compilation.

## Runtime: How Pods Use the Cache

At startup, `_build_startup_script()` in `run_interactive.py` copies cached `.so` files from `/opt/curobo-cache/<pyver>_<cutag>/` to the torch extensions directory. This lets curobo skip JIT compilation entirely.

The detection logic auto-selects the right cache based on the running Python and CUDA versions.

## Files

| File | Purpose |
|------|---------|
| `docker/Dockerfile.sim-interactive` | Base image definition (4.5.0/5.1.0 universal) |
| `docker/build.sh` | Orchestrates the 4-step build |
| `docker/compile_curobo.py` | Python script for GPU JIT compilation |
| `k8s/sim-interactive-1.yaml` | Pod manifest (hostNetwork, GPU allocation) |

## Current Images on Node 108

```
sim-interactive:4.5.0   15.2 GiB   py310_cu118   5 .so
sim-interactive:5.1.0   16.2 GiB   py311_cu128   5 .so
sim-interactive:latest   → 4.5.0
```
