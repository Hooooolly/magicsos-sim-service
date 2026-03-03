"""Compile curobo CUDA kernels via JIT and cache the resulting .so files.

Run inside a sim-interactive container with GPU access:
  source /isaac-sim/setup_python_env.sh
  PYTHONPATH=/opt/curobo-src:$PYTHONPATH python3 compile_curobo.py

Expects /opt/curobo-cache/ to be mounted as a volume for output.
"""

import os
import shutil
import subprocess
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

print(f"Python {sys.version}")

import torch

print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

print("Importing curobo MotionGen (triggers CUDA JIT)...")
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

print("MotionGen import OK")

# Build cache directory name from Python + CUDA versions
pyver = f"py{sys.version_info.major}{sys.version_info.minor}"
cutag = "cu" + torch.version.cuda.replace(".", "")
cache_dir = f"/opt/curobo-cache/{pyver}_{cutag}"
os.makedirs(cache_dir, exist_ok=True)
print(f"Cache dir: {cache_dir}")

# Copy compiled .so from torch extensions cache
result = subprocess.run(
    ["find", "/root/.cache/torch_extensions", "-name", "*.so"],
    capture_output=True,
    text=True,
)
count = 0
for so_path in result.stdout.strip().split("\n"):
    if so_path and so_path.endswith(".so"):
        ext_name = os.path.basename(os.path.dirname(so_path))
        dest_dir = os.path.join(cache_dir, ext_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(so_path, dest_dir)
        print(f"  cached: {os.path.basename(so_path)} -> {dest_dir}")
        count += 1

print(f"DONE: {count} .so files cached to {cache_dir}")
