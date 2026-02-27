# Pod Troubleshooting: v2 Hydra Black Screen + Curobo Plan Failure

**Date**: 2026-02-27
**Pods affected**: `sim-interactive-v2test-fix` (image: `humerlsl/magicphysics:v2`)
**Node**: 108 (RTX 2080 Ti, driver 560.35.03)

---

## Issue 1: WebRTC Stream Black Screen (Hydra RTX Renderer Failure)

### Symptom

v2test-fix stream shows black screen in browser. WebRTC ICE connects, but no video frames. GPU memory only 544 MiB (renderer not loaded). Working pods use 2400+ MiB.

### Logs

```
HydraEngine rtx failed creating scene renderer
unable to create shared hydra engine context
Invalid sync scope created. Failed to add Hydra engine
```

### Root Cause

The `humerlsl/magicphysics:v2` Docker image bakes in empty (0-byte) `libnvidia-rtcore.so` stub files for driver versions that don't match the host:

```
/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.545.23.08  → 0 bytes (empty stub!)
/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.560.35.03  → 70 MB (real, from host driver)
/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.580.95.05  → 0 bytes (empty stub!)
```

The NVIDIA K8s device plugin mounts the host driver (560.35.03) as read-only bind mount. The image's empty stubs for other versions remain and confuse the RTX renderer's library loading.

Comparison: `isaac-sim:4.5.0` base image does NOT include these stubs — curobo4 was unaffected.

### Fix

1. **Add `securityContext`** — v2 image runs as uid 1235, need root to delete stubs:
   ```yaml
   securityContext:
     runAsUser: 0
     runAsGroup: 0
   ```

2. **Startup script auto-removes empty stubs**:
   ```bash
   DRIVER_VER=$(cat /proc/driver/nvidia/version 2>/dev/null \
     | grep -oP 'Kernel Module  \K[0-9.]+' || echo "")
   if [ -n "$DRIVER_VER" ]; then
     for f in /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.*; do
       case "$f" in *"$DRIVER_VER") continue;; esac
       if [ -f "$f" ] && [ ! -s "$f" ]; then
         echo "[startup] Removing empty stub: $f"
         rm -f "$f"
       fi
     done
   fi
   ```

### Verification

After fix: GPU memory rises to 3247 MiB, Hydra error disappears, stream visible in browser.

---

## Issue 2: Curobo plan_single Always Returns success=False

### Symptom

Collection starts, curobo MotionGen imports successfully, but every `plan_single` call returns `success=False` after ~15 seconds (optimization timeout). Robot arm does not move. Collection finishes with `status=done_with_failures`.

```
Curobo plan_single returned success=False
Curobo full_approach: segment 1 (-> pre-grasp) plan failed
collect: attempt 1 Curobo full_approach failed; skipping attempt
```

### Root Cause: Pre-compiled CUDA Extensions Incompatible

`/sim-service/curobo/curobolib/` contains pre-compiled `.so` files:
```
geom_cu.cpython-311-x86_64-linux-gnu.so           (14.9 MB)
kinematics_fused_cu.cpython-311-x86_64-linux-gnu.so (10.2 MB)
lbfgs_step_cu.cpython-311-x86_64-linux-gnu.so     (14.1 MB)
line_search_cu.cpython-311-x86_64-linux-gnu.so     (10.1 MB)
tensor_step_cu.cpython-311-x86_64-linux-gnu.so     (11.0 MB)
```

These were compiled for **PyTorch 2.5.1+cu118** (Python 3.11).

| Pod | Python | PyTorch | Pre-compiled .so | What happens |
|-----|--------|---------|-------------------|--------------|
| curobo4 (isaac-sim:4.5.0) | 3.10 | 2.5.1+cu118 | cpython-311 → **ignored** (ABI mismatch) | Falls through to JIT compile → **works** |
| v2test-fix (magicphysics:v2) | 3.11 | 2.7.0+cu128 | cpython-311 → **loaded** (ABI matches) | `libc10.so` not found → **FAILS** |

The CUDA kernel loading fails silently (caught by try/except in curobo). The Python-level MotionGen import succeeds, but all CUDA operations produce garbage or no-ops, making every plan return False.

### Diagnosis

```python
# Test CUDA kernel loading:
from curobo.curobolib import kinematics_fused_cu
# → libc10.so: cannot open shared object file: No such file or directory

from curobo.curobolib import geom_cu
# → libc10.so: cannot open shared object file: No such file or directory
```

### Fix

1. **Remove incompatible pre-compiled `.so` files**:
   ```bash
   rm -f /sim-service/curobo/curobolib/*_cu.cpython-*.so
   ```
   Safe because curobo4 (Python 3.10) never uses these files anyway.

2. **Add ninja to PATH** — PyTorch JIT compilation requires ninja:
   ```bash
   export PATH=/isaac-sim/kit/python/bin:$CUDA_HOME/bin:$PATH
   ```
   The `pip install ninja` installs the binary to `/isaac-sim/kit/python/bin/ninja`, which is NOT in the default PATH of `/isaac-sim/python.sh`.

3. **JIT-compile at startup** (~7 min on RTX 2080 Ti):
   ```bash
   export CUDA_HOME=/usr/local/cuda        # v2 has CUDA 12.8 built-in
   export TORCH_CUDA_ARCH_LIST=7.5          # RTX 2080 Ti compute capability
   /isaac-sim/python.sh - <<'PY'
   import os
   os.environ['PATH'] = '/isaac-sim/kit/python/bin:' + os.environ['PATH']
   from curobo.wrap.reacher.motion_gen import MotionGen  # triggers JIT
   PY
   ```

4. **Use CUDA 12.8** (built into v2), NOT install CUDA 11.8:
   - v2: `/usr/local/cuda → /usr/local/cuda-12.8/` (matches PyTorch 2.7+cu128)
   - curobo4: must install CUDA 11.8 (matches PyTorch 2.5+cu118)

### JIT Compilation Details

Five CUDA extensions compiled, each takes 1-2 minutes:
1. `kinematics_fused_cu` — FK/IK kernels
2. `geom_cu` — collision geometry (sphere-OBB, self-collision)
3. `tensor_step_cu` — optimization step
4. `lbfgs_step_cu` — L-BFGS optimizer
5. `line_search_cu` — line search

Cached at `/root/.cache/torch_extensions/py311_cu128/` inside container. Cache is **ephemeral** — lost on pod restart, must recompile each time.

---

## Environment Comparison: isaac-sim:4.5.0 vs magicphysics:v2

| Property | isaac-sim:4.5.0 (curobo4) | magicphysics:v2 (v2test-fix) |
|----------|---------------------------|------------------------------|
| Kit | 106 | 107 |
| Python | 3.10 | 3.11 |
| PyTorch | 2.5.1+cu118 | 2.7.0+cu128 |
| CUDA (image) | none | 12.8 |
| CUDA (install) | 11.8 | not needed |
| Default user | root | user (uid 1235) |
| Livestream | v6.x (direct mode) | v7.x (NVCF mode) |
| libnvidia stubs | clean | empty stubs for 545/580 |
| g++ | needs install | 13.3.0 built-in |
| ninja | needs install | needs install (in Isaac Sim Python env) |

---

## Updated YAML Template for magicphysics:v2

Key startup script sections:

```yaml
securityContext:
  runAsUser: 0
  runAsGroup: 0
command: ["/bin/bash", "-c"]
args:
- |
  # 1. Remove empty NVIDIA driver stubs
  DRIVER_VER=$(cat /proc/driver/nvidia/version | grep -oP 'Kernel Module  \K[0-9.]+')
  for f in /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.*; do
    case "$f" in *"$DRIVER_VER") continue;; esac
    [ -f "$f" ] && [ ! -s "$f" ] && rm -f "$f"
  done

  # 2. Use built-in CUDA 12.8, add ninja to PATH
  export CUDA_HOME=/usr/local/cuda
  export PATH=/isaac-sim/kit/python/bin:$CUDA_HOME/bin:$PATH

  # 3. Install Python deps
  /isaac-sim/python.sh -m pip install -q flask pandas pyarrow imageio imageio-ffmpeg tqdm ninja warp-lang yourdfpy trimesh

  # 4. Remove incompatible pre-compiled curobo .so
  rm -f /sim-service/curobo/curobolib/*_cu.cpython-*.so

  # 5. JIT-compile curobo CUDA extensions (~7 min)
  TORCH_CUDA_ARCH_LIST=7.5 /isaac-sim/python.sh -c "
  import os; os.environ['PATH']='/isaac-sim/kit/python/bin:'+os.environ['PATH']
  from curobo.wrap.reacher.motion_gen import MotionGen
  print('Curobo MotionGen OK')
  "

  # 6. Launch
  exec /isaac-sim/python.sh -u /sim-service/run_interactive.py
```

---

## Issue 3: Curobo JIT Cache Not Shared Between Startup and Collect

### Symptom

First collect after pod restart "hangs" for ~6.5 minutes, then times out (180s default). Curobo logs show all 5 CUDA extensions JIT-compiling again during the collect, even though startup already compiled them.

```
[startup] Curobo MotionGen import OK       ← startup JIT succeeded
...
[curobo] geom_cu binary not found, jit compiling...     ← collect JIT-compiles AGAIN
[curobo] kinematics_fused_cu not found, JIT compiling...
[curobo] tensor_step_cu not found, jit compiling...
[curobo] lbfgs_step_cu not found, JIT compiling...
[curobo] line_search_cu not found, JIT compiling...
...
Curobo MotionGen initialised (device=cuda:0)
collect: episode 1 exceeded timeout 180.0s; requesting stop    ← 0 frames
```

### Root Cause: PyTorch JIT Cache Key Mismatch

The startup script runs JIT compilation in a child process:
```bash
/isaac-sim/python.sh - <<'PY'
from curobo.wrap.reacher.motion_gen import MotionGen  # triggers JIT
PY
exec /isaac-sim/python.sh -u /sim-service/run_interactive.py   # separate process
```

PyTorch's `torch.utils.cpp_extension.load()` caches compiled extensions at `~/.cache/torch_extensions/`. The cache key includes the build environment (compiler flags, include paths, library paths). `/isaac-sim/python.sh` internally modifies `LD_LIBRARY_PATH`, `PYTHONPATH`, and other environment variables, and these modifications differ slightly between:
- The child stdin process (`/isaac-sim/python.sh - <<'PY'`)
- The exec'd main process (`/isaac-sim/python.sh -u run_interactive.py`)

Because the build environment hash differs, the exec'd Isaac Sim process cannot find the startup's cached extensions and recompiles everything (~6.5 min on RTX 2080 Ti).

### Timeline

| Time | Event |
|------|-------|
| 05:50:57 | Collect starts, timeout=180s |
| 05:51:02 | `geom_cu` JIT compile begins |
| 05:52:26 | `kinematics_fused_cu` JIT begins |
| 05:53:34 | `tensor_step_cu` JIT begins |
| 05:54:41 | `lbfgs_step_cu` JIT begins |
| 05:55:51 | `line_search_cu` JIT begins |
| 05:57:01 | WorldConfig cuboid extraction works |
| 05:57:28 | `MG: GP Success` — planning SUCCEEDED |
| 05:57:32 | MotionGen warmup complete, BUT timeout exceeded → 0 frames |

### Fix

Add JIT warmup **inside `run_interactive.py`** (the Isaac Sim process), not in a separate child process:

```python
# In run_interactive.py, after warmup frames, before the main loop:
if os.environ.get("COLLECT_PLANNER_BACKEND", "").strip().lower() == "curobo":
    try:
        print("[interactive] Pre-warming curobo CUDA extensions (JIT) ...")
        from curobo.wrap.reacher.motion_gen import MotionGen
        print("[interactive] Curobo JIT warmup OK")
    except Exception as exc:
        print(f"[interactive] Curobo JIT warmup skipped: {exc}")
```

This ensures the JIT cache is built by the same process that runs collects, so the cache key matches.

### Verification

After fix: first collect no longer recompiles CUDA extensions. MotionGen initializes in seconds (uses cached .so), leaving full timeout for actual data collection.

---

## Issue 4: Curobo plan_single Returns valid_query=False (Collision World)

### Symptom

Every `plan_single` call returns `success=False`, `valid_query=False`. Robot arm does not move.

### Root Cause

`_extract_world_config()` used `get_obstacles_from_stage()` to extract mesh collision geometry from the USD stage. The resulting mesh was unreliable — incorrect position/size in robot frame, causing curobo to think every trajectory collides with the table.

### Fix

Rewrote `_extract_world_config()` to use **ground-truth bounding boxes** as cuboid primitives:
1. Compute axis-aligned bounding box via `UsdGeom.Imageable(prim).ComputeWorldBound()`
2. Transform center to robot frame
3. Create `Cuboid(name, pose, dims)` for table and pick object
4. Switch from `CollisionCheckerType.MESH` to `CollisionCheckerType.PRIMITIVE`
5. Reduced `collision_activation_distance` from 0.015 to 0.005
6. Enabled `enable_graph=True` with `enable_graph_attempt=3`

### Verification

```
Curobo cuboid 'table': center_world=[0,0,0.385] center_robot=[0.4,-0.1,-0.385] dims=[1.2,0.8,0.77]
Curobo WorldConfig: 2 cuboid(s) built from ground-truth bbox
MG: GP Success          ← Graph planning succeeded!
Warmup complete
Curobo MotionGen initialised (device=cuda:0)
```

---

## Lessons Learned

1. **Python ABI tag matching ≠ library compatibility**. A `.cpython-311` `.so` loads on any Python 3.11, but if it links against a different PyTorch version's `libc10.so`, it fails at dlopen.

2. **Silent CUDA kernel failure is dangerous**. Curobo catches ImportError and continues without CUDA acceleration. MotionGen appears to work (imports fine), but planning returns False every time. No obvious error message points to the CUDA kernel issue.

3. **Different Python versions accidentally create JIT fallback**. curobo4 (Python 3.10) "works" because it can't load the cpython-311 .so files — it falls through to JIT compilation. This is accidental, not by design.

4. **Isaac Sim Python environment is isolated**. System binaries in `/usr/bin/` are not always visible. Pip-installed binaries go to `/isaac-sim/kit/python/bin/`, which requires explicit PATH addition.

5. **Host-mounted volumes persist changes**. Deleting `.so` files from `/sim-service/curobo/curobolib/` removes them from the host at `/home/magics/sim-service/curobo/curobolib/`. This is permanent and affects all pods. Safe in this case because no pod uses the pre-compiled .so files correctly.

6. **PyTorch JIT cache keys include the build environment**. Two processes running `/isaac-sim/python.sh` can produce different cache keys if the script modifies env vars internally. Always JIT-compile in the SAME process that will use the cached extensions. Running `from curobo.wrap.reacher.motion_gen import MotionGen` in a child process does NOT warm the cache for the parent process.

7. **Ground-truth bounding boxes beat mesh extraction for collision**. In simulators like Isaac Sim, `ComputeWorldBound()` gives exact axis-aligned bounding boxes. Using these as `CollisionCheckerType.PRIMITIVE` cuboids is 4x faster and more reliable than mesh-based collision from `get_obstacles_from_stage()`.

8. **Collect timeout must account for one-time setup costs**. The 180s default timeout killed the collect because curobo JIT compilation (6.5 min) ran during the first collect. Fix: move one-time initialization to startup, or increase timeout for the first collect.
