"""Isaac Sim version compatibility layer.

Provides imports that work on both 4.5.0 (omni.isaac.*) and 5.1.0+ (isaacsim.*).
Try new name first, fall back to deprecated name.
"""

# SimulationApp
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp  # noqa: F401

# World
try:
    from isaacsim.core.api import World
except ImportError:
    from omni.isaac.core import World  # noqa: F401

# Stage utils
try:
    from isaacsim.core.utils.stage import add_reference_to_stage
except ImportError:
    from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: F401

try:
    from isaacsim.core.utils.prims import create_prim
except ImportError:
    from omni.isaac.core.utils.prims import create_prim  # noqa: F401

# Nucleus
try:
    from isaacsim.core.utils.nucleus import get_assets_root_path
except ImportError:
    try:
        from omni.isaac.core.utils.nucleus import get_assets_root_path  # noqa: F401
    except ImportError:
        def get_assets_root_path():
            return None

# Articulation
try:
    from isaacsim.core.api.articulations import Articulation
except ImportError:
    from omni.isaac.core.articulations import Articulation  # noqa: F401

# ArticulationAction
try:
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    from omni.isaac.core.utils.types import ArticulationAction  # noqa: F401

# AppLauncher (IsaacLab)
try:
    from omni.isaac.lab.app import AppLauncher
except ImportError:
    AppLauncher = None
