"""Python SDK for sim-service."""

from __future__ import annotations

from typing import Any, Optional

import requests


class SimServiceClient:
    """Requests-based client for sim-service endpoints."""

    def __init__(self, base_url: str = "http://sim-service.embodied.svc.cluster.local:5900", timeout: float = 30.0):
        """Initialize the client with service base URL and request timeout."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        response = requests.request(method, f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        """Fetch service health, sim readiness, GPU info, and version."""
        return self._request("GET", "/health")

    def load_scene(self, scene_dir: str) -> dict[str, Any]:
        """Generate scene YAML and request scene loading from sim-service."""
        return self._request("POST", "/scene/load", {"scene_dir": scene_dir})

    def scene_info(self) -> dict[str, Any]:
        """Return currently loaded scene object list and prim count."""
        return self._request("GET", "/scene/info")

    def spawn_robot(
        self,
        robot_type: str = "franka",
        position: Optional[list[float]] = None,
        orientation: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """Request robot spawn with robot type, position, and orientation."""
        return self._request(
            "POST",
            "/robot/spawn",
            {
                "robot_type": robot_type,
                "position": position or [0.0, 0.0, 0.0],
                "orientation": orientation or [0.0, 0.0, 0.0],
            },
        )

    def robot_state(self) -> dict[str, Any]:
        """Fetch joint positions, end-effector pose, and gripper state."""
        return self._request("GET", "/robot/state")

    def start_collection(
        self,
        skill: str = "Grasp",
        target_objects: Optional[list[str]] = None,
        output_dir: str = "",
    ) -> dict[str, Any]:
        """Start collection run with skill name, target objects, and output path."""
        return self._request(
            "POST",
            "/collect/start",
            {
                "skill": skill,
                "target_objects": target_objects or [],
                "output_dir": output_dir,
            },
        )

    def collection_status(self) -> dict[str, Any]:
        """Get collection progress and running status."""
        return self._request("GET", "/collect/status")

    def step(self, n: int = 1) -> dict[str, Any]:
        """Request stepping the simulator by n frames."""
        return self._request("POST", "/sim/step", {"n": n})

    def reset(self) -> dict[str, Any]:
        """Request simulator/environment reset."""
        return self._request("POST", "/sim/reset")
