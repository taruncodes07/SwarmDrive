from __future__ import annotations

import math
from environment.vehicle import Vehicle


class Scenario02Merge:
    def __init__(self, cfg: dict[str, float | int]) -> None:
        self.cfg = cfg
        self.x_merge = float(cfg.get("x_merge", 150.0))
        self.y_start = float(cfg.get("y_start", 3.5))

    def get_phase(self, timestep: int) -> str:
        if timestep <= int(self.cfg["steady_end"]):
            return "steady"
        if timestep <= int(self.cfg["merge_zone_end"]):
            return "merge_zone"
        return "post_merge"

    def lead_controls(self, lead_vehicle: Vehicle, phase: str) -> tuple[float, float]:
        # In this scenario, Agent 0 (lead) is just a reference or far ahead.
        # We can make it cruise at constant speed.
        cruise_speed = float(self.cfg["lead_cruise_speed"])
        if lead_vehicle.velocity < cruise_speed:
            return 0.1, 0.0
        return 0.0, 0.0

    def dynamics_modifiers(self, phase: str) -> dict[str, float]:
        return {
            "accel_scale": 1.0,
            "decel_scale": 1.0,
            "road_grip": 1.0,
            "road_grade": 0.0,
        }

    def get_y_position(self, vehicle: Vehicle) -> float:
        if vehicle.path_type == "straight":
            return 0.0
        
        # Merge path: cosine interpolation from y_start to 0
        # Starting merge at some distance before x_merge
        merge_start_x = self.x_merge - 60.0
        if vehicle.x < merge_start_x:
            return self.y_start
        if vehicle.x >= self.x_merge:
            return 0.0
        
        # Smooth transition
        ratio = (vehicle.x - merge_start_x) / (self.x_merge - merge_start_x)
        # Cosine interpolation for smoothness
        factor = 0.5 * (1.0 + math.cos(ratio * math.pi))
        return self.y_start * factor
