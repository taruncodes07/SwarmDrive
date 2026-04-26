from __future__ import annotations

from environment.vehicle import Vehicle


class Scenario03LowFriction:
    def __init__(self, cfg: dict[str, float | int]) -> None:
        self.cfg = cfg

    def get_phase(self, timestep: int) -> str:
        if timestep <= int(self.cfg["steady_end"]):
            return "steady"
        if timestep <= int(self.cfg["downhill_gain_end"]):
            return "downhill_gain"
        if timestep <= int(self.cfg["low_friction_end"]):
            return "low_friction"
        if timestep <= int(self.cfg["cutin_emergency_end"]):
            return "cutin_emergency"
        return "recover"

    def lead_controls(self, lead_vehicle: Vehicle, phase: str) -> tuple[float, float]:
        cruise_speed = float(self.cfg["lead_cruise_speed"])
        downhill_target = float(self.cfg["downhill_target_speed"])
        low_speed = float(self.cfg["lead_low_speed"])

        if phase == "steady":
            if lead_vehicle.velocity < cruise_speed:
                return float(self.cfg["lead_cruise_accel_pedal"]), 0.0
            return 0.0, 0.0

        if phase == "downhill_gain":
            if lead_vehicle.velocity < downhill_target:
                return float(self.cfg["downhill_accel_pedal"]), 0.0
            return 0.0, 0.0

        if phase == "low_friction":
            if lead_vehicle.velocity > low_speed + 2.0:
                return 0.0, float(self.cfg["lead_low_friction_brake_pedal"])
            return 0.0, 0.0

        if phase == "cutin_emergency":
            if lead_vehicle.velocity > low_speed:
                return 0.0, float(self.cfg["lead_emergency_brake_pedal"])
            return 0.0, 0.0

        if lead_vehicle.velocity < cruise_speed:
            return float(self.cfg["lead_recovery_accel_pedal"]), 0.0
        if lead_vehicle.velocity > cruise_speed:
            return 0.0, 0.08
        return 0.0, 0.0

    def dynamics_modifiers(self, phase: str) -> dict[str, float]:
        if phase == "downhill_gain":
            return {
                "accel_scale": 1.06,
                "decel_scale": 0.88,
                "road_grip": 0.9,
                "road_grade": -0.025,
            }
        if phase == "low_friction":
            return {
                "accel_scale": 0.95,
                "decel_scale": 0.6,
                "road_grip": 0.58,
                "road_grade": -0.01,
            }
        if phase == "cutin_emergency":
            return {
                "accel_scale": 0.9,
                "decel_scale": 0.68,
                "road_grip": 0.62,
                "road_grade": 0.0,
            }
        return {
            "accel_scale": 1.0,
            "decel_scale": 1.0,
            "road_grip": 1.0,
            "road_grade": 0.0,
        }
