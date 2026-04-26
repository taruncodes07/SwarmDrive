from __future__ import annotations

from environment.vehicle import Vehicle


class Scenario01Brake:
    def __init__(self, cfg: dict[str, float | int]) -> None:
        self.cfg = cfg

    def get_phase(self, timestep: int, vehicles: object = None) -> str:
        if timestep <= int(self.cfg["steady_end"]):
            return "steady"
        if timestep <= int(self.cfg["brake_end"]):
            return "brake_event"
        if timestep <= int(self.cfg["hold_low_end"]):
            return "hold_low"
        if timestep <= int(self.cfg["recovery_end"]):
            return "recovery"
        return "steady_2"

    def lead_controls(self, lead_vehicle: Vehicle, phase: str) -> tuple[float, float]:
        cruise_speed = float(self.cfg["lead_cruise_speed"])
        low_speed = float(self.cfg["lead_low_speed"])

        if phase == "steady":
            return float(self.cfg["lead_cruise_accel_pedal"]), 0.0

        if phase == "brake_event":
            if lead_vehicle.velocity > low_speed:
                return 0.0, float(self.cfg["lead_brake_pedal"])
            return 0.0, 0.0

        if phase == "hold_low":
            if lead_vehicle.velocity < low_speed:
                return 0.15, 0.0
            if lead_vehicle.velocity > low_speed:
                return 0.0, 0.15
            return 0.0, 0.0

        if phase == "recovery":
            if lead_vehicle.velocity < cruise_speed:
                return float(self.cfg["lead_recovery_accel_pedal"]), 0.0
            return 0.0, 0.0

        if lead_vehicle.velocity < cruise_speed:
            return 0.1, 0.0
        if lead_vehicle.velocity > cruise_speed:
            return 0.0, 0.1
        return 0.0, 0.0

    def dynamics_modifiers(self, phase: str) -> dict[str, float]:
        return {
            "accel_scale": 1.0,
            "decel_scale": 1.0,
            "road_grip": 1.0,
            "road_grade": 0.0,
        }
