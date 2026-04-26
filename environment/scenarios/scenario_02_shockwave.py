from __future__ import annotations

import math

from environment.vehicle import Vehicle


class Scenario02Shockwave:
    def __init__(self, cfg: dict[str, float | int]) -> None:
        self.cfg = cfg

    def get_phase(self, timestep: int) -> str:
        if timestep <= int(self.cfg["steady_end"]):
            return "steady"
        if timestep <= int(self.cfg["pulse_brake_end"]):
            return "pulse_brake"
        if timestep <= int(self.cfg["rebound_end"]):
            return "rebound"
        if timestep <= int(self.cfg["traffic_wave_end"]):
            return "traffic_wave"
        return "recover"

    def lead_controls(self, lead_vehicle: Vehicle, phase: str) -> tuple[float, float]:
        cruise_speed = float(self.cfg["lead_cruise_speed"])
        low_speed = float(self.cfg["lead_low_speed"])
        high_speed = float(self.cfg["lead_high_speed"])

        if phase == "steady":
            if lead_vehicle.velocity < cruise_speed:
                return float(self.cfg["lead_cruise_accel_pedal"]), 0.0
            return 0.0, 0.0

        if phase == "pulse_brake":
            if lead_vehicle.velocity > low_speed:
                return 0.0, float(self.cfg["lead_brake_pedal"])
            return 0.0, 0.0

        if phase == "rebound":
            if lead_vehicle.velocity < high_speed:
                return float(self.cfg["lead_rebound_accel_pedal"]), 0.0
            return 0.0, 0.0

        if phase == "traffic_wave":
            wave = 0.5 * (math.sin(lead_vehicle.x / 28.0) + 1.0)
            if lead_vehicle.velocity > cruise_speed + 0.8:
                return 0.0, 0.18 + (0.16 * wave)
            if lead_vehicle.velocity < cruise_speed - 0.8:
                return 0.16 + (0.24 * wave), 0.0
            return 0.0, 0.0

        if lead_vehicle.velocity < cruise_speed:
            return 0.12, 0.0
        if lead_vehicle.velocity > cruise_speed:
            return 0.0, 0.12
        return 0.0, 0.0

    def dynamics_modifiers(self, phase: str) -> dict[str, float]:
        if phase == "traffic_wave":
            return {
                "accel_scale": 0.96,
                "decel_scale": 1.03,
                "road_grip": 0.95,
                "road_grade": 0.005,
            }
        return {
            "accel_scale": 1.0,
            "decel_scale": 1.0,
            "road_grip": 1.0,
            "road_grade": 0.0,
        }
