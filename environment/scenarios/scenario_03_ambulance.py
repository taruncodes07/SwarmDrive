from __future__ import annotations

from environment.vehicle import Vehicle


class Scenario03Ambulance:
    """Three-lane road: lead traffic, two agents, ambulance approaching from behind."""

    def __init__(self, cfg: dict[str, float | int]) -> None:
        self.cfg = cfg
        self.lane_spacing_m = float(cfg.get("lane_spacing_m", 3.7))

    @staticmethod
    def lane_to_y(lane_index: int, lane_spacing_m: float) -> float:
        return float(lane_index - 1) * lane_spacing_m

    def get_phase(self, timestep: int, vehicles: dict[int, Vehicle] | None = None) -> str:
        if timestep <= int(self.cfg["steady_end"]):
            return "steady"
        if timestep <= int(self.cfg["approach_end"]):
            return "ambulance_approach"
        # Do not use pass_end for post_pass — time-only cutoff showed "post_pass" while AMB was still behind.
        if vehicles is not None and self._ambulance_cleared_both_followers(vehicles):
            return "post_pass"
        return "ambulance_pass"

    @staticmethod
    def _ambulance_cleared_both_followers(vehicles: dict[int, Vehicle]) -> bool:
        amb = vehicles.get(3)
        if amb is None:
            return False
        for aid in (1, 2):
            ego = vehicles.get(aid)
            if ego is None:
                return False
            if not (float(amb.x) > float(ego.x) + float(ego.length)):
                return False
        return True

    def lead_controls(self, lead_vehicle: Vehicle, phase: str) -> tuple[float, float]:
        cruise = float(self.cfg["lead_cruise_speed"])
        if lead_vehicle.velocity < cruise:
            return float(self.cfg.get("lead_cruise_accel_pedal", 0.1)), 0.0
        return 0.0, 0.0

    def ambulance_controls(self, amb: Vehicle, phase: str) -> tuple[float, float]:
        """Aggressive response driving; faster during pass phase."""
        v_target = float(self.cfg["ambulance_cruise_speed"])
        if phase in {"ambulance_approach", "ambulance_pass"}:
            v_target = float(self.cfg["ambulance_urgent_speed"])
        if amb.velocity < v_target - 0.5:
            return float(self.cfg.get("ambulance_accel_pedal", 0.85)), 0.0
        if amb.velocity > v_target + 1.0:
            return 0.0, 0.15
        return 0.0, 0.0

    def dynamics_modifiers(self, phase: str) -> dict[str, float]:
        return {
            "accel_scale": 1.0,
            "decel_scale": 1.0,
            "road_grip": 1.0,
            "road_grade": 0.0,
        }
