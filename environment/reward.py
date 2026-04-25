from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from environment.vehicle import Vehicle


@dataclass
class RewardTerms:
    collision_penalty: float
    gap_error_penalty: float
    speed_maintenance: float
    jerk_penalty: float
    recovery_bonus: float
    comfort_penalty: float
    alive_bonus: float
    gap_tracking_bonus: float
    speed_tracking_bonus: float

    @property
    def total(self) -> float:
        return (
            self.collision_penalty
            + self.gap_error_penalty
            + self.speed_maintenance
            + self.jerk_penalty
            + self.recovery_bonus
            + self.comfort_penalty
            + self.alive_bonus
            + self.gap_tracking_bonus
            + self.speed_tracking_bonus
        )


class RewardModel:
    def __init__(self, reward_cfg: dict[str, float], dt: float) -> None:
        self.reward_cfg = reward_cfg
        self.dt = dt

    def compute(
        self,
        ego: "Vehicle",
        front: "Vehicle",
        gap: float,
        desired_gap: float,
        phase: str,
    ) -> RewardTerms:
        gap_error = gap - desired_gap
        collision_penalty = self.reward_cfg["collision_penalty"] if gap <= 0.0 else 0.0

        gap_deadband = float(self.reward_cfg.get("gap_deadband_m", 0.5))
        speed_deadband = float(self.reward_cfg.get("speed_deadband_mps", 0.2))
        in_steady = phase in {"steady", "steady_2"}

        gap_abs = abs(gap_error)
        gap_over = max(0.0, gap_abs - (gap_deadband if in_steady else 0.0))
        gap_error_penalty = -min(gap_over, self.reward_cfg["gap_error_cap"]) * self.reward_cfg["gap_error_weight"]

        speed_abs = abs(ego.velocity - front.velocity)
        speed_over = max(0.0, speed_abs - (speed_deadband if in_steady else 0.0))
        speed_maintenance = -speed_over * self.reward_cfg["speed_error_weight"]

        jerk = abs((ego.net_acceleration - ego.last_net_acceleration) / self.dt)
        jerk_penalty = -min(jerk, self.reward_cfg["jerk_cap"]) * self.reward_cfg["jerk_weight"]

        recovery_bonus = (
            self.reward_cfg["steady2_recovery_bonus"]
            if phase == "steady_2" and abs(gap_error) < 1.0
            else 0.0
        )

        comfort_penalty = (
            self.reward_cfg["comfort_penalty"]
            if ego.accel_pedal > 0.0 and ego.brake_pedal > 0.0
            else 0.0
        )

        # Positive shaping so stable, safe trajectories can score above zero.
        alive_bonus = float(self.reward_cfg.get("alive_bonus", 0.0))
        gap_tracking_bonus = (
            float(self.reward_cfg.get("gap_tracking_bonus", 0.0))
            if gap_abs <= gap_deadband
            else 0.0
        )
        speed_tracking_bonus = (
            float(self.reward_cfg.get("speed_tracking_bonus", 0.0))
            if speed_abs <= speed_deadband
            else 0.0
        )

        return RewardTerms(
            collision_penalty=collision_penalty,
            gap_error_penalty=gap_error_penalty,
            speed_maintenance=speed_maintenance,
            jerk_penalty=jerk_penalty,
            recovery_bonus=recovery_bonus,
            comfort_penalty=comfort_penalty,
            alive_bonus=alive_bonus,
            gap_tracking_bonus=gap_tracking_bonus,
            speed_tracking_bonus=speed_tracking_bonus,
        )

    @staticmethod
    def desired_gap(ego_velocity: float, min_gap: float, headway_seconds: float) -> float:
        return float(max(min_gap, ego_velocity * headway_seconds))

    @staticmethod
    def gap_to_front(front: "Vehicle", ego: "Vehicle") -> float:
        return float(front.x - ego.x - ego.length)

    @staticmethod
    def parse_failure_penalty() -> float:
        return 0.0
