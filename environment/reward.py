from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    ttc_penalty: float
    merge_success_bonus: float
    merge_efficiency_reward: float
    merge_zipper_bonus: float
    merge_speed_match_bonus: float
    merge_rush_penalty: float
    merge_post_spacing_bonus: float
    merge_approach_patience_bonus: float
    ambulance_clear_lane_bonus: float
    ambulance_blocking_penalty: float
    ambulance_yield_bonus: float
    ambulance_pass_clear_bonus: float

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
            + self.ttc_penalty
            + self.merge_success_bonus
            + self.merge_efficiency_reward
            + self.merge_zipper_bonus
            + self.merge_speed_match_bonus
            + self.merge_rush_penalty
            + self.merge_post_spacing_bonus
            + self.merge_approach_patience_bonus
            + self.ambulance_clear_lane_bonus
            + self.ambulance_blocking_penalty
            + self.ambulance_yield_bonus
            + self.ambulance_pass_clear_bonus
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
        *,
        scenario_name: str = "scenario_01_brake",
        lateral_separation: float = 0.0,
        dist_to_merge: float | None = None,
        global_collision: bool = False,
        ambulance_ctx: dict[str, Any] | None = None,
    ) -> RewardTerms:
        gap_error = gap - desired_gap
        collision_penalty = (
            self.reward_cfg["collision_penalty"] if (gap <= 0.0 or global_collision) else 0.0
        )

        gap_deadband = float(self.reward_cfg.get("gap_deadband_m", 0.5))
        speed_deadband = float(self.reward_cfg.get("speed_deadband_mps", 0.2))
        in_steady = phase in {"steady", "steady_2"}

        gap_abs = abs(gap_error)
        eff_deadband = gap_deadband if in_steady else 0.0
        eff_gap_w = float(self.reward_cfg["gap_error_weight"])
        if scenario_name == "scenario_03_ambulance":
            eff_deadband = max(eff_deadband, 5.0)
            eff_gap_w *= 0.38
        gap_over = max(0.0, gap_abs - eff_deadband)
        gap_error_penalty = -min(gap_over, self.reward_cfg["gap_error_cap"]) * eff_gap_w

        speed_abs = abs(ego.velocity - front.velocity)
        speed_over = max(0.0, speed_abs - (speed_deadband if in_steady else 0.0))
        speed_maintenance = -speed_over * self.reward_cfg["speed_error_weight"]

        # After lane alignment, harsh gap+speed penalties fight "rejoin the platoon" — soften while still safe.
        if scenario_name == "scenario_02_merge" and phase == "post_merge":
            gap_error_penalty *= float(self.reward_cfg.get("post_merge_gap_penalty_scale", 0.28))
            if ego.velocity < front.velocity - 0.1:
                speed_maintenance *= float(self.reward_cfg.get("post_merge_speed_penalty_scale", 0.25))

        jerk = abs((ego.net_acceleration - ego.last_net_acceleration) / self.dt)
        jerk_penalty = -min(jerk, self.reward_cfg["jerk_cap"]) * self.reward_cfg["jerk_weight"]

        recovery_bonus = 0.0
        if phase == "steady_2" and abs(gap_error) < 1.0:
            recovery_bonus = float(self.reward_cfg.get("steady2_recovery_bonus", 3.0))
        if scenario_name == "scenario_01_brake" and phase == "recovery":
            if ego.velocity < front.velocity - 0.25 and gap_error > -1.5:
                recovery_bonus = max(
                    recovery_bonus,
                    float(self.reward_cfg.get("recovery_catchup_bonus", 0.32)),
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

        # Time-to-collision (TTC) safety shaping in hazard phases.
        ttc_penalty = 0.0
        closing_speed = ego.velocity - front.velocity
        if gap > 0.0 and closing_speed > 1e-6:
            ttc = gap / closing_speed
            ttc_threshold = float(self.reward_cfg.get("ttc_threshold_s", 1.8))
            shortfall = max(0.0, ttc_threshold - ttc)
            hazard_multiplier = 1.0
            if phase in {
                "pulse_brake",
                "traffic_wave",
                "low_friction",
                "cutin_emergency",
                "merge_zone",
                "ambulance_approach",
                "ambulance_overtaking",
            }:
                hazard_multiplier = float(self.reward_cfg.get("hazard_ttc_multiplier", 1.5))
            ttc_penalty = -shortfall * float(self.reward_cfg.get("ttc_weight", 0.5)) * hazard_multiplier

        # Merge specific rewards (scenario_02_merge)
        merge_success_bonus = 0.0
        merge_efficiency_reward = 0.0
        merge_zipper_bonus = 0.0
        merge_speed_match_bonus = 0.0
        merge_rush_penalty = 0.0
        merge_post_spacing_bonus = 0.0
        merge_approach_patience_bonus = 0.0

        if scenario_name == "scenario_02_merge":
            lat_ignore = float(self.reward_cfg.get("merge_lateral_ignore_gap_m", 1.85))
            in_merge_corridor = lateral_separation <= lat_ignore
            zip_min = float(self.reward_cfg.get("merge_zipper_gap_min_m", 4.5))
            zip_max = float(self.reward_cfg.get("merge_zipper_gap_max_m", 24.0))
            spd_band = float(self.reward_cfg.get("merge_speed_match_band_mps", 1.25))

            if phase == "merge_zone" and in_merge_corridor and zip_min <= gap <= zip_max:
                closing = ego.velocity - front.velocity
                if closing <= 0.8:
                    merge_zipper_bonus = float(self.reward_cfg.get("merge_zipper_bonus", 0.32))
                rush_d = float(self.reward_cfg.get("merge_rush_speed_delta_mps", 2.2))
                if gap < zip_min + 2.0 and closing > rush_d:
                    merge_rush_penalty = float(self.reward_cfg.get("merge_rush_penalty", -0.55))

            if phase == "merge_zone" and ego.path_type == "merge" and in_merge_corridor:
                dv = abs(ego.velocity - front.velocity)
                if dv <= spd_band:
                    merge_speed_match_bonus = float(self.reward_cfg.get("merge_speed_match_bonus", 0.15))
                w_eff = float(self.reward_cfg.get("merge_efficiency_weight", 0.22))
                merge_efficiency_reward = w_eff * float(np.clip(1.0 - dv / max(spd_band * 2.5, 0.1), 0.0, 1.0))

            if phase == "steady" and dist_to_merge is not None and dist_to_merge > 12.0:
                if ego.path_type == "merge" and ego.brake_pedal > 0.15 and ego.velocity <= front.velocity + 0.5:
                    merge_approach_patience_bonus = float(self.reward_cfg.get("merge_approach_patience_bonus", 0.08))

            if phase == "post_merge" and gap > 0.0:
                post_min = float(self.reward_cfg.get("merge_post_min_gap_m", 4.0))
                spd_def = float(front.velocity - ego.velocity)
                if ego.path_type == "merge":
                    if gap >= post_min:
                        merge_post_spacing_bonus = float(self.reward_cfg.get("merge_post_spacing_bonus", 0.45))
                    if ego.velocity < front.velocity - 0.25 and gap_error > -2.0:
                        merge_success_bonus = max(
                            merge_success_bonus,
                            float(self.reward_cfg.get("post_merge_catchup_bonus", 0.32)),
                        )
                    if spd_def > 0.35 and gap_error > 0.8 and gap >= post_min * 0.65:
                        rec = float(self.reward_cfg.get("post_merge_velocity_recover_bonus", 0.62))
                        scale = float(
                            np.clip((spd_def - 0.35) / 4.5, 0.0, 1.0)
                            * np.clip(gap_error / 14.0, 0.4, 1.6)
                        )
                        merge_success_bonus = max(merge_success_bonus, rec * scale)
                elif ego.path_type == "straight":
                    if ego.velocity < front.velocity - 0.3 and gap_error > -2.0:
                        merge_success_bonus = max(
                            merge_success_bonus,
                            float(self.reward_cfg.get("post_merge_main_catchup_bonus", 0.24)),
                        )
                    if spd_def > 0.4 and gap_error > 0.8 and gap >= post_min * 0.55:
                        rec = float(self.reward_cfg.get("post_merge_main_velocity_recover_bonus", 0.48))
                        scale = float(
                            np.clip((spd_def - 0.4) / 4.5, 0.0, 1.0)
                            * np.clip(gap_error / 16.0, 0.35, 1.5)
                        )
                        merge_success_bonus = max(merge_success_bonus, rec * scale)

        ambulance_clear_lane_bonus = 0.0
        ambulance_blocking_penalty = 0.0
        ambulance_yield_bonus = 0.0
        ambulance_pass_clear_bonus = 0.0

        if scenario_name == "scenario_03_ambulance" and ambulance_ctx:
            heard = bool(ambulance_ctx.get("heard_siren"))
            amb_lane = int(ambulance_ctx.get("ambulance_lane", 1))
            ego_lane = int(ambulance_ctx.get("ego_lane", 1))
            blocking = bool(ambulance_ctx.get("blocking_ambulance_lane"))
            closing = bool(ambulance_ctx.get("ambulance_closing_fast"))
            passed = bool(ambulance_ctx.get("ambulance_passed"))
            changed_lane = bool(ambulance_ctx.get("changed_lane_this_step"))

            if heard and ego_lane != amb_lane and phase in {"ambulance_approach", "ambulance_overtaking"}:
                ambulance_clear_lane_bonus = float(self.reward_cfg.get("ambulance_clear_lane_bonus", 0.55))

            if heard and blocking and closing:
                ambulance_blocking_penalty = float(self.reward_cfg.get("ambulance_blocking_penalty", -1.2))

            if heard and changed_lane and ego_lane != amb_lane:
                ambulance_yield_bonus = float(self.reward_cfg.get("ambulance_yield_lane_bonus", 0.45))

            if passed and ego_lane != amb_lane:
                ambulance_pass_clear_bonus = float(self.reward_cfg.get("ambulance_pass_clear_bonus", 0.35))

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
            ttc_penalty=ttc_penalty,
            merge_success_bonus=merge_success_bonus,
            merge_efficiency_reward=merge_efficiency_reward,
            merge_zipper_bonus=merge_zipper_bonus,
            merge_speed_match_bonus=merge_speed_match_bonus,
            merge_rush_penalty=merge_rush_penalty,
            merge_post_spacing_bonus=merge_post_spacing_bonus,
            merge_approach_patience_bonus=merge_approach_patience_bonus,
            ambulance_clear_lane_bonus=ambulance_clear_lane_bonus,
            ambulance_blocking_penalty=ambulance_blocking_penalty,
            ambulance_yield_bonus=ambulance_yield_bonus,
            ambulance_pass_clear_bonus=ambulance_pass_clear_bonus,
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
