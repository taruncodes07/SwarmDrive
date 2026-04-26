from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from openenv import Environment
except Exception:  # pragma: no cover - fallback when OpenEnv is unavailable locally
    class Environment:  # type: ignore[no-redef]
        pass

from config.settings import ROOT_DIR, load_settings
from environment.communication import BroadcastLayer
from environment.reward import RewardModel
from environment.scenarios import Scenario01Brake, Scenario02Merge, Scenario03Ambulance
from environment.vehicle import Vehicle

ACTION_REGEX = re.compile(
    r"ACTION:\s*accel_pedal:\s*([0-9]*\.?[0-9]+)\s*brake_pedal:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE | re.MULTILINE,
)
LANE_CHANGE_REGEX = re.compile(r"lane_change:\s*(stay|left|right)", re.IGNORECASE)
TARGET_LANE_REGEX = re.compile(r"target_lane:\s*([012])", re.IGNORECASE)
MOVE_LEFT_REGEX = re.compile(r"move_left:\s*(0|1|false|true|no|yes)\b", re.IGNORECASE)
MOVE_RIGHT_REGEX = re.compile(r"move_right:\s*(0|1|false|true|no|yes)\b", re.IGNORECASE)
_LOOSE_ACCEL = re.compile(r"accel(?:_pedal)?\s*[:=]\s*([+\-]?[0-9]*\.?[0-9]+)", re.IGNORECASE)
_LOOSE_BRAKE = re.compile(r"brake(?:_pedal)?\s*[:=]\s*([+\-]?[0-9]*\.?[0-9]+)", re.IGNORECASE)
_FLOATS = re.compile(r"[+\-]?[0-9]*\.?[0-9]+")


def _normalize_action_text(raw: str) -> str:
    """Repair frequent malformed ACTION blocks from LLM output (see results/metrics.jsonl)."""
    if not raw:
        return ""
    t = raw
    # Typo: "accel_pedal: brake_pedal: 0.85" — treat trailing number as brake, zero accel.
    t = re.sub(
        r"(?im)accel_pedal:\s*brake_pedal:\s*([0-9]*\.?[0-9]+)",
        r"accel_pedal: 0.0\nbrake_pedal: \1",
        t,
    )
    # Duplicate label lines the model sometimes emits.
    t = re.sub(r"(?im)(accel_pedal:\s*)accel_pedal:\s*", r"\1", t)
    t = re.sub(r"(?im)(brake_pedal:\s*)brake_pedal:\s*", r"\1", t)
    return t


class PlatoonEnv(Environment):
    def __init__(self, settings_path: Path | None = None, scenario_name: str | None = None) -> None:
        self.settings = load_settings(settings_path)
        self._validate_manifest(ROOT_DIR / "openenv.yaml")

        sim_cfg = self.settings["simulation"]
        self.dt = float(sim_cfg["dt"])
        self.max_steps = int(sim_cfg["max_steps"])
        self.v_min = float(sim_cfg["v_min"])
        self.v_max = float(sim_cfg["v_max"])
        self.max_acceleration = float(sim_cfg["max_acceleration"])
        self.max_deceleration = float(sim_cfg["max_deceleration"])
        self.min_desired_gap = float(sim_cfg["min_desired_gap"])
        self.headway_seconds = float(sim_cfg["headway_seconds"])
        self.follower_safety_clamp = bool(sim_cfg.get("follower_safety_clamp", True))

        self.scenario_name = scenario_name or str(self.settings.get("scenario_active", "scenario_01_brake"))
        self.scenario = self._build_scenario(self.scenario_name)
        self.reward_model = RewardModel(self.settings["reward"], dt=self.dt)
        self.broadcast_layer = BroadcastLayer()

        self.metrics_path = ROOT_DIR / self.settings["logging"]["metrics_path"]
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        self.timestep = 0
        self.phase = "steady"
        self.vehicles: dict[int, Vehicle] = {}
        self._dynamics = {
            "accel_scale": 1.0,
            "decel_scale": 1.0,
            "road_grip": 1.0,
            "road_grade": 0.0,
        }

    def _build_scenario(self, scenario_name: str) -> Any:
        if scenario_name == "scenario_01_brake":
            return Scenario01Brake(self.settings["scenario_01"])
        if scenario_name == "scenario_02_merge":
            return Scenario02Merge(self.settings["scenario_02"])
        if scenario_name == "scenario_03_ambulance":
            return Scenario03Ambulance(self.settings["scenario_03"])
        raise ValueError(f"Unknown scenario: {scenario_name}")

    def _compute_phase(self) -> str:
        if self.scenario_name == "scenario_03_ambulance":
            return self.scenario.get_phase(self.timestep, self.vehicles)
        return self.scenario.get_phase(self.timestep)

    def _validate_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise ValueError(f"openenv.yaml missing: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            try:
                yaml.safe_load(handle)
            except yaml.YAMLError as exc:
                raise ValueError(f"openenv.yaml invalid: {exc}") from exc

    def reset(self, seed: int | None = None) -> dict[str, str]:
        if seed is None:
            seed = int(self.settings["simulation"]["seed"])
        np.random.seed(seed)

        sim = self.settings["simulation"]
        if self.scenario_name == "scenario_02_merge":
            sc = self.settings["scenario_02"]
            xm = float(sc["x_merge"])
            d = float(sc.get("merge_distance_m", 40.0))
            v0 = float(sc.get("approach_speed_mps", sc.get("lead_cruise_speed", 15.0)))
            lead_off = float(sc.get("lead_past_merge_m", 68.0))
            x_main = xm - d
            x_lead = xm + lead_off
            self.vehicles = {
                0: Vehicle(
                    car_id=0,
                    x=x_lead,
                    velocity=v0,
                    path_type="straight",
                    length=float(sim["vehicle_length"]),
                    width=float(sim["vehicle_width"]),
                ),
                1: Vehicle(
                    car_id=1,
                    x=x_main,
                    velocity=v0,
                    path_type="straight",
                    length=float(sim["vehicle_length"]),
                    width=float(sim["vehicle_width"]),
                ),
                2: Vehicle(
                    car_id=2,
                    x=x_main,
                    velocity=v0,
                    path_type="merge",
                    length=float(sim["vehicle_length"]),
                    width=float(sim["vehicle_width"]),
                ),
            }
        elif self.scenario_name == "scenario_03_ambulance":
            sc = self.settings["scenario_03"]
            rng = np.random.default_rng(seed)
            lane_sp = float(sc["lane_spacing_m"])
            lead_x = float(sc.get("x_lead_min", 195.0)) + float(rng.uniform(0.0, 10.0))
            x1 = lead_x - float(rng.uniform(24.0, 40.0))
            x2 = x1 - float(rng.uniform(16.0, 32.0))
            x3 = max(6.0, x2 - float(rng.uniform(48.0, 88.0)))
            v_traffic = float(sc["lead_cruise_speed"])
            v_amb0 = float(sc["ambulance_cruise_speed"]) * 0.92
            lane_lead = 1
            lane_1 = int(rng.integers(0, 3))
            lane_2 = int(rng.integers(0, 3))
            lane_amb = int(rng.integers(0, 3))
            vl = float(sim["vehicle_length"])
            vw = float(sim["vehicle_width"])
            self.vehicles = {
                0: Vehicle(
                    car_id=0,
                    x=lead_x,
                    velocity=v_traffic,
                    path_type="straight",
                    length=vl,
                    width=vw,
                    lane=lane_lead,
                    vehicle_role="passenger",
                    emergency_siren=False,
                    y=Scenario03Ambulance.lane_to_y(lane_lead, lane_sp),
                ),
                1: Vehicle(
                    car_id=1,
                    x=x1,
                    velocity=v_traffic,
                    path_type="straight",
                    length=vl,
                    width=vw,
                    lane=lane_1,
                    vehicle_role="passenger",
                    emergency_siren=False,
                    y=Scenario03Ambulance.lane_to_y(lane_1, lane_sp),
                ),
                2: Vehicle(
                    car_id=2,
                    x=x2,
                    velocity=v_traffic,
                    path_type="straight",
                    length=vl,
                    width=vw,
                    lane=lane_2,
                    vehicle_role="passenger",
                    emergency_siren=False,
                    y=Scenario03Ambulance.lane_to_y(lane_2, lane_sp),
                ),
                3: Vehicle(
                    car_id=3,
                    x=x3,
                    velocity=v_amb0,
                    path_type="straight",
                    length=vl,
                    width=vw,
                    lane=lane_amb,
                    vehicle_role="ambulance",
                    emergency_siren=True,
                    y=Scenario03Ambulance.lane_to_y(lane_amb, lane_sp),
                ),
            }
        else:
            init = self.settings["initial_conditions"]
            self.vehicles = {
                0: Vehicle(
                    car_id=0,
                    x=float(init["car_0"]["x"]),
                    velocity=float(init["car_0"]["velocity"]),
                    path_type="straight",
                    length=float(sim["vehicle_length"]),
                    width=float(sim["vehicle_width"]),
                ),
                1: Vehicle(
                    car_id=1,
                    x=float(init["car_1"]["x"]),
                    velocity=float(init["car_1"]["velocity"]),
                    path_type="straight",
                    length=float(sim["vehicle_length"]),
                    width=float(sim["vehicle_width"]),
                ),
                2: Vehicle(
                    car_id=2,
                    x=float(init["car_2"]["x"]),
                    velocity=float(init["car_2"]["velocity"]),
                    path_type="straight",
                    length=float(sim["vehicle_length"]),
                    width=float(sim["vehicle_width"]),
                ),
            }

        self.timestep = 0
        self.phase = self._compute_phase()
        self.broadcast_layer.clear()

        for vehicle in self.vehicles.values():
            if self.scenario_name == "scenario_03_ambulance":
                continue
            if hasattr(self.scenario, "get_y_position"):
                vehicle.y = self.scenario.get_y_position(vehicle)
            else:
                vehicle.y = 0.0

        return {
            "agent_1": self._build_observation(agent_id=1),
            "agent_2": self._build_observation(agent_id=2),
        }

    def step(
        self, actions: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, float], dict[str, bool], dict[str, dict[str, Any]]]:
        self.phase = self._compute_phase()
        self._dynamics = self.scenario.dynamics_modifiers(self.phase)
        accel_scale = float(self._dynamics.get("accel_scale", 1.0))
        decel_scale = float(self._dynamics.get("decel_scale", 1.0))
        step_max_accel = self.max_acceleration * accel_scale
        step_max_decel = self.max_deceleration * decel_scale

        lead_accel, lead_brake = self.scenario.lead_controls(self.vehicles[0], self.phase)
        self.vehicles[0].apply_action(
            lead_accel,
            lead_brake,
            dt=self.dt,
            max_acceleration=step_max_accel,
            max_deceleration=step_max_decel,
            v_min=self.v_min,
            v_max=self.v_max,
        )

        if self.scenario_name == "scenario_03_ambulance":
            sc3 = self.settings["scenario_03"]
            v_amb_max = float(sc3.get("ambulance_v_max", 32.0))
            aa, ab = self.scenario.ambulance_controls(self.vehicles[3], self.phase)
            self.vehicles[3].apply_action(
                aa,
                ab,
                dt=self.dt,
                max_acceleration=step_max_accel * 1.12,
                max_deceleration=step_max_decel,
                v_min=self.v_min,
                v_max=v_amb_max,
            )

        parse_logs: list[dict[str, Any]] = []
        parsed_actions: dict[int, tuple[float, float]] = {}
        agent_prev_lane: dict[int, int] = {}
        for agent_id in (1, 2):
            raw_action = actions.get(f"agent_{agent_id}", "")
            agent_prev_lane[agent_id] = self.vehicles[agent_id].lane
            accel, brake, parse_info = self._parse_action(raw_action, agent_id)
            if self.follower_safety_clamp:
                accel, brake = self._safety_clamp_follower(agent_id, accel, brake)
            parsed_actions[agent_id] = (accel, brake)
            if parse_info is not None:
                parse_logs.append(parse_info)
            if self.scenario_name == "scenario_03_ambulance":
                self._apply_lane_intent(self.vehicles[agent_id], raw_action)

        for agent_id in (1, 2):
            accel, brake = parsed_actions[agent_id]
            self.vehicles[agent_id].apply_action(
                accel,
                brake,
                dt=self.dt,
                max_acceleration=step_max_accel,
                max_deceleration=step_max_decel,
                v_min=self.v_min,
                v_max=self.v_max,
            )

        for vehicle in self.vehicles.values():
            if self.scenario_name == "scenario_03_ambulance":
                self._sync_lateral_to_lane(vehicle)
            elif hasattr(self.scenario, "get_y_position"):
                vehicle.y = self.scenario.get_y_position(vehicle)
            else:
                vehicle.y = 0.0

        self.broadcast_layer.update([vehicle.to_broadcast_packet() for vehicle in self.vehicles.values()])

        rewards: dict[str, float] = {}
        infos: dict[str, dict[str, Any]] = {}
        collision = self._pairwise_collision_any()

        for agent_id in (1, 2):
            ego = self.vehicles[agent_id]
            front = self.vehicles[agent_id - 1]
            gap_raw = self.reward_model.gap_to_front(front, ego)
            gap = self._effective_gap_for_merge(front, ego, gap_raw)
            desired_gap = self.reward_model.desired_gap(
                ego_velocity=ego.velocity,
                min_gap=self.min_desired_gap,
                headway_seconds=self.headway_seconds,
            )
            dist_merge = None
            lateral_sep = abs(ego.y - front.y)
            if hasattr(self.scenario, "x_merge"):
                dist_merge = float(self.scenario.x_merge - ego.x)
            amb_ctx = None
            if self.scenario_name == "scenario_03_ambulance":
                amb_ctx = self._ambulance_context(agent_id, agent_prev_lane[agent_id])
            terms = self.reward_model.compute(
                ego=ego,
                front=front,
                gap=gap,
                desired_gap=desired_gap,
                phase=self.phase,
                scenario_name=self.scenario_name,
                lateral_separation=lateral_sep,
                dist_to_merge=dist_merge,
                global_collision=self._agent_in_collision(agent_id),
                ambulance_ctx=amb_ctx,
            )
            rewards[f"agent_{agent_id}"] = terms.total
            infos[f"agent_{agent_id}"] = {
                "gap": gap_raw,
                "gap_effective": gap,
                "desired_gap": desired_gap,
                "gap_error": gap - desired_gap,
                "reward_terms": {
                    "collision_penalty": terms.collision_penalty,
                    "gap_error_penalty": terms.gap_error_penalty,
                    "speed_maintenance": terms.speed_maintenance,
                    "jerk_penalty": terms.jerk_penalty,
                    "recovery_bonus": terms.recovery_bonus,
                    "comfort_penalty": terms.comfort_penalty,
                    "alive_bonus": terms.alive_bonus,
                    "gap_tracking_bonus": terms.gap_tracking_bonus,
                    "speed_tracking_bonus": terms.speed_tracking_bonus,
                    "ttc_penalty": terms.ttc_penalty,
                    "merge_success_bonus": terms.merge_success_bonus,
                    "merge_efficiency_reward": terms.merge_efficiency_reward,
                    "merge_zipper_bonus": terms.merge_zipper_bonus,
                    "merge_speed_match_bonus": terms.merge_speed_match_bonus,
                    "merge_rush_penalty": terms.merge_rush_penalty,
                    "merge_post_spacing_bonus": terms.merge_post_spacing_bonus,
                    "merge_approach_patience_bonus": terms.merge_approach_patience_bonus,
                    "ambulance_clear_lane_bonus": terms.ambulance_clear_lane_bonus,
                    "ambulance_blocking_penalty": terms.ambulance_blocking_penalty,
                    "ambulance_yield_bonus": terms.ambulance_yield_bonus,
                    "ambulance_pass_clear_bonus": terms.ambulance_pass_clear_bonus,
                },
            }

        for parse_log in parse_logs:
            self._append_metric(parse_log)

        self.timestep += 1
        self.phase = self._compute_phase()
        done = collision or self.timestep >= self.max_steps

        obs = {
            "agent_1": self._build_observation(agent_id=1),
            "agent_2": self._build_observation(agent_id=2),
        }

        dones = {"agent_1": done, "agent_2": done}
        return obs, rewards, dones, infos

    def state(self) -> dict[str, Any]:
        collision = self._pairwise_collision_any()
        merge_layout = None
        road_layout = None
        if self.scenario_name == "scenario_02_merge":
            merge_layout = {
                "x_merge": float(self.settings["scenario_02"]["x_merge"]),
                "y_start": float(self.settings["scenario_02"].get("y_start", 3.5)),
            }
        if self.scenario_name == "scenario_03_ambulance":
            road_layout = {
                "kind": "three_lane",
                "lane_spacing_m": float(self.settings["scenario_03"]["lane_spacing_m"]),
            }
        ambulance_clearance = None
        if self.scenario_name == "scenario_03_ambulance":
            c1 = self._ambulance_context(1, self.vehicles[1].lane)
            c2 = self._ambulance_context(2, self.vehicles[2].lane)
            ambulance_clearance = {
                "agent_1": bool(c1["ambulance_passed"]),
                "agent_2": bool(c2["ambulance_passed"]),
            }
        return {
            "timestep": self.timestep,
            "max_steps": self.max_steps,
            "phase": self.phase,
            "scenario": self.scenario_name,
            "collision": collision,
            "ambulance_clearance": ambulance_clearance,
            "dynamics": self._dynamics,
            "merge_layout": merge_layout,
            "road_layout": road_layout,
            "vehicles": {
                car_id: {
                    "x": vehicle.x,
                    "y": vehicle.y,
                    "velocity": vehicle.velocity,
                    "accel_pedal": vehicle.accel_pedal,
                    "brake_pedal": vehicle.brake_pedal,
                    "net_acceleration": vehicle.net_acceleration,
                    "length": vehicle.length,
                    "width": vehicle.width,
                    "path_type": vehicle.path_type,
                    "lane": vehicle.lane,
                    "vehicle_role": vehicle.vehicle_role,
                    "emergency_siren": vehicle.emergency_siren,
                }
                for car_id, vehicle in self.vehicles.items()
            },
            "broadcast_buffer": self.broadcast_layer.buffer,
        }

    def close(self) -> None:
        return None

    def _apply_lane_intent(self, vehicle: Vehicle, raw_action: str) -> None:
        text = raw_action or ""
        vehicle.last_lateral = "—"

        def _truthy(token: str) -> bool:
            return token.strip().lower() in ("1", "true", "yes")

        tm = TARGET_LANE_REGEX.search(text)
        if tm:
            vehicle.lane = int(np.clip(int(tm.group(1)), 0, 2))
            vehicle.last_lateral = f"target_lane:{vehicle.lane}"
            return
        ml = MOVE_LEFT_REGEX.search(text)
        mr = MOVE_RIGHT_REGEX.search(text)
        if ml or mr:
            v_l = bool(ml and _truthy(ml.group(1)))
            v_r = bool(mr and _truthy(mr.group(1)))
            if v_l and v_r:
                vehicle.last_lateral = "move_conflict"
                return
            if v_l:
                vehicle.lane = max(0, vehicle.lane - 1)
                vehicle.last_lateral = "move_left"
                return
            if v_r:
                vehicle.lane = min(2, vehicle.lane + 1)
                vehicle.last_lateral = "move_right"
                return
        lm = LANE_CHANGE_REGEX.search(text)
        if lm:
            word = lm.group(1).lower()
            if word == "left":
                vehicle.lane = max(0, vehicle.lane - 1)
                vehicle.last_lateral = "lane_left"
            elif word == "right":
                vehicle.lane = min(2, vehicle.lane + 1)
                vehicle.last_lateral = "lane_right"
            else:
                vehicle.last_lateral = "lane_stay"

    def _sync_lateral_to_lane(self, vehicle: Vehicle) -> None:
        sc = self.settings["scenario_03"]
        lane_sp = float(sc["lane_spacing_m"])
        ty = Scenario03Ambulance.lane_to_y(int(vehicle.lane), lane_sp)
        max_dy = float(sc.get("max_lateral_speed_mps", 9.0)) * self.dt
        vehicle.y += float(np.clip(ty - vehicle.y, -max_dy, max_dy))

    def _ambulance_context(self, ego_id: int, prev_lane: int) -> dict[str, Any]:
        amb = self.vehicles[3]
        ego = self.vehicles[ego_id]
        sc = self.settings["scenario_03"]
        r_comm = float(sc["proximity_comm_range_m"])
        dist = float(np.hypot(ego.x - amb.x, ego.y - amb.y))
        heard = dist <= r_comm and bool(amb.emergency_siren)
        amb_lane = int(amb.lane)
        ego_lane = int(ego.lane)
        long_cut = float(sc.get("blocking_longitudinal_m", 55.0))
        blocking = bool(
            ego_lane == amb_lane and amb.x < ego.x and (ego.x - amb.x) < long_cut
        )
        closing = bool((ego.velocity - amb.velocity) < -1.5)
        passed = bool(amb.x > ego.x + ego.length)
        changed = bool(ego.lane != prev_lane)
        return {
            "heard_siren": heard,
            "ambulance_lane": amb_lane,
            "ego_lane": ego_lane,
            "blocking_ambulance_lane": blocking,
            "ambulance_closing_fast": closing and blocking,
            "ambulance_passed": passed,
            "changed_lane_this_step": changed,
        }

    def _effective_gap_for_merge(self, front: Vehicle, ego: Vehicle, gap_raw: float) -> float:
        if self.scenario_name != "scenario_02_merge":
            return gap_raw
        lat = abs(ego.y - front.y)
        ignore = float(self.settings["reward"].get("merge_lateral_ignore_gap_m", 1.85))
        nominal = float(self.settings["reward"].get("merge_nominal_gap_when_offset_m", 20.0))
        if lat > ignore:
            return max(gap_raw, nominal)
        return gap_raw

    @staticmethod
    def _vehicles_collide_2d(a: Vehicle, b: Vehicle) -> bool:
        lat = abs(a.y - b.y)
        half_w = (a.width + b.width) * 0.5 * 0.92
        if lat > half_w:
            return False
        overlap = min(a.x + a.length, b.x + b.length) - max(a.x, b.x)
        return overlap > -0.4

    def _pairwise_collision_any(self) -> bool:
        ids = sorted(self.vehicles.keys())
        for i, ia in enumerate(ids):
            for ib in ids[i + 1 :]:
                if self._vehicles_collide_2d(self.vehicles[ia], self.vehicles[ib]):
                    return True
        return False

    def _agent_in_collision(self, agent_id: int) -> bool:
        v = self.vehicles[agent_id]
        for oid, o in self.vehicles.items():
            if oid != agent_id and self._vehicles_collide_2d(v, o):
                return True
        return False

    def _build_observation(self, agent_id: int) -> str:
        ego = self.vehicles[agent_id]
        front = self.vehicles[agent_id - 1]

        gap_raw = self.reward_model.gap_to_front(front, ego)
        gap_to_front = self._effective_gap_for_merge(front, ego, gap_raw)
        # In merge scenario, we want to show distance to merge point as well
        merge_info = ""
        if hasattr(self.scenario, "x_merge"):
            dist_to_merge = self.scenario.x_merge - ego.x
            merge_info = f"dist_to_merge: {dist_to_merge:.2f} m\n"

        desired_gap = self.reward_model.desired_gap(
            ego_velocity=ego.velocity,
            min_gap=self.min_desired_gap,
            headway_seconds=self.headway_seconds,
        )
        gap_error = gap_to_front - desired_gap
        gap_note = ""
        if self.scenario_name == "scenario_02_merge" and gap_raw != gap_to_front:
            gap_note = f"(raw longitudinal gap {gap_raw:.2f} m — lateral offset; effective gap for learning: {gap_to_front:.2f} m)\n"
        front_velocity = front.velocity if agent_id > 0 else -1.0

        phase = self._compute_phase()
        road_grip = float(self._dynamics.get("road_grip", 1.0))
        road_grade = float(self._dynamics.get("road_grade", 0.0))
        lane_info = ""
        amb_hint = ""
        if self.scenario_name == "scenario_03_ambulance":
            lane_info = (
                f"ego_lane: {ego.lane} (0=left, 1=center, 2=right)\n"
                "layout: three_lane_highway\n"
            )
            amb = self.vehicles[3]
            dist_amb = float(np.hypot(ego.x - amb.x, ego.y - amb.y))
            amb_hint = (
                f"distance_to_ambulance_m: {dist_amb:.1f} | ambulance_lane: {amb.lane} | ambulance_siren: {amb.emergency_siren}\n"
                "Peer list only includes ambulance packets when within proximity_comm_range (emergency V2V).\n"
            )
        peer_lines = []
        for packet in self.broadcast_layer.receive_for(agent_id):
            if self.scenario_name == "scenario_03_ambulance":
                sc3 = self.settings["scenario_03"]
                if str(packet.get("vehicle_role")) == "ambulance":
                    dist = float(
                        np.hypot(
                            ego.x - float(packet["x_position"]),
                            ego.y - float(packet["y_position"]),
                        )
                    )
                    if dist > float(sc3["proximity_comm_range_m"]):
                        continue
            peer_lines.append(
                "Car {sender_id} | role={vehicle_role} | lane={lane_index} | siren={emergency_siren} | "
                "x={x_position:.2f} m | y={y_position:.2f} m | vel={velocity:.2f} m/s | "
                "path={path_type} | accel_pedal={accel_pedal:.2f} | brake_pedal={brake_pedal:.2f} | "
                "net_accel={net_acceleration:+.2f} m/s^2".format(**packet)
            )
        peer_block = "\n".join(peer_lines) if peer_lines else "<no broadcasts yet>"

        action_tail = (
            "Respond with your action in the exact format shown below. accel_pedal and brake_pedal cannot both be non-zero.\n"
            "ACTION:\n"
            "accel_pedal: <float 0.0-1.0>\n"
            "brake_pedal: <float 0.0-1.0>\n"
        )
        if self.scenario_name == "scenario_03_ambulance":
            action_tail += (
                "move_left: <false|true>   (optional; 0/1 also accepted)\n"
                "move_right: <false|true>   (optional; at most one of move_left/move_right true)\n"
                "lane_change: <stay|left|right>   OR   target_lane: <0|1|2>\n"
                "(optional; default stay). At most one discrete lane shift intent per step.\n"
            )

        return (
            f"[OBSERVATION - Agent {agent_id} - Step {self.timestep}]\n"
            f"scenario_name: {self.scenario_name}\n"
            f"scenario_phase: {phase}\n"
            f"{lane_info}"
            f"{amb_hint}"
            f"{merge_info}"
            f"road_grip:      {road_grip:.2f}\n"
            f"road_grade:     {road_grade:+.3f}\n"
            f"ego_velocity:    {ego.velocity:.2f} m/s\n"
            f"ego_y:           {ego.y:.2f} m\n"
            f"ego_path:        {ego.path_type}\n"
            f"ego_accel_pedal: {ego.accel_pedal:.2f}\n"
            f"ego_brake_pedal: {ego.brake_pedal:.2f}\n"
            f"ego_x:           {ego.x:.2f} m\n"
            "ego_length: 4.5 m  |  ego_width: 1.8 m\n"
            f"{gap_note}"
            f"gap_to_front:  {gap_to_front:.2f} m\n"
            f"desired_gap:   {desired_gap:.2f} m   (gap_error: {gap_error:+.2f} m)\n"
            f"front_velocity: {front_velocity:.2f} m/s\n"
            f"[PEER BROADCASTS - physical state from end of prior step]\n"
            f"{peer_block}\n"
            f"{action_tail}"
        )

    def _parse_action_loose(self, text: str) -> tuple[float, float] | None:
        am = _LOOSE_ACCEL.search(text)
        bm = _LOOSE_BRAKE.search(text)
        if am and bm:
            return float(am.group(1)), float(bm.group(1))
        if am and not bm:
            return float(am.group(1)), 0.0
        if bm and not am:
            return 0.0, float(bm.group(1))
        nums = [float(x) for x in _FLOATS.findall(text)]
        if len(nums) >= 2:
            return nums[0], nums[1]
        if len(nums) == 1:
            return nums[0], 0.0
        return None

    def _safety_clamp_follower(self, agent_id: int, accel: float, brake: float) -> tuple[float, float]:
        """Raise braking when bumper gap is critically tight (longitudinal rear-end guard)."""
        ego = self.vehicles[agent_id]
        front = self.vehicles[agent_id - 1]
        gap_raw = self.reward_model.gap_to_front(front, ego)
        gap = self._effective_gap_for_merge(front, ego, gap_raw)
        desired = self.reward_model.desired_gap(
            ego_velocity=ego.velocity,
            min_gap=self.min_desired_gap,
            headway_seconds=self.headway_seconds,
        )
        gap_error = gap - desired
        closing = float(ego.velocity - front.velocity)
        a = float(np.clip(accel, 0.0, 1.0))
        b = float(np.clip(brake, 0.0, 1.0))
        if gap_error < -1.0:
            need = float(
                np.clip(
                    (-gap_error) / 14.0 + max(0.0, closing - 0.4) / 10.0,
                    0.25,
                    0.95,
                )
            )
            if b < need:
                b = need
                a = min(a, 0.12)
        if gap_error < -3.5:
            b = max(b, 0.5)
            a = 0.0
        if gap_error < -6.0:
            b = max(b, 0.72)
            a = 0.0
        if a > 0.0 and b > 0.0:
            a = 0.0
        return float(np.clip(a, 0.0, 1.0)), float(np.clip(b, 0.0, 1.0))

    def _parse_action(self, raw_action: str, agent_id: int) -> tuple[float, float, dict[str, Any] | None]:
        text = _normalize_action_text(raw_action or "")
        match = ACTION_REGEX.search(text)
        if not match:
            loose = self._parse_action_loose(text)
            if loose is None:
                return 0.0, 0.0, {
                    "event": "parse_failure",
                    "timestep": self.timestep,
                    "agent_id": agent_id,
                    "raw_action": raw_action or "",
                    "resolution": "default_coast",
                }
            accel = float(np.clip(loose[0], 0.0, 1.0))
            brake = float(np.clip(loose[1], 0.0, 1.0))
            if accel > 0.0 and brake > 0.0:
                accel = 0.0
            return accel, brake, {
                "event": "parse_recovered",
                "timestep": self.timestep,
                "agent_id": agent_id,
                "resolution": "loose_parser",
            }

        accel = float(np.clip(float(match.group(1)), 0.0, 1.0))
        brake = float(np.clip(float(match.group(2)), 0.0, 1.0))

        if accel > 0.0 and brake > 0.0:
            log_item = {
                "event": "constraint_violation",
                "timestep": self.timestep,
                "agent_id": agent_id,
                "raw_action": text,
                "resolution": "set_accel_to_zero_keep_brake",
            }
            return 0.0, brake, log_item

        return accel, brake, None

    def _append_metric(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def _run_smoke_test(scenario_name: str | None = None) -> None:
    env = PlatoonEnv(scenario_name=scenario_name)
    obs = env.reset(seed=123)
    print("Smoke test start. Initial observation keys:", list(obs.keys()))

    required_broadcast_fields = {
        "sender_id",
        "x_position",
        "y_position",
        "velocity",
        "path_type",
        "accel_pedal",
        "brake_pedal",
        "net_acceleration",
        "length",
        "width",
        "lane_index",
        "lateral_intent",
        "vehicle_role",
        "emergency_siren",
    }

    done = False
    step_count = 0
    while not done:
        state = env.state()
        veh = state["vehicles"]
        actions: dict[str, str] = {}
        for agent_id in (1, 2):
            ego = veh[agent_id]
            front = veh[agent_id - 1]
            gap = front["x"] - ego["x"] - ego["length"]
            desired_gap = max(5.0, ego["velocity"] * 2.0)
            gap_error = gap - desired_gap
            closing_speed = ego["velocity"] - front["velocity"]

            accel = 0.0
            brake = 0.0
            if gap_error < 0.0 or closing_speed > 0.7:
                brake = float(np.clip(((-gap_error) / 12.0) + (max(0.0, closing_speed) / 10.0), 0.0, 1.0))
            elif gap_error > 2.0:
                accel = float(np.clip(gap_error / 25.0, 0.0, 1.0))

            lane_line = ""
            if env.scenario_name == "scenario_03_ambulance":
                amb_lane = veh[3]["lane"]
                dist_a = float(np.hypot(ego["x"] - veh[3]["x"], ego["y"] - veh[3]["y"]))
                lc = "stay"
                if dist_a < float(env.settings["scenario_03"]["proximity_comm_range_m"]) and int(ego["lane"]) == int(
                    amb_lane
                ):
                    lc = "right" if int(ego["lane"]) < 2 else "left"
                lane_line = f"lane_change: {lc}\n"

            actions[f"agent_{agent_id}"] = (
                f"ACTION:\naccel_pedal: {accel:.2f}\nbrake_pedal: {brake:.2f}\n{lane_line}"
            )

        _, rewards, dones, infos = env.step(actions)
        step_count += 1

        for packet in env.state()["broadcast_buffer"]:
            if not required_broadcast_fields.issubset(set(packet.keys())):
                raise RuntimeError(f"Broadcast packet missing fields: {packet}")

        if step_count % 50 == 0 or dones["agent_1"]:
            print(
                f"step={step_count} phase={env.phase} "
                f"r1={rewards['agent_1']:.3f} r2={rewards['agent_2']:.3f} "
                f"g1={infos['agent_1']['gap']:.3f} g2={infos['agent_2']['gap']:.3f}"
            )

        done = dones["agent_1"]

    print(f"Smoke test complete. Total steps executed: {step_count}")


def _run_bad_action_test(scenario_name: str | None = None) -> None:
    env = PlatoonEnv(scenario_name=scenario_name)
    env.reset(seed=321)
    _ = env.step(
        {
            "agent_1": "I am malformed output",
            "agent_2": "ACTION:\naccel_pedal: 0.30\nbrake_pedal: 0.70",
        }
    )
    print("Bad action test complete. Check results/metrics.jsonl for parse_failure and constraint_violation logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--test-bad-action", action="store_true")
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=["scenario_01_brake", "scenario_02_merge", "scenario_03_ambulance"],
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test(scenario_name=args.scenario)
    elif args.test_bad_action:
        _run_bad_action_test(scenario_name=args.scenario)
    else:
        parser.print_help()
