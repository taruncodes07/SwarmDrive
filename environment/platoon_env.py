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
from environment.scenarios import Scenario01Brake, Scenario02Merge, Scenario03LowFriction
from environment.vehicle import Vehicle

ACTION_REGEX = re.compile(
    r"ACTION:\s*accel_pedal:\s*([0-9]*\.?[0-9]+)\s*brake_pedal:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE | re.MULTILINE,
)


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
        if scenario_name == "scenario_03_low_friction":
            return Scenario03LowFriction(self.settings["scenario_03"])
        raise ValueError(f"Unknown scenario: {scenario_name}")

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
        self.phase = self.scenario.get_phase(self.timestep)
        self.broadcast_layer.clear()

        for vehicle in self.vehicles.values():
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
        self.phase = self.scenario.get_phase(self.timestep)
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

        parse_logs: list[dict[str, Any]] = []
        parsed_actions: dict[int, tuple[float, float]] = {}
        for agent_id in (1, 2):
            raw_action = actions.get(f"agent_{agent_id}", "")
            accel, brake, parse_info = self._parse_action(raw_action, agent_id)
            parsed_actions[agent_id] = (accel, brake)
            if parse_info is not None:
                parse_logs.append(parse_info)

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
            if hasattr(self.scenario, "get_y_position"):
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
            terms = self.reward_model.compute(
                ego=ego,
                front=front,
                gap=gap,
                desired_gap=desired_gap,
                phase=self.phase,
                scenario_name=self.scenario_name,
                lateral_separation=lateral_sep,
                dist_to_merge=dist_merge,
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
                },
            }

        for parse_log in parse_logs:
            self._append_metric(parse_log)

        self.timestep += 1
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
        if self.scenario_name == "scenario_02_merge":
            merge_layout = {
                "x_merge": float(self.settings["scenario_02"]["x_merge"]),
                "y_start": float(self.settings["scenario_02"].get("y_start", 3.5)),
            }
        return {
            "timestep": self.timestep,
            "max_steps": self.max_steps,
            "phase": self.phase,
            "scenario": self.scenario_name,
            "collision": collision,
            "dynamics": self._dynamics,
            "merge_layout": merge_layout,
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
                }
                for car_id, vehicle in self.vehicles.items()
            },
            "broadcast_buffer": self.broadcast_layer.buffer,
        }

    def close(self) -> None:
        return None

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

        phase = self.scenario.get_phase(self.timestep)
        road_grip = float(self._dynamics.get("road_grip", 1.0))
        road_grade = float(self._dynamics.get("road_grade", 0.0))
        peer_lines = []
        for packet in self.broadcast_layer.receive_for(agent_id):
            peer_lines.append(
                "Car {sender_id} | x={x_position:.2f} m | y={y_position:.2f} m | vel={velocity:.2f} m/s | "
                "path={path_type} | accel_pedal={accel_pedal:.2f} | brake_pedal={brake_pedal:.2f} | "
                "net_accel={net_acceleration:+.2f} m/s^2".format(**packet)
            )
        peer_block = "\n".join(peer_lines) if peer_lines else "<no broadcasts yet>"

        return (
            f"[OBSERVATION - Agent {agent_id} - Step {self.timestep}]\n"
            f"scenario_name: {self.scenario_name}\n"
            f"scenario_phase: {phase}\n"
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
            f"[PEER BROADCASTS - physical state from end of step {self.timestep - 1}]\n"
            f"{peer_block}\n"
            "Respond with your action in the exact format shown below. accel_pedal and brake_pedal cannot both be non-zero.\n"
            "ACTION:\n"
            "accel_pedal: <float 0.0-1.0>\n"
            "brake_pedal: <float 0.0-1.0>\n"
        )

    def _parse_action(self, raw_action: str, agent_id: int) -> tuple[float, float, dict[str, Any] | None]:
        text = raw_action or ""
        match = ACTION_REGEX.search(text)
        if not match:
            return 0.0, 0.0, {
                "event": "parse_failure",
                "timestep": self.timestep,
                "agent_id": agent_id,
                "raw_action": text,
                "resolution": "default_coast",
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
    }

    done = False
    step_count = 0
    while not done:
        state = env.state()
        actions: dict[str, str] = {}
        for agent_id in (1, 2):
            ego = state["vehicles"][agent_id]
            front = state["vehicles"][agent_id - 1]
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

            actions[f"agent_{agent_id}"] = (
                f"ACTION:\naccel_pedal: {accel:.2f}\nbrake_pedal: {brake:.2f}"
            )

        _, rewards, dones, infos = env.step(actions)
        step_count += 1

        for packet in env.state()["broadcast_buffer"]:
            if set(packet.keys()) != required_broadcast_fields:
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
        choices=["scenario_01_brake", "scenario_02_merge", "scenario_03_low_friction"],
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test(scenario_name=args.scenario)
    elif args.test_bad_action:
        _run_bad_action_test(scenario_name=args.scenario)
    else:
        parser.print_help()
