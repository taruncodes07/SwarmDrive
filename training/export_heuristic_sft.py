"""Generate SFT JSONL via the same heuristic used in platoon_env smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config.settings import ROOT_DIR
from environment.platoon_env import PlatoonEnv


def _heuristic_actions(env: PlatoonEnv) -> dict[str, str]:
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
    return actions


def _scenario_tag(name: str) -> str:
    if name == "scenario_01_brake":
        return "scenario_01"
    if name == "scenario_02_merge":
        return "scenario_02"
    if name == "scenario_03_ambulance":
        return "scenario_03"
    raise ValueError(name)


def export_one_episode(env: PlatoonEnv, seed: int, out_handle, run_id: str) -> int:
    obs = env.reset(seed=seed)
    rows = 0
    done = False
    while not done:
        t = env.timestep
        phase = env.phase
        actions = _heuristic_actions(env)
        for agent_id in (1, 2):
            reasoning = (
                f"Heuristic gap-keeping with phase={phase}; "
                f"merge/ambulance rules applied when relevant for {env.scenario_name}."
            )
            rec = {
                "id": f"{run_id}_s{seed}_t{t}_a{agent_id}",
                "scenario": _scenario_tag(env.scenario_name),
                "phase": phase,
                "agent_id": agent_id,
                "timestep": t,
                "observation_text": obs[f"agent_{agent_id}"],
                "reasoning": reasoning,
                "action_text": actions[f"agent_{agent_id}"],
            }
            out_handle.write(json.dumps(rec) + "\n")
            rows += 1

        obs, _, dones, _ = env.step(actions)
        done = bool(dones["agent_1"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export heuristic trajectories as SFT jsonl")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        metavar="NAME",
        help="Scenario (repeatable). Default: all three.",
    )
    parser.add_argument("--seeds", type=int, default=4, help="Random seeds per scenario (starting at --seed-base)")
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data" / "sft",
        help="Directory for scenario_XX.jsonl files",
    )
    args = parser.parse_args()

    # Default: only 02/03 so we do not overwrite the curated data/sft/scenario_01.jsonl.
    scenarios = args.scenarios or [
        "scenario_02_merge",
        "scenario_03_ambulance",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sc in scenarios:
        path = args.output_dir / (
            "scenario_01.jsonl"
            if sc == "scenario_01_brake"
            else "scenario_02.jsonl"
            if sc == "scenario_02_merge"
            else "scenario_03.jsonl"
        )
        total = 0
        with path.open("w", encoding="utf-8") as handle:
            for i in range(args.seeds):
                seed = args.seed_base + i * 17 + hash(sc) % 997
                env = PlatoonEnv(scenario_name=sc)
                n = export_one_episode(env, seed, handle, run_id=_scenario_tag(sc))
                total += n
        print(f"Wrote {total} rows to {path}")


if __name__ == "__main__":
    main()
