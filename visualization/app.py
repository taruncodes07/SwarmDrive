from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

try:
    import spaces
except Exception:  # pragma: no cover
    class _SpacesFallback:
        @staticmethod
        def GPU(func):
            return func

    spaces = _SpacesFallback()  # type: ignore[assignment]

from agents.llm_agent import LLMAgent
from config.settings import ROOT_DIR
from environment.platoon_env import PlatoonEnv
from visualization.renderer import build_road_svg

AVAILABLE_SCENARIOS = ["scenario_01_brake", "scenario_02_merge", "scenario_03_low_friction"]


@dataclass
class AppRuntime:
    env_trained: PlatoonEnv
    env_untrained: PlatoonEnv
    obs_trained: dict[str, str]
    obs_untrained: dict[str, str]
    trained_agent: LLMAgent
    untrained_agent: LLMAgent
    mode: str = "Trained (RL)"
    scenario_name: str = "scenario_01_brake"
    done_trained: bool = False
    done_untrained: bool = False
    is_playing: bool = False
    history_trained: list[dict[str, Any]] | None = None
    history_untrained: list[dict[str, Any]] | None = None
    history_side: list[tuple[dict[str, Any], dict[str, Any]]] | None = None


def _gpu_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"GPU active: {name} ({vram:.1f} GB VRAM)"
    return "GPU unavailable. Running on CPU (demo may be slower)."


def _checkpoint_step(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint-"):
        tail = name.split("checkpoint-", maxsplit=1)[1]
        if tail.isdigit():
            return int(tail)
    return -1


def _resolve_adapter_dir(root: Path) -> Path | None:
    # Accept direct adapter folder first.
    if (root / "adapter_config.json").exists():
        return root

    candidates: list[Path] = []
    for cfg in root.rglob("adapter_config.json"):
        parent = cfg.parent
        if (parent / "adapter_model.safetensors").exists() or (parent / "adapter_model.bin").exists():
            candidates.append(parent)

    if not candidates:
        return None

    # Prefer the newest numbered checkpoint when checkpoints are present.
    candidates.sort(key=lambda p: (_checkpoint_step(p), len(p.parts)), reverse=True)
    return candidates[0]


def _try_download_adapter(repo_id: str) -> Path | None:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    try:
        local_dir = snapshot_download(repo_id=repo_id, token=hf_token)
        return _resolve_adapter_dir(Path(local_dir))
    except Exception:
        return None


def _load_runtime() -> tuple[AppRuntime, str]:
    load_dotenv(ROOT_DIR / ".env")
    hf_username = os.getenv("HF_USERNAME", "").strip()

    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    banner_parts = [_gpu_info()]

    rl_adapter = None
    sft_adapter = None
    if hf_username and hf_username != "your_hf_username":
        rl_adapter = _try_download_adapter(f"{hf_username}/platoon-qwen-rl")
        if rl_adapter is not None:
            banner_parts.append("Loaded RL adapter from Hugging Face Hub.")
        else:
            banner_parts.append("RL adapter not found; trying SFT adapter.")

        if rl_adapter is None:
            sft_adapter = _try_download_adapter(f"{hf_username}/platoon-qwen-sft")
            if sft_adapter is not None:
                banner_parts.append("Loaded SFT adapter fallback.")
            else:
                banner_parts.append("SFT adapter not found; falling back to base model.")
    else:
        banner_parts.append("HF_USERNAME missing in .env; using base model only.")

    trained_adapter = str(rl_adapter or sft_adapter) if (rl_adapter or sft_adapter) else None

    trained_agent = LLMAgent(base_model_name=base_model, adapter_path=trained_adapter)
    untrained_agent = LLMAgent(base_model_name=base_model, adapter_path=None)

    default_scenario = "scenario_01_brake"
    env_trained = PlatoonEnv(scenario_name=default_scenario)
    env_untrained = PlatoonEnv(scenario_name=default_scenario)

    runtime = AppRuntime(
        env_trained=env_trained,
        env_untrained=env_untrained,
        obs_trained=env_trained.reset(seed=123),
        obs_untrained=env_untrained.reset(seed=123),
        trained_agent=trained_agent,
        untrained_agent=untrained_agent,
        scenario_name=default_scenario,
        history_trained=[],
        history_untrained=[],
        history_side=[],
    )

    return runtime, "\n".join(banner_parts)


RUNTIME, STARTUP_BANNER = _load_runtime()


def _state_json(state: dict[str, Any], agent_id: int) -> dict[str, Any]:
    car = state["vehicles"][agent_id]
    return {
        "phase": state["phase"],
        "timestep": state["timestep"],
        "x": car["x"],
        "velocity": car["velocity"],
        "accel_pedal": car["accel_pedal"],
        "brake_pedal": car["brake_pedal"],
        "net_acceleration": car["net_acceleration"],
    }


def _broadcast_table(state: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for packet in state.get("broadcast_buffer", [])[-5:]:
        rows.append(
            [
                int(packet["sender_id"]),
                float(packet["x_position"]),
                float(packet["velocity"]),
                float(packet["accel_pedal"]),
                float(packet["brake_pedal"]),
                float(packet["net_acceleration"]),
            ]
        )
    return rows


@spaces.GPU
def _step_once(mode: str, delay: float) -> tuple[str, str, list[list[Any]], dict[str, Any], dict[str, Any], str, str]:
    RUNTIME.mode = mode

    if mode == "Trained (RL)":
        stepped = False
        if not RUNTIME.done_trained:
            out1 = RUNTIME.trained_agent.act(RUNTIME.obs_trained["agent_1"], temperature=0.0)
            out2 = RUNTIME.trained_agent.act(RUNTIME.obs_trained["agent_2"], temperature=0.0)
            RUNTIME.obs_trained, rewards, dones, infos = RUNTIME.env_trained.step(
                {"agent_1": out1.action_text, "agent_2": out2.action_text}
            )
            RUNTIME.done_trained = dones["agent_1"]
            stepped = True
            summary = (
                f"Total logged steps: {len(RUNTIME.history_trained) + 1}\n"
                f"Last reward A1={rewards['agent_1']:.3f}, A2={rewards['agent_2']:.3f}\n"
                f"Final gap errors A1={infos['agent_1']['gap_error']:.2f}, A2={infos['agent_2']['gap_error']:.2f}"
            )
        else:
            summary = "Episode finished. Click Reset."

        state = RUNTIME.env_trained.state()
        if stepped:
            RUNTIME.history_trained.append(state)
        if delay > 0.0:
            time.sleep(delay)
        return (
            build_road_svg(state, title="Trained Agent"),
            "",
            _broadcast_table(state),
            _state_json(state, 1),
            _state_json(state, 2),
            state["phase"],
            summary,
        )

    if mode == "Untrained (base)":
        stepped = False
        if not RUNTIME.done_untrained:
            out1 = RUNTIME.untrained_agent.act(RUNTIME.obs_untrained["agent_1"], temperature=0.0)
            out2 = RUNTIME.untrained_agent.act(RUNTIME.obs_untrained["agent_2"], temperature=0.0)
            RUNTIME.obs_untrained, rewards, dones, infos = RUNTIME.env_untrained.step(
                {"agent_1": out1.action_text, "agent_2": out2.action_text}
            )
            RUNTIME.done_untrained = dones["agent_1"]
            stepped = True
            summary = (
                f"Total logged steps: {len(RUNTIME.history_untrained) + 1}\n"
                f"Last reward A1={rewards['agent_1']:.3f}, A2={rewards['agent_2']:.3f}\n"
                f"Gap errors A1={infos['agent_1']['gap_error']:.2f}, A2={infos['agent_2']['gap_error']:.2f}"
            )
        else:
            summary = "Episode finished. Click Reset."

        state = RUNTIME.env_untrained.state()
        if stepped:
            RUNTIME.history_untrained.append(state)
        if delay > 0.0:
            time.sleep(delay)
        return (
            build_road_svg(state, title="Untrained Agent"),
            "",
            _broadcast_table(state),
            _state_json(state, 1),
            _state_json(state, 2),
            state["phase"],
            summary,
        )

    # Side-by-side
    stepped = False
    if not RUNTIME.done_trained:
        out1_t = RUNTIME.trained_agent.act(RUNTIME.obs_trained["agent_1"], temperature=0.0)
        out2_t = RUNTIME.trained_agent.act(RUNTIME.obs_trained["agent_2"], temperature=0.0)
        RUNTIME.obs_trained, rewards_t, dones_t, infos_t = RUNTIME.env_trained.step(
            {"agent_1": out1_t.action_text, "agent_2": out2_t.action_text}
        )
        RUNTIME.done_trained = dones_t["agent_1"]
        stepped = True
    else:
        rewards_t = {"agent_1": 0.0, "agent_2": 0.0}
        infos_t = {"agent_1": {"gap_error": 0.0}, "agent_2": {"gap_error": 0.0}}

    if not RUNTIME.done_untrained:
        out1_u = RUNTIME.untrained_agent.act(RUNTIME.obs_untrained["agent_1"], temperature=0.0)
        out2_u = RUNTIME.untrained_agent.act(RUNTIME.obs_untrained["agent_2"], temperature=0.0)
        RUNTIME.obs_untrained, rewards_u, dones_u, infos_u = RUNTIME.env_untrained.step(
            {"agent_1": out1_u.action_text, "agent_2": out2_u.action_text}
        )
        RUNTIME.done_untrained = dones_u["agent_1"]
        stepped = True
    else:
        rewards_u = {"agent_1": 0.0, "agent_2": 0.0}
        infos_u = {"agent_1": {"gap_error": 0.0}, "agent_2": {"gap_error": 0.0}}

    state_t = RUNTIME.env_trained.state()
    state_u = RUNTIME.env_untrained.state()
    if stepped:
        RUNTIME.history_side.append((state_t, state_u))
    if delay > 0.0:
        time.sleep(delay)

    summary = (
        "Side-by-side\n"
        f"Trained rewards: A1={rewards_t['agent_1']:.3f}, A2={rewards_t['agent_2']:.3f}\n"
        f"Untrained rewards: A1={rewards_u['agent_1']:.3f}, A2={rewards_u['agent_2']:.3f}\n"
        f"Trained final gap err A1={infos_t['agent_1']['gap_error']:.2f}, A2={infos_t['agent_2']['gap_error']:.2f}\n"
        f"Untrained final gap err A1={infos_u['agent_1']['gap_error']:.2f}, A2={infos_u['agent_2']['gap_error']:.2f}"
    )

    return (
        build_road_svg(state_t, title="Trained"),
        build_road_svg(state_u, title="Untrained"),
        _broadcast_table(state_t),
        _state_json(state_t, 1),
        _state_json(state_t, 2),
        state_t["phase"],
        summary,
    )


def _episode_done_for_mode(mode: str) -> bool:
    if mode == "Trained (RL)":
        return RUNTIME.done_trained
    if mode == "Untrained (base)":
        return RUNTIME.done_untrained
    return RUNTIME.done_trained and RUNTIME.done_untrained


def _history_for_playback(mode: str) -> list[Any]:
    if mode == "Trained (RL)":
        return RUNTIME.history_trained
    if mode == "Untrained (base)":
        return RUNTIME.history_untrained
    return RUNTIME.history_side


def _play_loop(
    mode: str,
    delay: float,
    steps_per_frame: int,
    seed: float,
    max_steps: int = 2000,
):
    RUNTIME.is_playing = True
    hist: list[Any] = _history_for_playback(mode)
    sim_seed = int(seed) if seed is not None else 123

    if not hist:
        yield _reset(sim_seed, interrupt_playback=False)
        sim_guard = 0
        while not _episode_done_for_mode(mode) and sim_guard < max_steps:
            if not RUNTIME.is_playing:
                break
            for _ in range(max(1, int(steps_per_frame))):
                if _episode_done_for_mode(mode):
                    break
                _step_once(mode, 0.0)
                sim_guard += 1
            st = RUNTIME.env_trained.state()
            su = RUNTIME.env_untrained.state()
            if mode == "Side-by-Side":
                yield (
                    build_road_svg(st, title=f"Recording… (t={st['timestep']})"),
                    build_road_svg(su, title=f"Recording… (t={su['timestep']})"),
                    _broadcast_table(st),
                    _state_json(st, 1),
                    _state_json(st, 2),
                    st["phase"],
                    f"Recording episode… step {st['timestep']}. Please wait.",
                )
            elif mode == "Untrained (base)":
                yield (
                    build_road_svg(su, title=f"Recording… (t={su['timestep']})"),
                    "",
                    _broadcast_table(su),
                    _state_json(su, 1),
                    _state_json(su, 2),
                    su["phase"],
                    f"Recording episode… step {su['timestep']}. Please wait.",
                )
            else:
                yield (
                    build_road_svg(st, title=f"Recording… (t={st['timestep']})"),
                    "",
                    _broadcast_table(st),
                    _state_json(st, 1),
                    _state_json(st, 2),
                    st["phase"],
                    f"Recording episode… step {st['timestep']}. Please wait.",
                )
        hist = _history_for_playback(mode)

    if mode == "Side-by-Side":
        for st, su in hist:
            if not RUNTIME.is_playing:
                break
            yield (
                build_road_svg(st, title="Trained (Playback)"),
                build_road_svg(su, title="Untrained (Playback)"),
                _broadcast_table(st),
                _state_json(st, 1),
                _state_json(st, 2),
                st["phase"],
                f"Playback: step {st['timestep']} / {len(hist)} (pair)",
            )
            if delay > 0.0:
                time.sleep(delay)
    else:
        for frame_state in hist:
            if not RUNTIME.is_playing:
                break
            title = "Trained Agent (Playback)" if mode == "Trained (RL)" else "Untrained Agent (Playback)"
            yield (
                build_road_svg(frame_state, title=title),
                "",
                _broadcast_table(frame_state),
                _state_json(frame_state, 1),
                _state_json(frame_state, 2),
                frame_state["phase"],
                f"Playback: step {frame_state['timestep']} / {len(hist)}",
            )
            if delay > 0.0:
                time.sleep(delay)

    RUNTIME.is_playing = False


def _reset(
    seed: int,
    *,
    interrupt_playback: bool = True,
) -> tuple[str, str, list[list[Any]], dict[str, Any], dict[str, Any], str, str]:
    if interrupt_playback:
        RUNTIME.is_playing = False
    RUNTIME.done_trained = False
    RUNTIME.done_untrained = False
    RUNTIME.history_trained = []
    RUNTIME.history_untrained = []
    RUNTIME.history_side = []

    RUNTIME.obs_trained = RUNTIME.env_trained.reset(seed=seed)
    RUNTIME.obs_untrained = RUNTIME.env_untrained.reset(seed=seed)

    state_t = RUNTIME.env_trained.state()
    return (
        build_road_svg(state_t, title="Trained Agent"),
        "",
        _broadcast_table(state_t),
        _state_json(state_t, 1),
        _state_json(state_t, 2),
        state_t["phase"],
        "Episode reset.",
    )


def _pause() -> str:
    RUNTIME.is_playing = False
    return "Paused. Click Play to continue stepping."


def _set_scenario(
    scenario_name: str,
    seed: int,
) -> tuple[str, str, list[list[Any]], dict[str, Any], dict[str, Any], str, str]:
    RUNTIME.is_playing = False
    RUNTIME.done_trained = False
    RUNTIME.done_untrained = False
    RUNTIME.history_trained = []
    RUNTIME.history_untrained = []
    RUNTIME.history_side = []
    RUNTIME.scenario_name = scenario_name

    RUNTIME.env_trained = PlatoonEnv(scenario_name=scenario_name)
    RUNTIME.env_untrained = PlatoonEnv(scenario_name=scenario_name)
    RUNTIME.obs_trained = RUNTIME.env_trained.reset(seed=int(seed))
    RUNTIME.obs_untrained = RUNTIME.env_untrained.reset(seed=int(seed))

    state_t = RUNTIME.env_trained.state()
    return (
        build_road_svg(state_t, title="Trained Agent"),
        "",
        _broadcast_table(state_t),
        _state_json(state_t, 1),
        _state_json(state_t, 2),
        state_t["phase"],
        f"Scenario switched to {scenario_name}. Episode reset.",
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Platoon RL Env") as demo:
        gr.Markdown("# Platoon RL Environment Demo")
        gr.Markdown(STARTUP_BANNER)

        with gr.Row():
            mode = gr.Radio(
                choices=["Trained (RL)", "Untrained (base)", "Side-by-Side"],
                value="Trained (RL)",
                label="Mode Selector",
            )
            scenario = gr.Dropdown(
                choices=AVAILABLE_SCENARIOS,
                value="scenario_01_brake",
                label="Scenario",
            )
            speed = gr.Slider(0.0, 0.5, value=0.033, step=0.001, label="Playback delay (s) [30 FPS approx 0.033]")
            steps_per_frame = gr.Slider(1, 4, value=2, step=1, label="Simulation steps per frame")
            seed = gr.Number(value=123, precision=0, label="Episode seed")

        with gr.Row():
            road_left = gr.HTML(label="Road Canvas")
            road_right = gr.HTML(label="Side-by-Side Canvas")

        with gr.Row():
            phase = gr.Label(label="Phase Banner")
            stats = gr.Markdown(label="Stats Panel")

        with gr.Row():
            broadcast = gr.Dataframe(
                headers=["sender", "x", "velocity", "accel_pedal", "brake_pedal", "net_accel"],
                datatype=["number", "number", "number", "number", "number", "number"],
                row_count=(5, "dynamic"),
                col_count=(6, "fixed"),
                label="Broadcast Feed",
            )

        with gr.Row():
            agent1 = gr.JSON(label="Agent 1 State")
            agent2 = gr.JSON(label="Agent 2 State")

        with gr.Row():
            play = gr.Button("Play")
            pause = gr.Button("Pause")
            reset = gr.Button("Reset")

        play.click(
            fn=_play_loop,
            inputs=[mode, speed, steps_per_frame, seed],
            outputs=[road_left, road_right, broadcast, agent1, agent2, phase, stats],
        )

        scenario.change(
            fn=_set_scenario,
            inputs=[scenario, seed],
            outputs=[road_left, road_right, broadcast, agent1, agent2, phase, stats],
        )

        pause.click(fn=_pause, outputs=[stats])

        reset.click(
            fn=_reset,
            inputs=[seed],
            outputs=[road_left, road_right, broadcast, agent1, agent2, phase, stats],
        )

    return demo


# Hugging Face Spaces imports this module and expects a Gradio app in `demo`.
# Queue is required for generator/streaming event handlers (e.g. Play).
demo = build_app()
demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
