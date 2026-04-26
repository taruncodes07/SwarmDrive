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

AVAILABLE_SCENARIOS = ["scenario_01_brake", "scenario_02_merge", "scenario_03_ambulance"]


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
    history_trained: list[Any] | None = None
    history_untrained: list[Any] | None = None
    history_side: list[Any] | None = None
    collision_ever_trained: bool = False
    collision_ever_untrained: bool = False
    last_rewards_trained: dict[str, float] | None = None
    last_infos_trained: dict[str, Any] | None = None
    last_rewards_untrained: dict[str, float] | None = None
    last_infos_untrained: dict[str, Any] | None = None


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


def _history_frame_state(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict) and "state" in entry:
        return entry["state"]
    return entry  # raw state dict (legacy)


def _history_frame_rewards_infos(entry: Any) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    if isinstance(entry, dict) and "rewards" in entry:
        r = entry.get("rewards")
        i = entry.get("infos")
        return (
            r if isinstance(r, dict) else None,
            i if isinstance(i, dict) else None,
        )
    return None, None


def _reward_terms_lines(infos: dict[str, Any] | None, prefix: str) -> list[str]:
    if not infos:
        return []
    lines: list[str] = []
    for aid in ("agent_1", "agent_2"):
        block = infos.get(aid)
        if not isinstance(block, dict):
            continue
        terms = block.get("reward_terms")
        if not isinstance(terms, dict):
            continue
        nonzero = {k: float(v) for k, v in terms.items() if abs(float(v)) > 1e-8}
        if not nonzero:
            continue
        ordered = sorted(nonzero.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
        parts = ", ".join(f"`{k}` {v:+.4f}" for k, v in ordered[:14])
        more = f" (+{len(ordered) - 14} more)" if len(ordered) > 14 else ""
        lines.append(f"- **{prefix}{aid}** reward terms: {parts}{more}")
    return lines


def _format_step_stats(
    state: dict[str, Any],
    rewards: dict[str, float] | None,
    infos: dict[str, Any] | None,
    *,
    collision_ever: bool,
    episode_done: bool,
    subtitle: str = "",
) -> str:
    t = int(state.get("timestep", 0))
    phase = state.get("phase", "")
    coll_now = bool(state.get("collision"))
    lines: list[str] = [
        f"### Step {t} · phase `{phase}`" + (f" · {subtitle}" if subtitle else ""),
        "",
    ]
    if rewards:
        r1 = float(rewards.get("agent_1", 0.0))
        r2 = float(rewards.get("agent_2", 0.0))
        lines.append(f"**This step — total reward** · Agent 1: `{r1:+.4f}` · Agent 2: `{r2:+.4f}`")
        lines.append("")
        lines.extend(_reward_terms_lines(infos, ""))
        if lines[-1] != "":
            lines.append("")
    else:
        lines.append("*No environment step recorded yet for this episode (after reset, click Play or step once).*")
        lines.append("")

    if coll_now:
        lines.append("- **Collision in current state:** yes")
    elif collision_ever:
        lines.append("- **Collision this episode:** recorded earlier")
    else:
        lines.append("- **Collision this episode:** none so far")
    lines.append("")

    if episode_done:
        if collision_ever or coll_now:
            lines.append("### ⚠️ Episode ended — **collision occurred**")
        else:
            scen = state.get("scenario")
            ac = state.get("ambulance_clearance")
            if scen == "scenario_03_ambulance" and isinstance(ac, dict):
                a1 = bool(ac.get("agent_1"))
                a2 = bool(ac.get("agent_2"))
                if a1 and a2:
                    lines.append(
                        "### ✅ Episode complete — **no collision**; **ambulance passed both agents** (success)"
                    )
                else:
                    lines.append(
                        "### ⚠️ Episode complete — **no collision**, but **ambulance did not pass both agents** "
                        f"(agent 1: {'yes' if a1 else 'no'}, agent 2: {'yes' if a2 else 'no'})"
                    )
            else:
                lines.append("### ✅ Episode complete — **no collision** (success)")
    return "\n".join(lines)


def _format_side_step_stats(
    state_t: dict[str, Any],
    rewards_t: dict[str, float] | None,
    infos_t: dict[str, Any] | None,
    state_u: dict[str, Any],
    rewards_u: dict[str, float] | None,
    infos_u: dict[str, Any] | None,
    *,
    collision_ever_t: bool,
    collision_ever_u: bool,
    episode_done_t: bool,
    episode_done_u: bool,
) -> str:
    st = _format_step_stats(
        state_t,
        rewards_t,
        infos_t,
        collision_ever=collision_ever_t,
        episode_done=False,
        subtitle="trained (left)",
    )
    su = _format_step_stats(
        state_u,
        rewards_u,
        infos_u,
        collision_ever=collision_ever_u,
        episode_done=False,
        subtitle="untrained (right)",
    )
    parts = ["## Side-by-side", "", st, "", "---", "", su, ""]

    def _amb_ok(stx: dict[str, Any]) -> bool:
        if stx.get("scenario") != "scenario_03_ambulance":
            return True
        ac = stx.get("ambulance_clearance")
        if not isinstance(ac, dict):
            return False
        return bool(ac.get("agent_1")) and bool(ac.get("agent_2"))

    if episode_done_t and episode_done_u:
        ok = not collision_ever_t and not collision_ever_u
        if ok and _amb_ok(state_t) and _amb_ok(state_u):
            parts.append("### ✅ Both episodes finished — **no collisions**; **ambulance passed both agents on each side** (success)")
        elif ok:
            parts.append(
                "### ⚠️ Both episodes finished — **no collisions**, but **ambulance scenario incomplete** on one or both sides "
                "(requires ambulance to pass agent 1 **and** agent 2)"
            )
        else:
            parts.append("### ⚠️ One or both episodes had a **collision**")
    elif episode_done_t and not collision_ever_t:
        if _amb_ok(state_t):
            parts.append("### ✅ Trained episode — **no collision**; **ambulance passed both agents** (success)")
        else:
            parts.append("### ⚠️ Trained episode — **no collision**, but **ambulance did not pass both agents**")
    elif episode_done_u and not collision_ever_u:
        if _amb_ok(state_u):
            parts.append("### ✅ Untrained episode — **no collision**; **ambulance passed both agents** (success)")
        else:
            parts.append("### ⚠️ Untrained episode — **no collision**, but **ambulance did not pass both agents**")
    return "\n".join(parts)


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
        rewards: dict[str, float] | None = None
        infos: dict[str, Any] | None = None
        if not RUNTIME.done_trained:
            out1 = RUNTIME.trained_agent.act(RUNTIME.obs_trained["agent_1"], temperature=0.0)
            out2 = RUNTIME.trained_agent.act(RUNTIME.obs_trained["agent_2"], temperature=0.0)
            RUNTIME.obs_trained, rewards, dones, infos = RUNTIME.env_trained.step(
                {"agent_1": out1.action_text, "agent_2": out2.action_text}
            )
            RUNTIME.done_trained = dones["agent_1"]
            stepped = True
            RUNTIME.last_rewards_trained = rewards
            RUNTIME.last_infos_trained = infos

        state = RUNTIME.env_trained.state()
        if stepped:
            if state.get("collision"):
                RUNTIME.collision_ever_trained = True
            RUNTIME.history_trained.append({"state": state, "rewards": rewards, "infos": infos})
        rw = rewards if stepped else RUNTIME.last_rewards_trained
        inf = infos if stepped else RUNTIME.last_infos_trained
        summary = _format_step_stats(
            state,
            rw,
            inf,
            collision_ever=RUNTIME.collision_ever_trained,
            episode_done=RUNTIME.done_trained,
        )
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
        rewards: dict[str, float] | None = None
        infos: dict[str, Any] | None = None
        if not RUNTIME.done_untrained:
            out1 = RUNTIME.untrained_agent.act(RUNTIME.obs_untrained["agent_1"], temperature=0.0)
            out2 = RUNTIME.untrained_agent.act(RUNTIME.obs_untrained["agent_2"], temperature=0.0)
            RUNTIME.obs_untrained, rewards, dones, infos = RUNTIME.env_untrained.step(
                {"agent_1": out1.action_text, "agent_2": out2.action_text}
            )
            RUNTIME.done_untrained = dones["agent_1"]
            stepped = True
            RUNTIME.last_rewards_untrained = rewards
            RUNTIME.last_infos_untrained = infos

        state = RUNTIME.env_untrained.state()
        if stepped:
            if state.get("collision"):
                RUNTIME.collision_ever_untrained = True
            RUNTIME.history_untrained.append({"state": state, "rewards": rewards, "infos": infos})
        rw = rewards if stepped else RUNTIME.last_rewards_untrained
        inf = infos if stepped else RUNTIME.last_infos_untrained
        summary = _format_step_stats(
            state,
            rw,
            inf,
            collision_ever=RUNTIME.collision_ever_untrained,
            episode_done=RUNTIME.done_untrained,
        )
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
    rewards_t: dict[str, float] | None = None
    infos_t: dict[str, Any] | None = None
    rewards_u: dict[str, float] | None = None
    infos_u: dict[str, Any] | None = None
    if not RUNTIME.done_trained:
        out1_t = RUNTIME.trained_agent.act(RUNTIME.obs_trained["agent_1"], temperature=0.0)
        out2_t = RUNTIME.trained_agent.act(RUNTIME.obs_trained["agent_2"], temperature=0.0)
        RUNTIME.obs_trained, rewards_t, dones_t, infos_t = RUNTIME.env_trained.step(
            {"agent_1": out1_t.action_text, "agent_2": out2_t.action_text}
        )
        RUNTIME.done_trained = dones_t["agent_1"]
        stepped = True
        RUNTIME.last_rewards_trained = rewards_t
        RUNTIME.last_infos_trained = infos_t
    else:
        rewards_t = RUNTIME.last_rewards_trained
        infos_t = RUNTIME.last_infos_trained

    if not RUNTIME.done_untrained:
        out1_u = RUNTIME.untrained_agent.act(RUNTIME.obs_untrained["agent_1"], temperature=0.0)
        out2_u = RUNTIME.untrained_agent.act(RUNTIME.obs_untrained["agent_2"], temperature=0.0)
        RUNTIME.obs_untrained, rewards_u, dones_u, infos_u = RUNTIME.env_untrained.step(
            {"agent_1": out1_u.action_text, "agent_2": out2_u.action_text}
        )
        RUNTIME.done_untrained = dones_u["agent_1"]
        stepped = True
        RUNTIME.last_rewards_untrained = rewards_u
        RUNTIME.last_infos_untrained = infos_u
    else:
        rewards_u = RUNTIME.last_rewards_untrained
        infos_u = RUNTIME.last_infos_untrained

    state_t = RUNTIME.env_trained.state()
    state_u = RUNTIME.env_untrained.state()
    if stepped:
        if state_t.get("collision"):
            RUNTIME.collision_ever_trained = True
        if state_u.get("collision"):
            RUNTIME.collision_ever_untrained = True
        RUNTIME.history_side.append(
            (
                {"state": state_t, "rewards": rewards_t, "infos": infos_t},
                {"state": state_u, "rewards": rewards_u, "infos": infos_u},
            )
        )
    if delay > 0.0:
        time.sleep(delay)

    summary = _format_side_step_stats(
        state_t,
        rewards_t,
        infos_t,
        state_u,
        rewards_u,
        infos_u,
        collision_ever_t=RUNTIME.collision_ever_trained,
        collision_ever_u=RUNTIME.collision_ever_untrained,
        episode_done_t=RUNTIME.done_trained,
        episode_done_u=RUNTIME.done_untrained,
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


def _stats_after_step(mode: str) -> str:
    if mode == "Trained (RL)":
        if not RUNTIME.history_trained:
            return "### Recording…\n\n*Waiting for first step.*"
        e = RUNTIME.history_trained[-1]
        st = _history_frame_state(e)
        rw, inf = _history_frame_rewards_infos(e)
        return _format_step_stats(
            st, rw, inf, collision_ever=RUNTIME.collision_ever_trained, episode_done=RUNTIME.done_trained
        )
    if mode == "Untrained (base)":
        if not RUNTIME.history_untrained:
            return "### Recording…\n\n*Waiting for first step.*"
        e = RUNTIME.history_untrained[-1]
        st = _history_frame_state(e)
        rw, inf = _history_frame_rewards_infos(e)
        return _format_step_stats(
            st, rw, inf, collision_ever=RUNTIME.collision_ever_untrained, episode_done=RUNTIME.done_untrained
        )
    if not RUNTIME.history_side:
        return "### Recording…\n\n*Waiting for first step.*"
    rt, ru = RUNTIME.history_side[-1]
    return _format_side_step_stats(
        rt["state"],
        rt.get("rewards"),
        rt.get("infos"),
        ru["state"],
        ru.get("rewards"),
        ru.get("infos"),
        collision_ever_t=RUNTIME.collision_ever_trained,
        collision_ever_u=RUNTIME.collision_ever_untrained,
        episode_done_t=RUNTIME.done_trained,
        episode_done_u=RUNTIME.done_untrained,
    )


def _playback_frame_stats(mode: str, item: Any, index: int, total: int) -> str:
    is_last = index == total - 1 and total > 0
    if mode == "Side-by-Side":
        rt, ru = item
        return _format_side_step_stats(
            rt["state"],
            rt.get("rewards"),
            rt.get("infos"),
            ru["state"],
            ru.get("rewards"),
            ru.get("infos"),
            collision_ever_t=RUNTIME.collision_ever_trained,
            collision_ever_u=RUNTIME.collision_ever_untrained,
            episode_done_t=is_last and RUNTIME.done_trained,
            episode_done_u=is_last and RUNTIME.done_untrained,
        )
    fr = _history_frame_state(item)
    rw, inf = _history_frame_rewards_infos(item)
    if mode == "Trained (RL)":
        return _format_step_stats(
            fr,
            rw,
            inf,
            collision_ever=RUNTIME.collision_ever_trained,
            episode_done=is_last and RUNTIME.done_trained,
        )
    return _format_step_stats(
        fr,
        rw,
        inf,
        collision_ever=RUNTIME.collision_ever_untrained,
        episode_done=is_last and RUNTIME.done_untrained,
    )


def _scrub_slider_update(
    mode: str,
    *,
    index: int | None = None,
    interactive: bool,
) -> dict:
    hist = _history_for_playback(mode)
    n = len(hist)
    mx = max(0, n - 1)
    if n == 0:
        return gr.update(minimum=0, maximum=0, value=0, interactive=interactive)
    v = int(index) if index is not None else mx
    v = max(0, min(v, mx))
    return gr.update(minimum=0, maximum=mx, value=v, interactive=interactive)


def _live_env_outputs(mode: str) -> tuple[str, str, list[list[Any]], dict[str, Any], dict[str, Any], str, str]:
    if mode == "Side-by-Side":
        st = RUNTIME.env_trained.state()
        su = RUNTIME.env_untrained.state()
        summary = _format_side_step_stats(
            st,
            RUNTIME.last_rewards_trained,
            RUNTIME.last_infos_trained,
            su,
            RUNTIME.last_rewards_untrained,
            RUNTIME.last_infos_untrained,
            collision_ever_t=RUNTIME.collision_ever_trained,
            collision_ever_u=RUNTIME.collision_ever_untrained,
            episode_done_t=RUNTIME.done_trained,
            episode_done_u=RUNTIME.done_untrained,
        )
        return (
            build_road_svg(st, title="Trained"),
            build_road_svg(su, title="Untrained"),
            _broadcast_table(st),
            _state_json(st, 1),
            _state_json(st, 2),
            st["phase"],
            summary,
        )
    if mode == "Untrained (base)":
        su = RUNTIME.env_untrained.state()
        summary = _format_step_stats(
            su,
            RUNTIME.last_rewards_untrained,
            RUNTIME.last_infos_untrained,
            collision_ever=RUNTIME.collision_ever_untrained,
            episode_done=RUNTIME.done_untrained,
        )
        return (
            build_road_svg(su, title="Untrained Agent"),
            "",
            _broadcast_table(su),
            _state_json(su, 1),
            _state_json(su, 2),
            su["phase"],
            summary,
        )
    st = RUNTIME.env_trained.state()
    summary = _format_step_stats(
        st,
        RUNTIME.last_rewards_trained,
        RUNTIME.last_infos_trained,
        collision_ever=RUNTIME.collision_ever_trained,
        episode_done=RUNTIME.done_trained,
    )
    return (
        build_road_svg(st, title="Trained Agent"),
        "",
        _broadcast_table(st),
        _state_json(st, 1),
        _state_json(st, 2),
        st["phase"],
        summary,
    )


def _render_history_frame(mode: str, frame_index: float | int) -> tuple[str, str, list[list[Any]], dict[str, Any], dict[str, Any], str, str]:
    hist = _history_for_playback(mode)
    if not hist:
        return _live_env_outputs(mode)
    idx = max(0, min(int(frame_index), len(hist) - 1))
    item = hist[idx]
    if mode == "Side-by-Side":
        rt, ru = item
        st = rt["state"]
        su = ru["state"]
        stats_pb = _playback_frame_stats(mode, item, idx, len(hist))
        return (
            build_road_svg(st, title="Trained (scrub)"),
            build_road_svg(su, title="Untrained (scrub)"),
            _broadcast_table(st),
            _state_json(st, 1),
            _state_json(st, 2),
            st["phase"],
            stats_pb,
        )
    frame_state = _history_frame_state(item)
    title = "Trained Agent (scrub)" if mode == "Trained (RL)" else "Untrained Agent (scrub)"
    stats_pb = _playback_frame_stats(mode, item, idx, len(hist))
    return (
        build_road_svg(frame_state, title=title),
        "",
        _broadcast_table(frame_state),
        _state_json(frame_state, 1),
        _state_json(frame_state, 2),
        frame_state["phase"],
        stats_pb,
    )


def _on_playback_scrub(mode: str, frame_index: float) -> tuple[str, str, list[list[Any]], dict[str, Any], dict[str, Any], str, str]:
    return _render_history_frame(mode, frame_index)


def _on_mode_change(mode: str) -> tuple[str, str, list[list[Any]], dict[str, Any], dict[str, Any], str, str, dict]:
    RUNTIME.mode = mode
    hist = _history_for_playback(mode)
    idx = max(0, len(hist) - 1) if hist else 0
    vis = _render_history_frame(mode, idx)
    return (*vis, _scrub_slider_update(mode, index=idx, interactive=True))


def _playback_pass_once(mode: str, delay: float, hist: list[Any]):
    if not hist:
        return
    stopped_mid = False
    if mode == "Side-by-Side":
        for i, pair in enumerate(hist):
            if not RUNTIME.is_playing:
                idx = (i - 1) if i > 0 else 0
                yield (
                    *_render_history_frame(mode, idx),
                    _scrub_slider_update(mode, index=idx, interactive=True),
                )
                stopped_mid = True
                break
            rt, ru = pair
            st = rt["state"]
            su = ru["state"]
            stats_pb = _playback_frame_stats(mode, pair, i, len(hist))
            yield (
                build_road_svg(st, title="Trained (Playback)"),
                build_road_svg(su, title="Untrained (Playback)"),
                _broadcast_table(st),
                _state_json(st, 1),
                _state_json(st, 2),
                st["phase"],
                stats_pb,
                _scrub_slider_update(mode, index=i, interactive=False),
            )
            if delay > 0.0:
                time.sleep(delay)
    else:
        for i, item in enumerate(hist):
            if not RUNTIME.is_playing:
                idx = (i - 1) if i > 0 else 0
                yield (
                    *_render_history_frame(mode, idx),
                    _scrub_slider_update(mode, index=idx, interactive=True),
                )
                stopped_mid = True
                break
            frame_state = _history_frame_state(item)
            title = "Trained Agent (Playback)" if mode == "Trained (RL)" else "Untrained Agent (Playback)"
            stats_pb = _playback_frame_stats(mode, item, i, len(hist))
            yield (
                build_road_svg(frame_state, title=title),
                "",
                _broadcast_table(frame_state),
                _state_json(frame_state, 1),
                _state_json(frame_state, 2),
                frame_state["phase"],
                stats_pb,
                _scrub_slider_update(mode, index=i, interactive=False),
            )
            if delay > 0.0:
                time.sleep(delay)
    if stopped_mid:
        return
    if not RUNTIME.is_playing:
        li = len(hist) - 1
        yield (
            *_render_history_frame(mode, li),
            _scrub_slider_update(mode, index=li, interactive=True),
        )
        return
    li = len(hist) - 1
    yield (
        *_render_history_frame(mode, li),
        _scrub_slider_update(mode, index=li, interactive=True),
    )


def _replay_playback_loop(mode: str, delay: float):
    RUNTIME.mode = mode
    hist = _history_for_playback(mode)
    if not hist:
        RUNTIME.is_playing = False
        live = list(_live_env_outputs(mode))
        live[-1] = (
            live[-1]
            + "\n\n*No recording in this mode — run **Play** first to record an episode, then use **Replay recording**.*"
        )
        yield (
            *live,
            gr.update(minimum=0, maximum=0, value=0, interactive=True),
        )
        return
    RUNTIME.is_playing = True
    yield from _playback_pass_once(mode, delay, hist)
    RUNTIME.is_playing = False


def _play_loop(
    mode: str,
    delay: float,
    steps_per_frame: int,
    seed: float,
    max_steps: int = 2000,
):
    RUNTIME.mode = mode
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
            rec_stats = _stats_after_step(mode)
            scrub_u = _scrub_slider_update(mode, interactive=False)
            if mode == "Side-by-Side":
                yield (
                    build_road_svg(st, title=f"Recording… (t={st['timestep']})"),
                    build_road_svg(su, title=f"Recording… (t={su['timestep']})"),
                    _broadcast_table(st),
                    _state_json(st, 1),
                    _state_json(st, 2),
                    st["phase"],
                    rec_stats,
                    scrub_u,
                )
            elif mode == "Untrained (base)":
                yield (
                    build_road_svg(su, title=f"Recording… (t={su['timestep']})"),
                    "",
                    _broadcast_table(su),
                    _state_json(su, 1),
                    _state_json(su, 2),
                    su["phase"],
                    rec_stats,
                    scrub_u,
                )
            else:
                yield (
                    build_road_svg(st, title=f"Recording… (t={st['timestep']})"),
                    "",
                    _broadcast_table(st),
                    _state_json(st, 1),
                    _state_json(st, 2),
                    st["phase"],
                    rec_stats,
                    scrub_u,
                )
        hist = _history_for_playback(mode)

    if not hist:
        RUNTIME.is_playing = False
        live = _live_env_outputs(mode)
        yield (
            *live,
            gr.update(minimum=0, maximum=0, value=0, interactive=True),
        )
        return

    yield from _playback_pass_once(mode, delay, hist)
    RUNTIME.is_playing = False


def _reset(
    seed: int,
    *,
    interrupt_playback: bool = True,
) -> tuple[str, str, list[list[Any]], dict[str, Any], dict[str, Any], str, str, dict]:
    if interrupt_playback:
        RUNTIME.is_playing = False
    RUNTIME.done_trained = False
    RUNTIME.done_untrained = False
    RUNTIME.history_trained = []
    RUNTIME.history_untrained = []
    RUNTIME.history_side = []
    RUNTIME.collision_ever_trained = False
    RUNTIME.collision_ever_untrained = False
    RUNTIME.last_rewards_trained = None
    RUNTIME.last_infos_trained = None
    RUNTIME.last_rewards_untrained = None
    RUNTIME.last_infos_untrained = None

    RUNTIME.obs_trained = RUNTIME.env_trained.reset(seed=seed)
    RUNTIME.obs_untrained = RUNTIME.env_untrained.reset(seed=seed)

    state_t = RUNTIME.env_trained.state()
    stats_md = _format_step_stats(
        state_t,
        None,
        None,
        collision_ever=False,
        episode_done=False,
    )
    return (
        build_road_svg(state_t, title="Trained Agent"),
        "",
        _broadcast_table(state_t),
        _state_json(state_t, 1),
        _state_json(state_t, 2),
        state_t["phase"],
        "### Episode reset\n\n" + stats_md,
        gr.update(minimum=0, maximum=0, value=0, interactive=True),
    )


def _pause() -> str:
    RUNTIME.is_playing = False
    return "Paused. Click Play to continue stepping."


def _set_scenario(
    scenario_name: str,
    seed: int,
) -> tuple[str, str, list[list[Any]], dict[str, Any], dict[str, Any], str, str, dict]:
    RUNTIME.is_playing = False
    RUNTIME.done_trained = False
    RUNTIME.done_untrained = False
    RUNTIME.history_trained = []
    RUNTIME.history_untrained = []
    RUNTIME.history_side = []
    RUNTIME.collision_ever_trained = False
    RUNTIME.collision_ever_untrained = False
    RUNTIME.last_rewards_trained = None
    RUNTIME.last_infos_trained = None
    RUNTIME.last_rewards_untrained = None
    RUNTIME.last_infos_untrained = None
    RUNTIME.scenario_name = scenario_name

    RUNTIME.env_trained = PlatoonEnv(scenario_name=scenario_name)
    RUNTIME.env_untrained = PlatoonEnv(scenario_name=scenario_name)
    RUNTIME.obs_trained = RUNTIME.env_trained.reset(seed=int(seed))
    RUNTIME.obs_untrained = RUNTIME.env_untrained.reset(seed=int(seed))

    state_t = RUNTIME.env_trained.state()
    stats_md = _format_step_stats(
        state_t,
        None,
        None,
        collision_ever=False,
        episode_done=False,
    )
    return (
        build_road_svg(state_t, title="Trained Agent"),
        "",
        _broadcast_table(state_t),
        _state_json(state_t, 1),
        _state_json(state_t, 2),
        state_t["phase"],
        f"### Scenario switched to `{scenario_name}`\n\n" + stats_md,
        gr.update(minimum=0, maximum=0, value=0, interactive=True),
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
            replay = gr.Button("Replay recording")
            pause = gr.Button("Pause")
            reset = gr.Button("Reset")

        playback_scrub = gr.Slider(
            minimum=0,
            maximum=0,
            value=0,
            step=1,
            label="Playback frame (scrub recorded episode; drag and release)",
        )

        play.click(
            fn=_play_loop,
            inputs=[mode, speed, steps_per_frame, seed],
            outputs=[road_left, road_right, broadcast, agent1, agent2, phase, stats, playback_scrub],
        )

        replay.click(
            fn=_replay_playback_loop,
            inputs=[mode, speed],
            outputs=[road_left, road_right, broadcast, agent1, agent2, phase, stats, playback_scrub],
        )

        scenario.change(
            fn=_set_scenario,
            inputs=[scenario, seed],
            outputs=[road_left, road_right, broadcast, agent1, agent2, phase, stats, playback_scrub],
        )

        mode.change(
            fn=_on_mode_change,
            inputs=[mode],
            outputs=[road_left, road_right, broadcast, agent1, agent2, phase, stats, playback_scrub],
        )

        pause.click(fn=_pause, outputs=[stats])

        reset.click(
            fn=_reset,
            inputs=[seed],
            outputs=[road_left, road_right, broadcast, agent1, agent2, phase, stats, playback_scrub],
        )

        playback_scrub.release(
            fn=_on_playback_scrub,
            inputs=[mode, playback_scrub],
            outputs=[road_left, road_right, broadcast, agent1, agent2, phase, stats],
        )

    return demo


# Hugging Face Spaces imports this module and expects a Gradio app in `demo`.
# Queue is required for generator/streaming event handlers (e.g. Play).
demo = build_app()
demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
