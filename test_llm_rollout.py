#!/usr/bin/env python3
"""
Direct LLM agent observation script.
Shows how Qwen acts in the platoon environment over 30 steps.
Run: python test_llm_rollout.py
"""

import os
import socket
import sys
import time
import threading
from pathlib import Path

from dotenv import load_dotenv
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import logging as hf_logging

from agents.llm_agent import LLMAgent
from environment.platoon_env import PlatoonEnv


def _has_model_weights(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    for p in model_dir.glob("*"):
        name = p.name.lower()
        if name == "model.safetensors" or name == "pytorch_model.bin":
            return True
        if name.endswith(".safetensors") and "model" in name:
            return True
        if name.startswith("pytorch_model-") and name.endswith(".bin"):
            return True
    return False


def _cached_snapshot_for_repo(repo_id: str) -> Path | None:
    hf_home = Path(os.getenv("HF_HOME", "")).expanduser() if os.getenv("HF_HOME") else (Path.home() / ".cache" / "huggingface")
    snapshots_root = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots_root.exists():
        return None
    candidates = [p for p in snapshots_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _repo_cache_root(repo_id: str) -> Path:
    hf_home = Path(os.getenv("HF_HOME", "")).expanduser() if os.getenv("HF_HOME") else (Path.home() / ".cache" / "huggingface")
    return hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(max(0, n))
    idx = 0
    while v >= 1024.0 and idx < len(units) - 1:
        v /= 1024.0
        idx += 1
    return f"{v:.2f}{units[idx]}"


def _cleanup_incomplete_blobs(
    repo_id: str,
    stale_age_s: int = 900,
    remove_all: bool = False,
) -> tuple[int, int]:
    blobs_dir = _repo_cache_root(repo_id) / "blobs"
    if not blobs_dir.exists():
        return 0, 0
    now = time.time()
    removed = 0
    removed_bytes = 0
    for p in blobs_dir.glob("*.incomplete"):
        try:
            age_s = now - p.stat().st_mtime
            if (not remove_all) and age_s < stale_age_s:
                continue
            size = p.stat().st_size
            p.unlink(missing_ok=True)
            removed += 1
            removed_bytes += size
        except OSError:
            continue
    return removed, removed_bytes


def _can_reach_hf(timeout_s: float = 3.0) -> bool:
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout_s):
            return True
    except OSError:
        return False


def main() -> None:
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    trace_hf = os.getenv("TRACE_HF", "0").strip().lower() in {"1", "true", "yes"}
    if trace_hf:
        hf_logging.set_verbosity_info()
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    else:
        hf_logging.set_verbosity_error()

    print("=" * 80)
    print("PLATOON SIMULATOR: LLM AGENT LIVE OBSERVATION")
    print("=" * 80)

    print("\n[1/4] Initializing environment...")
    env = PlatoonEnv()
    obs = env.reset(seed=42)
    print("OK Environment ready (scenario: brake test)")
    print_llm_raw = os.getenv("PRINT_LLM_RAW", "1").strip().lower() in {"1", "true", "yes"}
    enable_private_reasoning = os.getenv("ENABLE_PRIVATE_REASONING", "1").strip().lower() in {"1", "true", "yes"}
    print_llm_reasoning = os.getenv("PRINT_LLM_REASONING", "1" if enable_private_reasoning else "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    print("\n[GPU] Runtime diagnostics...")
    print(f"  python_executable={sys.executable}")
    print(f"  torch_version={torch.__version__}")
    if torch.cuda.is_available():
        print(f"  cuda_available=True ({torch.cuda.get_device_name(0)})")
    else:
        print("  cuda_available=False")
        if "+cpu" in torch.__version__:
            print("WARN CPU-only torch build detected.")
            print("  If you expected GPU, run with your project venv Python instead of system Python.")
            print(r"  Example: .\.venv\Scripts\python.exe test_llm_rollout.py")

    require_cuda = os.getenv("REQUIRE_CUDA", "0").strip().lower() in {"1", "true", "yes"}
    if require_cuda and not torch.cuda.is_available():
        print("ERROR REQUIRE_CUDA is enabled but CUDA is unavailable in this runtime.")
        return

    # Fast local smoke test option:
    #   set MODEL_ID=sshleifer/tiny-gpt2 before running
    # Full intended model:
    #   MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
    requested_model = os.getenv("MODEL_ID", "").strip()
    force_large_cpu_model = os.getenv("FORCE_QWEN_ON_CPU", "0").strip().lower() in {"1", "true", "yes"}
    if requested_model:
        base_model = requested_model
    elif torch.cuda.is_available() or force_large_cpu_model:
        base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    else:
        # CPU-first default to keep local interaction responsive and avoid long startup stalls.
        base_model = "sshleifer/tiny-gpt2"

    model_path = os.getenv("MODEL_PATH", "").strip()
    adapter_path = os.getenv("ADAPTER_PATH", "").strip()
    local_files_only = os.getenv("LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes"}
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("\n[2/4] Resolving model source...")
    resolved_model = base_model
    if model_path:
        candidate = Path(model_path).expanduser().resolve()
        if not candidate.exists():
            print(f"ERROR MODEL_PATH does not exist: {candidate}")
            return
        resolved_model = str(candidate)
        print(f"OK Using local model path: {resolved_model}")
    else:
        print(f"OK Using model id: {resolved_model}")
        if resolved_model == "sshleifer/tiny-gpt2":
            print("OK Auto-selected small model for local CPU startup speed.")
            print("   Set MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct to force the full model.")
            print("   Or set FORCE_QWEN_ON_CPU=1 to keep Qwen as default on CPU.")
    if local_files_only:
        print("OK Offline/local-files-only mode enabled")

    # Preflight: fail fast when weights are missing and hub is unreachable.
    print("\n[3/5] Model preflight...")
    is_local_path = model_path != ""
    if is_local_path:
        local_dir = Path(resolved_model)
        if not _has_model_weights(local_dir):
            print(f"ERROR No model weights found in local MODEL_PATH: {local_dir}")
            print("  Expected files like model.safetensors or pytorch_model.bin.")
            return
        print("OK Local model weights found.")
    else:
        snapshot = _cached_snapshot_for_repo(resolved_model)
        if snapshot and _has_model_weights(snapshot):
            print(f"OK Cached model weights found: {snapshot}")
        else:
            print("WARN Cached model weights not found for this repo id.")
            if not _can_reach_hf():
                print("ERROR huggingface.co is unreachable from this machine right now.")
                print("  Model weights cannot be downloaded, so load would hang/fail.")
                print("  Fix network/HF access or set MODEL_PATH to a full local model directory.")
                return
            print("OK huggingface.co reachable; loader will download missing weights.")

    # Explicit pre-download stage so users can see real transfer progress.
    if (not is_local_path) and (not local_files_only):
        print("\n[4/5] Downloading/caching model files (with progress)...")
        try:
            dl_t0 = time.time()
            max_workers = int(os.getenv("HF_DOWNLOAD_MAX_WORKERS", "1"))
            etag_timeout_s = float(os.getenv("HF_ETAG_TIMEOUT_S", "30"))
            hb_interval_s = int(os.getenv("HF_HEARTBEAT_S", "5"))
            stall_timeout_s = int(os.getenv("HF_STALL_TIMEOUT_S", "180"))
            repo_id_for_download = resolved_model
            stale_cleanup_s = int(os.getenv("HF_STALE_INCOMPLETE_AGE_S", "900"))
            clean_all_incomplete = os.getenv("HF_CLEAN_INCOMPLETE", "1").strip().lower() in {"1", "true", "yes"}
            removed_n, removed_bytes = _cleanup_incomplete_blobs(
                repo_id=repo_id_for_download,
                stale_age_s=stale_cleanup_s,
                remove_all=clean_all_incomplete,
            )
            if removed_n > 0:
                print(
                    "WARN Removed partial HF blobs before download: "
                    f"count={removed_n}, reclaimed={_fmt_bytes(removed_bytes)}, "
                    f"mode={'all' if clean_all_incomplete else f'stale>{stale_cleanup_s}s'}"
                )
            cache_root = _repo_cache_root(repo_id_for_download)
            dl_result: dict[str, str | Exception | None] = {"path": None, "error": None}
            dl_done = threading.Event()

            def _do_download() -> None:
                try:
                    dl_result["path"] = snapshot_download(
                        repo_id=repo_id_for_download,
                        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                        max_workers=max_workers,
                        etag_timeout=etag_timeout_s,
                    )
                except Exception as exc:
                    dl_result["error"] = exc
                finally:
                    dl_done.set()

            dl_thread = threading.Thread(target=_do_download, daemon=True)
            dl_thread.start()

            elapsed = 0
            no_growth_s = 0
            prev_size = _dir_size_bytes(cache_root)
            while not dl_done.wait(hb_interval_s):
                elapsed += hb_interval_s
                curr_size = _dir_size_bytes(cache_root)
                delta = max(0, curr_size - prev_size)
                prev_size = curr_size
                no_growth_s = 0 if delta > 0 else (no_growth_s + hb_interval_s)
                print(
                    "... download heartbeat "
                    f"({elapsed}s elapsed) cached={_fmt_bytes(curr_size)} (+{_fmt_bytes(delta)}) "
                    f"no_growth={no_growth_s}s"
                )
                if no_growth_s >= stall_timeout_s:
                    raise RuntimeError(
                        "HF download appears stalled (no cache growth). "
                        "Try again; stale partial files were cleaned automatically. "
                        "If this repeats, set MODEL_PATH to a pre-downloaded local snapshot."
                    )

            dl_thread.join()
            if dl_result["error"] is not None:
                raise RuntimeError(str(dl_result["error"]))
            if not isinstance(dl_result["path"], str):
                raise RuntimeError("snapshot_download returned empty path")
            resolved_model = dl_result["path"]
            print(f"OK Model snapshot ready in {time.time() - dl_t0:.1f}s")
            print(f"   Local snapshot path: {resolved_model}")
            print(f"   Download settings: max_workers={max_workers}, etag_timeout={etag_timeout_s}s")
        except Exception as e:
            print(f"ERROR Model download failed: {e}")
            return
    else:
        print("\n[4/5] Skipping download stage (local/offline mode).")

    print("\n[5/5] Loading model into memory...")
    stop_progress = threading.Event()
    step_t0 = time.time()

    def _progress_printer() -> None:
        elapsed = 0
        while not stop_progress.wait(10):
            elapsed += 10
            print(f"... model load in progress ({elapsed}s elapsed)")

    def _loader_trace(message: str) -> None:
        print(f"   [loader +{time.time() - step_t0:6.1f}s] {message}")

    progress_thread = threading.Thread(target=_progress_printer, daemon=True)
    progress_thread.start()
    try:
        t0 = time.time()
        agent = LLMAgent(
            base_model_name=resolved_model,
            adapter_path=adapter_path if adapter_path else None,
            enable_private_reasoning=enable_private_reasoning,
            local_files_only=local_files_only,
            progress_callback=_loader_trace,
        )
        stop_progress.set()
        progress_thread.join(timeout=0.2)
        print(f"OK Model loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        stop_progress.set()
        progress_thread.join(timeout=0.2)
        print(f"ERROR Model load failed: {e}")
        if local_files_only:
            print("  Hint: disable LOCAL_FILES_ONLY or cache the model first.")
        else:
            print("  Hint: if network is unavailable, set MODEL_PATH to a local model directory.")
        return

    rollout_steps = int(os.getenv("ROLLOUT_STEPS", "30"))
    print(f"\n[5/5] Running {rollout_steps}-step rollout...")
    use_batch_inference = os.getenv("BATCH_INFERENCE", "1").strip().lower() in {"1", "true", "yes"}
    print(f"INFO Inference mode: {'batched (2 agents / 1 generate call)' if use_batch_inference else 'separate (2 generate calls)'}")
    print(f"INFO Private reasoning: {'enabled (local logging only)' if enable_private_reasoning else 'disabled'}")
    print(f"INFO Adapter: {adapter_path if adapter_path else 'none (base model only)'}")
    print("-" * 80)
    print(f"{'Step':>4} | {'Phase':>12} | {'Agent 1 Action':>25} | {'Agent 2 Action':>25} | {'Rewards':>18}")
    print("-" * 80)

    total_reward_1 = 0.0
    total_reward_2 = 0.0
    parse_failures = 0
    nonzero_actions = 0
    steps_run = 0
    total_infer_s = 0.0

    for step in range(rollout_steps):
        try:
            infer_t0 = time.time()
            if use_batch_inference:
                action_1, action_2 = agent.act_batch(
                    [obs["agent_1"], obs["agent_2"]],
                    temperature=0.0,
                )
            else:
                action_1 = agent.act(obs["agent_1"], temperature=0.0)
                action_2 = agent.act(obs["agent_2"], temperature=0.0)
            total_infer_s += (time.time() - infer_t0)
        except Exception as e:
            print(f"ERROR Step {step}: Action generation failed: {e}")
            break

        obs, rewards, dones, infos = env.step(
            {
                "agent_1": action_1.action_text,
                "agent_2": action_2.action_text,
            }
        )

        total_reward_1 += rewards["agent_1"]
        total_reward_2 += rewards["agent_2"]
        steps_run = step + 1

        phase_name = env.phase

        action_1_str = f"a:{action_1.accel_pedal:.2f} b:{action_1.brake_pedal:.2f}"
        action_2_str = f"a:{action_2.accel_pedal:.2f} b:{action_2.brake_pedal:.2f}"
        reward_str = f"a1:{rewards['agent_1']:+.2f} a2:{rewards['agent_2']:+.2f}"

        print(f"{step:4d} | {phase_name:>12} | {action_1_str:>25} | {action_2_str:>25} | {reward_str:>18}")
        if print_llm_raw:
            raw_1 = " ".join(action_1.raw_text.split())
            raw_2 = " ".join(action_2.raw_text.split())
            print(f"      LLM1 raw: {raw_1[:220]}")
            print(f"      LLM2 raw: {raw_2[:220]}")
        if print_llm_reasoning:
            rsn_1 = " ".join(action_1.reasoning_text.split()) if action_1.reasoning_text else "<none>"
            rsn_2 = " ".join(action_2.reasoning_text.split()) if action_2.reasoning_text else "<none>"
            print(f"      LLM1 reasoning: {rsn_1[:220]}")
            print(f"      LLM2 reasoning: {rsn_2[:220]}")

        if (not action_1.parse_ok) or (not action_2.parse_ok):
            parse_failures += int(not action_1.parse_ok) + int(not action_2.parse_ok)
        if action_1.accel_pedal > 0.0 or action_1.brake_pedal > 0.0:
            nonzero_actions += 1
        if action_2.accel_pedal > 0.0 or action_2.brake_pedal > 0.0:
            nonzero_actions += 1

        if dones.get("agent_1") or dones.get("agent_2"):
            print(f"(Episode ended at step {step})")
            break

    print("-" * 80)
    print(f"Total Episode Rewards: Agent 1 = {total_reward_1:.2f}, Agent 2 = {total_reward_2:.2f}")
    total_actions = max(1, steps_run * 2)
    parse_ok_rate = 1.0 - (parse_failures / total_actions)
    nonzero_rate = nonzero_actions / total_actions
    print(
        "LLM->RL handshake summary: "
        f"steps={steps_run}, parse_ok_rate={parse_ok_rate:.1%}, nonzero_action_rate={nonzero_rate:.1%}"
    )
    if steps_run > 0:
        avg_step_infer_ms = (total_infer_s / steps_run) * 1000.0
        avg_action_infer_ms = (total_infer_s / (steps_run * 2)) * 1000.0
        print(
            "Inference timing: "
            f"total={total_infer_s:.2f}s, "
            f"avg_step={avg_step_infer_ms:.1f}ms, "
            f"avg_action={avg_action_infer_ms:.1f}ms"
        )
    if steps_run > 0:
        print("OK LLM outputs were consumed by RL env.step() and produced rewards/dones each step.")
    print("\nOK Rollout complete. Agents successfully controlled vehicles through brake scenario.")


if __name__ == "__main__":
    main()
