from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from config.settings import ROOT_DIR, load_settings
from environment.platoon_env import PlatoonEnv

ACTION_REGEX = re.compile(
    r"ACTION:\s*accel_pedal:\s*([0-9]*\.?[0-9]+)\s*brake_pedal:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE | re.MULTILINE,
)

GAP_ERROR_REGEX = re.compile(r"gap_error:\s*([+\-]?[0-9]*\.?[0-9]+)")
EGO_VEL_REGEX = re.compile(r"ego_velocity:\s*([0-9]*\.?[0-9]+)")
FRONT_VEL_REGEX = re.compile(r"front_velocity:\s*([+\-]?[0-9]*\.?[0-9]+)")


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency torch. Install requirements in WSL2 and retry."
        ) from exc
    return torch


def _import_training_stack() -> dict[str, Any]:
    try:
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing ML training dependencies. Install requirements.txt in WSL2 before running training."
        ) from exc

    return {
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "TrainingArguments": TrainingArguments,
        "SFTTrainer": SFTTrainer,
    }


@dataclass
class EpisodeMetrics:
    episode: int
    steps: int
    collision: bool
    total_reward_agent_1: float
    total_reward_agent_2: float
    mean_reward: float
    final_gap_error_agent_1: float
    final_gap_error_agent_2: float
    mean_jerk: float
    parse_failures: int
    parse_failure_rate: float


@dataclass
class RolloutSample:
    prompt: str
    action_text: str
    reward: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train platoon RL model")
    parser.add_argument("--sft", action="store_true", help="Run SFT flow")
    parser.add_argument("--rl", action="store_true", help="Run RL flow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--grpo-update-every", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    default_base_model = os.getenv("MODEL_PATH", "").strip().strip('"').strip("'") or "Qwen/Qwen2.5-1.5B-Instruct"
    parser.add_argument("--base-model", type=str, default=default_base_model)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--lr-sft", type=float, default=2e-4)
    parser.add_argument("--lr-rl", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument(
        "--report-to",
        type=str,
        default="auto",
        choices=["auto", "none", "wandb"],
        help="Experiment tracking backend",
    )
    parser.add_argument("--reset-metrics", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch = _import_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def read_hf_username() -> str:
    return os.getenv("HF_USERNAME", "").strip()


def resolve_report_to(report_to_arg: str) -> list[str]:
    if report_to_arg == "none":
        return []
    if report_to_arg == "wandb":
        return ["wandb"]

    # auto mode: enable wandb only when API key is present.
    if os.getenv("WANDB_API_KEY", "").strip():
        return ["wandb"]
    return []


def maybe_upload(local_dir: Path, repo_id: str, commit_message: str) -> None:
    if not local_dir.exists() or not repo_id:
        return
    try:
        from huggingface_hub import create_repo, upload_folder

        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
    except Exception as exc:
        print(f"[WARN] HF upload failed for {repo_id}: {exc}")


def _checkpoint_step(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint-"):
        suffix = name.split("checkpoint-", maxsplit=1)[1]
        if suffix.isdigit():
            return int(suffix)
    return -1


def resolve_adapter_dir(path: str | Path | None) -> str | None:
    if not path:
        return None
    root = Path(path)
    if not root.exists():
        return None
    if (root / "adapter_config.json").exists():
        return str(root)

    candidates: list[Path] = []
    for cfg in root.rglob("adapter_config.json"):
        parent = cfg.parent
        if (parent / "adapter_model.safetensors").exists() or (parent / "adapter_model.bin").exists():
            candidates.append(parent)

    if not candidates:
        return None

    candidates.sort(key=lambda p: (_checkpoint_step(p), len(p.parts)), reverse=True)
    return str(candidates[0])


def load_base_model_and_tokenizer(
    base_model: str,
    max_seq_len: int,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    torch = _import_torch()
    stack = _import_training_stack()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    AutoTokenizer = stack["AutoTokenizer"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    LoraConfig = stack["LoraConfig"]
    PeftModel = stack["PeftModel"]
    get_peft_model = stack["get_peft_model"]

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        token=hf_token,
    )

    if adapter_path:
        try:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
            print(f"Loaded adapter into training model: {adapter_path}")
        except TypeError:
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"Loaded adapter into training model (no is_trainable flag): {adapter_path}")
        except Exception as exc:
            print(f"[WARN] Failed to load adapter into training model ({adapter_path}): {exc}")

    # For SFT (no adapter input), initialize fresh LoRA trainable adapters.
    if not isinstance(model, PeftModel):
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    tokenizer.model_max_length = max_seq_len
    return model, tokenizer


def build_sft_dataset(dataset_path: Path) -> Dataset:
    Dataset = _import_training_stack()["Dataset"]

    if not dataset_path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {dataset_path}")

    rows: list[dict[str, str]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            required = {
                "id",
                "scenario",
                "phase",
                "agent_id",
                "timestep",
                "observation_text",
                "reasoning",
                "action_text",
            }
            missing = required - set(item.keys())
            if missing:
                raise ValueError(f"Invalid SFT record at line {index}, missing {sorted(missing)}")

            rows.append(
                {
                    "text": (
                        f"{item['observation_text']}\n"
                        f"Reasoning:\n{item['reasoning']}\n"
                        f"{item['action_text']}"
                    )
                }
            )

    if not rows:
        raise ValueError("SFT dataset is empty")

    return Dataset.from_list(rows)


def run_sft(args: argparse.Namespace) -> None:
    torch = _import_torch()
    stack = _import_training_stack()
    TrainingArguments = stack["TrainingArguments"]
    SFTTrainer = stack["SFTTrainer"]

    print("Starting SFT fine-tuning")
    data_path = ROOT_DIR / "data" / "sft" / "scenario_01.jsonl"
    output_dir = ROOT_DIR / "checkpoints" / "sft_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_sft_dataset(data_path)
    model, tokenizer = load_base_model_and_tokenizer(args.base_model, args.max_seq_len)

    report_to = resolve_report_to(args.report_to)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr_sft,
        num_train_epochs=args.epochs,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        report_to=report_to,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    sft_sig = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in sft_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in sft_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in sft_sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_sig.parameters:
        trainer_kwargs["max_seq_length"] = args.max_seq_len

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    hf_username = read_hf_username()
    if hf_username and hf_username != "your_hf_username":
        maybe_upload(output_dir, f"{hf_username}/platoon-qwen-sft", "Upload SFT adapter")

    print(f"SFT finished. Saved checkpoint: {output_dir}")


def score_action_from_prompt(prompt: str, action_text: str) -> float:
    match = ACTION_REGEX.search(action_text)
    if not match:
        return -8.0

    accel = float(np.clip(float(match.group(1)), 0.0, 1.0))
    brake = float(np.clip(float(match.group(2)), 0.0, 1.0))

    if accel > 0.0 and brake > 0.0:
        accel = 0.0

    gap_error_match = GAP_ERROR_REGEX.search(prompt)
    ego_vel_match = EGO_VEL_REGEX.search(prompt)
    front_vel_match = FRONT_VEL_REGEX.search(prompt)

    if not gap_error_match or not ego_vel_match or not front_vel_match:
        return -4.0

    gap_error = float(gap_error_match.group(1))
    ego_vel = float(ego_vel_match.group(1))
    front_vel = float(front_vel_match.group(1))

    net_accel = (accel * 3.0) - (brake * 8.0)
    relative_speed = ego_vel - front_vel

    target_brake = max(0.0, min(1.0, (max(0.0, -gap_error) / 10.0) + (max(0.0, relative_speed) / 12.0)))
    target_accel = max(0.0, min(1.0, gap_error / 18.0)) if gap_error > 1.0 else 0.0

    action_mismatch = abs(brake - target_brake) + abs(accel - target_accel)
    jerk_proxy = abs(net_accel)
    safety_penalty = 8.0 if gap_error < -6.0 and brake < 0.4 else 0.0

    reward = -action_mismatch - (0.06 * jerk_proxy) - safety_penalty
    if abs(gap_error) < 1.0:
        reward += 1.5
    return float(reward)


def choose_group_best_actions(agent: LLMAgent, prompts: list[str], group_size: int) -> list[RolloutSample]:
    selected: list[RolloutSample] = []
    for prompt in prompts:
        candidates: list[RolloutSample] = []
        for _ in range(group_size):
            out = agent.act(prompt, temperature=0.7)
            reward = score_action_from_prompt(prompt, out.action_text)
            candidates.append(RolloutSample(prompt=prompt, action_text=out.action_text, reward=reward))

        best = sorted(candidates, key=lambda row: row.reward, reverse=True)[0]
        selected.append(best)
    return selected


def apply_grpo_style_update(
    model: Any,
    tokenizer: Any,
    samples: list[RolloutSample],
    output_dir: Path,
    lr: float,
    max_seq_len: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    torch = _import_torch()
    stack = _import_training_stack()
    Dataset = stack["Dataset"]
    TrainingArguments = stack["TrainingArguments"]
    SFTTrainer = stack["SFTTrainer"]

    if not samples:
        return

    data = [{"text": f"{sample.prompt}\n{sample.action_text}"} for sample in samples]
    dataset = Dataset.from_list(data)

    report_to = resolve_report_to(os.getenv("REPORT_TO", "auto"))
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="no",
        report_to=report_to,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    sft_sig = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in sft_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in sft_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in sft_sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_sig.parameters:
        trainer_kwargs["max_seq_length"] = max_seq_len

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()


def try_native_grpo_update(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    output_dir: Path,
    lr: float,
    batch_size: int,
    grad_accum: int,
    group_size: int,
) -> tuple[bool, str]:
    if not prompts:
        return False, "no_prompts"

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except Exception:
        return False, "grpo_unavailable"

    prompt_dataset = Dataset.from_list([{"prompt": prompt} for prompt in prompts])

    def reward_fn(completions: list[str], prompts: list[str], **_: Any) -> list[float]:
        return [score_action_from_prompt(prompt, completion) for prompt, completion in zip(prompts, completions)]

    try:
        grpo_args = GRPOConfig(
            output_dir=str(output_dir),
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_generations=max(2, group_size),
            max_completion_length=64,
            logging_steps=5,
            report_to=[],
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            train_dataset=prompt_dataset,
            args=grpo_args,
            processing_class=tokenizer,
        )
        trainer.train()
    except Exception as exc:
        return False, f"grpo_runtime_error:{exc}"

    return True, "grpo_native"


def run_episode(
    env: PlatoonEnv,
    agent: LLMAgent,
    episode_seed: int,
    temperature: float,
    collect_prompts: bool,
) -> tuple[EpisodeMetrics, list[str]]:
    obs = env.reset(seed=episode_seed)
    done = False
    step_count = 0

    total_r1 = 0.0
    total_r2 = 0.0
    parse_failures = 0
    jerk_values: list[float] = []
    final_gap_err_1 = 0.0
    final_gap_err_2 = 0.0
    prompts: list[str] = []

    while not done:
        if collect_prompts:
            prompts.append(obs["agent_1"])
            prompts.append(obs["agent_2"])

        out_1 = agent.act(obs["agent_1"], temperature=temperature)
        out_2 = agent.act(obs["agent_2"], temperature=temperature)
        if not out_1.parse_ok:
            parse_failures += 1
        if not out_2.parse_ok:
            parse_failures += 1

        obs, rewards, dones, infos = env.step(
            {
                "agent_1": out_1.action_text,
                "agent_2": out_2.action_text,
            }
        )

        total_r1 += rewards["agent_1"]
        total_r2 += rewards["agent_2"]
        final_gap_err_1 = float(infos["agent_1"]["gap_error"])
        final_gap_err_2 = float(infos["agent_2"]["gap_error"])

        jerk_values.append(abs(float(env.state()["vehicles"][1]["net_acceleration"])))
        jerk_values.append(abs(float(env.state()["vehicles"][2]["net_acceleration"])))

        step_count += 1
        done = bool(dones["agent_1"])

    vehicle_state = env.state()["vehicles"]
    collision = (
        vehicle_state[1]["x"] + vehicle_state[1]["length"] >= vehicle_state[0]["x"]
        or vehicle_state[2]["x"] + vehicle_state[2]["length"] >= vehicle_state[1]["x"]
    )

    metrics = EpisodeMetrics(
        episode=-1,
        steps=step_count,
        collision=collision,
        total_reward_agent_1=total_r1,
        total_reward_agent_2=total_r2,
        mean_reward=(total_r1 + total_r2) / 2.0,
        final_gap_error_agent_1=final_gap_err_1,
        final_gap_error_agent_2=final_gap_err_2,
        mean_jerk=float(np.mean(jerk_values)) if jerk_values else 0.0,
        parse_failures=parse_failures,
        parse_failure_rate=(parse_failures / (2.0 * max(step_count, 1))),
    )
    return metrics, prompts


def evaluate(agent: LLMAgent, eval_seeds: list[int]) -> dict[str, float]:
    env = PlatoonEnv()
    rows: list[EpisodeMetrics] = []
    for seed in eval_seeds:
        episode_metrics, _ = run_episode(env, agent, episode_seed=seed, temperature=0.0, collect_prompts=False)
        rows.append(episode_metrics)

    if not rows:
        return {
            "collision_rate": 1.0,
            "mean_episode_reward": -999.0,
            "mean_gap_error_final": 999.0,
            "mean_jerk": 999.0,
            "parse_failure_rate": 1.0,
        }

    return {
        "collision_rate": float(np.mean([1.0 if row.collision else 0.0 for row in rows])),
        "mean_episode_reward": float(np.mean([row.mean_reward for row in rows])),
        "mean_gap_error_final": float(
            np.mean(
                [
                    (abs(row.final_gap_error_agent_1) + abs(row.final_gap_error_agent_2)) / 2.0
                    for row in rows
                ]
            )
        ),
        "mean_jerk": float(np.mean([row.mean_jerk for row in rows])),
        "parse_failure_rate": float(np.mean([row.parse_failure_rate for row in rows])),
    }


def plot_training_curves(metrics_path: Path, reward_png: Path, loss_png: Path) -> None:
    if not metrics_path.exists():
        return

    episodes: list[int] = []
    rewards: list[float] = []
    losses: list[float] = []

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("event") != "episode_end":
                continue
            episodes.append(int(item["episode"]))
            rewards.append(float(item["mean_reward"]))
            losses.append(float(item.get("proxy_loss", max(0.0, -item["mean_reward"] / 300.0))))

    if not episodes:
        return

    reward_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, color="tab:blue")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.title("Reward Curve")
    plt.tight_layout()
    plt.savefig(reward_png)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, losses, color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Proxy Loss")
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(loss_png)
    plt.close()


def run_rl(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    from agents.llm_agent import LLMAgent

    print("Starting RL training")
    metrics_path = ROOT_DIR / settings["logging"]["metrics_path"]
    reward_png = ROOT_DIR / "results" / "reward_curve.png"
    loss_png = ROOT_DIR / "results" / "loss_curve.png"

    if args.reset_metrics and metrics_path.exists():
        metrics_path.unlink()

    sft_path = ROOT_DIR / "checkpoints" / "sft_final"
    raw_adapter = args.adapter if args.adapter else (str(sft_path) if sft_path.exists() else None)
    adapter_path = resolve_adapter_dir(raw_adapter)
    if raw_adapter and not adapter_path:
        print(f"[WARN] Could not find adapter_config.json under {raw_adapter}; RL will start from base model")
    elif adapter_path:
        print(f"Using adapter for RL init: {adapter_path}")
    model, tokenizer = load_base_model_and_tokenizer(
        args.base_model,
        args.max_seq_len,
        adapter_path=adapter_path,
    )
    agent = LLMAgent(
        base_model_name=args.base_model,
        adapter_path=adapter_path,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    env = PlatoonEnv()
    eval_seeds = settings.get("evaluation", {}).get("seeds", list(range(1001, 1011)))
    max_prompts_per_update = int(settings.get("training_runtime", {}).get("max_prompts_per_update", 400))
    prompt_buffer: list[str] = []

    low_collision_windows = 0

    for episode in range(1, args.episodes + 1):
        episode_seed = args.seed + episode
        print(f"[RL] episode {episode}/{args.episodes} start seed={episode_seed}", flush=True)
        metrics, prompts = run_episode(
            env,
            agent,
            episode_seed=episode_seed,
            temperature=0.35,
            collect_prompts=True,
        )
        metrics.episode = episode
        prompt_buffer.extend(prompts)

        proxy_loss = max(0.0, -metrics.mean_reward / 300.0)
        row = asdict(metrics)
        row.update({"event": "episode_end", "proxy_loss": proxy_loss})
        append_jsonl(metrics_path, row)
        print(
            f"[RL] episode {episode} done steps={metrics.steps} "
            f"mean_reward={metrics.mean_reward:.3f} collision={metrics.collision} "
            f"parse_fail_rate={metrics.parse_failure_rate:.3f}",
            flush=True,
        )

        if episode % args.grpo_update_every == 0 and prompt_buffer:
            print(f"[RL] episode {episode} update start prompts={len(prompt_buffer[:max_prompts_per_update])}", flush=True)
            update_prompts = prompt_buffer[:max_prompts_per_update]
            native_ok, native_mode = try_native_grpo_update(
                model=model,
                tokenizer=tokenizer,
                prompts=update_prompts,
                output_dir=ROOT_DIR / "checkpoints" / "tmp_grpo_update",
                lr=args.lr_rl,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                group_size=args.group_size,
            )

            selected_count = 0
            if not native_ok:
                selected = choose_group_best_actions(
                    agent=agent,
                    prompts=update_prompts,
                    group_size=max(2, args.group_size),
                )
                selected_count = len(selected)

                apply_grpo_style_update(
                    model=model,
                    tokenizer=tokenizer,
                    samples=selected,
                    output_dir=ROOT_DIR / "checkpoints" / "tmp_rl_update",
                    lr=args.lr_rl,
                    max_seq_len=args.max_seq_len,
                    batch_size=args.batch_size,
                    grad_accum=args.grad_accum,
                )
            print(
                f"[RL] episode {episode} update done mode={native_mode} native={native_ok} "
                f"selected={selected_count or len(update_prompts)}",
                flush=True,
            )

            prompt_buffer.clear()
            append_jsonl(
                metrics_path,
                {
                    "event": "grpo_update",
                    "episode": episode,
                    "native_grpo": native_ok,
                    "mode": native_mode,
                    "sample_count": selected_count,
                    "group_size": args.group_size,
                },
            )

        if episode % args.eval_every == 0:
            eval_metrics = evaluate(agent, eval_seeds=eval_seeds)
            append_jsonl(metrics_path, {"event": "evaluation", "episode": episode, **eval_metrics})
            print(
                f"[RL] episode {episode} eval collision_rate={eval_metrics['collision_rate']:.3f} "
                f"mean_reward={eval_metrics['mean_episode_reward']:.3f} "
                f"mean_gap_error={eval_metrics['mean_gap_error_final']:.3f}",
                flush=True,
            )

            if eval_metrics["collision_rate"] < 0.05:
                low_collision_windows += 1
            else:
                low_collision_windows = 0

            if low_collision_windows >= 3:
                append_jsonl(
                    metrics_path,
                    {
                        "event": "early_stop",
                        "episode": episode,
                        "reason": "collision_rate_below_0_05_for_3_windows",
                    },
                )
                print("Early stopping criterion met.")
                break

        if episode % args.checkpoint_every == 0:
            ckpt_dir = ROOT_DIR / "checkpoints" / f"rl_ep{episode:03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"[RL] episode {episode} checkpoint saved to {ckpt_dir}", flush=True)

            hf_username = read_hf_username()
            if hf_username and hf_username != "your_hf_username":
                maybe_upload(ckpt_dir, f"{hf_username}/platoon-qwen-rl", f"Upload RL checkpoint episode {episode}")
                print(f"[RL] episode {episode} checkpoint uploaded to {hf_username}/platoon-qwen-rl", flush=True)

    plot_training_curves(metrics_path, reward_png, loss_png)
    print("RL training completed")


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    args = parse_args()
    os.environ["REPORT_TO"] = args.report_to

    if args.sft == args.rl:
        raise ValueError("Specify exactly one mode: --sft or --rl")

    settings = load_settings()
    seed_everything(args.seed)

    if args.sft:
        run_sft(args)
    if args.rl:
        run_rl(args, settings)


if __name__ == "__main__":
    main()
