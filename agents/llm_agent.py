from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ACTION_REGEX = re.compile(
    r"ACTION:\s*accel_pedal:\s*([0-9]*\.?[0-9]+)\s*brake_pedal:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE | re.MULTILINE,
)
ACCEL_REGEX = re.compile(r"accel(?:_pedal)?\s*[:=]\s*([+\-]?[0-9]*\.?[0-9]+)", re.IGNORECASE)
BRAKE_REGEX = re.compile(r"brake(?:_pedal)?\s*[:=]\s*([+\-]?[0-9]*\.?[0-9]+)", re.IGNORECASE)
FLOAT_REGEX = re.compile(r"([+\-]?[0-9]*\.?[0-9]+)")


@dataclass
class AgentOutput:
    raw_text: str
    reasoning_text: str
    action_text: str
    accel_pedal: float
    brake_pedal: float
    parse_ok: bool


class LLMAgent:
    def __init__(
        self,
        base_model_name: str,
        adapter_path: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: str = "cuda",
        max_new_tokens: int = 32,
        enable_private_reasoning: bool = False,
        local_files_only: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._progress_callback = progress_callback
        load_t0 = time.time()
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.enable_private_reasoning = enable_private_reasoning
        self.device = device if torch.cuda.is_available() else "cpu"
        self.local_files_only = local_files_only
        self._system_instruction = (
            "You are a vehicle controller for platoon safety. "
            "Respond with ONLY this exact 3-line format and nothing else:\n"
            "ACTION:\n"
            "accel_pedal: <float_0_to_1>\n"
            "brake_pedal: <float_0_to_1>\n"
            "Never output explanations, markdown, roleplay text, or extra lines."
        )
        self._reasoning_system_instruction = (
            "You explain a vehicle action in one concise sentence for local debugging. "
            "Do not invent hidden state. No markdown."
        )
        self._log(
            "Runtime: "
            f"torch={torch.__version__}, "
            f"cuda_available={torch.cuda.is_available()}, "
            f"selected_device={self.device}, "
            f"local_files_only={self.local_files_only}"
        )

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self._log("Using preloaded model/tokenizer")
            if adapter_path:
                self._log("Adapter path provided with preloaded model; expected adapter already loaded.")
        else:
            self._log(f"Tokenizer load start: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            self._log("Tokenizer load complete")

            self._log("Model load start (weights/config)")
            loaded_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True,
            )
            self._log("Model load complete")

            if adapter_path:
                adapter_dir = Path(adapter_path)
                if (adapter_dir / "adapter_config.json").exists():
                    self._log(f"Adapter load start: {adapter_path}")
                    self.model = PeftModel.from_pretrained(loaded_model, adapter_path)
                    self._log("Adapter load complete")
                else:
                    self.model = loaded_model
                    self._log(f"Adapter missing adapter_config.json at {adapter_path}; using base model")
            else:
                self.model = loaded_model
                self._log("No adapter configured")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        if hasattr(self.model, "generation_config"):
            # Keep greedy decoding truly greedy and silence invalid-generation-flag warnings.
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        self._log(f"Model set to eval (total init {time.time() - load_t0:.1f}s)")

    def _system_for_observation(self, observation_text: str) -> str:
        if "scenario_03_ambulance" in observation_text:
            return (
                "You control a vehicle on a three-lane highway with an emergency ambulance. "
                "Reply with ONLY an ACTION block (no markdown, no explanation).\n"
                "First lines must be exactly:\n"
                "ACTION:\n"
                "accel_pedal: <float 0.0-1.0>\n"
                "brake_pedal: <float 0.0-1.0>\n"
                "Then optionally add ANY of these lines you need (omit if not changing lane):\n"
                "move_left: false OR true\n"
                "move_right: false OR true\n"
                "lane_change: stay OR left OR right\n"
                "target_lane: 0 OR 1 OR 2\n"
                "At most one of move_left / move_right may be true."
            )
        return self._system_instruction

    @staticmethod
    def _slice_action_block(text: str) -> str:
        lo = text.lower().find("action:")
        if lo < 0:
            return ""
        return text[lo:].strip()

    def act(self, observation_text: str, temperature: float = 0.0) -> AgentOutput:
        return self.act_batch([observation_text], temperature=temperature)[0]

    def act_batch(self, observation_texts: list[str], temperature: float = 0.0) -> list[AgentOutput]:
        if not observation_texts:
            return []

        prompts = [self._build_prompt(text) for text in observation_texts]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        max_new = self.max_new_tokens
        if any("scenario_03_ambulance" in (t or "") for t in observation_texts):
            max_new = max(max_new, 96)

        generation_kwargs = {
            "max_new_tokens": max_new,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = max(temperature, 1e-3)
            generation_kwargs["top_p"] = 0.95
            generation_kwargs["top_k"] = 50
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        # Each row may have a different prompt length due to batch padding.
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        results: list[AgentOutput] = []
        for i, prompt_len in enumerate(prompt_lengths):
            new_tokens = generated[i][int(prompt_len):]
            tail = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            normalized_tail = self._normalize_tail(tail)
            parsed = self._parse_action(normalized_tail)
            if parsed is None:
                action_text = "ACTION:\naccel_pedal: 0.00\nbrake_pedal: 0.00"
                results.append(
                    AgentOutput(
                        raw_text=normalized_tail,
                        reasoning_text="",
                        action_text=action_text,
                        accel_pedal=0.0,
                        brake_pedal=0.0,
                        parse_ok=False,
                    )
                )
                continue

            accel, brake = parsed
            block = LLMAgent._slice_action_block(normalized_tail)
            if block.lower().startswith("action:"):
                action_text = block
            else:
                action_text = f"ACTION:\naccel_pedal: {accel:.2f}\nbrake_pedal: {brake:.2f}"
            results.append(
                AgentOutput(
                    raw_text=normalized_tail,
                    reasoning_text="",
                    action_text=action_text,
                    accel_pedal=accel,
                    brake_pedal=brake,
                    parse_ok=True,
                )
            )
        if self.enable_private_reasoning:
            self._populate_private_reasoning(observation_texts, results)
        return results

    def _build_prompt(self, observation_text: str) -> str:
        user_prompt = (
            f"{observation_text.rstrip()}\n"
            "Return ONLY the required action format. No extra text.\n"
            "ACTION:\n"
            "accel_pedal: "
        )
        system_msg = self._system_for_observation(observation_text)
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{system_msg}\n\n{user_prompt}"
        return prompt

    @staticmethod
    def _normalize_tail(tail: str) -> str:
        # Trim at obvious conversation/format spillovers if present.
        for marker in ("```", "\nHuman:", "\nUser:", "\nAssistant:", "\n[OBSERVATION"):
            idx = tail.find(marker)
            if idx != -1:
                tail = tail[:idx].strip()
        return tail if "action" in tail.lower() else f"ACTION:\naccel_pedal: {tail}"

    def _build_reasoning_prompt(self, observation_text: str, action_text: str) -> str:
        user_prompt = (
            f"Observation:\n{observation_text.rstrip()}\n\n"
            f"Chosen action:\n{action_text}\n\n"
            "Explain briefly why this action is reasonable."
        )
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self._reasoning_system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{self._reasoning_system_instruction}\n\n{user_prompt}"

    def _populate_private_reasoning(
        self,
        observation_texts: list[str],
        outputs: list[AgentOutput],
    ) -> None:
        prompts = [
            self._build_reasoning_prompt(obs, out.action_text)
            for obs, out in zip(observation_texts, outputs, strict=False)
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=48,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for i, prompt_len in enumerate(prompt_lengths):
            reason = self.tokenizer.decode(generated[i][int(prompt_len):], skip_special_tokens=True).strip()
            reason = self._sanitize_reasoning_text(reason)
            outputs[i].reasoning_text = reason

    @staticmethod
    def _sanitize_reasoning_text(text: str) -> str:
        for marker in ("```", "\nHuman:", "\nUser:", "\nAssistant:", "\nObservation:", "\nChosen action:"):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx].strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        first = lines[0]
        if first.lower().startswith("reasoning:"):
            first = first[len("reasoning:"):].strip()
        return first

    @staticmethod
    def _parse_action(text: str) -> tuple[float, float] | None:
        # Prefer parsing from the model's final ACTION block when present.
        action_idx = text.lower().rfind("action")
        parse_text = text[action_idx:] if action_idx != -1 else text

        # Strict canonical parser.
        match = ACTION_REGEX.search(parse_text)
        if not match:
            # Flexible fallback parser for minor formatting drift.
            accel_match = ACCEL_REGEX.search(parse_text)
            brake_match = BRAKE_REGEX.search(parse_text)
            if accel_match and brake_match:
                accel = float(accel_match.group(1))
                brake = float(brake_match.group(1))
            elif accel_match and not brake_match:
                accel = float(accel_match.group(1))
                brake = 0.0
            elif brake_match and not accel_match:
                accel = 0.0
                brake = float(brake_match.group(1))
            else:
                # Last-resort numeric extraction to keep parser robust for training rollouts.
                numbers = [float(x) for x in FLOAT_REGEX.findall(parse_text)]
                if len(numbers) >= 2:
                    accel, brake = numbers[0], numbers[1]
                elif len(numbers) == 1:
                    accel, brake = numbers[0], 0.0
                else:
                    return None
        else:
            accel = float(match.group(1))
            brake = float(match.group(2))

        accel = min(max(accel, 0.0), 1.0)
        brake = min(max(brake, 0.0), 1.0)

        if accel > 0.0 and brake > 0.0:
            accel = 0.0

        return accel, brake

    def to(self, device: str) -> None:
        self.device = device
        self.model.to(device)

    def save_adapter(self, output_dir: str) -> None:
        save_path = output_dir
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def info(self) -> dict[str, Any]:
        return {
            "base_model_name": self.base_model_name,
            "adapter_path": self.adapter_path,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "enable_private_reasoning": self.enable_private_reasoning,
            "local_files_only": self.local_files_only,
        }

    def _log(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)
