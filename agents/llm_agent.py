from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable
from typing import Any

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
    action_text: str
    accel_pedal: float
    brake_pedal: float
    parse_ok: bool


class LLMAgent:
    def __init__(
        self,
        base_model_name: str,
        adapter_path: str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 32,
        local_files_only: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._progress_callback = progress_callback
        load_t0 = time.time()
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
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
        self._log(
            "Runtime: "
            f"torch={torch.__version__}, "
            f"cuda_available={torch.cuda.is_available()}, "
            f"selected_device={self.device}, "
            f"local_files_only={self.local_files_only}"
        )

        self._log(f"Tokenizer load start: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self._log("Tokenizer load complete")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._log("Model load start (weights/config)")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            low_cpu_mem_usage=True,
        )
        self._log("Model load complete")

        if adapter_path:
            self._log(f"Adapter load start: {adapter_path}")
            self.model = PeftModel.from_pretrained(model, adapter_path)
            self._log("Adapter load complete")
        else:
            self.model = model
            self._log("No adapter configured")

        self.model.eval()
        if hasattr(self.model, "generation_config"):
            # Keep greedy decoding truly greedy and silence invalid-generation-flag warnings.
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        self._log(f"Model set to eval (total init {time.time() - load_t0:.1f}s)")

    def act(self, observation_text: str, temperature: float = 0.0) -> AgentOutput:
        user_prompt = (
            f"{observation_text.rstrip()}\n"
            "Return ONLY the required action format. No extra text.\n"
            "ACTION:\n"
            "accel_pedal: "
        )
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self._system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{self._system_instruction}\n\n{user_prompt}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
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

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        # Slice by token length (not string length) to avoid prompt-text leakage in decoded tail.
        prompt_len = int(inputs["input_ids"].shape[1])
        new_tokens = generated[0][prompt_len:]
        tail = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Trim at obvious conversation/format spillovers if present.
        for marker in ("```", "\nHuman:", "\nUser:", "\nAssistant:", "\n[OBSERVATION"):
            idx = tail.find(marker)
            if idx != -1:
                tail = tail[:idx].strip()
        normalized_tail = f"ACTION:\naccel_pedal: {tail}" if not tail.lower().startswith("action") else tail

        parsed = self._parse_action(normalized_tail)
        if parsed is None:
            action_text = "ACTION:\naccel_pedal: 0.00\nbrake_pedal: 0.00"
            return AgentOutput(
                raw_text=normalized_tail,
                action_text=action_text,
                accel_pedal=0.0,
                brake_pedal=0.0,
                parse_ok=False,
            )

        accel, brake = parsed
        action_text = f"ACTION:\naccel_pedal: {accel:.2f}\nbrake_pedal: {brake:.2f}"
        return AgentOutput(
            raw_text=normalized_tail,
            action_text=action_text,
            accel_pedal=accel,
            brake_pedal=brake,
            parse_ok=True,
        )

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
            "local_files_only": self.local_files_only,
        }

    def _log(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)
