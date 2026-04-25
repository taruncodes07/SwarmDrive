from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from unsloth import FastLanguageModel as _FastLanguageModel  # type: ignore[import]
except Exception:
    _FastLanguageModel = None  # type: ignore[assignment]

ACTION_REGEX = re.compile(
    r"ACTION:\s*accel_pedal:\s*([0-9]*\.?[0-9]+)\s*brake_pedal:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE | re.MULTILINE,
)


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
        use_unsloth: bool = False,
    ) -> None:
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.device = device if torch.cuda.is_available() else "cpu"

        if use_unsloth and _FastLanguageModel is not None:
            load_name = adapter_path if adapter_path else base_model_name
            model, tokenizer = _FastLanguageModel.from_pretrained(
                model_name=load_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            _FastLanguageModel.for_inference(model)
            self.tokenizer = tokenizer
            self.model = model
            self.model.eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            if adapter_path:
                self.model = PeftModel.from_pretrained(model, adapter_path)
            else:
                self.model = model

            self.model.eval()

    def act(self, observation_text: str, temperature: float = 0.0) -> AgentOutput:
        inputs = self.tokenizer(observation_text, return_tensors="pt", truncation=True, max_length=2048)
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        tail = full_text[len(observation_text) :].strip() if full_text.startswith(observation_text) else full_text.strip()

        parsed = self._parse_action(tail)
        if parsed is None:
            action_text = "ACTION:\naccel_pedal: 0.00\nbrake_pedal: 0.00"
            return AgentOutput(
                raw_text=tail,
                action_text=action_text,
                accel_pedal=0.0,
                brake_pedal=0.0,
                parse_ok=False,
            )

        accel, brake = parsed
        action_text = f"ACTION:\naccel_pedal: {accel:.2f}\nbrake_pedal: {brake:.2f}"
        return AgentOutput(
            raw_text=tail,
            action_text=action_text,
            accel_pedal=accel,
            brake_pedal=brake,
            parse_ok=True,
        )

    @staticmethod
    def _parse_action(text: str) -> tuple[float, float] | None:
        match = ACTION_REGEX.search(text)
        if not match:
            return None

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
        }
