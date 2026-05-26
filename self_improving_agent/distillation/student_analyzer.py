"""
QLoRA student failure analyzer.

The student is a LoRA adapter trained from Grok-labeled failed traces. It
implements the same high-level analyzer role as the LLM failure analyzer, but
returns a corrective strategy directly so the runtime can avoid a second
teacher-model call when the local adapter is available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..analysis.failure_analyzer import FailureAnalyzer


FAILURE_TYPE_MAP = {
    "tool_usage_error": "tool_misuse",
    "reasoning_error": "incorrect_reasoning",
    "environment_misread": "environmental_change",
    "loop_detected": "circular_loop",
    "path_error": "environmental_change",
    "schema_error": "tool_misuse",
    "constraint_violation": "instruction_misinterpretation",
    "timeout": "context_truncation",
    "unknown": "other",
}


class StudentFailureAnalyzer:
    """
    Local failure analyzer using a QLoRA fine-tuned LLaMA-3.2-1B adapter.

    Expected output shape matches the project failure-analysis contract:
    {
        "failure_type": str,
        "failed_steps": list[int],
        "pattern_summary": str,
        "raw_trace_excerpt": str,
        "corrective_strategy": str,
        "tags": list[str],
        "analysis_source": "student_qlora",
    }
    """

    def __init__(
        self,
        adapter_path: str | Path = "models/failure_analyzer_lora",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> None:
        self.adapter_path = Path(adapter_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.fallback = FailureAnalyzer()
        self._model = None
        self._tokenizer = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StudentFailureAnalyzer":
        distill_cfg = config.get("distillation", {})
        adapter_path = os.getenv(
            "STUDENT_ADAPTER_PATH",
            distill_cfg.get("student_adapter_path", "models/failure_analyzer_lora"),
        )
        return cls(
            adapter_path=adapter_path,
            max_new_tokens=int(distill_cfg.get("max_new_tokens", 256)),
            temperature=float(distill_cfg.get("temperature", 0.1)),
        )

    def analyze(self, task_or_trace: Any, trace: Any | None = None) -> Dict[str, Any]:
        """
        Analyze either `(task, trace)` or `trace`.

        The trace-only signature keeps this compatible with DatasetCollector,
        while the `(task, trace)` signature lets the main evaluator pass the
        task description explicitly.
        """
        if trace is None:
            trace = task_or_trace
            task = {
                "id": getattr(trace, "task_id", "unknown"),
                "description": getattr(trace, "task_description", ""),
            }
        else:
            task = task_or_trace if isinstance(task_or_trace, dict) else {}

        heuristic_hint = self.fallback.analyze(trace)
        prompt = self._format_prompt(task, trace, heuristic_hint)
        raw = self._infer(prompt)
        parsed = self._parse_response(raw)

        teacher_category = parsed.get("category") or parsed.get("failure_type") or "unknown"
        failure_type = FAILURE_TYPE_MAP.get(str(teacher_category), str(teacher_category))
        if not failure_type:
            failure_type = heuristic_hint.get("failure_type", "other")

        failed_steps = parsed.get("failed_steps") or heuristic_hint.get("failed_steps", [])
        failed_steps = [int(s) for s in failed_steps if str(s).isdigit()]

        corrective_strategy = str(
            parsed.get("corrective_strategy")
            or parsed.get("strategy_text")
            or "Retry with explicit state verification before acting."
        ).strip()

        pattern_summary = str(
            parsed.get("root_cause")
            or parsed.get("pattern_summary")
            or heuristic_hint.get("pattern_summary", "")
        ).strip()

        tags = [failure_type]
        tags.extend(str(tag).strip().lower().replace(" ", "_") for tag in parsed.get("tags", []) if tag)

        return {
            "failure_type": failure_type,
            "failed_steps": failed_steps,
            "pattern_summary": pattern_summary,
            "raw_trace_excerpt": self._trace_to_text(trace, max_steps=10),
            "corrective_strategy": corrective_strategy,
            "tags": list(dict.fromkeys(tags))[:8],
            "analysis_source": "student_qlora",
            "raw_student_response": raw,
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Install distillation dependencies with: "
                "pip install -r requirements-distillation.txt"
            ) from exc

        manifest_path = self.adapter_path / "training_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No training manifest found at {manifest_path}. "
                "Train the adapter with self_improving_agent.distillation.qlora_trainer first."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base_model = manifest["base_model"]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            token=os.getenv("HF_TOKEN"),
        )
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def _infer(self, prompt: str) -> str:
        import torch

        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = ids[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _format_prompt(
        self,
        task: Dict[str, Any],
        trace: Any,
        heuristic_hint: Dict[str, Any],
    ) -> str:
        task_description = task.get("description") or getattr(trace, "task_description", "")
        if not task_description:
            task_description = f"Task id: {task.get('id') or getattr(trace, 'task_id', 'unknown')}"

        user_content = (
            f"Task: {task_description}\n\n"
            f"Heuristic hint: {json.dumps(heuristic_hint, ensure_ascii=False)}\n\n"
            f"Execution trace:\n{self._trace_to_text(trace, max_steps=15)}\n\n"
            "Respond with JSON only:\n"
            "{"
            '"category": "...", '
            '"failed_steps": [1], '
            '"root_cause": "...", '
            '"corrective_strategy": "...", '
            '"tags": ["..."]'
            "}"
        )
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are an expert long-horizon agent failure analyst. Respond with JSON only."
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_content}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        return {}

    @classmethod
    def _trace_to_text(cls, trace: Any, max_steps: int = 15) -> str:
        lines: List[str] = []
        for step in list(cls._iter_steps(trace))[:max_steps]:
            observation = step["observation"]
            if len(observation) > 300:
                observation = observation[:300] + "..."
            lines.append(
                f"Step {step['step']}:\n"
                f"  Thought: {step['thought'][:180]}\n"
                f"  Action: {step['action'][:180]}\n"
                f"  Observation: {observation}"
            )
        return "\n".join(lines) if lines else "(no trace steps)"

    @staticmethod
    def _iter_steps(trace: Any) -> Iterable[Dict[str, str]]:
        steps = trace.get("steps", []) if isinstance(trace, dict) else getattr(trace, "steps", [])
        for raw_step in steps:
            if isinstance(raw_step, dict):
                action = raw_step.get("action", "")
                thought = raw_step.get("thought") or raw_step.get("reasoning", "")
                observation = raw_step.get("observation", "")
                step_num = raw_step.get("step", "?")
            else:
                action = getattr(raw_step, "action", "")
                thought = getattr(raw_step, "thought", getattr(raw_step, "reasoning", ""))
                observation = getattr(raw_step, "observation", "")
                step_num = getattr(raw_step, "step", "?")

            if isinstance(action, dict):
                action = action.get("action") or action.get("tool") or json.dumps(action)

            yield {
                "step": str(step_num),
                "thought": str(thought or ""),
                "action": str(action or ""),
                "observation": str(observation or ""),
            }
