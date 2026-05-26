"""
Student Analyzer: loads the QLoRA-fine-tuned LLaMA-3.2-1B adapter and provides
the same failure-analysis interface as the Grok-4 teacher — but locally, with no API cost.

Usage:
    from distillation.student_analyzer import StudentAnalyzer
    analyzer = StudentAnalyzer(adapter_path="models/student_lora")
    result = analyzer.analyze(trace, task_description)
"""

import json
from pathlib import Path
from typing import Optional

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_NEW_TOKENS = 300

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with input. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

INSTRUCTION = (
    "Analyze this failed agent execution trace. "
    "Identify the failure category and provide a corrective strategy."
)


class StudentAnalyzer:
    """
    Local failure analyzer using the QLoRA-fine-tuned LLaMA student.
    Drops in as a replacement for the Grok-4 teacher API call — ~95% cheaper.
    """

    def __init__(self, adapter_path: str, base_model: str = BASE_MODEL):
        self.adapter_path = Path(adapter_path)
        self.base_model = base_model
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load the model and adapter on first use."""
        if self._model is not None:
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}. "
                "Install with: pip install torch transformers peft bitsandbytes"
            )

        print(f"Loading student model from {self.adapter_path} ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.adapter_path), trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        print("Student model loaded.")

    def analyze(self, trace: dict, task_description: str) -> Optional[dict]:
        """
        Analyze a failed trace. Returns dict with failure_category,
        root_cause, corrective_strategy — same schema as grok_teacher output.
        """
        self._load()
        import torch

        input_text = (
            f"Task: {task_description}\n"
            f"Steps taken: {len(trace.get('steps', []))}\n"
            f"Final status: {trace.get('status', 'failed')}\n"
            f"Last actions: {json.dumps(trace.get('steps', [])[-3:])[:600]}"
        )
        prompt = PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input=input_text)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        return self._parse_response(response)

    def _parse_response(self, response: str) -> dict:
        """Parse free-text response into structured dict."""
        result = {
            "failure_category": "reasoning_error",
            "root_cause": "",
            "corrective_strategy": response,
        }
        lines = response.splitlines()
        for line in lines:
            low = line.lower()
            if low.startswith("failure category:"):
                result["failure_category"] = line.split(":", 1)[1].strip()
            elif low.startswith("root cause:"):
                result["root_cause"] = line.split(":", 1)[1].strip()
            elif low.startswith("corrective strategy:"):
                result["corrective_strategy"] = line.split(":", 1)[1].strip()
        return result

    @classmethod
    def from_manifest(cls, output_dir: str) -> "StudentAnalyzer":
        """Load student using training_manifest.json written by qlora_trainer."""
        manifest_path = Path(output_dir) / "training_manifest.json"
        base_model = BASE_MODEL
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            base_model = manifest.get("base_model", BASE_MODEL)
        return cls(adapter_path=output_dir, base_model=base_model)
