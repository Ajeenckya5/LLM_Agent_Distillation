"""
Grok-4 Teacher: annotates failure traces with root-cause category and corrective strategy.
Outputs instruction-following JSONL for QLoRA training of the LLaMA-3.2-1B student.

Usage:
    python -m distillation.grok_teacher \
        --traces-dir results/traces/ \
        --out data/train.jsonl
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_BASE_URL = "https://api.x.ai/v1"
TEACHER_MODEL = "grok-4"

SYSTEM_PROMPT = """You are an expert AI agent failure analyst.
Given a failed task execution trace, produce a JSON object with:
  - "failure_category": one of [repeated_action, circular_loop, missing_tool,
    wrong_tool_order, context_loss, environment_error, reasoning_error]
  - "root_cause": one sentence describing what went wrong
  - "corrective_strategy": a concrete, actionable strategy (2-4 sentences) to avoid
    this failure class in future tasks of the same type
Return only valid JSON, no markdown fences."""

FAILURE_CATEGORIES = [
    "repeated_action",
    "circular_loop",
    "missing_tool",
    "wrong_tool_order",
    "context_loss",
    "environment_error",
    "reasoning_error",
]


def _call_teacher(trace_text: str, task_description: str) -> Optional[dict]:
    """Call Grok-4 to annotate a single failure trace."""
    user_prompt = (
        f"Task: {task_description}\n\n"
        f"Execution trace (failed):\n{trace_text[:3000]}"
    )
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    try:
        resp = httpx.post(
            f"{XAI_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  [teacher] error: {e}")
        return None


def _trace_to_instruction(trace: dict, annotation: dict) -> dict:
    """Format as an instruction-following sample for QLoRA training."""
    instruction = (
        "Analyze this failed agent execution trace. "
        "Identify the failure category and provide a corrective strategy."
    )
    input_text = (
        f"Task: {trace.get('task_description', 'unknown')}\n"
        f"Steps taken: {len(trace.get('steps', []))}\n"
        f"Final status: {trace.get('status', 'failed')}\n"
        f"Last actions: {json.dumps(trace.get('steps', [])[-3:], indent=2)[:800]}"
    )
    output_text = (
        f"Failure category: {annotation['failure_category']}\n"
        f"Root cause: {annotation['root_cause']}\n"
        f"Corrective strategy: {annotation['corrective_strategy']}"
    )
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "failure_category": annotation["failure_category"],
    }


def annotate_traces(traces_dir: str, out_path: str, max_traces: int = 500) -> int:
    """
    Read JSONL failure traces, annotate with Grok-4, write training JSONL.
    Returns number of successfully annotated samples.
    """
    traces_path = Path(traces_dir)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    trace_files = list(traces_path.glob("*.jsonl")) + list(traces_path.glob("*.json"))
    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        return 0

    samples_written = 0
    with out_file.open("w") as f_out:
        for trace_file in trace_files:
            traces = []
            with trace_file.open() as tf:
                for line in tf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        if t.get("status") == "failed":
                            traces.append(t)
                    except json.JSONDecodeError:
                        continue

            print(f"Processing {trace_file.name}: {len(traces)} failures")
            for i, trace in enumerate(traces[:max_traces]):
                if samples_written >= max_traces:
                    break
                task_desc = trace.get("task_description", "complete the given task")
                trace_text = json.dumps(trace.get("steps", []), indent=2)

                annotation = _call_teacher(trace_text, task_desc)
                if annotation and "failure_category" in annotation:
                    sample = _trace_to_instruction(trace, annotation)
                    f_out.write(json.dumps(sample) + "\n")
                    samples_written += 1
                    if (i + 1) % 10 == 0:
                        print(f"  Annotated {i+1}/{len(traces)}")
                    time.sleep(0.5)  # rate limit

    print(f"Done. Wrote {samples_written} training samples to {out_path}")
    return samples_written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grok-4 teacher annotation")
    parser.add_argument("--traces-dir", required=True, help="Directory with JSONL trace files")
    parser.add_argument("--out", required=True, help="Output JSONL path for training data")
    parser.add_argument("--max-traces", type=int, default=500)
    args = parser.parse_args()

    annotate_traces(args.traces_dir, args.out, args.max_traces)
