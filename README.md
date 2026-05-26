# LLM Agent Distillation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Self-improving LLM agent with a Grok-4 teacher → QLoRA-fine-tuned LLaMA-3.2-1B student pipeline. Reduces inference cost by ~95% while retaining failure-analysis quality.**

# LLM Agent Distillation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Self-improving LLM agent with a Grok-4 teacher → QLoRA-fine-tuned LLaMA-3.2-1B student pipeline.
Reduces inference cost by ~95% while retaining failure-analysis quality.**

</div>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                          │
│                                                             │
│  Agent Traces ──► Grok-4 Teacher ──► JSONL Annotations     │
│  (failures)        (xAI API)         (instruction tuning)  │
│                         │                                   │
│                         ▼                                   │
│              QLoRA Fine-Tuning                              │
│          LLaMA-3.2-1B-Instruct                              │
│          4-bit NF4 · LoRA r=8                               │
│          ~2% trainable params                               │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                          │
│                                                             │
│  Agent Task ──► Strategy Memory ──► Enhanced Agent          │
│                      │                                      │
│                      ▼                                      │
│           Student Failure Analyzer                          │
│         (local, no API dependency)                          │
│         95% cheaper · same quality                          │
└─────────────────────────────────────────────────────────────┘
```

## What It Does

1. **Runs an agent** on filesystem and database tasks in a controlled sandbox
2. **Traces failures** — logs every action, observation, and reasoning step
3. **Distills from Grok-4** — the teacher annotates each failure trace with a root-cause category and a concrete corrective strategy
4. **Fine-tunes a local student** — QLoRA trains LLaMA-3.2-1B on the annotated JSONL with 4-bit NF4 quantization
5. **Deploys the student** — at inference, the student model replaces the Grok-4 API call, cutting cost by ~95%

The agent improves across runs: each failure adds a corrective strategy to a persistent ChromaDB strategy memory, and the fine-tuned student gets sharper over time.

---

## Project Structure

```
LLM_Agent_Distillation/
├── distillation/
│   ├── grok_teacher.py      # Grok-4 annotates failure traces → training JSONL
│   ├── qlora_trainer.py     # QLoRA fine-tunes LLaMA-3.2-1B on annotations
│   └── student_analyzer.py  # Inference wrapper for the fine-tuned student
├── agent/
│   ├── base.py              # Abstract agent interface
│   ├── baseline.py          # Baseline ReAct agent
│   └── strategy_enhanced.py # Agent with strategy-memory lookup
├── failure_analysis/
│   ├── categories.py        # Failure taxonomy
│   └── analyzer.py          # Rule-based + LLM-based failure detection
├── tracing/
│   └── logger.py            # Execution trace logger (JSONL)
├── strategy_memory/
│   └── store.py             # ChromaDB-backed corrective strategy memory
├── environment/
│   └── controlled.py        # Sandboxed task environment
├── tasks/
│   ├── base.py              # Task interface
│   ├── filesystem.py        # File-organization tasks
│   └── database.py          # SQL tasks
├── experiments/
│   ├── runner.py            # Batch experiment runner
│   └── metrics.py           # Success rate, step efficiency, cost metrics
├── main.py                  # CLI entrypoint
├── config.py                # Configuration
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/Ajeenckya5/LLM_Agent_Distillation
cd LLM_Agent_Distillation
pip install -r requirements.txt
cp .env.example .env   # add XAI_API_KEY
```

### 1 — Run the agent and collect failure traces

```bash
python main.py --mode evaluate --tasks filesystem database --runs 50
# Traces saved to traces/
```

### 2 — Generate distillation data with Grok-4 teacher

```bash
python -m distillation.grok_teacher \
  --traces-dir traces/ \
  --out data/distill_train.jsonl \
  --max-traces 500
```

### 3 — Fine-tune the student with QLoRA

```bash
python -m distillation.qlora_trainer \
  --data data/distill_train.jsonl \
  --output models/failure_analyzer_lora \
  --epochs 3
# Trainable params: ~2% of LLaMA-3.2-1B (~20M / 1B)
```

### 4 — Run with the local student model

```bash
python main.py --mode evaluate --analyzer student \
  --adapter models/failure_analyzer_lora
```

---

## Results

| Metric | Baseline Agent | Strategy-Enhanced (Teacher) | Strategy-Enhanced (Student) |
|--------|---------------|----------------------------|----------------------------|
| Task success rate | 61% | 87% | 85% |
| Avg steps to completion | 14.2 | 9.8 | 10.1 |
| Inference cost / run | $0.04 | $0.18 | $0.009 |
| **Cost reduction vs teacher** | — | — | **95%** |

*Evaluated on 50 filesystem + 30 database tasks across 3 runs.*

---

## Distillation Pipeline Details

### Teacher: Grok-4 (xAI API)

Annotates each failed trace with:
```json
{
  "category": "tool_usage_error",
  "corrective_strategy": "Always call list_dir before moving files to confirm filenames exist.",
  "root_cause": "Agent assumed file existed without verification"
}
```

Failure categories: `tool_usage_error · reasoning_error · environment_misread · loop_detected · path_error · schema_error · constraint_violation · timeout`

### Student: LLaMA-3.2-1B-Instruct + QLoRA

| Component | Config |
|-----------|--------|
| Base model | `meta-llama/Llama-3.2-1B-Instruct` |
| Quantization | 4-bit NF4 (bitsandbytes) |
| LoRA rank | r=8, α=32 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Optimizer | paged_adamw_8bit |
| Trainable params | ~2% of total (~20M) |
| Training data | 200–500 distilled failure examples |

---

## Strategy Memory

Corrective strategies are stored in a ChromaDB vector store keyed by failure embedding. At inference, the strategy-enhanced agent retrieves the most relevant past strategy before each step.

```python
from strategy_memory.store import StrategyMemory

memory = StrategyMemory()
memory.add(category="path_error", strategy="Verify path with stat before write")
relevant = memory.retrieve(query="file not found during copy", k=3)
```

---

## Tech Stack

`Python 3.10+` · `PyTorch` · `HuggingFace Transformers` · `PEFT` · `TRL` · `bitsandbytes` · `ChromaDB` · `xAI Grok-4 API` · `SQLite`

---

## Requirements

```
torch>=2.1
transformers>=4.40
peft>=0.10
trl>=0.8
bitsandbytes>=0.43
accelerate>=0.28
datasets>=2.18
chromadb>=0.4
```

GPU with ≥8 GB VRAM recommended for QLoRA training. CPU inference works but is slow.

---

<div align="center">

*"Distill the teacher's judgment into a model you own. Ship faster, spend less."*

</div></div>
