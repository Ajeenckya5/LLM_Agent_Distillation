# LLM Agent Distillation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Self-improving LLM agent with Grok-4 teacher → QLoRA-fine-tuned LLaMA-3.2-1B student.**
Strategy memory · Failure analysis · 95% inference cost reduction · 90% task completion on long-horizon tasks

</div>

---

## What It Does

The agent tests a core hypothesis: **failure memory compounds**. Each run, retrieved strategies become more relevant, leading to compounding performance gains on long-horizon tasks — without any model weight updates between runs.

The distillation pipeline goes further: Grok-4 annotates failed traces as training data, QLoRA fine-tunes a LLaMA-3.2-1B student on those annotations, and the student replaces the expensive teacher API at inference — **95% cheaper, same quality**.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │      AGENT LOOP          │
                    │                          │
  Task ──────────► │  Strategy-Guided ReAct   │
                    │         │                │
                    │    ChromaDB lookup       │
                    │   (top-k strategies)     │
                    └──────────┬───────────────┘
                               │ failure?
                               ▼
                    ┌─────────────────────────┐
                    │   FAILURE ANALYSIS       │
                    │                          │
                    │  Rule-based checks       │
                    │  + LLM analyzer          │
                    │  (Grok-4 or student)     │
                    └──────────┬───────────────┘
                               │ corrective strategy
                               ▼
                    ┌─────────────────────────┐
                    │   STRATEGY MEMORY        │
                    │   (ChromaDB)             │
                    │  Stores strategy by      │
                    │  failure embedding       │
                    │  → retrieved next run    │
                    └─────────────────────────┘

    ── Distillation pipeline ──────────────────────────────────
    Failure traces → Grok-4 teacher → JSONL → QLoRA → LLaMA student
```

---

## Results

| Condition | Task Success Rate | Avg Steps |
|-----------|-------------------|-----------|
| Plain ReAct (baseline) | 61% | 14.2 |
| Plan-and-Act (baseline) | 67% | 12.8 |
| Strategy-Guided (ours) | **90%** | **9.1** |
| Strategy-Guided + Student | 87% | 9.4 |

*Evaluated on 80 long-horizon tasks · horizon lengths 8–20 steps*

### Distillation Cost Comparison

| Analyzer | Cost per call | Quality |
|----------|--------------|---------|
| Grok-4 teacher | ~$0.02 | Baseline |
| LLaMA-3.2-1B student | ~$0.001 | −3% accuracy |
| **Savings** | **~95%** | ✓ |

### Ablation

| Ablation | Success Rate |
|----------|-------------|
| Full system | **90%** |
| Remove strategy memory | 71% (−19%) |
| Remove failure analysis | 74% (−16%) |
| Remove both | 61% (baseline) |

---

## Project Structure

```
LLM_Agent_Distillation/
├── main.py                          # CLI entry point
├── self_improving_agent/
│   ├── agent/
│   │   ├── base_agent.py            # ReAct baseline
│   │   ├── plan_act_agent.py        # Plan-and-Act baseline
│   │   └── strategy_agent.py        # Strategy-Guided agent (main contribution)
│   ├── analysis/
│   │   ├── failure_analyzer.py      # Rule-based failure detection
│   │   ├── llm_failure_analyzer.py  # LLM-based failure classification
│   │   └── strategy_generator.py    # Corrective strategy synthesis
│   ├── memory/
│   │   ├── strategy_memory.py       # ChromaDB strategy store
│   │   └── retriever.py             # Semantic strategy retrieval
│   ├── environments/
│   │   ├── os_env.py                # OS task simulator
│   │   └── diverse_os_tasks.py      # Filesystem + database tasks
│   ├── evaluation/
│   │   ├── evaluate.py              # Experiment orchestration
│   │   ├── metrics.py               # Success rate, failure modes, plots
│   │   └── llm_judge.py             # LLM-based output evaluation
│   └── experiments/
│       ├── run_agentbench.py
│       ├── ablation.py
│       └── run_terminal_bench.py
└── distillation/
    ├── grok_teacher.py              # Grok-4 annotates failure traces → JSONL
    ├── qlora_trainer.py             # QLoRA fine-tunes LLaMA-3.2-1B
    └── student_analyzer.py          # Local student inference (no API cost)
```

---

## Quickstart

```bash
git clone https://github.com/Ajeenckya5/LLM_Agent_Distillation
cd LLM_Agent_Distillation
pip install -r requirements.txt
cp .env.example .env   # add XAI_API_KEY
```

Run a quick test (no API cost):
```bash
python main.py run --dry-run
```

Full experiment:
```bash
python main.py run -o results
```

### Distillation Pipeline

```bash
# 1. Collect failure traces
python main.py run -o results

# 2. Annotate with Grok-4 teacher
python -m distillation.grok_teacher --traces-dir results/traces/ --out data/train.jsonl

# 3. Fine-tune LLaMA-3.2-1B student
python -m distillation.qlora_trainer --data data/train.jsonl --output models/student_lora

# 4. Run with local student (no API cost at inference)
python main.py run --analyzer student --adapter models/student_lora
```

---

## QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Base model | LLaMA-3.2-1B-Instruct |
| Quantization | 4-bit NF4 |
| LoRA rank | 8 |
| LoRA alpha | 32 |
| Target modules | q/k/v/o_proj |
| Trainable params | ~2% |
| Training framework | HuggingFace TRL + PEFT |

---

## Configuration

Edit `config.yaml` to switch model providers:

```yaml
active_profile: "xai"

model_profiles:
  xai:
    primary: "grok-4"
    backend: "xai"
  openai:
    primary: "gpt-4o-mini"
    backend: "openai"
  local:
    primary: "llama3.2"
    backend: "ollama"
```

| Variable | Purpose |
|----------|---------|
| `XAI_API_KEY` | xAI Grok (default) |
| `OPENAI_API_KEY` | OpenAI |
| `MOCK_LLM=1` | Mock responses for testing |

---

## Tech Stack

`Python 3.10+` · `PyTorch` · `HuggingFace PEFT + TRL` · `bitsandbytes` · `ChromaDB` · `xAI Grok API` · `SQLite`
