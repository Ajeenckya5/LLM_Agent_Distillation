# LLM Agent Distillation

Full self-improving long-horizon agent project with a dedicated
teacher-student distillation path.

The project compares a baseline LLM agent against a strategy-enhanced agent
on multi-step filesystem and SQL tasks. Failed runs are logged as traces,
classified by a failure analyzer, converted into corrective strategies, and
stored in vector memory for later retrieval. The newer distillation pipeline
uses a Grok teacher to label failed traces, then fine-tunes a local
LLaMA-3.2-1B student with QLoRA so failure analysis can run without repeated
teacher API calls.

## What This Repo Contains

- Long-horizon task suite with automatic verifiers.
- Controlled filesystem and SQLite execution environment.
- Baseline Plan-Act-Observe agent.
- Strategy-enhanced agent with ChromaDB strategy memory.
- Trace logging for every attempted task.
- Rule-based and LLM-based failure analysis.
- Grok teacher data generation for failed traces.
- QLoRA fine-tuning pipeline for a local student failure analyzer.
- Scripted benchmark that validates the task/environment layer without any API key.
- Original project report and presentation assets under `docs/`.

## Architecture

```text
filesystem / SQL tasks
        |
        v
controlled environment + verifiers
        |
        v
baseline agent ------------------------------+
                                            |
strategy-enhanced agent + retrieved memory --+--> execution traces
                                                   |
                                                   v
                                      failure analysis
                                      | rule checks
                                      | OpenAI analyzer fallback
                                      | optional QLoRA student
                                                   |
                                                   v
                                      corrective strategies
                                                   |
                                                   v
                                      ChromaDB strategy memory

Distillation path:
failed traces -> Grok teacher labels -> JSONL training data
              -> QLoRA LLaMA-3.2-1B adapter
              -> local student failure analyzer
```

## Repository Layout

```text
agent/                 Agent interfaces and baseline/strategy-enhanced agents
distillation/          Grok teacher, QLoRA trainer, local student analyzer
environment/           Controlled task execution environment
experiments/           Experiment runner, metrics, and plotting
failure_analysis/      Failure taxonomy and analyzer
strategy_memory/       ChromaDB-backed corrective strategy retrieval
tasks/                 Filesystem and database task definitions/verifiers
tracing/               JSON trace logger
docs/                  Project report and presentation assets
bench_scripted.py      No-API benchmark for environment and task correctness
config.py              Runtime paths, model names, and feature flags
main.py                CLI entrypoint for LLM experiments
requirements.txt       Base project dependencies
requirements-distillation.txt
                       Optional QLoRA training dependencies
```

## Setup

```bash
git clone https://github.com/Ajeenckya5/LLM_Agent_Distillation.git
cd LLM_Agent_Distillation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add keys to `.env` only for the workflows that need them:

```text
OPENAI_API_KEY=...
XAI_API_KEY=...
```

## Validate Without API Keys

Run the deterministic scripted benchmark:

```bash
python bench_scripted.py
```

This exercises all six current tasks, the controlled environment, task
verifiers, and strategy-memory retrieval without calling an LLM.

## Run LLM Experiments

```bash
python main.py run --attempts 2 --max-steps 15
```

Useful options:

```bash
python main.py run --no-plot
python main.py run --output results
python main.py run --sandbox-dir /tmp/agent_sandbox
```

The runner executes the baseline agent first. Failed traces are analyzed and
converted into strategies, then the strategy-enhanced agent retrieves the most
relevant strategies before attempting similar tasks.

## Distillation Pipeline

Install optional training dependencies in a CUDA-capable environment:

```bash
pip install -r requirements-distillation.txt
```

Generate teacher labels from failed traces:

```bash
python -m distillation.grok_teacher \
  --traces-dir traces \
  --out data/distill_train.jsonl \
  --max-traces 500
```

Fine-tune the local student with QLoRA:

```bash
python -m distillation.qlora_trainer \
  --data data/distill_train.jsonl \
  --output models/failure_analyzer_lora \
  --epochs 3
```

Use the student analyzer at runtime:

```bash
USE_STUDENT_ANALYZER=1 python main.py run
```

If `USE_STUDENT_ANALYZER=1` is not set, or if the adapter is missing, the
runner falls back to the standard `FailureAnalyzer`.

## Failure Categories

The base analyzer classifies failed traces into:

- `planning_error`
- `memory_limitation`
- `instruction_misinterpretation`
- `environmental_change`
- `false_assumption`

The distillation teacher prompt also supports more operational labels such as
`tool_usage_error`, `path_error`, `schema_error`, `constraint_violation`, and
`timeout` for richer student training examples.

## Generated Files

The following runtime artifacts are intentionally ignored:

- `.env`
- `sandbox/`
- `traces/`
- `results/`
- `chroma_db/`
- `models/`
- `data/*.jsonl`
- Python cache files

This keeps the repo focused on reproducible source code and documentation.
