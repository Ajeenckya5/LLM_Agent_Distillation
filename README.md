# LLM Agent Distillation

Full self-improving LLM agent project for long-horizon tasks, with the
dedicated Grok teacher -> QLoRA student distillation pipeline included in the
same clean repository.

The base system learns from failed task executions without retraining the main
agent. It records traces, analyzes failures, generates corrective strategies,
stores them in persistent memory, and retrieves relevant strategies for future
tasks. The distillation path adds a local student failure analyzer so repeated
failure-analysis calls can be served by a fine-tuned adapter instead of a
teacher API call.

## What This Project Does

- Runs three agent strategies: ReAct, Plan-and-Act, and Strategy-Guided.
- Evaluates long-horizon filesystem, database, OS, WebArena-style, and Terminal-Bench tasks.
- Logs execution traces and labels failures.
- Stores corrective strategies in persistent memory with semantic retrieval.
- Runs prompt/evaluator ablations for failure analysis and strategy generation.
- Generates Grok-labeled distillation data from failed traces.
- Fine-tunes a LLaMA-3.2-1B student with QLoRA.
- Lets the runtime use the student analyzer through `analysis.failure_analyzer: "student"` or `USE_STUDENT_ANALYZER=1`.

## Quick Start

```bash
git clone https://github.com/Ajeenckya5/LLM_Agent_Distillation.git
cd LLM_Agent_Distillation
python -m venv .venv
source .venv/bin/activate
pip install -r self_improving_agent/requirements.txt
cp .env.example .env
```

Set one backend in `.env`:

```bash
XAI_API_KEY=xai-your-key
# or OPENAI_API_KEY=...
# or ANTHROPIC_API_KEY=...
# or OLLAMA_BASE_URL=http://localhost:11434
# or MOCK_LLM=1 for tests/smoke runs
```

## Run Experiments

Controlled filesystem/database tasks:

```bash
python main.py run --dry-run
python main.py run --attempts 1 -o results
```

AgentBench OS tasks:

```bash
python main.py agentbench --dry-run
python main.py agentbench --n-tasks 100
```

Terminal-Bench:

```bash
python main.py terminalbench --n-tasks 10 --n-concurrent 2
```

Ablation and prompt evaluation:

```bash
python main.py ablation --dry-run
python main.py member2-eval --profile xai
python main.py member2-ablation --profile xai
```

## Distillation And QLoRA

Install optional training dependencies in a CUDA-capable environment:

```bash
pip install -r requirements-distillation.txt
```

Generate teacher labels from failed traces:

```bash
python -m self_improving_agent.distillation.grok_teacher \
  --traces-dir results \
  --out data/distill_train.jsonl \
  --max-traces 500
```

Fine-tune the local student:

```bash
python -m self_improving_agent.distillation.qlora_trainer \
  --data data/distill_train.jsonl \
  --output models/failure_analyzer_lora \
  --epochs 3
```

Use the student analyzer:

```bash
USE_STUDENT_ANALYZER=1 python main.py run --dry-run
```

Equivalent config:

```yaml
analysis:
  failure_analyzer: "student"

distillation:
  student_adapter_path: "models/failure_analyzer_lora"
```

If the student adapter is not trained yet, keep `analysis.failure_analyzer` set
to `llm` or `heuristic`.

## Project Structure

```text
LLM_Agent_Distillation/
├── main.py                         Unified CLI entry point
├── requirements-distillation.txt   Optional QLoRA dependencies
├── self_improving_agent/
│   ├── agent/                      ReAct, Plan-and-Act, Strategy-Guided agents
│   ├── analysis/                   Heuristic and LLM failure analysis
│   ├── distillation/               Grok teacher, QLoRA trainer, student analyzer
│   ├── memory/                     SQLite strategy memory and semantic retriever
│   ├── environments/               OS, controlled, and web environments
│   ├── evaluation/                 Metrics, plots, experiment loop
│   ├── experiments/                AgentBench, WebArena, Terminal-Bench, ablations
│   ├── prompts/                    Failure-analysis and strategy prompts
│   ├── tasks/                      Controlled filesystem/database task suite
│   ├── tests/                      Pytest suite with mock LLM support
│   └── config.yaml                 Model profiles and experiment settings
└── fangkai/                        Additional mock-agent experiments
```

## Tests

No API keys are needed for the test suite:

```bash
MOCK_LLM=1 python -m pytest self_improving_agent/tests/ -v
```

Compile check:

```bash
python -m compileall main.py self_improving_agent fangkai
```

## Configuration

Edit `self_improving_agent/config.yaml` to change:

- Active model profile: `xai`, `haiku`, `groq`, or `ollama`.
- Agent settings: `max_steps`, `temperature`, `max_tokens`.
- Memory settings: `top_k`, `similarity_threshold`, database path.
- Analysis mode: `llm`, `heuristic`, or `student`.
- Distillation paths: teacher model, base model, adapter output, training JSONL.

## Runtime Outputs

Generated files are intentionally ignored:

- `.env`
- `sandbox/`
- `results/`
- `traces/`
- `logs/`
- `models/`
- `data/`
- local SQLite databases and model checkpoints

## References

```bibtex
@inproceedings{madaan2023selfrefine,
  title={{SELF-REFINE}: Iterative Refinement with Self-Feedback},
  author={Madaan, Aman and Tandon, Niket and Gupta, Prakhar and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@article{liu2023agentbench,
  title={{AgentBench}: Evaluating LLMs as Agents},
  author={Liu, Xiao and Yu, Hao and Zhang, Hanchen and others},
  journal={arXiv preprint arXiv:2308.03688},
  year={2023}
}

@inproceedings{zhou2024webarena,
  title={{WebArena}: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and others},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
