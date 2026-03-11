# LLMJudgeTempCausal

Causal analysis framework for evaluating how **temperature** affects **LLM-as-a-Judge** stability and performance, using the [MT-Bench Human Judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) dataset (3,355 pairwise human annotations across 80 multi-turn questions and 6 models) as ground truth.

## Core Research Questions

1. **Score & ranking stability** — Are absolute scores and relative rankings from LLM judges stable across temperatures?
2. **Judge type impact** — How do pairwise comparison, single-answer scoring (1–10), and reference-guided scoring differ in agreement and performance at different temperatures?
3. **Prompt engineering effects** — How do position swap, few-shot, chain-of-thought (CoT), reference-guided, and multi-turn prompting strategies affect judge performance?
4. **Model size effects** — How does judge model size (e.g., 1B vs 7B vs 70B) interact with temperature sensitivity?

## Causal Framework

The project builds a **Directed Acyclic Graph (DAG)** and uses **do-calculus** to isolate the causal effect of temperature:

```
Temperature ──→ Judgment ──→ Metrics
     ↑              ↑           ↑
JudgeModelSize ─────┤           │
PromptType ─────────┘           │
InputResponses ────→ Judgment   │
HumanJudgment ─────────────→ Metrics
```

**ATE estimation**: `ATE = E[Metrics | do(T=0.0)] − E[Metrics | do(T=1.0)]`

Methods: Simple difference-in-means, ANOVA, Double Machine Learning (DML via `econml`), stratified analysis by prompt type and model size.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Agreement S1** | Match rate with human labels (ties & disagreements count as agreement) |
| **Agreement S2** | Match rate among non-tie cases only |
| **Consistency** | Same verdict when A/B positions are swapped |
| **Error Rate** | Fraction of outputs that fail to parse (not `[[A]]`/`[[B]]`/`[[C]]` or `[[1-10]]`) |
| **Position Bias** | `P(first wins) − P(second wins)` — deviation from fair 50/50 |
| **Score Statistics** | Mean, std of absolute scores (single-answer & reference-guided) |
| **Ranking Correlation** | Kendall τ and Spearman ρ between judge scores and human preference |

## Installation

Requires **Python ≥ 3.11** and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url> && cd LLMJudgeTempCausal
uv sync
```

This installs all dependencies: `datasets`, `openai`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `econml`, `pgmpy`, `matplotlib`, `seaborn`, `tqdm`, `click`.

## Quick Start

### 1. Start a local LLM server

Using **vLLM**:
```bash
uv run vllm serve google/gemma-3-1b-it --gpu-memory-utilization 0.5
# Server at http://localhost:8000
```

Or **SGLang**:
```bash
python -m sglang.launch_server --model meta-llama/Llama-3-8B-Instruct --port 8001
```

### 2. Run a quick test

```bash
uv run python main.py --quick-test
```

This runs a minimal experiment (5 samples, 2 repeats, 3 temperatures, 2 judge types, 2 prompt variants) and outputs to `results_test/`.

### 3. Run a full experiment

```bash
uv run llmjudge run \
  -m google/gemma-3-1b-it \
  -u http://localhost:8000 \
  -s 1B \
  -b vllm \
  -n 100 \
  -r 10 \
  -o results
```

## CLI Reference

### `llmjudge run` — Run experiment

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | *(required, repeatable)* | Model name (e.g., `google/gemma-3-1b-it`) |
| `-u, --base-url` | *(required, repeatable)* | Server URL (e.g., `http://localhost:8000`) |
| `-k, --api-key` | `EMPTY` | API key per model |
| `-b, --backend` | `vllm` | Backend: `vllm`, `sglang`, `openai` |
| `-s, --model-size` | *(auto)* | Size label: `1B`, `7B`, `70B` |
| `-t, --temperatures` | `0.0,0.2,0.4,0.6,0.8,1.0` | Comma-separated temperatures |
| `-j, --judge-types` | `pairwise,single_answer,reference_guided` | Judge types |
| `-p, --prompt-variants` | `baseline,position_swap,few_shot,cot,reference_guided,multi_turn` | Prompt variants |
| `-r, --num-repeats` | `10` | Repeats per config |
| `-n, --sample-size` | `100` | Pairs to sample from dataset |
| `--seed` | `42` | Random seed |
| `--top-p` | `0.95` | Top-p sampling |
| `--max-tokens` | `1024` | Max generation tokens |
| `-o, --output-dir` | `results` | Output directory |
| `-v, --verbose` | off | Debug logging |

### `llmjudge analyze` — Re-analyze existing results

```bash
uv run llmjudge analyze -d results
```

Re-runs metric aggregation, causal analysis (DAG + DML + ATE), and generates all plots from a saved `results.csv`.

### `llmjudge dag` — Generate causal DAG only

```bash
uv run llmjudge dag -o results/causal_dag.png
```

## Multi-Model Example

```bash
# Compare a small and large model
uv run llmjudge run \
  -m google/gemma-3-1b-it -u http://localhost:8000 -s 1B -b vllm \
  -m meta-llama/Llama-3-70B-Instruct -u http://server2:8001 -s 70B -b vllm \
  -n 500 -r 5 -o results_multi

# Use OpenAI API
uv run llmjudge run \
  -m gpt-4o -u https://api.openai.com -k "$OPENAI_API_KEY" -s GPT4o -b openai \
  -n 200 -r 10 -o results_openai
```

## Output Files

After running an experiment, the output directory contains:

### Data
| File | Content |
|------|---------|
| `results.csv` | Raw per-sample judge results |
| `metrics_by_temperature.csv` | Metrics aggregated by temperature |
| `metrics_by_temp_model.csv` | Metrics by temperature × model |
| `metrics_by_temp_prompt.csv` | Metrics by temperature × prompt variant |
| `metrics_by_temp_judgetype.csv` | Metrics by temperature × judge type |
| `consistency_orig.csv` | Original-order pairwise results |
| `consistency_swap.csv` | Swapped-order pairwise results |
| `analysis.json` | Full causal analysis (ATE, DML, ANOVA, stratified effects) |

### Visualizations
| File | Content |
|------|---------|
| `causal_dag.png` | Causal DAG |
| `plot_metrics_by_temperature.png` | All metrics vs temperature (line plots) |
| `plot_causal_forest.png` | Forest plot of ATE estimates |
| `plot_consistency.png` | Consistency by temperature |
| `plot_position_bias.png` | Position bias by temperature |
| `plot_scores_*.png` | Score distributions by temperature (box plots) |
| `heatmap_agreement_*.png` | Agreement heatmaps (T × Prompt, T × Model) |
| `heatmap_error_rate_*.png` | Error rate heatmaps |
| `heatmap_jt_*.png` | Judge type × temperature heatmaps |
| `heatmap_model_*.png` | Model × temperature heatmaps |

## Project Architecture

```
src/llmjudgetempcausal/
├── config.py       # Configuration: enums, ModelConfig, ExperimentConfig
├── data.py         # MT-Bench dataset loading & pair sampling
├── client.py       # Unified LLM client (vLLM/SGLang/OpenAI + completions fallback)
├── prompts.py      # Prompt templates: 6 variants × 3 judge types
├── judge.py        # Judge execution, output parsing ([[A]], [[B]], [[C]], [[1-10]])
├── metrics.py      # Agreement, Consistency, Error rate, Position bias, score stats
├── causal.py       # DAG (pgmpy), Simple ATE, DML ATE (econml), ANOVA, stratified
├── experiment.py   # Orchestrator: runs all configs, saves results, triggers analysis
├── visualize.py    # 8 plot generators: heatmaps, line plots, forest, box plots
└── cli.py          # Click CLI: run, analyze, dag commands
```

## Supported Configurations

### Backends
- **vLLM** — Local GPU server via `vllm serve`
- **SGLang** — Local GPU server via `sglang.launch_server`
- **OpenAI** — Cloud API (GPT-4o, etc.)

All backends use the OpenAI-compatible chat completions API, with automatic fallback to text completions if chat endpoint fails.

### Judge Types
- **Pairwise** — "Which response is better? Output `[[A]]`, `[[B]]`, or `[[C]]` for tie"
- **Single-answer** — "Rate this response 1–10. Output `[[score]]`"
- **Reference-guided** — "Compare to reference answer and rate 1–10. Output `[[score]]`"

### Prompt Variants
- **Baseline** — Standard judge prompt
- **Position swap** — Randomly flip A/B order to measure position bias
- **Few-shot** — Include 3 example judgments
- **CoT (Chain-of-thought)** — "First compare helpfulness, then coherence, then decide"
- **Reference-guided** — Include factual accuracy hint
- **Multi-turn** — Include full conversation context (turn 1 + turn 2)

## Analysis Pipeline

1. **Descriptive statistics** — Metrics by temperature, model, prompt, judge type
2. **Causal inference** — DML-estimated ATE of temperature on agreement, controlling for prompt/model
3. **Stability testing** — Spearman ρ (scores across temperatures), ANOVA for significance
4. **Stratified effects** — Heterogeneous treatment effects by prompt variant and model size
5. **Visualizations** — Heatmaps, forest plots, line charts, box plots

## License

MIT