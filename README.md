# LLMJudgeTempCausal

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Package](https://img.shields.io/badge/package-src%2Fllmjudgetempcausal-informational)](src/llmjudgetempcausal)
[![Hugging Face Profile](https://img.shields.io/badge/HuggingFace-Volavion-yellow?logo=huggingface)](https://huggingface.co/Volavion)
[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/Volavion/eval_temperatures_bench)

A causal analysis and evaluation framework for studying how decoding temperature affects LLM-as-a-Judge reliability, agreement with human preference, consistency, and bias.

This project combines:
- structured LLM judging (pairwise, single-answer, reference-guided),
- prompt strategy interventions (baseline, position-swap, few-shot, CoT, reference-guided, multi-turn),
- multi-temperature sweeps,
- causal effect estimation (simple ATE + DML),
- and reproducible result artifacts (CSV/JSON/plots).

## Hugging Face (Homepage Links)

- Team/Profile: https://huggingface.co/Volavion
- Project dataset: https://huggingface.co/datasets/Volavion/eval_temperatures_bench
- Upstream benchmark used by this project: https://huggingface.co/datasets/lmsys/mt_bench_human_judgments

## Table of Contents

- [1. Why This Project](#1-why-this-project)
- [2. Research Questions](#2-research-questions)
- [3. Causal Design](#3-causal-design)
- [4. Metrics](#4-metrics)
- [5. Repository Layout](#5-repository-layout)
- [6. Requirements](#6-requirements)
- [7. Installation](#7-installation)
- [8. Quick Start](#8-quick-start)
- [9. CLI Reference](#9-cli-reference)
- [10. Script-Based Workflows](#10-script-based-workflows)
- [11. Data Sources and Input Format](#11-data-sources-and-input-format)
- [12. Output Artifacts](#12-output-artifacts)
- [13. Reproducibility Checklist](#13-reproducibility-checklist)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Contributing](#15-contributing)
- [16. Citation](#16-citation)
- [17. License](#17-license)

## 1. Why This Project

LLM-as-a-Judge is now common in evaluation pipelines, but its behavior is sensitive to generation settings. Temperature is often tuned for diversity, yet even small changes can alter:

- absolute scores,
- pairwise winner decisions,
- model ranking order,
- parse validity,
- and judge-side bias patterns.

LLMJudgeTempCausal is designed to quantify these effects with both descriptive and causal analysis, so evaluation choices are traceable and defensible.

## 2. Research Questions

1. How stable are judge decisions and scores across temperatures?
2. Do different judge types react differently to temperature?
3. How much do prompt strategies mediate temperature effects?
4. Are there systematic shifts in consistency, error rate, or position bias?
5. After controlling confounders (prompt/model), what is the causal effect of temperature on agreement?

## 3. Causal Design

The project models temperature as treatment and agreement-like quality metrics as outcomes, while controlling for confounders such as prompt strategy and model identity/size.

Conceptual DAG:

    Temperature ------> Judge Output ------> Metrics
         ^                 ^                   ^
         |                 |                   |
    Model/Size ------------|                   |
    Prompt Variant --------|                   |
    Input Pair ------------------------------->|
    Human Label ------------------------------->|

Core estimands and analysis routines:

- simple ATE on transformed analysis table,
- DML-based ATE via econml,
- stratified analysis by prompt and model label,
- consistency analysis under position swap.

## 4. Metrics

| Metric | Meaning |
|---|---|
| Agreement S1 | Agreement with human signal under project-defined matching policy |
| Agreement S2 | Agreement on stricter subsets (for robustness views) |
| Consistency | Stability when A/B positions are swapped |
| Error Rate | Fraction of unparsable judge outputs |
| Position Bias | Difference between first-position and second-position win tendency |
| Score Stats | Mean/std for score-based judge modes |

## 5. Repository Layout

Top-level:

    .
    |- main.py
    |- exp_main.py
    |- exp_main_batch_async.py
    |- exp_main_batch_async_copy.py
    |- exp_main_batch_async_supplementary.py
    |- input/
    |- output/
    |- results_test/
    |- src/
    |- pyproject.toml
    |- README.md

Package modules:

    src/llmjudgetempcausal/
    |- cli.py          # Click commands: run / analyze / dag
    |- experiment.py   # Orchestration and end-to-end pipeline
    |- data.py         # Dataset loading and sampling
    |- client.py       # LLM client wrapper (OpenAI-compatible backends)
    |- prompts.py      # Prompt builders across judge modes/variants
    |- judge.py        # Output parsing and judge result handling
    |- metrics.py      # Aggregation and metric computation
    |- causal.py       # Causal graph + ATE/DML estimators
    |- visualize.py    # Plot generation
    |- config.py       # Experiment/model dataclasses and enums

## 6. Requirements

- Python 3.11+
- uv (recommended)
- A running inference backend endpoint (vLLM, SGLang, or OpenAI-compatible server/API)

Main dependencies (from pyproject.toml):

- datasets
- openai
- jinja2
- numpy
- pandas
- scipy
- scikit-learn
- econml
- pgmpy
- dowhy
- matplotlib
- seaborn
- tqdm
- click
- vllm

## 7. Installation

### Option A (recommended, uv)

~~~bash
git clone <your-repo-url>
cd LLMJudgeTempCausal
uv sync
~~~

### Option B (pip/venv)

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
~~~

## 8. Quick Start

### Step 1: Launch a model server

Example with vLLM:

~~~bash
uv run vllm serve google/gemma-3-1b-it --gpu-memory-utilization 0.5
# default endpoint: http://localhost:8000
~~~

### Step 2: Run a smoke test

~~~bash
uv run python main.py --quick-test
~~~

This executes a small configuration and writes artifacts to results_test/.

### Step 3: Run a full configurable experiment

~~~bash
uv run llmjudge run \
  -m google/gemma-3-1b-it \
  -u http://localhost:8000 \
  -s 1B \
  -b vllm \
  -n 100 \
  -r 10 \
  -o results
~~~

### Step 4: Re-analyze existing outputs

~~~bash
uv run llmjudge analyze -d results
~~~

### Step 5: Export causal DAG only

~~~bash
uv run llmjudge dag -o results/causal_dag.png
~~~

## 9. CLI Reference

### llmjudge run

| Option | Type | Default | Description |
|---|---|---|---|
| -m, --model | repeatable str | required | Model name per endpoint |
| -u, --base-url | repeatable str | required | Base URL per model |
| -k, --api-key | repeatable str | EMPTY | API key list (aligned by index) |
| -b, --backend | repeatable str | vllm | vllm / sglang / openai |
| -s, --model-size | repeatable str | auto | Label for stratified analysis |
| -t, --temperatures | csv float | 0.0,0.2,0.4,0.6,0.8,1.0 | Treatment grid |
| -j, --judge-types | csv str | pairwise,single_answer,reference_guided | Judge modes |
| -p, --prompt-variants | csv str | baseline,position_swap,few_shot,cot,reference_guided,multi_turn | Prompt interventions |
| -r, --num-repeats | int | 10 | Repeat count |
| -n, --sample-size | int | 100 | Number of sampled pairs |
| --seed | int | 42 | Sampling seed |
| --top-p | float | 0.95 | Decoding top-p |
| --max-tokens | int | 1024 | Generation limit |
| -o, --output-dir | str | results | Output folder |
| -v, --verbose | flag | false | Debug logging |

### llmjudge analyze

~~~bash
uv run llmjudge analyze -d <results_dir>
~~~

Rebuilds aggregate CSV files, analysis.json, and figures from results.csv.

### llmjudge dag

~~~bash
uv run llmjudge dag -o <output_png>
~~~

## 10. Script-Based Workflows

In addition to CLI, this repository includes script-driven pipelines for high-throughput or custom supplementary studies.

### exp_main.py

- Stream-style experiment writing JSONL outputs.
- Useful for custom local runs and iterative debugging.

Run:

~~~bash
uv run python exp_main.py
~~~

### exp_main_batch_async.py

- Async batched requests using AsyncOpenAI-compatible endpoint.
- Includes chunking, batching, fallback behavior, and resumable writes.

Run:

~~~bash
uv run python exp_main_batch_async.py
~~~

### exp_main_batch_async_copy.py

- Variant of the async batch runner tuned for another model endpoint setup.

Run:

~~~bash
uv run python exp_main_batch_async_copy.py
~~~

### exp_main_batch_async_supplementary.py

- Supplementary controlled experiments (position bias, verbosity bias, human-alignment framing).

Run:

~~~bash
uv run python exp_main_batch_async_supplementary.py
~~~

Note: exp_main_batch.py currently exists as an empty placeholder file.

## 11. Data Sources and Input Format

### Upstream benchmark loader

The default package loader uses:

- dataset id: lmsys/mt_bench_human_judgments

### Local TempBench-style loader

The local JSON/JSONL path-based flow supports rows with fields like:

- row_idx or question_id
- model_a, model_b
- winner
- conversation_a, conversation_b
- reference_answer (optional)

Example local files in this repo:

- input/combined_dataset_with_reference_good_row_idx.json
- src/llmjudgetempcausal/assets/mmlu_pro_judged_stream.jsonl (if generated/populated)

## 12. Output Artifacts

A typical run directory includes:

### Core tables

- results.csv
- metrics_by_temperature.csv
- metrics_by_temp_model.csv
- metrics_by_temp_prompt.csv
- metrics_by_temp_judgetype.csv
- consistency_orig.csv
- consistency_swap.csv
- analysis.json

### Figures

- causal_dag.png
- plot_metrics_by_temperature.png
- plot_causal_forest.png
- plot_consistency.png
- plot_position_bias.png
- plot_scores_single_answer.png
- heatmap_agreement_s1.png
- heatmap_agreement_s2.png
- heatmap_error_rate_pairwise.png
- heatmap_jt_error_rate_pairwise.png
- heatmap_jt_error_rate_single_answer.png
- heatmap_model_agreement_s1.png
- heatmap_model_agreement_s2.png
- heatmap_model_error_rate_pairwise.png

You can inspect a concrete sample under results_test/.

## 13. Reproducibility Checklist

1. Pin environment with uv sync.
2. Record model name, endpoint, backend, and decoding params.
3. Set seed and keep it in logs/output.
4. Keep raw result files (results.csv or JSONL stream outputs).
5. Re-run llmjudge analyze from saved outputs.
6. Version-control plots and analysis.json for paper figures.

## 14. Troubleshooting

### Endpoint connectivity errors

- Verify server is reachable from your runtime host.
- Confirm base URL and port mapping.
- Test with a minimal chat/completions request first.

### High parse_error rate

- Lower temperature.
- Tighten prompt format instructions.
- Increase max_tokens if outputs are being truncated.

### Slow throughput

- Use async batch scripts for large sweeps.
- Increase batch size carefully based on GPU memory.
- Subsample pairs during iteration, then scale up.

### Analyze command fails due to missing files

- Ensure results.csv exists in target directory.
- Run full experiment first, then analyze.

## 15. Contributing

Contributions are welcome.

Recommended workflow:

1. Create a feature branch.
2. Add/modify code in src/llmjudgetempcausal or scripts.
3. Validate with a small local run (main.py --quick-test).
4. Open a PR with:
   - motivation,
   - reproducible command,
   - and before/after artifacts when relevant.

## 16. Citation

If you use this project in academic work, please cite:

~~~bibtex
@software{llmjudgetempcausal,
  title  = {LLMJudgeTempCausal: Causal Analysis of Temperature Effects in LLM-as-a-Judge},
  author = {LLMJudgeTempCausal Contributors},
  year   = {2026},
  url    = {https://github.com/<your-org>/LLMJudgeTempCausal}
}
~~~

## 17. License

No root LICENSE file is currently committed in this repository.

If you plan to open-source this project publicly, add a LICENSE file (for example MIT, Apache-2.0, or BSD-3-Clause) before release.