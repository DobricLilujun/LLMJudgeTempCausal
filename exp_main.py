"""Streaming experiment runner for local single-endpoint evaluation.

This script writes one JSON line per evaluated configuration row and supports
resume-by-key behavior. It is intentionally explicit (not hidden behind CLI)
so researchers can tweak loop order and output schema for case studies.
"""

import logging
import os
import json
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from llmjudgetempcausal.client import LLMClient
from llmjudgetempcausal.data import load_temp_bench, sample_pairs
from llmjudgetempcausal.judge import parse_judge_reason, parse_pairwise, parse_score
from llmjudgetempcausal.prompts import build_messages


# Ensure the project root is on the path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Reduce SDK transport noise like: HTTP Request ... "200 OK"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

print("Project root:", project_root)



from llmjudgetempcausal.config import (
    BackendType,
    JudgeType,
    ModelConfig,
    PromptVariant,
)

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
BASE_URL = "http://localhost:8000"
model_size = "14B"
TEMPERATURES = [0.01, 0.5, 1.0, 1.5, 2.0, 3.0]
PATH_INPUT = f"{project_root}/src/llmjudgetempcausal/assets/combined_dataset_with_reference_good_row_idx.json"

N_REPEATS = 10
BASE_SEED = 42
MAX_TOKENS = 1024
TOP_P = 0.95

JUDGE_COMBOS = [
    (JudgeType.PAIRWISE, PromptVariant.BASELINE),
    (JudgeType.PAIRWISE, PromptVariant.COT),
    (JudgeType.SINGLE_ANSWER, PromptVariant.BASELINE),
    (JudgeType.SINGLE_ANSWER, PromptVariant.COT),
    (JudgeType.REFERENCE_GUIDED, PromptVariant.BASELINE),
    (JudgeType.REFERENCE_GUIDED, PromptVariant.COT),
]

SEEDS = [random.Random(BASE_SEED + i).randint(0, 2**31 - 1) for i in range(N_REPEATS)]


# Qwen 2.5-14B on localhost:8000
model_cfg = ModelConfig(
    model_name=MODEL_NAME,
    base_url=BASE_URL,
    api_key="token-abc123",
    backend=BackendType.VLLM,
    model_size_label=model_size,
)




all_pairs = load_temp_bench(path=PATH_INPUT)  # 默认读 assets/mmlu_pro_judged_stream.jsonl
print(f"Total pairs loaded: {len(all_pairs)}")

pairs = sample_pairs(all_pairs, n=1, seed=42)
print(f"Sampled pairs: {len(pairs)}")

# Inspect first pair
pair = pairs[0]
print(f"Question ID: {pair.question_id}")
print(f"Model A: {pair.model_a}")
print(f"Model B: {pair.model_b}")
print(f"Human winner: {pair.human_winner}")
print(f"Turn: {pair.turn}")
print(f"\nQuestion text:\n{pair.question_text[:500]}")
print(f"\nResponse A (first 300 chars):\n{pair.response_a[:500]}")
print(f"\nResponse B (first 300 chars):\n{pair.response_b[:500]}")


# Initialize the active judge client once and reuse it across all runs.
client = LLMClient(model_cfg)
print(f"Qwen client:  {client.model_name}  @ {client._get_base_url()}")
active_client = client  # swap this variable if multiple clients are tested

OUTPUT_JSONL = Path(project_root) / "output" / f"test_main_eval_stream_{active_client.model_name}.jsonl"
OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)



def make_run_key(question_id: int, judge_type: str, prompt_variant: str, temperature: float, repeat_id: int) -> str:
    """Build a deterministic id used for resume-safe deduplication.

    One logical experiment row corresponds to one run key. For single-answer
    mode, A/B scoring is still stored in one logical row.
    """
    return f"{question_id}|{judge_type}|{prompt_variant}|{temperature}|{repeat_id}"


# Resume: only count rows without row_error as completed
processed = set()
if OUTPUT_JSONL.exists():
    with OUTPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("row_error"):
                continue

            run_key = obj.get("run_key")
            if not run_key:
                run_key = make_run_key(
                    question_id=int(obj["question_id"]),
                    judge_type=str(obj["judge_type"]),
                    prompt_variant=str(obj["prompt_variant"]),
                    temperature=float(obj["temperature"]),
                    repeat_id=int(obj["repeat_id"]),
                )
            processed.add(run_key)

expected_total = len(TEMPERATURES) * N_REPEATS * len(JUDGE_COMBOS) * len(pairs)
print(f"Output JSONL: {OUTPUT_JSONL}")
print(f"Seeds: {SEEDS}")
print(f"Resume state: {len(processed)} / {expected_total} already completed")


with OUTPUT_JSONL.open("a", encoding="utf-8") as f:
    overall_pbar = tqdm(
        total=expected_total,
        initial=len(processed),
        desc="overall",
        unit="run",
        dynamic_ncols=True,
    )
    new_written = 0
    new_errors = 0

    # Loop order requested:
    # outer: repeat_id -> middle: judge combo -> inner: temperature
    for repeat_id in range(N_REPEATS):
        seed = SEEDS[repeat_id]

        for judge_type, prompt_variant in JUDGE_COMBOS:
            for temp in TEMPERATURES:
                for p in pairs:
                    run_key = make_run_key(
                        question_id=p.question_id,
                        judge_type=judge_type.value,
                        prompt_variant=prompt_variant.value,
                        temperature=temp,
                        repeat_id=repeat_id,
                    )
                    if run_key in processed:
                        continue

                    base = {
                        "run_key": run_key,
                        "question_id": p.question_id,
                        "model_a": p.model_a,
                        "model_b": p.model_b,
                        "human_winner": p.human_winner,
                        "judge_model": active_client.model_name,
                        "judge_type": judge_type.value,
                        "prompt_variant": prompt_variant.value,
                        "temperature": temp,
                        "repeat_id": repeat_id,
                        "seed": seed,
                    }

                    try:
                        if judge_type == JudgeType.SINGLE_ANSWER:
                            # SINGLE_ANSWER: score A and B, but keep them in ONE row.
                            msgs_a = build_messages(
                                p,
                                judge_type,
                                prompt_variant,
                                which_response="a",
                                model_name=active_client.model_name,
                            )
                            raw_a = active_client.generate(
                                msgs_a,
                                temperature=temp,
                                top_p=TOP_P,
                                max_tokens=MAX_TOKENS,
                                seed=seed,
                            )

                            msgs_b = build_messages(
                                p,
                                judge_type,
                                prompt_variant,
                                which_response="b",
                                model_name=active_client.model_name,
                            )
                            raw_b = active_client.generate(
                                msgs_b,
                                temperature=temp,
                                top_p=TOP_P,
                                max_tokens=MAX_TOKENS,
                                seed=seed,
                            )

                            row = {
                                **base,
                                "which_response": "both",
                                "score_a": parse_score(raw_a),
                                "score_b": parse_score(raw_b),
                                "judge_reason_a": parse_judge_reason(raw_a),
                                "judge_reason_b": parse_judge_reason(raw_b),
                                "raw_output_a": raw_a,
                                "raw_output_b": raw_b,
                                "pairwise_winner": None,
                            }

                        else:
                            msgs = build_messages(
                                p,
                                judge_type,
                                prompt_variant,
                                model_name=active_client.model_name,
                            )
                            raw = active_client.generate(
                                msgs,
                                temperature=temp,
                                top_p=TOP_P,
                                max_tokens=MAX_TOKENS,
                                seed=seed,
                            )

                            row = {
                                **base,
                                "which_response": None,
                                "score_a": None,
                                "score_b": None,
                                "judge_reason_a": None,
                                "judge_reason_b": None,
                                "raw_output_a": None,
                                "raw_output_b": None,
                                "raw_output": raw,
                                "judge_reason": parse_judge_reason(raw),
                                "pairwise_winner": parse_pairwise(raw),
                            }

                    except Exception as e:
                        new_errors += 1
                        row = {
                            **base,
                            "row_error": str(e),
                        }

                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()

                    new_written += 1
                    overall_pbar.update(1)
                    overall_pbar.set_postfix(new=new_written, errors=new_errors)

                    if "row_error" not in row:
                        processed.add(run_key)

    overall_pbar.close()


# Reload successful rows for quick sanity-check analysis in-memory.
rows = []
error_count = 0
with OUTPUT_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj.get("row_error"):
            error_count += 1
            continue
        rows.append(obj)

exp_df = pd.DataFrame(rows)
print(f"Done. Success rows: {len(exp_df)} | Error rows: {error_count} | Completed keys: {len(processed)}/{expected_total}")
