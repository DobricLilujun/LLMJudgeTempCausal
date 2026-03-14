import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from llmjudgetempcausal.client import LLMClient
from llmjudgetempcausal.config import BackendType, JudgeType, ModelConfig, PromptVariant
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


# -----------------------------------------------------------------------------
# User config
# -----------------------------------------------------------------------------
MODEL_NAME = "google/gemma-3-27b-it"
BASE_URL = "http://10.6.32.18:8001"
MODEL_SIZE = "27B"
API_KEY = "token-abc123"

PATH_INPUT = f"{project_root}/input/combined_dataset_with_reference_good_row_idx.json"

TEMPERATURES = [0.01, 0.5, 1.0, 1.5, 2.0, 3.0]
N_REPEATS = 10
BASE_SEED = 42
MAX_TOKENS = 1024
TOP_P = 0.95
BATCH_SIZE = 64  # vLLM batch size per request
MAX_CONCURRENT_BATCHES = 8  # async in-flight batch requests
SAMPLE_N = 1

JUDGE_COMBOS = [
    (JudgeType.PAIRWISE, PromptVariant.BASELINE),
    (JudgeType.PAIRWISE, PromptVariant.COT),
    (JudgeType.SINGLE_ANSWER, PromptVariant.BASELINE),
    (JudgeType.SINGLE_ANSWER, PromptVariant.COT),
    (JudgeType.REFERENCE_GUIDED, PromptVariant.BASELINE),
    (JudgeType.REFERENCE_GUIDED, PromptVariant.COT),
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    """Convert chat messages to a single prompt string for completions API."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<start_of_turn>system\\n{content}<end_of_turn>")
        elif role == "user":
            parts.append(f"<start_of_turn>user\\n{content}<end_of_turn>")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\\n{content}<end_of_turn>")
    parts.append("<start_of_turn>model\\n")
    return "\\n".join(parts)


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


async def _single_completion(
    async_client: AsyncOpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
) -> str:
    extra = {"seed": seed} if seed is not None else {}
    response = await async_client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<end_of_turn>"],
        **extra,
    )
    return response.choices[0].text or ""


async def batch_generate_prompts(
    async_client: AsyncOpenAI,
    model_name: str,
    prompts: list[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
) -> list[str]:
    """True vLLM batch inference via one completions call with prompt list.

    Falls back to per-prompt calls if the batch request fails.
    """
    if not prompts:
        return []

    extra = {"seed": seed} if seed is not None else {}

    try:
        response = await async_client.completions.create(
            model=model_name,
            prompt=prompts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<end_of_turn>"],
            **extra,
        )

        outputs = ["" for _ in prompts]
        for choice in response.choices:
            idx = int(getattr(choice, "index", 0))
            if 0 <= idx < len(outputs):
                outputs[idx] = choice.text or ""

        # Fill any unexpected holes with single requests.
        missing_indices = [i for i, text in enumerate(outputs) if text == ""]
        if missing_indices:
            fallback_results = await asyncio.gather(
                *[
                    _single_completion(
                        async_client,
                        model_name,
                        prompts[i],
                        temperature,
                        top_p,
                        max_tokens,
                        seed,
                    )
                    for i in missing_indices
                ],
                return_exceptions=True,
            )
            for idx, result in zip(missing_indices, fallback_results):
                if isinstance(result, Exception):
                    outputs[idx] = f"ERROR: {result}"
                else:
                    outputs[idx] = result

        return outputs

    except Exception as e:
        logging.warning("Batch completions failed, fallback to single calls (%d prompts): %s", len(prompts), e)
        fallback_results = await asyncio.gather(
            *[
                _single_completion(
                    async_client,
                    model_name,
                    prompt,
                    temperature,
                    top_p,
                    max_tokens,
                    seed,
                )
                for prompt in prompts
            ],
            return_exceptions=True,
        )
        outputs = []
        for result in fallback_results:
            if isinstance(result, Exception):
                outputs.append(f"ERROR: {result}")
            else:
                outputs.append(result)
        return outputs


async def process_chunk(
    async_client: AsyncOpenAI,
    model_name: str,
    chunk: list[tuple],
    judge_type: JudgeType,
    prompt_variant: PromptVariant,
    temp: float,
    seed: int,
) -> tuple[list[tuple[str, dict]], int]:
    """Process one chunk and return (rows_with_keys, error_count)."""
    rows_with_keys: list[tuple[str, dict]] = []
    error_count = 0

    if judge_type == JudgeType.SINGLE_ANSWER:
        prompts_a = []
        prompts_b = []
        for p, _, _ in chunk:
            msgs_a = build_messages(
                p,
                judge_type,
                prompt_variant,
                which_response="a",
                model_name=model_name,
            )
            msgs_b = build_messages(
                p,
                judge_type,
                prompt_variant,
                which_response="b",
                model_name=model_name,
            )
            prompts_a.append(_messages_to_prompt(msgs_a))
            prompts_b.append(_messages_to_prompt(msgs_b))

        raw_as, raw_bs = await asyncio.gather(
            batch_generate_prompts(
                async_client,
                model_name,
                prompts_a,
                temperature=temp,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                seed=seed,
            ),
            batch_generate_prompts(
                async_client,
                model_name,
                prompts_b,
                temperature=temp,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                seed=seed,
            ),
        )

        for (_, base, run_key), raw_a, raw_b in zip(chunk, raw_as, raw_bs):
            row_error_parts = []
            if raw_a.startswith("ERROR:"):
                row_error_parts.append(f"response_a: {raw_a}")
            if raw_b.startswith("ERROR:"):
                row_error_parts.append(f"response_b: {raw_b}")

            row = {
                **base,
                "which_response": "both",
                "score_a": parse_score(raw_a),
                "score_b": parse_score(raw_b),
                "judge_reason_a": parse_judge_reason(raw_a),
                "judge_reason_b": parse_judge_reason(raw_b),
                "raw_output_a": raw_a,
                "raw_output_b": raw_b,
                "raw_output": None,
                "judge_reason": None,
                "pairwise_winner": None,
            }
            if row_error_parts:
                error_count += 1
                row["row_error"] = " | ".join(row_error_parts)

            rows_with_keys.append((run_key, row))

        return rows_with_keys, error_count

    prompts = []
    for p, _, _ in chunk:
        msgs = build_messages(
            p,
            judge_type,
            prompt_variant,
            model_name=model_name,
        )
        prompts.append(_messages_to_prompt(msgs))

    raws = await batch_generate_prompts(
        async_client,
        model_name,
        prompts,
        temperature=temp,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        seed=seed,
    )

    for (_, base, run_key), raw in zip(chunk, raws):
        if raw.startswith("ERROR:"):
            error_count += 1
            row = {
                **base,
                "row_error": raw,
            }
        else:
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

        rows_with_keys.append((run_key, row))

    return rows_with_keys, error_count


async def process_chunk_with_semaphore(
    semaphore: asyncio.Semaphore,
    async_client: AsyncOpenAI,
    model_name: str,
    chunk: list[tuple],
    judge_type: JudgeType,
    prompt_variant: PromptVariant,
    temp: float,
    seed: int,
) -> tuple[list[tuple[str, dict]], int]:
    async with semaphore:
        try:
            return await process_chunk(
                async_client,
                model_name,
                chunk,
                judge_type,
                prompt_variant,
                temp,
                seed,
            )
        except Exception as e:
            rows_with_keys = []
            for _, base, run_key in chunk:
                rows_with_keys.append((run_key, {**base, "row_error": f"ERROR: {e}"}))
            return rows_with_keys, len(chunk)


def make_run_key(question_id: int, judge_type: str, prompt_variant: str, temperature: float, repeat_id: int) -> str:
    # One logical experiment row per key. SINGLE_ANSWER keeps A/B in the same row.
    return f"{question_id}|{judge_type}|{prompt_variant}|{temperature}|{repeat_id}"


# -----------------------------------------------------------------------------
# Setup model/client/data
# -----------------------------------------------------------------------------
model_cfg = ModelConfig(
    model_name=MODEL_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    backend=BackendType.VLLM,
    model_size_label=MODEL_SIZE,
)

active_client = LLMClient(model_cfg)
print(f"Client: {active_client.model_name} @ {active_client._get_base_url()}")
async_client = AsyncOpenAI(
    base_url=active_client._get_base_url(),
    api_key=model_cfg.api_key,
)

all_pairs = load_temp_bench(path=PATH_INPUT)
print(f"Total pairs loaded: {len(all_pairs)}")

pairs = all_pairs

safe_model_name = active_client.model_name.replace("/", "__")
OUTPUT_JSONL = Path(project_root) / "output" / f"test_main_eval_stream_batch_{safe_model_name}.jsonl"
OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

SEEDS = [random.Random(BASE_SEED + i).randint(0, 2**31 - 1) for i in range(N_REPEATS)]


# -----------------------------------------------------------------------------
# Resume state
# -----------------------------------------------------------------------------
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
print(f"Batch size: {BATCH_SIZE}")
print(f"Seeds: {SEEDS}")
print(f"Resume state: {len(processed)} / {expected_total} already completed")


# -----------------------------------------------------------------------------
# Main batch loop
# -----------------------------------------------------------------------------
async def run_batches() -> None:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

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

        # Loop order: outer repeat_id -> middle judge combo -> inner temperature
        for repeat_id in range(N_REPEATS):
            seed = SEEDS[repeat_id]

            for judge_type, prompt_variant in JUDGE_COMBOS:
                for temp in TEMPERATURES:
                    pending = []
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
                        pending.append((p, base, run_key))

                    if not pending:
                        continue

                    tasks = [
                        asyncio.create_task(
                            process_chunk_with_semaphore(
                                semaphore,
                                async_client,
                                active_client.model_name,
                                chunk,
                                judge_type,
                                prompt_variant,
                                temp,
                                seed,
                            )
                        )
                        for chunk in _chunks(pending, BATCH_SIZE)
                    ]

                    for task in asyncio.as_completed(tasks):
                        rows_with_keys, chunk_error_count = await task
                        new_errors += chunk_error_count

                        for run_key, row in rows_with_keys:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            f.flush()

                            new_written += 1
                            overall_pbar.update(1)
                            overall_pbar.set_postfix(new=new_written, errors=new_errors)

                            if "row_error" not in row:
                                processed.add(run_key)

        overall_pbar.close()


asyncio.run(run_batches())


# -----------------------------------------------------------------------------
# Reload successful rows for analysis
# -----------------------------------------------------------------------------
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
print(
    f"Done. Success rows: {len(exp_df)} | "
    f"Error rows: {error_count} | "
    f"Completed keys: {len(processed)}/{expected_total}"
)
