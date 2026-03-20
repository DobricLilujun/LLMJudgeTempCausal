"""Quick-start script for LLM-as-a-Judge Temperature Causal Analysis.

Usage examples:

  # Test with local vLLM server (gemma-3-1b-it)
  uv run python main.py --quick-test

  # Full experiment (customize as needed)
  uv run llmjudge run -m google/gemma-3-1b-it -u http://localhost:8000 -n 20 -r 2

  # Re-analyze existing results
  uv run llmjudge analyze -d results
"""

import logging
import sys

from llmjudgetempcausal.config import (
    BackendType,
    ExperimentConfig,
    JudgeType,
    ModelConfig,
    PromptVariant,
)
from llmjudgetempcausal.experiment import ExperimentRunner


def quick_test():
    """Run a small end-to-end smoke test.

    The quick test intentionally uses a tiny configuration so contributors can
    validate environment setup, endpoint connectivity, parsing, aggregation,
    and plotting in a few minutes before launching larger sweeps.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Keep this config intentionally tiny to minimize runtime and API cost.
    config = ExperimentConfig(
        temperatures=[0.0, 0.5, 1.0],
        judge_types=[JudgeType.PAIRWISE, JudgeType.SINGLE_ANSWER],
        prompt_variants=[PromptVariant.BASELINE, PromptVariant.COT],
        num_repeats=2,
        sample_size=5,
        output_dir="results_test",
        models=[
            ModelConfig(
                model_name="google/gemma-3-1b-it",
                base_url="http://localhost:8000",
                backend=BackendType.VLLM,
                model_size_label="1B",
            ),
        ],
    )

    # Run the full pipeline (judge -> metrics -> causal analysis -> plots).
    runner = ExperimentRunner(config)
    runner.run_all()


if __name__ == "__main__":
    # ``--quick-test`` is a convenient shortcut for local validation.
    # Otherwise, delegate to the full Click CLI entrypoint.
    if "--quick-test" in sys.argv:
        quick_test()
    else:
        from llmjudgetempcausal.cli import main
        main()

