"""Experiment runner: orchestrates the full evaluation pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .causal import (
    build_causal_dag,
    estimate_ate_dml,
    estimate_ate_simple,
    prepare_dml_data,
    stratified_analysis,
    visualize_dag,
)
from .client import LLMClient
from .config import (
    ExperimentConfig,
    JudgeType,
    ModelConfig,
    PromptVariant,
)
from .data import JudgePair, load_mt_bench_human, sample_pairs
from .judge import JudgeResult, run_judge_pair_consistency, run_judge_single
from .metrics import aggregate_metrics_by_group, compute_all_metrics

logger = logging.getLogger(__name__)


def _result_to_dict(r: JudgeResult) -> dict:
    return {
        "question_id": r.question_id,
        "model_a": r.model_a,
        "model_b": r.model_b,
        "human_winner": r.human_winner,
        "judge_type": r.judge_type,
        "prompt_variant": r.prompt_variant,
        "temperature": r.temperature,
        "model_name": r.model_name,
        "model_size_label": r.model_size_label,
        "repeat_id": r.repeat_id,
        "raw_output": r.raw_output,
        "pairwise_winner": r.pairwise_winner,
        "score_a": r.score_a,
        "score_b": r.score_b,
        "is_swapped": r.is_swapped,
        "parse_error": r.parse_error,
    }


class ExperimentRunner:
    """Orchestrates the full experiment pipeline."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load and sample data
        logger.info("Loading MT-Bench human judgments...")
        all_pairs = load_mt_bench_human()
        self.pairs = sample_pairs(all_pairs, config.sample_size, config.random_seed)
        logger.info("Sampled %d pairs from %d total", len(self.pairs), len(all_pairs))

        # Build LLM clients
        self.clients: list[tuple[ModelConfig, LLMClient]] = []
        for mc in config.models:
            self.clients.append((mc, LLMClient(mc)))

        self.all_results: list[dict] = []
        self.consistency_results_orig: list[dict] = []
        self.consistency_results_swap: list[dict] = []

    def run_all(self):
        """Run the complete experiment."""
        logger.info("=" * 60)
        logger.info("Starting experiment with %d models, %d temperatures, "
                     "%d judge types, %d prompt variants, %d repeats",
                     len(self.clients), len(self.config.temperatures),
                     len(self.config.judge_types), len(self.config.prompt_variants),
                     self.config.num_repeats)
        logger.info("=" * 60)

        total_configs = (
            len(self.clients)
            * len(self.config.temperatures)
            * len(self.config.judge_types)
            * len(self.config.prompt_variants)
            * self.config.num_repeats
        )

        pbar = tqdm(total=total_configs * len(self.pairs), desc="Judging")

        for mc, client in self.clients:
            for temp in self.config.temperatures:
                for jt in self.config.judge_types:
                    for pv in self.config.prompt_variants:
                        for rep in range(self.config.num_repeats):
                            self._run_single_config(
                                client, mc, temp, jt, pv, rep, pbar
                            )

        pbar.close()

        # Run consistency checks (original vs swapped pairwise)
        logger.info("Running consistency checks (position swap)...")
        self._run_consistency_checks()

        # Save results
        self._save_results()

        # Run analysis
        self._run_analysis()

    def _run_single_config(
        self,
        client: LLMClient,
        mc: ModelConfig,
        temp: float,
        jt: JudgeType,
        pv: PromptVariant,
        repeat_id: int,
        pbar,
    ):
        """Run one configuration across all pairs."""
        for pair in self.pairs:
            try:
                results = run_judge_single(
                    client=client,
                    pair=pair,
                    judge_type=jt,
                    prompt_variant=pv,
                    temperature=temp,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    repeat_id=repeat_id,
                    model_size_label=mc.model_size_label,
                )
                for r in results:
                    self.all_results.append(_result_to_dict(r))
            except Exception as e:
                logger.error("Error judging pair %d: %s", pair.question_id, e)
            pbar.update(1)

    def _run_consistency_checks(self):
        """Run position-swap consistency checks for each model/temp/repeat."""
        for mc, client in self.clients:
            for temp in self.config.temperatures:
                for rep in range(min(self.config.num_repeats, 3)):  # limit repeats for consistency
                    for pair in self.pairs[:min(50, len(self.pairs))]:  # subsample for speed
                        try:
                            r_orig, r_swap = run_judge_pair_consistency(
                                client=client,
                                pair=pair,
                                prompt_variant=PromptVariant.BASELINE,
                                temperature=temp,
                                top_p=self.config.top_p,
                                max_tokens=self.config.max_tokens,
                                repeat_id=rep,
                                model_size_label=mc.model_size_label,
                            )
                            self.consistency_results_orig.append(_result_to_dict(r_orig))
                            self.consistency_results_swap.append(_result_to_dict(r_swap))
                        except Exception as e:
                            logger.error("Consistency check error: %s", e)

    def _save_results(self):
        """Save all results to CSV and JSON."""
        df = pd.DataFrame(self.all_results)
        csv_path = self.output_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved %d results to %s", len(df), csv_path)

        if self.consistency_results_orig:
            df_orig = pd.DataFrame(self.consistency_results_orig)
            df_swap = pd.DataFrame(self.consistency_results_swap)
            df_orig.to_csv(self.output_dir / "consistency_orig.csv", index=False)
            df_swap.to_csv(self.output_dir / "consistency_swap.csv", index=False)

    def _run_analysis(self):
        """Run causal analysis and aggregated metrics."""
        df = pd.DataFrame(self.all_results)
        if len(df) == 0:
            logger.warning("No results to analyze")
            return

        analysis = {}

        # 1. Descriptive statistics by temperature
        logger.info("Computing descriptive statistics...")
        if "temperature" in df.columns:
            agg = aggregate_metrics_by_group(df, ["temperature"])
            agg.to_csv(self.output_dir / "metrics_by_temperature.csv", index=False)
            analysis["by_temperature"] = agg.to_dict(orient="records")

        # 2. By temperature × model
        if "model_size_label" in df.columns:
            agg2 = aggregate_metrics_by_group(df, ["temperature", "model_size_label"])
            agg2.to_csv(self.output_dir / "metrics_by_temp_model.csv", index=False)

        # 3. By temperature × prompt variant
        if "prompt_variant" in df.columns:
            agg3 = aggregate_metrics_by_group(df, ["temperature", "prompt_variant"])
            agg3.to_csv(self.output_dir / "metrics_by_temp_prompt.csv", index=False)

        # 4. By temperature × judge type
        if "judge_type" in df.columns:
            agg4 = aggregate_metrics_by_group(df, ["temperature", "judge_type"])
            agg4.to_csv(self.output_dir / "metrics_by_temp_judgetype.csv", index=False)

        # 5. Causal analysis
        logger.info("Running causal analysis...")
        try:
            dag = build_causal_dag()
            visualize_dag(dag, str(self.output_dir / "causal_dag.png"))
        except Exception as e:
            logger.error("DAG visualization failed: %s", e)

        dml_df = prepare_dml_data(df)
        if len(dml_df) > 0:
            # Simple ATE
            simple_ate = estimate_ate_simple(dml_df, outcome_col="agreement")
            analysis["simple_ate"] = simple_ate

            # DML ATE
            dml_ate = estimate_ate_dml(dml_df)
            analysis["dml_ate"] = dml_ate

            # Stratified by prompt
            strat_prompt = stratified_analysis(dml_df, "prompt_variant", "agreement")
            analysis["stratified_by_prompt"] = {
                str(k): v for k, v in strat_prompt.items()
            }

            # Stratified by model
            strat_model = stratified_analysis(dml_df, "model_size_label", "agreement")
            analysis["stratified_by_model"] = {
                str(k): v for k, v in strat_model.items()
            }

        # 6. Consistency analysis
        if self.consistency_results_orig:
            df_orig = pd.DataFrame(self.consistency_results_orig)
            df_swap = pd.DataFrame(self.consistency_results_swap)
            from .metrics import compute_consistency
            consistency_by_temp = {}
            for temp in self.config.temperatures:
                orig_t = df_orig[df_orig["temperature"] == temp]
                swap_t = df_swap[df_swap["temperature"] == temp]
                if len(orig_t) > 0:
                    consistency_by_temp[str(temp)] = compute_consistency(orig_t, swap_t)
            analysis["consistency_by_temperature"] = consistency_by_temp

        # Save analysis
        analysis_path = self.output_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info("Analysis saved to %s", analysis_path)

        # Generate visualizations
        logger.info("Generating visualizations...")
        try:
            from .visualize import generate_all_plots
            generate_all_plots(self.output_dir)
        except Exception as e:
            logger.error("Visualization failed: %s", e)

        logger.info("Experiment complete! Results in %s", self.output_dir)
