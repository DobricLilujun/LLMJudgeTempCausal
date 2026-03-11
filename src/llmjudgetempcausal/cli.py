"""CLI entry point for LLM Judge Temperature Causal Analysis."""

from __future__ import annotations

import json
import logging
import sys

import click

from .config import (
    BackendType,
    ExperimentConfig,
    JudgeType,
    ModelConfig,
    PromptVariant,
)
from .experiment import ExperimentRunner


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
def main():
    """LLM-as-a-Judge Temperature Causal Analysis."""
    pass


@main.command()
@click.option("--model", "-m", multiple=True, required=True,
              help="Model name (e.g., google/gemma-3-1b-it)")
@click.option("--base-url", "-u", multiple=True, required=True,
              help="Base URL for each model (e.g., http://localhost:8000)")
@click.option("--api-key", "-k", multiple=True, default=None,
              help="API key for each model (default: EMPTY)")
@click.option("--backend", "-b", multiple=True, default=None,
              help="Backend type for each model: vllm, sglang, openai")
@click.option("--model-size", "-s", multiple=True, default=None,
              help="Model size label for each model (e.g., 1B, 7B, 70B)")
@click.option("--temperatures", "-t", default="0.0,0.2,0.4,0.6,0.8,1.0",
              help="Comma-separated temperature values")
@click.option("--judge-types", "-j", default="pairwise,single_answer,reference_guided",
              help="Comma-separated judge types")
@click.option("--prompt-variants", "-p",
              default="baseline,position_swap,few_shot,cot,reference_guided,multi_turn",
              help="Comma-separated prompt variants")
@click.option("--num-repeats", "-r", default=10, type=int,
              help="Number of repeats per configuration")
@click.option("--sample-size", "-n", default=100, type=int,
              help="Number of pairs to sample from dataset")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--output-dir", "-o", default="results", help="Output directory")
@click.option("--top-p", default=0.95, type=float, help="Top-p sampling parameter")
@click.option("--max-tokens", default=1024, type=int, help="Max tokens for generation")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def run(
    model, base_url, api_key, backend, model_size,
    temperatures, judge_types, prompt_variants,
    num_repeats, sample_size, seed, output_dir, top_p, max_tokens, verbose,
):
    """Run the full experiment pipeline."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Validate model/url count match
    if len(model) != len(base_url):
        click.echo("Error: --model and --base-url must have the same count", err=True)
        sys.exit(1)

    # Build model configs
    if not api_key:
        api_key = tuple("EMPTY" for _ in model)
    if not backend:
        backend = tuple("vllm" for _ in model)
    if not model_size:
        model_size = tuple("" for _ in model)

    # Pad to match model count
    api_key = tuple(api_key) + ("EMPTY",) * (len(model) - len(api_key))
    backend = tuple(backend) + ("vllm",) * (len(model) - len(backend))
    model_size = tuple(model_size) + ("",) * (len(model) - len(model_size))

    models = []
    for i in range(len(model)):
        models.append(ModelConfig(
            model_name=model[i],
            base_url=base_url[i],
            api_key=api_key[i],
            backend=BackendType(backend[i]),
            model_size_label=model_size[i],
        ))

    config = ExperimentConfig(
        temperatures=[float(t) for t in temperatures.split(",")],
        top_p=top_p,
        max_tokens=max_tokens,
        judge_types=[JudgeType(jt.strip()) for jt in judge_types.split(",")],
        prompt_variants=[PromptVariant(pv.strip()) for pv in prompt_variants.split(",")],
        num_repeats=num_repeats,
        sample_size=sample_size,
        random_seed=seed,
        output_dir=output_dir,
        models=models,
    )

    logger.info("Experiment config:")
    logger.info("  Models: %s", [m.model_name for m in models])
    logger.info("  Temperatures: %s", config.temperatures)
    logger.info("  Judge types: %s", [jt.value for jt in config.judge_types])
    logger.info("  Prompt variants: %s", [pv.value for pv in config.prompt_variants])
    logger.info("  Repeats: %d, Sample size: %d", config.num_repeats, config.sample_size)

    runner = ExperimentRunner(config)
    runner.run_all()


@main.command()
@click.option("--results-dir", "-d", default="results", help="Results directory to analyze")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def analyze(results_dir, verbose):
    """Re-run analysis on existing results."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from pathlib import Path
    import pandas as pd
    from .causal import (
        build_causal_dag, visualize_dag,
        prepare_dml_data, estimate_ate_simple, estimate_ate_dml,
        stratified_analysis,
    )
    from .metrics import aggregate_metrics_by_group, compute_consistency
    from .visualize import generate_all_plots

    output_dir = Path(results_dir)
    csv_path = output_dir / "results.csv"
    if not csv_path.exists():
        click.echo(f"Error: {csv_path} not found", err=True)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d results from %s", len(df), csv_path)

    # Re-run aggregation
    for group_cols, fname in [
        (["temperature"], "metrics_by_temperature.csv"),
        (["temperature", "model_size_label"], "metrics_by_temp_model.csv"),
        (["temperature", "prompt_variant"], "metrics_by_temp_prompt.csv"),
        (["temperature", "judge_type"], "metrics_by_temp_judgetype.csv"),
    ]:
        try:
            agg = aggregate_metrics_by_group(df, group_cols)
            agg.to_csv(output_dir / fname, index=False)
        except Exception as e:
            logger.error("Aggregation failed for %s: %s", fname, e)

    # Causal analysis
    analysis = {}
    try:
        dag = build_causal_dag()
        visualize_dag(dag, str(output_dir / "causal_dag.png"))
    except Exception as e:
        logger.error("DAG visualization failed: %s", e)

    dml_df = prepare_dml_data(df)
    if len(dml_df) > 0:
        analysis["simple_ate"] = estimate_ate_simple(dml_df)
        analysis["dml_ate"] = estimate_ate_dml(dml_df)
        analysis["stratified_by_prompt"] = {
            str(k): v for k, v in stratified_analysis(dml_df, "prompt_variant").items()
        }
        analysis["stratified_by_model"] = {
            str(k): v for k, v in stratified_analysis(dml_df, "model_size_label").items()
        }

    # Consistency
    orig_path = output_dir / "consistency_orig.csv"
    swap_path = output_dir / "consistency_swap.csv"
    if orig_path.exists() and swap_path.exists():
        df_orig = pd.read_csv(orig_path)
        df_swap = pd.read_csv(swap_path)
        temps = sorted(df_orig["temperature"].unique())
        consistency = {}
        for temp in temps:
            orig_t = df_orig[df_orig["temperature"] == temp]
            swap_t = df_swap[df_swap["temperature"] == temp]
            if len(orig_t) > 0:
                consistency[str(temp)] = compute_consistency(orig_t, swap_t)
        analysis["consistency_by_temperature"] = consistency

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    generate_all_plots(output_dir)
    logger.info("Analysis complete!")


@main.command()
@click.option("--output", "-o", default="results/causal_dag.png", help="Output path")
def dag(output):
    """Generate and save the causal DAG visualization."""
    setup_logging()
    from pathlib import Path
    from .causal import build_causal_dag, visualize_dag

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model = build_causal_dag()
    visualize_dag(model, output)
    click.echo(f"DAG saved to {output}")


if __name__ == "__main__":
    main()
