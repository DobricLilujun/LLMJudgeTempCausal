"""Visualization: heatmaps, line plots, forest plots for causal analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _load_csv_safe(path: Path) -> pd.DataFrame | None:
    """Load a CSV file if present, otherwise return None.

    Plot functions use this helper to stay resilient when some intermediate
    artifacts are intentionally absent (for example, partial reruns).
    """
    if path.exists():
        return pd.read_csv(path)
    return None


def plot_metrics_by_temperature(output_dir: Path):
    """Line plot: each metric vs temperature."""
    df = _load_csv_safe(output_dir / "metrics_by_temperature.csv")
    if df is None or len(df) == 0:
        return

    metric_cols = [c for c in df.columns if c not in ("temperature", "n_samples")]
    n_metrics = len(metric_cols)
    if n_metrics == 0:
        return

    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(metric_cols):
        ax = axes[i]
        vals = pd.to_numeric(df[col], errors="coerce")
        ax.plot(df["temperature"], vals, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Temperature")
        ax.set_ylabel(col)
        ax.set_title(col.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Metrics vs Temperature", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_metrics_by_temperature.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap_temp_prompt(output_dir: Path):
    """Heatmap: Temperature × Prompt Variant for key metrics."""
    df = _load_csv_safe(output_dir / "metrics_by_temp_prompt.csv")
    if df is None or len(df) == 0:
        return

    for metric in ["agreement_s1", "agreement_s2", "error_rate_pairwise"]:
        if metric not in df.columns:
            continue

        pivot = df.pivot_table(
            index="prompt_variant", columns="temperature",
            values=metric, aggfunc="first"
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Temperature × Prompt")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Prompt Variant")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_heatmap_temp_model(output_dir: Path):
    """Heatmap: Temperature × Model for key metrics."""
    df = _load_csv_safe(output_dir / "metrics_by_temp_model.csv")
    if df is None or len(df) == 0:
        return

    for metric in ["agreement_s1", "agreement_s2", "error_rate_pairwise"]:
        if metric not in df.columns:
            continue

        pivot = df.pivot_table(
            index="model_size_label", columns="temperature",
            values=metric, aggfunc="first"
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Temperature × Model")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Model")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_model_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_heatmap_temp_judgetype(output_dir: Path):
    """Heatmap: Temperature × Judge Type for error rates."""
    df = _load_csv_safe(output_dir / "metrics_by_temp_judgetype.csv")
    if df is None or len(df) == 0:
        return

    # Collect all error rate columns
    error_cols = [c for c in df.columns if "error_rate" in c]
    for metric in error_cols:
        pivot = df.pivot_table(
            index="judge_type", columns="temperature",
            values=metric, aggfunc="first"
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Reds", ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Temperature × Judge Type")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_jt_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_consistency_by_temperature(output_dir: Path):
    """Bar plot: consistency rate by temperature."""
    analysis_path = output_dir / "analysis.json"
    if not analysis_path.exists():
        return

    with open(analysis_path) as f:
        analysis = json.load(f)

    consistency = analysis.get("consistency_by_temperature", {})
    if not consistency:
        return

    temps = sorted(consistency.keys(), key=float)
    values = [consistency[t] for t in temps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([float(t) for t in temps], values, width=0.15, color="#5DADE2", edgecolor="black")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Consistency Rate")
    ax.set_title("Position Swap Consistency by Temperature")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_consistency.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_position_bias(output_dir: Path):
    """Plot position bias (P(first) - P(second)) by temperature."""
    df = _load_csv_safe(output_dir / "metrics_by_temperature.csv")
    if df is None or "position_bias" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["temperature"], df["position_bias"], width=0.15, color="#F39C12", edgecolor="black")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Position Bias (P(first) - P(second))")
    ax.set_title("Position Bias by Temperature")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_position_bias.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_causal_forest(output_dir: Path):
    """Forest plot of ATE estimates (simple + DML) for overview."""
    analysis_path = output_dir / "analysis.json"
    if not analysis_path.exists():
        return

    with open(analysis_path) as f:
        analysis = json.load(f)

    estimates = []

    simple = analysis.get("simple_ate", {})
    if "ate_simple" in simple:
        estimates.append(("Simple ATE\n(T=0.0 vs T=1.0)", simple["ate_simple"], None, None))

    dml = analysis.get("dml_ate", {})
    if "ate" in dml:
        estimates.append(("DML ATE", dml["ate"], dml.get("ate_ci_lower"), dml.get("ate_ci_upper")))

    # Stratified estimates
    for strat_key in ("stratified_by_prompt", "stratified_by_model"):
        strat = analysis.get(strat_key, {})
        for name, vals in strat.items():
            if "ate_simple" in vals:
                label = f"{strat_key.replace('stratified_by_', '').title()}: {name}"
                estimates.append((label, vals["ate_simple"], None, None))

    if not estimates:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(estimates) * 0.6)))
    y_positions = list(range(len(estimates)))
    labels = [e[0] for e in estimates]
    values = [e[1] for e in estimates]

    for i, (label, val, ci_lo, ci_hi) in enumerate(estimates):
        color = "#2E86C1" if "DML" in label else "#E74C3C" if "Simple" in label else "#27AE60"
        ax.plot(val, i, "o", color=color, markersize=8)
        if ci_lo is not None and ci_hi is not None:
            ax.hlines(i, ci_lo, ci_hi, color=color, linewidth=2)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("ATE (Agreement difference: T=0.0 vs T=1.0)")
    ax.set_title("Causal Effect Estimates: Temperature on Agreement")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_causal_forest.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distribution(output_dir: Path):
    """Box plot of score distributions by temperature for single-answer scoring."""
    df = _load_csv_safe(output_dir / "results.csv")
    if df is None:
        return

    for jt in ("single_answer", "reference_guided"):
        sub = df[df["judge_type"] == jt].copy()
        if len(sub) == 0:
            continue

        sub["score_a"] = pd.to_numeric(sub["score_a"], errors="coerce")
        valid = sub.dropna(subset=["score_a"])
        if len(valid) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        valid.boxplot(column="score_a", by="temperature", ax=ax)
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Score (1-10)")
        ax.set_title(f"Score Distribution by Temperature ({jt})")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(output_dir / f"plot_scores_{jt}.png", dpi=150, bbox_inches="tight")
        plt.close()


def generate_all_plots(output_dir: Path | str):
    """Generate all visualization plots and continue on per-plot failures."""
    output_dir = Path(output_dir)

    plot_funcs = [
        plot_metrics_by_temperature,
        plot_heatmap_temp_prompt,
        plot_heatmap_temp_model,
        plot_heatmap_temp_judgetype,
        plot_consistency_by_temperature,
        plot_position_bias,
        plot_causal_forest,
        plot_score_distribution,
    ]

    for func in plot_funcs:
        try:
            func(output_dir)
            logger.info("Generated: %s", func.__name__)
        except Exception as e:
            logger.error("Failed %s: %s", func.__name__, e)
