"""Metrics computation: Agreement, Consistency, Error rate, Position bias."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


def _pairwise_to_human_label(human_winner: str) -> str:
    """Convert dataset winner to A/B/C label."""
    if human_winner == "model_a":
        return "A"
    elif human_winner == "model_b":
        return "B"
    else:
        return "C"  # tie


def compute_agreement_s1(df: pd.DataFrame) -> float:
    """Agreement S1: (agreements + ties) / total.

    Tie or disagreement both count as 'tie' agreement.
    """
    if len(df) == 0:
        return 0.0
    df = df[~df["pairwise_winner"].isna()].copy()
    if len(df) == 0:
        return 0.0
    df["human_label"] = df["human_winner"].apply(_pairwise_to_human_label)

    agree = 0
    for _, row in df.iterrows():
        judge = row["pairwise_winner"]
        human = row["human_label"]
        if judge == human:
            agree += 1
        elif judge == "C" or human == "C":
            agree += 1  # tie counted as agreement

    return agree / len(df)


def compute_agreement_s2(df: pd.DataFrame) -> float:
    """Agreement S2: agreements / non-tie total.

    Only consider cases where both human and judge gave non-tie.
    """
    if len(df) == 0:
        return 0.0
    df = df[~df["pairwise_winner"].isna()].copy()
    if len(df) == 0:
        return 0.0
    df["human_label"] = df["human_winner"].apply(_pairwise_to_human_label)

    non_tie = df[(df["human_label"] != "C") & (df["pairwise_winner"] != "C")]
    if len(non_tie) == 0:
        return 0.0
    agree = (non_tie["pairwise_winner"] == non_tie["human_label"]).sum()
    return agree / len(non_tie)


def compute_consistency(df_orig: pd.DataFrame, df_swap: pd.DataFrame) -> float:
    """Consistency: fraction of pairs where original and swapped order agree.

    Both dataframes should have the same question_ids and repeat_ids.
    After swapping, the winner should have been flipped back already.
    """
    if len(df_orig) == 0 or len(df_swap) == 0:
        return 0.0

    merged = pd.merge(
        df_orig[["question_id", "repeat_id", "pairwise_winner"]],
        df_swap[["question_id", "repeat_id", "pairwise_winner"]],
        on=["question_id", "repeat_id"],
        suffixes=("_orig", "_swap"),
    )
    if len(merged) == 0:
        return 0.0

    valid = merged.dropna(subset=["pairwise_winner_orig", "pairwise_winner_swap"])
    if len(valid) == 0:
        return 0.0

    consistent = (valid["pairwise_winner_orig"] == valid["pairwise_winner_swap"]).sum()
    return consistent / len(valid)


def compute_error_rate(df: pd.DataFrame) -> float:
    """Error rate: fraction of outputs that failed to parse."""
    if len(df) == 0:
        return 0.0
    return df["parse_error"].mean()


def compute_position_bias(df: pd.DataFrame) -> dict[str, float]:
    """Compute position bias metrics.

    Returns dict with:
        - p_first: P(judge picks first position / A)
        - p_second: P(judge picks second position / B)
        - bias: p_first - p_second
        - abs_bias: |p_first - 0.5| (deviation from fair)
    """
    valid = df[df["pairwise_winner"].isin(["A", "B"])].copy()
    if len(valid) == 0:
        return {"p_first": 0.0, "p_second": 0.0, "bias": 0.0, "abs_bias": 0.0}

    p_first = (valid["pairwise_winner"] == "A").mean()
    p_second = (valid["pairwise_winner"] == "B").mean()
    return {
        "p_first": p_first,
        "p_second": p_second,
        "bias": p_first - p_second,
        "abs_bias": abs(p_first - 0.5),
    }


def compute_score_stats(df: pd.DataFrame) -> dict[str, float]:
    """Compute scoring statistics for single-answer / reference-guided judges.

    Returns mean, std, and range for score_a and score_b.
    """
    result = {}
    for col in ("score_a", "score_b"):
        scores = df[col].dropna()
        if len(scores) == 0:
            result[f"{col}_mean"] = float("nan")
            result[f"{col}_std"] = float("nan")
        else:
            result[f"{col}_mean"] = scores.mean()
            result[f"{col}_std"] = scores.std()
    return result


def compute_ranking_correlation(df: pd.DataFrame, score_col: str = "score_a") -> dict[str, float]:
    """Compute Kendall tau and Spearman correlation between judge scores and human preference.

    Maps human_winner to a numeric scale: model_a=1, tie=0.5, model_b=0
    Maps judge score to the score_a value.
    """
    valid = df.dropna(subset=[score_col])
    if len(valid) < 3:
        return {"kendall_tau": float("nan"), "spearman_rho": float("nan")}

    human_score = valid["human_winner"].map({"model_a": 1.0, "tie": 0.5, "model_b": 0.0})
    judge_score = valid[score_col].astype(float)

    tau, _ = stats.kendalltau(judge_score, human_score)
    rho, _ = stats.spearmanr(judge_score, human_score)
    return {"kendall_tau": tau, "spearman_rho": rho}


def compute_all_metrics(
    results_df: pd.DataFrame,
    consistency_df_orig: Optional[pd.DataFrame] = None,
    consistency_df_swap: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """Compute all metrics for a given results dataframe.

    Returns a dict with all metric values.
    """
    metrics = {}

    # Pairwise metrics
    pw = results_df[results_df["judge_type"] == "pairwise"]
    if len(pw) > 0:
        metrics["agreement_s1"] = compute_agreement_s1(pw)
        metrics["agreement_s2"] = compute_agreement_s2(pw)
        metrics["error_rate_pairwise"] = compute_error_rate(pw)
        bias = compute_position_bias(pw)
        metrics.update({f"position_{k}": v for k, v in bias.items()})

    # Reference-guided pairwise metrics
    rg = results_df[results_df["judge_type"] == "reference_guided"]
    if len(rg) > 0:
        metrics["reference_guided_agreement_s1"] = compute_agreement_s1(rg)
        metrics["reference_guided_agreement_s2"] = compute_agreement_s2(rg)
        metrics["error_rate_reference_guided"] = compute_error_rate(rg)
        rg_bias = compute_position_bias(rg)
        metrics.update({f"reference_guided_position_{k}": v for k, v in rg_bias.items()})

    # Consistency (requires paired original/swap data)
    if consistency_df_orig is not None and consistency_df_swap is not None:
        metrics["consistency"] = compute_consistency(consistency_df_orig, consistency_df_swap)

    # Scoring metrics
    for jt in ("single_answer",):
        sc = results_df[results_df["judge_type"] == jt]
        if len(sc) > 0:
            score_stats = compute_score_stats(sc)
            metrics.update({f"{jt}_{k}": v for k, v in score_stats.items()})
            metrics[f"error_rate_{jt}"] = compute_error_rate(sc)
            rank_corr = compute_ranking_correlation(sc, "score_a")
            metrics.update({f"{jt}_{k}": v for k, v in rank_corr.items()})

    return metrics


def aggregate_metrics_by_group(
    results_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Aggregate metrics grouped by specified columns (e.g., temperature, model, prompt)."""
    records = []
    for group_key, group_df in results_df.groupby(group_cols):
        if isinstance(group_key, tuple):
            group_dict = dict(zip(group_cols, group_key))
        else:
            group_dict = {group_cols[0]: group_key}

        metrics = compute_all_metrics(group_df)
        group_dict.update(metrics)
        group_dict["n_samples"] = len(group_df)
        records.append(group_dict)

    return pd.DataFrame(records)
