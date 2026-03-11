"""Causal analysis: DAG specification, ATE estimation via DML, statistical tests."""

from __future__ import annotations

import logging
from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAG specification using pgmpy
# ---------------------------------------------------------------------------

def build_causal_dag():
    """Build the causal DAG for the LLM-as-a-judge temperature experiment.

    Nodes:
        Temperature (T) - treatment
        JudgeModelSize (M) - covariate
        PromptType (P) - covariate
        InputResponses (X) - baseline (measured)
        Judgment (J) - outcome
        HumanJudgment (H) - ground truth
        Metrics (Y) - derived outcome

    Edges:
        T -> J  (direct causal effect of temperature on judgment)
        M -> J  (model size affects judgment quality)
        P -> J  (prompt engineering affects judgment)
        X -> J  (input content affects judgment)
        J -> Y  (judgment determines metrics)
        H -> Y  (human judgment used to compute metrics)
        M -> Y  (model size moderates metric quality)
        P -> Y  (prompt type moderates metrics)
    """
    from pgmpy.models import DiscreteBayesianNetwork

    edges = [
        ("Temperature", "Judgment"),
        ("JudgeModelSize", "Judgment"),
        ("PromptType", "Judgment"),
        ("InputResponses", "Judgment"),
        ("Judgment", "Metrics"),
        ("HumanJudgment", "Metrics"),
        ("JudgeModelSize", "Metrics"),
        ("PromptType", "Metrics"),
        ("Temperature", "Metrics"),  # indirect via Judgment + direct
    ]

    model = DiscreteBayesianNetwork(edges)
    return model


def visualize_dag(model, output_path: str = "results/causal_dag.png"):
    """Visualize the causal DAG and save to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph(model.edges())

    pos = {
        "Temperature": (0, 2),
        "JudgeModelSize": (2, 3),
        "PromptType": (2, 1),
        "InputResponses": (-2, 2),
        "Judgment": (1, 2),
        "HumanJudgment": (3, 3),
        "Metrics": (3, 2),
    }

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    nx.draw(
        G, pos, ax=ax,
        with_labels=True,
        node_color="#AED6F1",
        node_size=3000,
        font_size=9,
        font_weight="bold",
        arrowsize=20,
        edge_color="#5D6D7E",
        width=2,
    )
    # Color the treatment node differently
    nx.draw_networkx_nodes(G, pos, nodelist=["Temperature"], node_color="#F5B041", node_size=3000, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=["Metrics"], node_color="#82E0AA", node_size=3000, ax=ax)

    ax.set_title("Causal DAG: Temperature → LLM-as-a-Judge → Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("DAG saved to %s", output_path)


# ---------------------------------------------------------------------------
# Causal effect estimation via Double Machine Learning (DML)
# ---------------------------------------------------------------------------

def prepare_dml_data(results_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for DML estimation.

    Encodes categorical variables and computes the outcome (agreement with human).
    """
    df = results_df.copy()

    # Compute binary agreement outcome for pairwise
    pw = df[df["judge_type"] == "pairwise"].copy()
    if len(pw) == 0:
        return pd.DataFrame()

    def _agrees(row):
        human = {"model_a": "A", "model_b": "B", "tie": "C"}.get(row["human_winner"])
        return 1.0 if row["pairwise_winner"] == human else 0.0

    pw["agreement"] = pw.apply(_agrees, axis=1)
    pw["parse_error_num"] = pw["parse_error"].astype(float)

    # Encode categoricals
    pw["prompt_code"] = pd.Categorical(pw["prompt_variant"]).codes.astype(float)
    pw["model_code"] = pd.Categorical(pw["model_size_label"]).codes.astype(float)

    return pw


def estimate_ate_dml(
    df: pd.DataFrame,
    treatment_col: str = "temperature",
    outcome_col: str = "agreement",
    covariate_cols: list[str] | None = None,
) -> dict:
    """Estimate Average Treatment Effect using Double Machine Learning (LinearDML).

    Compares treatment levels to basline (T=0.0).
    Returns ATE estimates and confidence intervals.
    """
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV, LogisticRegressionCV

    if covariate_cols is None:
        covariate_cols = ["prompt_code", "model_code"]

    df = df.dropna(subset=[outcome_col, treatment_col] + covariate_cols)
    if len(df) < 20:
        logger.warning("Too few samples (%d) for DML estimation", len(df))
        return {"error": "insufficient_data", "n": len(df)}

    Y = df[outcome_col].values
    T = df[treatment_col].values.reshape(-1, 1)
    X = df[covariate_cols].values

    try:
        est = LinearDML(
            model_y=LassoCV(cv=3, max_iter=5000),
            model_t=LassoCV(cv=3, max_iter=5000),
            random_state=42,
        )
        est.fit(Y, T, X=X)

        ate = est.ate(X=X)
        ate_interval = est.ate_interval(X=X, alpha=0.05)

        # Heterogeneous effects (CATE at mean covariates)
        X_mean = X.mean(axis=0).reshape(1, -1)
        cate = est.effect(X=X_mean)

        return {
            "ate": float(ate),
            "ate_ci_lower": float(ate_interval[0]),
            "ate_ci_upper": float(ate_interval[1]),
            "cate_at_mean": float(cate[0]),
            "n_samples": len(df),
        }
    except Exception as e:
        logger.error("DML estimation failed: %s", e)
        return {"error": str(e), "n": len(df)}


def estimate_ate_simple(df: pd.DataFrame, outcome_col: str = "agreement") -> dict:
    """Simple ATE estimation: E[Y|do(T=0.0)] - E[Y|do(T=1.0)].

    Also computes per-temperature means and ANOVA test.
    """
    if len(df) == 0:
        return {}

    temp_groups = df.groupby("temperature")[outcome_col]
    means = temp_groups.mean().to_dict()
    stds = temp_groups.std().to_dict()
    counts = temp_groups.count().to_dict()

    # ATE: T=0.0 vs T=1.0
    t0 = df[df["temperature"] == 0.0][outcome_col]
    t1 = df[df["temperature"] == 1.0][outcome_col]
    ate = t0.mean() - t1.mean() if len(t0) > 0 and len(t1) > 0 else float("nan")

    # ANOVA across all temperature groups
    groups = [g.values for _, g in temp_groups if len(g) > 1]
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
    else:
        f_stat, p_value = float("nan"), float("nan")

    # Spearman correlation: temperature vs outcome
    if len(df) > 5:
        rho, rho_p = stats.spearmanr(df["temperature"], df[outcome_col])
    else:
        rho, rho_p = float("nan"), float("nan")

    return {
        "ate_simple": ate,
        "means_by_temp": means,
        "stds_by_temp": stds,
        "counts_by_temp": counts,
        "anova_f": f_stat,
        "anova_p": p_value,
        "spearman_rho": rho,
        "spearman_p": rho_p,
    }


def stratified_analysis(
    df: pd.DataFrame,
    stratify_col: str,
    outcome_col: str = "agreement",
) -> dict:
    """Run ATE estimation stratified by a covariate (e.g., prompt_variant, model_size_label)."""
    results = {}
    for stratum, sdf in df.groupby(stratify_col):
        results[stratum] = estimate_ate_simple(sdf, outcome_col)
    return results
