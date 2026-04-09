"""
eda.py
======
Exploratory Data Analysis — distributions, correlations, class-level comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional


def plot_eda_overview(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str = "income_label",
    save_path: Optional[str] = None,
) -> None:
    """
    Generate a 2×2 EDA overview figure:
      - Income class distribution
      - Age distribution by income
      - Correlation matrix (top numeric features)
      - Top 10 education categories

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list of str
    target_col : str
    save_path : str, optional
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── 1. Income Class Distribution ─────────────────────────────────────
    ax = axes[0, 0]
    counts = df[target_col].value_counts().sort_index()
    bars = ax.bar(
        [f"≤50K ({counts.index[0]})", f">50K ({counts.index[1]})"],
        counts.values,
        color=["#3498db", "#e74c3c"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Count")
    ax.set_title("Income Class Distribution")
    ax.set_xlabel("income_label")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1000,
            f"{val:,}",
            ha="center",
            fontsize=10,
        )

    # ── 2. Age Distribution by Income ────────────────────────────────────
    ax = axes[0, 1]
    for label, color, name in [(0, "#3498db", "≤50K"), (1, "#e74c3c", ">50K")]:
        subset = df[df[target_col] == label]["age"]
        ax.hist(subset, bins=50, alpha=0.6, color=color, label=name, edgecolor="white")
    ax.set_xlabel("age")
    ax.set_ylabel("Count")
    ax.set_title("age Distribution by Income")
    ax.legend()

    # ── 3. Correlation Matrix ────────────────────────────────────────────
    ax = axes[1, 0]
    # Select top 10 numeric features (exclude weight, year)
    corr_cols = [
        c for c in numeric_cols
        if c not in ["weight", "year"] and c in df.columns
    ][:10]
    corr = df[corr_cols].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, ax=ax, annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix (Top Numeric Features)")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    # ── 4. Top 10 Education Categories ───────────────────────────────────
    ax = axes[1, 1]
    if "education" in df.columns:
        edu_counts = df["education"].value_counts().head(10).sort_values()
        edu_counts.plot.barh(ax=ax, color="#2ecc71", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Count")
        ax.set_title("Top 10: education")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved EDA overview to {save_path}")

    plt.close(fig)


def print_categorical_summary(
    df: pd.DataFrame,
    categorical_cols: List[str],
    top_n: int = 5,
) -> None:
    """Print value counts for each categorical column."""
    print("\n--- Categorical Feature Value Counts (Top {}) ---".format(top_n))
    for col in categorical_cols:
        print(f"\n  {col}:")
        print(df[col].value_counts().head(top_n).to_string())
