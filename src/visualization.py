"""
visualization.py
================
All plotting functions for classification results, segmentation, and model comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


# ── Color palette ────────────────────────────────────────────────────────
COLORS = {
    "Logistic Regression": "#3498db",
    "Random Forest": "#e67e22",
    "Gradient Boosting": "#2ecc71",
}


def plot_classification_results(
    results: Dict,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot ROC curves, PR curves, and model comparison bar chart (3 subplots).
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── 1. ROC Curves ────────────────────────────────────────────────────
    ax = axes[0]
    for name, res in results.items():
        ax.plot(
            res["fpr"], res["tpr"],
            label=f'{name} (AUC={res["auc_roc"]:.3f})',
            color=COLORS.get(name, "gray"),
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")

    # ── 2. Precision-Recall Curves ───────────────────────────────────────
    ax = axes[1]
    for name, res in results.items():
        ax.plot(
            res["recall"], res["precision"],
            label=name,
            color=COLORS.get(name, "gray"),
            linewidth=2,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")

    # ── 3. Model Comparison Bar Chart ────────────────────────────────────
    ax = axes[2]
    metrics = ["accuracy", "f1", "auc_roc"]
    metric_labels = ["Accuracy", "F1", "AUC-ROC"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, (name, res) in enumerate(results.items()):
        values = [res[m] for m in metrics]
        ax.bar(
            x + i * width, values, width,
            label=name, color=COLORS.get(name, "gray"),
            edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model Comparison")
    ax.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved classification results to {save_path}")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Gradient Boosting",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a heatmap confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["≤50K", ">50K"],
        yticklabels=["≤50K", ">50K"],
        ax=ax,
        annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved confusion matrix to {save_path}")
    plt.close(fig)


def plot_feature_importance(
    fi_df: pd.DataFrame,
    model_name: str = "Gradient Boosting",
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart of top feature importances.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    fi_sorted = fi_df.sort_values("Importance", ascending=True)
    ax.barh(
        fi_sorted["Feature"], fi_sorted["Importance"],
        color="#2ecc71", edgecolor="black", linewidth=0.5,
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(fi_df)} Feature Importances - {model_name}")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved feature importance to {save_path}")
    plt.close(fig)


def plot_cluster_optimization(
    inertias: Dict[int, float],
    silhouettes: Dict[int, float],
    save_path: Optional[str] = None,
) -> None:
    """
    Elbow method + Silhouette analysis (2 subplots).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ks = sorted(inertias.keys())

    # Elbow
    ax = axes[0]
    ax.plot(ks, [inertias[k] for k in ks], "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")

    # Silhouette
    ax = axes[1]
    ax.plot(ks, [silhouettes[k] for k in ks], "ro-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved cluster optimization to {save_path}")
    plt.close(fig)


def plot_segment_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    profiles: Dict[str, pd.DataFrame],
    target_col: str = "income_label",
    save_path: Optional[str] = None,
) -> None:
    """
    4-panel segment profile figure:
      - Income >50K proportion by segment
      - Age distribution by segment
      - Segment size pie chart
      - Normalized feature means by segment
    """
    df = df.copy()
    df["Cluster"] = labels

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    n_clusters = len(df["Cluster"].unique())
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # ── 1. Income proportion ─────────────────────────────────────────────
    ax = axes[0, 0]
    income_df = profiles["income"]
    clusters = income_df.index
    proportions = income_df["Proportion >50K"]
    bars = ax.bar(clusters, proportions, color=cluster_colors[:n_clusters],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, proportions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.2f}", ha="center", fontsize=10)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion >50K")
    ax.set_title("Income >50K Proportion by Segment")

    # ── 2. Age distribution ──────────────────────────────────────────────
    ax = axes[0, 1]
    for i in range(n_clusters):
        subset = df[df["Cluster"] == i]["age"]
        ax.hist(subset, bins=40, alpha=0.5, label=f"Cluster {i}",
                color=cluster_colors[i], edgecolor="white")
    ax.set_xlabel("age")
    ax.set_ylabel("Count")
    ax.set_title("age Distribution by Segment")
    ax.legend()

    # ── 3. Segment size pie chart ────────────────────────────────────────
    ax = axes[1, 0]
    sizes = income_df["Size"].values
    pcts = sizes / sizes.sum() * 100
    labels_pie = [f"Segment {i}\n(n={s:,})" for i, s in zip(clusters, sizes)]
    ax.pie(
        pcts, labels=labels_pie, autopct="%.1f%%",
        colors=cluster_colors[:n_clusters],
        startangle=90, textprops={"fontsize": 10},
    )
    ax.set_title("Segment Size Distribution")

    # ── 4. Normalized feature means ──────────────────────────────────────
    ax = axes[1, 1]
    num_means = profiles["numeric_means"]
    # Select interesting features
    plot_cols = [c for c in [
        "age", "detailed industry recode", "detailed occupation recode",
        "wage per hour", "capital gains", "capital losses",
    ] if c in num_means.columns]

    if plot_cols:
        plot_data = num_means[plot_cols].copy()
        # Normalize to [0, 1] for comparison
        for col in plot_data.columns:
            col_range = plot_data[col].max() - plot_data[col].min()
            if col_range > 0:
                plot_data[col] = (plot_data[col] - plot_data[col].min()) / col_range

        plot_data.plot.bar(ax=ax, width=0.8)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Normalized Mean")
        ax.set_title("Normalized Feature Means by Segment")
        ax.legend(fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved segment profiles to {save_path}")
    plt.close(fig)


def plot_cluster_scatter(
    X_pca: np.ndarray,
    labels: np.ndarray,
    y: np.ndarray = None,
    save_path: Optional[str] = None,
) -> None:
    """
    2D PCA scatter plot: clusters (left) and income overlay (right).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: Cluster assignments ────────────────────────────────────────
    ax = axes[0]
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap="Set1", alpha=0.4, s=5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Customer Segments (K={len(np.unique(labels))})")
    plt.colorbar(scatter, ax=ax, label="Cluster")

    # ── Right: Income overlay ────────────────────────────────────────────
    ax = axes[1]
    if y is not None:
        scatter2 = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=y, cmap="RdYlBu", alpha=0.4, s=5,
        )
        plt.colorbar(scatter2, ax=ax, label="Income (0=≤50K, 1=>50K)")
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c="gray", alpha=0.3, s=5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Income Class Overlay")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved cluster scatter to {save_path}")
    plt.close(fig)
