"""
segmentation.py
===============
Unsupervised customer segmentation using K-Means + PCA.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_pca(
    X_processed: np.ndarray,
    variance_threshold: float = 0.95,
) -> Tuple[np.ndarray, PCA]:
    """
    Fit PCA retaining `variance_threshold` of total variance.

    Parameters
    ----------
    X_processed : np.ndarray
        Already preprocessed feature matrix.
    variance_threshold : float

    Returns
    -------
    X_pca : np.ndarray
    pca : fitted PCA object
    """
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_processed)

    n_comp = pca.n_components_
    ratios = pca.explained_variance_ratio_

    print(f"\nPCA components retaining {variance_threshold*100:.0f}% variance: {n_comp}")
    print(f"Explained variance ratios (first 10): "
          f"{[round(r, 4) for r in ratios[:10]]}")

    return X_pca, pca


def find_optimal_k(
    X_pca: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Tuple[Dict[int, float], Dict[int, float], int]:
    """
    Evaluate K using Elbow (inertia) and Silhouette methods.

    Parameters
    ----------
    X_pca : np.ndarray
    k_range : range
    random_state : int

    Returns
    -------
    inertias, silhouettes, best_k_silhouette
    """
    print("\nFinding optimal K using Elbow + Silhouette methods...")

    inertias = {}
    silhouettes = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(X_pca)
        inertias[k] = km.inertia_
        sil = silhouette_score(X_pca, labels, sample_size=min(10000, len(X_pca)))
        silhouettes[k] = sil
        print(f"  K={k}: Inertia={km.inertia_:.0f}, Silhouette={sil:.4f}")

    best_k = max(silhouettes, key=silhouettes.get)
    print(f"\nOptimal K by silhouette: {best_k}")

    return inertias, silhouettes, best_k


def fit_kmeans(
    X_pca: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[KMeans, np.ndarray]:
    """
    Fit K-Means with chosen K.

    Returns
    -------
    km : fitted KMeans
    labels : cluster assignments
    """
    print(f"\nFitting K-Means with K={n_clusters}...")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(X_pca)

    print(f"\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c:,} observations ({c/len(labels)*100:.1f}%)")

    return km, labels


def profile_segments(
    df_original: pd.DataFrame,
    labels: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: str = "income_label",
) -> Dict[str, pd.DataFrame]:
    """
    Build comprehensive profiles for each segment.

    Parameters
    ----------
    df_original : pd.DataFrame (original data, pre-preprocessing)
    labels : np.ndarray (cluster assignments, aligned with df_original)
    numeric_cols : list of str
    categorical_cols : list of str
    target_col : str

    Returns
    -------
    profiles : dict with keys 'income', 'numeric_means', 'categorical_modes'
    """
    df = df_original.copy()
    df["Cluster"] = labels

    print("\n" + "=" * 70)
    print("SEGMENT PROFILES")
    print("=" * 70)

    # ── Income composition ───────────────────────────────────────────────
    income_by_cluster = df.groupby("Cluster")[target_col].agg(["mean", "count"])
    income_by_cluster.columns = ["Proportion >50K", "Size"]
    income_by_cluster["Proportion ≤50K"] = 1 - income_by_cluster["Proportion >50K"]

    print("\n--- Income Composition by Cluster ---")
    print(income_by_cluster)

    # ── Numeric feature means ────────────────────────────────────────────
    valid_numeric = [c for c in numeric_cols if c in df.columns]
    numeric_means = df.groupby("Cluster")[valid_numeric].mean().round(2)

    print("\n--- Numeric Feature Means by Cluster ---")
    print(numeric_means.T)

    # ── Categorical modes ────────────────────────────────────────────────
    valid_cat = [c for c in categorical_cols if c in df.columns][:10]
    cat_modes = df.groupby("Cluster")[valid_cat].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A"
    )

    print("\n--- Categorical Feature Modes by Cluster ---")
    print(cat_modes.T)

    return {
        "income": income_by_cluster,
        "numeric_means": numeric_means,
        "categorical_modes": cat_modes,
    }


def print_segment_recommendations(
    profiles: Dict[str, pd.DataFrame],
    df_original: pd.DataFrame,
    labels: np.ndarray,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> None:
    """
    Print human-readable segment descriptions and marketing recommendations.
    """
    df = df_original.copy()
    df["Cluster"] = labels

    income_df = profiles["income"]
    num_means = profiles["numeric_means"]

    print("\n" + "=" * 70)
    print("MARKETING SEGMENT DESCRIPTIONS & RECOMMENDATIONS")
    print("=" * 70)

    # ── Recommendation logic (based on income proportion) ────────────────
    for cluster_id in sorted(income_df.index):
        row = income_df.loc[cluster_id]
        size = int(row["Size"])
        pct = size / len(df) * 100
        high_income_pct = row["Proportion >50K"] * 100

        print(f"\n--- Segment {cluster_id} ---")
        print(f"  Size: {size:,} ({pct:.1f}% of population)")
        print(f"  High income (>50K): {high_income_pct:.1f}%")

        # Print top categorical modes
        valid_cat = [c for c in categorical_cols if c in df.columns]
        for col in valid_cat[:5]:
            mode_val = df[df["Cluster"] == cluster_id][col].mode()
            if len(mode_val) > 0:
                print(f"  Most common {col}: {mode_val.iloc[0]}")

        # Print key numeric means
        valid_num = [c for c in numeric_cols if c in df.columns]
        for col in valid_num[:5]:
            mean_val = df[df["Cluster"] == cluster_id][col].mean()
            print(f"  Mean {col}: {mean_val:.2f}")

        # Marketing recommendation
        if high_income_pct > 10:
            rec = ("Premium targeting - offer luxury goods, investment services, "
                   "career development programs")
        elif high_income_pct > 3:
            rec = ("Mid-tier targeting - offer quality products with value "
                   "proposition, skill-building opportunities")
        else:
            rec = ("Value-focused marketing - target with deals, essentials, "
                   "budget-friendly products")
        print(f"  >> RECOMMENDATION: {rec}")
