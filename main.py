#!/usr/bin/env python3
"""
main.py
=======
Census Income Classification & Customer Segmentation — Full Pipeline

Usage:
    python main.py --data_path data/census-income.data --output_dir outputs
    python main.py --data_path data/census-income.data --steps eda classification
    python main.py --help
"""

import argparse
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Local imports ────────────────────────────────────────────────────────
from src.data_loader import load_data, inspect_data, encode_target
from src.eda import plot_eda_overview
from src.preprocessing import (
    split_features_target,
    get_train_test,
    build_preprocessor,
)
from src.classification import (
    get_models,
    train_and_evaluate,
    cross_validate_models,
    get_feature_importance,
)
from src.segmentation import (
    run_pca,
    find_optimal_k,
    fit_kmeans,
    profile_segments,
    print_segment_recommendations,
)
from src.visualization import (
    plot_classification_results,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_cluster_optimization,
    plot_segment_profiles,
    plot_cluster_scatter,
)

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════
# Argument Parser
# ═════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Census Income Classification & Segmentation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the census income data file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory for saving figures and results (default: outputs).",
    )

    # Pipeline control
    parser.add_argument(
        "--steps", nargs="+",
        default=["eda", "classification", "segmentation"],
        choices=["eda", "classification", "segmentation"],
        help="Which pipeline steps to run (default: all).",
    )

    # Train/test
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=5)

    # Classification hyperparameters
    parser.add_argument("--gb_n_estimators", type=int, default=200)
    parser.add_argument("--gb_max_depth", type=int, default=6)
    parser.add_argument("--gb_learning_rate", type=float, default=0.1)
    parser.add_argument("--gb_subsample", type=float, default=0.8)
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=20)

    # Segmentation hyperparameters
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--pca_variance", type=float, default=0.95)

    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # Create output directories
    fig_dir = Path(args.output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CENSUS INCOME CLASSIFICATION & CUSTOMER SEGMENTATION")
    print("=" * 70)
    print(f"  Data path:    {args.data_path}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Steps:        {args.steps}")
    print(f"  Random state: {args.random_state}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 0: Load & Inspect Data
    # ─────────────────────────────────────────────────────────────────────
    df = load_data(args.data_path)
    metadata = inspect_data(df)
    df = encode_target(df)

    numeric_cols = metadata["numeric_cols"]
    categorical_cols = metadata["categorical_cols"]

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: EDA
    # ─────────────────────────────────────────────────────────────────────
    if "eda" in args.steps:
        print("\n" + "=" * 70)
        print("STEP 1: EXPLORATORY DATA ANALYSIS")
        print("=" * 70)

        plot_eda_overview(
            df, numeric_cols,
            target_col="income_label",
            save_path=str(fig_dir / "eda_overview.png"),
        )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Preprocessing + Feature Engineering
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING")
    print("=" * 70)

    X, y, w = split_features_target(df)
    X_train, X_test, y_train, y_test, w_train, w_test = get_train_test(
        X, y, w,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    preprocessor, num_cols, low_card, high_card = build_preprocessor(X_train)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Classification
    # ─────────────────────────────────────────────────────────────────────
    if "classification" in args.steps:
        print("\n" + "=" * 70)
        print("STEP 3: CLASSIFICATION MODEL TRAINING & EVALUATION")
        print("=" * 70)

        models = get_models(
            random_state=args.random_state,
            gb_n_estimators=args.gb_n_estimators,
            gb_max_depth=args.gb_max_depth,
            gb_learning_rate=args.gb_learning_rate,
            gb_subsample=args.gb_subsample,
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=args.rf_max_depth,
        )

        # Train & evaluate
        results = train_and_evaluate(
            preprocessor, models,
            X_train, X_test, y_train, y_test, w_train,
        )

        # Cross-validation
        cv_results = cross_validate_models(
            preprocessor, models,
            X_train, y_train,
            cv=args.cv_folds,
            random_state=args.random_state,
        )

        # Best model
        best_name = max(results, key=lambda k: results[k]["auc_roc"])
        best_result = results[best_name]
        print(f"\n★ Best model: {best_name} (AUC-ROC = {best_result['auc_roc']:.4f})")

        # Feature importance
        fi_df = get_feature_importance(
            best_result["pipeline"], preprocessor, X_train, top_n=20,
        )

        # ── Plots ────────────────────────────────────────────────────────
        plot_classification_results(
            results,
            save_path=str(fig_dir / "classification_results.png"),
        )

        plot_confusion_matrix(
            best_result["cm"],
            model_name=best_name,
            save_path=str(fig_dir / "confusion_matrix.png"),
        )

        if not fi_df.empty:
            plot_feature_importance(
                fi_df,
                model_name=best_name,
                save_path=str(fig_dir / "feature_importance.png"),
            )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: Segmentation
    # ─────────────────────────────────────────────────────────────────────
    if "segmentation" in args.steps:
        print("\n" + "=" * 70)
        print("STEP 4: SEGMENTATION MODEL")
        print("=" * 70)

        # Use a sample for segmentation (speed up for large datasets)
        seg_sample_size = min(50000, len(X_train))
        np.random.seed(args.random_state)
        seg_idx = np.random.choice(len(X_train), seg_sample_size, replace=False)

        X_seg = X_train.iloc[seg_idx]
        y_seg = y_train.iloc[seg_idx]

        # Preprocess
        X_seg_processed = preprocessor.fit_transform(X_seg)

        # PCA
        X_pca, pca = run_pca(X_seg_processed, variance_threshold=args.pca_variance)

        # Find optimal K
        inertias, silhouettes, best_k_sil = find_optimal_k(
            X_pca, random_state=args.random_state,
        )

        chosen_k = args.n_clusters
        print(f"Chosen K for marketing segmentation: {chosen_k}")

        # Fit K-Means
        km, cluster_labels = fit_kmeans(
            X_pca, n_clusters=chosen_k, random_state=args.random_state,
        )

        # Profile segments
        # Use original (unprocessed) data for interpretability
        df_seg = df.iloc[X_seg.index].copy()

        profiles = profile_segments(
            df_seg, cluster_labels,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        print_segment_recommendations(
            profiles, df_seg, cluster_labels,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
        )

        # ── Plots ────────────────────────────────────────────────────────
        plot_cluster_optimization(
            inertias, silhouettes,
            save_path=str(fig_dir / "cluster_optimization.png"),
        )

        plot_segment_profiles(
            df_seg, cluster_labels, profiles,
            save_path=str(fig_dir / "segment_profiles.png"),
        )

        # 2D scatter (use first 2 PCA components)
        plot_cluster_scatter(
            X_pca[:, :2], cluster_labels,
            y=y_seg.values,
            save_path=str(fig_dir / "segmentation_visualization.png"),
        )

    # ─────────────────────────────────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print(f"   Figures saved to: {fig_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
