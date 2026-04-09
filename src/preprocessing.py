"""
preprocessing.py
================
Feature engineering, train/test split, and sklearn ColumnTransformer pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "income_label",
    weight_col: str = "weight",
    drop_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Separate features, target, and sample weights.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    weight_col : str
    drop_cols : list of str
        Additional columns to drop from features.

    Returns
    -------
    X, y, sample_weights
    """
    if drop_cols is None:
        drop_cols = []

    # Columns to exclude from features
    exclude = [target_col, weight_col, "label"] + drop_cols
    exclude = [c for c in exclude if c in df.columns]

    y = df[target_col].copy()
    w = df[weight_col].copy() if weight_col in df.columns else pd.Series(
        np.ones(len(df))
    )
    X = df.drop(columns=exclude, errors="ignore")

    print(f"\nFeature columns: {X.shape[1]}")
    numeric_feats = X.select_dtypes(include=np.number).columns.tolist()
    cat_feats = X.select_dtypes(include="object").columns.tolist()
    print(f"Numeric features ({len(numeric_feats)}): {numeric_feats}")
    print(f"Categorical features ({len(cat_feats)}): {cat_feats}")

    return X, y, w


def get_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """
    Stratified train/test split preserving class distribution.

    Returns
    -------
    X_train, X_test, y_train, y_test, w_train, w_test
    """
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"\nTrain set: {X_train.shape[0]:,} samples")
    print(f"Test set:  {X_test.shape[0]:,} samples")
    print(f"Train class dist: {y_train.value_counts().to_dict()}")
    print(f"Test class dist:  {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, w_train, w_test


def build_preprocessor(
    X: pd.DataFrame,
    high_card_threshold: int = 20,
    max_ohe_categories: int = 20,
) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    """
    Build a ColumnTransformer for mixed-type data.

    - Numeric: median imputation + standard scaling
    - Low-cardinality categorical: mode imputation + one-hot encoding
    - High-cardinality categorical: mode imputation + frequency encoding

    Parameters
    ----------
    X : pd.DataFrame
    high_card_threshold : int
    max_ohe_categories : int

    Returns
    -------
    preprocessor (ColumnTransformer), numeric_cols, low_card_cols, high_card_cols
    """
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    low_card_cols = [c for c in cat_cols if X[c].nunique() <= high_card_threshold]
    high_card_cols = [c for c in cat_cols if X[c].nunique() > high_card_threshold]

    print(f"\n--- Preprocessor Configuration ---")
    print(f"  Numeric cols ({len(numeric_cols)}): {numeric_cols}")
    print(f"  Low-card categorical ({len(low_card_cols)}): {low_card_cols}")
    print(f"  High-card categorical ({len(high_card_cols)}): {high_card_cols}")

    # ── Numeric pipeline ─────────────────────────────────────────────────
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # ── Low-cardinality categorical pipeline ─────────────────────────────
    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            max_categories=max_ohe_categories,
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    # ── High-cardinality categorical pipeline ────────────────────────────
    # Frequency encoding: replace each category with its proportion
    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("freq_encode", FunctionTransformer(
            func=_frequency_encode,
            validate=False,
        )),
    ])

    # ── Combine ──────────────────────────────────────────────────────────
    transformers = [
        ("num", numeric_pipeline, numeric_cols),
    ]
    if low_card_cols:
        transformers.append(("low_cat", low_card_pipeline, low_card_cols))
    if high_card_cols:
        transformers.append(("high_cat", high_card_pipeline, high_card_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor, numeric_cols, low_card_cols, high_card_cols


def _frequency_encode(X: np.ndarray) -> np.ndarray:
    """
    Replace each category with its frequency (proportion) in the column.
    Works on 2D numpy arrays (from SimpleImputer output).
    """
    X_out = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        unique, counts = np.unique(col, return_counts=True)
        freq_map = dict(zip(unique, counts / len(col)))
        X_out[:, j] = np.array([freq_map.get(v, 0.0) for v in col])
    return X_out
