"""
data_loader.py
==============
Load census income data, inspect schema, and perform initial validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


# ── Default column names (42 columns for Census Income KDD dataset) ──────────
COLUMN_NAMES = [
    "age", "class of worker", "detailed industry recode",
    "detailed occupation recode", "education", "wage per hour",
    "enroll in edu inst last wk", "marital stat", "major industry code",
    "major occupation code", "race", "hispanic origin", "sex",
    "member of a labor union", "reason for unemployment",
    "full or part time employment stat", "capital gains", "capital losses",
    "dividends from stocks", "tax filer stat", "region of previous residence",
    "state of previous residence", "detailed household and family stat",
    "detailed household summary in household", "weight",
    "migration code-change in msa", "migration code-change in reg",
    "migration code-move within reg", "live in this house 1 year ago",
    "migration prev res in sunbelt", "num persons worked for employer",
    "family members under 18", "country of birth father",
    "country of birth mother", "country of birth self", "citizenship",
    "own business or self employed",
    "fill inc questionnaire for veteran's admin", "veterans benefits",
    "weeks worked in year", "year", "label",
]

TARGET_COL = "label"
WEIGHT_COL = "weight"
TARGET_MAP = {"- 50000": 0, "50000+": 1}


def load_data(
    filepath: str,
    column_names: Optional[List[str]] = None,
    sep: str = ",",
) -> pd.DataFrame:
    """
    Load the census income dataset from a CSV / .data file.

    Parameters
    ----------
    filepath : str
        Path to the raw data file.
    column_names : list of str, optional
        Column names. Uses COLUMN_NAMES if None.
    sep : str
        Delimiter (default comma).

    Returns
    -------
    pd.DataFrame
    """
    cols = column_names or COLUMN_NAMES
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading data from {filepath} ...")
    df = pd.read_csv(filepath, names=cols, sep=sep, skipinitialspace=True)
    print(f"  → Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Print comprehensive data summary and return metadata dict.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict  with keys: 'shape', 'numeric_cols', 'categorical_cols',
          'missing', 'target_dist'
    """
    print("\n" + "=" * 70)
    print("DATA EXPLORATION")
    print("=" * 70)

    # ── Shape ────────────────────────────────────────────────────────────
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    # ── First rows ───────────────────────────────────────────────────────
    print("\n--- First 5 Rows ---")
    print(df.head())

    # ── Data types ───────────────────────────────────────────────────────
    print("\n--- Data Types ---")
    print(df.dtypes)

    # ── Statistics ───────────────────────────────────────────────────────
    print("\n--- Statistical Summary (Numeric) ---")
    print(df.describe())

    print("\n--- Statistical Summary (Categorical) ---")
    print(df.describe(include="object"))

    # ── Column classification ────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    # Remove target and weight from feature lists
    feature_numeric = [c for c in numeric_cols if c not in [TARGET_COL, WEIGHT_COL]]
    feature_categorical = [c for c in categorical_cols if c != TARGET_COL]

    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # ── Target ───────────────────────────────────────────────────────────
    print(f"\nTarget column: '{TARGET_COL}'")
    print(f"Target value distribution:")
    print(df[TARGET_COL].value_counts())

    # ── Missing values ───────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing[missing > 0],
        "Missing %": missing_pct[missing > 0],
    }).sort_values("Missing %", ascending=False)

    print("\n--- Missing Values ---")
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("  No missing values detected (may be encoded as '?').")

    # ── Replace '?' with NaN ─────────────────────────────────────────────
    n_question = (df == "?").sum().sum()
    if n_question > 0:
        print(f"\n  Found {n_question:,} '?' entries → replacing with NaN")
        df.replace("?", np.nan, inplace=True)
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            "Missing Count": missing[missing > 0],
            "Missing %": missing_pct[missing > 0],
        }).sort_values("Missing %", ascending=False)
        print(missing_df)

    print(f"\nTotal columns with missing values: {(df.isnull().sum() > 0).sum()}")

    # ── Cardinality ──────────────────────────────────────────────────────
    print("\n--- Categorical Feature Cardinality ---")
    for col in feature_categorical:
        n_unique = df[col].nunique()
        print(f"  {col}: {n_unique} unique values")

    high_card = [c for c in feature_categorical if df[c].nunique() > 20]
    low_card = [c for c in feature_categorical if df[c].nunique() <= 20]
    print(f"\nHigh-cardinality categoricals (>20 unique): {high_card}")
    print(f"Low-cardinality categoricals (≤20 unique): {low_card}")

    return {
        "shape": df.shape,
        "numeric_cols": feature_numeric,
        "categorical_cols": feature_categorical,
        "high_card_cols": high_card,
        "low_card_cols": low_card,
        "missing_df": missing_df,
    }


def encode_target(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    """
    Map the label column to binary 0/1.

    Parameters
    ----------
    df : pd.DataFrame
    col : str

    Returns
    -------
    pd.DataFrame with new column 'income_label'
    """
    # Clean label strings
    df[col] = df[col].str.replace(".", "", regex=False).str.strip()

    unique_vals = df[col].unique()
    print(f"\nUnique target values: {unique_vals}")
    print(f"Target mapping: {TARGET_MAP}")

    df["income_label"] = df[col].map(TARGET_MAP)

    print(f"\nEncoded target distribution:")
    print(df["income_label"].value_counts())
    print(f"\nClass balance:")
    print(df["income_label"].value_counts(normalize=True))

    return df
