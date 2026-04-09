"""
classification.py
=================
Train and evaluate supervised classifiers for income prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)


# ── Model definitions ────────────────────────────────────────────────────
def get_models(random_state: int = 42, **kwargs) -> Dict[str, Any]:
    """
    Return a dict of model name → sklearn estimator.

    Parameters
    ----------
    random_state : int
    **kwargs : override hyperparameters
        gb_n_estimators, gb_max_depth, gb_learning_rate, gb_subsample,
        rf_n_estimators, rf_max_depth

    Returns
    -------
    dict
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
            C=1.0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=kwargs.get("rf_n_estimators", 300),
            max_depth=kwargs.get("rf_max_depth", 20),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=kwargs.get("gb_n_estimators", 200),
            max_depth=kwargs.get("gb_max_depth", 6),
            learning_rate=kwargs.get("gb_learning_rate", 0.1),
            subsample=kwargs.get("gb_subsample", 0.8),
            random_state=random_state,
        ),
    }
    return models


# ── Training ─────────────────────────────────────────────────────────────
def train_and_evaluate(
    preprocessor: ColumnTransformer,
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    w_train: pd.Series = None,
) -> Dict[str, Dict]:
    """
    Train each model in a Pipeline(preprocessor → model), evaluate on test set.

    Parameters
    ----------
    preprocessor : ColumnTransformer
    models : dict of name → estimator
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    w_train : pd.Series, optional

    Returns
    -------
    results : dict of name → {
        'pipeline', 'accuracy', 'f1', 'auc_roc',
        'y_pred', 'y_proba', 'report', 'cm',
        'fpr', 'tpr', 'precision', 'recall'
    }
    """
    results = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # Build pipeline
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        # Fit
        fit_params = {}
        if w_train is not None:
            fit_params["classifier__sample_weight"] = w_train.values

        pipe.fit(X_train, y_train, **fit_params)

        # Predict
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)

        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc_roc:.4f}")

        report = classification_report(
            y_test, y_pred, target_names=["≤50K", ">50K"]
        )
        print(f"\nClassification Report:\n{report}")

        cm = confusion_matrix(y_test, y_pred)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        # Precision-Recall curve
        prec, rec, _ = precision_recall_curve(y_test, y_proba)

        results[name] = {
            "pipeline": pipe,
            "accuracy": acc,
            "f1": f1,
            "auc_roc": auc_roc,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "report": report,
            "cm": cm,
            "fpr": fpr,
            "tpr": tpr,
            "precision": prec,
            "recall": rec,
        }

    return results


# ── Cross-validation ─────────────────────────────────────────────────────
def cross_validate_models(
    preprocessor: ColumnTransformer,
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Perform stratified K-fold CV for AUC-ROC.

    Returns
    -------
    cv_results : dict of name → (mean_auc, std_auc)
    """
    print(f"\n--- {cv}-Fold Stratified Cross-Validation (AUC-ROC) ---")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_results = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=skf, scoring="roc_auc", n_jobs=-1,
        )

        mean_auc = scores.mean()
        std_auc = scores.std()
        cv_results[name] = (mean_auc, std_auc)
        print(f"{name}: Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    return cv_results


# ── Feature importance ───────────────────────────────────────────────────
def get_feature_importance(
    pipeline: Pipeline,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract feature importances from the best model (tree-based).

    Parameters
    ----------
    pipeline : fitted Pipeline
    preprocessor : fitted ColumnTransformer
    X_train : pd.DataFrame (for column names)
    top_n : int

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'Importance']
    """
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        print("  Model does not support feature_importances_.")
        return pd.DataFrame()

    importances = classifier.feature_importances_

    # Reconstruct feature names
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "low_cat":
            # Get OHE feature names
            ohe = transformer.named_steps["onehot"]
            ohe_names = ohe.get_feature_names_out(cols).tolist()
            feature_names.extend(ohe_names)
        elif name == "high_cat":
            feature_names.extend(cols)

    # Align lengths
    if len(feature_names) != len(importances):
        print(f"  Warning: {len(feature_names)} names vs {len(importances)} importances")
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)

    print(f"\n--- Feature Importance (Best Model) ---")
    print(fi_df.to_string(index=False))

    return fi_df
