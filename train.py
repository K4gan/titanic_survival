"""
Train and compare Logistic Regression vs Random Forest on Seaborn Titanic data.
Saves sklearn Pipelines under models/ for predict.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET_COLUMN = "survived"

NUMERIC_FEATURES = ["age", "fare", "sibsp", "parch"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked"]

MODELS_DIR = Path(__file__).resolve().parent / "models"


def load_dataset() -> pd.DataFrame:
    df = sns.load_dataset("titanic")
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    df = df.dropna(subset=["embarked"])
    return df


def build_pipeline(estimator) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def evaluate_model(
    name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    y_pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Titanic survival models.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    lr_pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=args.random_state)
    )
    rf_pipeline = build_pipeline(
        RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            random_state=args.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
    )

    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    metrics_lr = evaluate_model("logistic_regression", lr_pipeline, X_test, y_test)
    metrics_rf = evaluate_model("random_forest", rf_pipeline, X_test, y_test)

    comparison = {
        "test_metrics": [metrics_lr, metrics_rf],
        "feature_columns": FEATURE_COLUMNS,
        "random_state": args.random_state,
        "test_size": args.test_size,
    }

    joblib.dump(lr_pipeline, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(rf_pipeline, MODELS_DIR / "random_forest.joblib")

    with open(MODELS_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    header = f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    print("\nTest set comparison:\n")
    print(header)
    print("-" * len(header))
    for m in comparison["test_metrics"]:
        print(
            f"{m['model']:<22} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}"
        )

    best = max(comparison["test_metrics"], key=lambda row: row["accuracy"])
    print(f"\nBest by accuracy: {best['model']}")
    print(f"Artifacts: {MODELS_DIR}")


if __name__ == "__main__":
    main()
