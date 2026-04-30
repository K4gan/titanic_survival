"""
CLI inference for Titanic survival using pipelines saved by train.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILES = {
    "lr": "logistic_regression.joblib",
    "rf": "random_forest.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Titanic survival (train with train.py first).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FILES.keys()),
        default="rf",
        help="lr = logistic regression, rf = random forest",
    )
    parser.add_argument("--pclass", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--sex", type=str, required=True, choices=["male", "female"])
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--sibsp", type=int, required=True)
    parser.add_argument("--parch", type=int, required=True)
    parser.add_argument("--fare", type=float, required=True)
    parser.add_argument(
        "--embarked",
        type=str,
        required=True,
        choices=["S", "C", "Q"],
        help="S=Southampton, C=Cherbourg, Q=Queenstown",
    )
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Print P(survived)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = MODELS_DIR / MODEL_FILES[args.model]
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Run: python train.py"
        )

    pipeline = joblib.load(path)
    row = pd.DataFrame(
        [
            {
                "pclass": args.pclass,
                "sex": args.sex,
                "age": args.age,
                "sibsp": args.sibsp,
                "parch": args.parch,
                "fare": args.fare,
                "embarked": args.embarked,
            }
        ]
    )

    survived = int(pipeline.predict(row)[0])
    label = "survived" if survived == 1 else "did_not_survive"
    print(f"prediction: {survived} ({label})")

    if args.proba:
        print(f"P(survived): {float(pipeline.predict_proba(row)[0, 1]):.4f}")


if __name__ == "__main__":
    main()
