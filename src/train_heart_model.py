from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
DATASET_PAGE = "https://doi.org/10.24432/C52P4X"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]
FEATURES = COLUMNS[:-1]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]


def download_dataset(destination: Path) -> pd.DataFrame:
    response = requests.get(DATASET_URL, timeout=(10, 30))
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        filename = next(
            name for name in archive.namelist() if name.endswith("processed.cleveland.data")
        )
        content = archive.read(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return pd.read_csv(io.BytesIO(content), names=COLUMNS, na_values="?")


def train(output_dir: Path, random_state: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "processed.cleveland.data"
    frame = (
        pd.read_csv(dataset_path, names=COLUMNS, na_values="?")
        if dataset_path.exists()
        else download_dataset(dataset_path)
    )
    X = frame[FEATURES]
    y = (frame["num"] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "dataset": "UCI Heart Disease (Cleveland)",
        "dataset_url": DATASET_PAGE,
        "license": "CC BY 4.0",
        "instances": int(len(frame)),
        "test_instances": int(len(y_test)),
        "random_state": random_state,
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "precision": round(float(precision_score(y_test, prediction)), 4),
        "recall": round(float(recall_score(y_test, prediction)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "limitations": (
            "Small historical Cleveland cohort with specialist test variables; performance does "
            "not establish current clinical validity or generalizability."
        ),
        "notebook_note": (
            "The supplied notebook informed the preprocessing and evaluation structure, but its "
            "unavailable synthetic Kaggle dataset and perfect synthetic scores were not used."
        ),
    }
    joblib.dump(pipeline, output_dir / "heart_risk_model.joblib")
    (output_dir / "heart_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {output_dir / 'heart_risk_model.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the experimental heart screening model")
    parser.add_argument("--output", type=Path, default=Path("data/ml"))
    args = parser.parse_args()
    train(args.output)
