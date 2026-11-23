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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/529/"
    "early+stage+diabetes+risk+prediction+dataset.zip"
)
DATASET_PAGE = "https://doi.org/10.24432/C5VG8H"

FEATURES = [
    "age",
    "gender",
    "polyuria",
    "polydipsia",
    "sudden_weight_loss",
    "weakness",
    "polyphagia",
    "genital_thrush",
    "visual_blurring",
    "itching",
    "irritability",
    "delayed_healing",
    "partial_paresis",
    "muscle_stiffness",
    "alopecia",
    "obesity",
]
CATEGORICAL = FEATURES[1:]


def normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [column.strip().lower().replace(" ", "_") for column in frame.columns]
    return frame


def download_dataset(destination: Path) -> pd.DataFrame:
    response = requests.get(DATASET_URL, timeout=(10, 30))
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        content = archive.read(csv_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return pd.read_csv(io.BytesIO(content))


def train(output_dir: Path, random_state: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "diabetes_data_upload.csv"
    frame = pd.read_csv(dataset_path) if dataset_path.exists() else download_dataset(dataset_path)
    frame = normalized_columns(frame)
    missing = set(FEATURES + ["class"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    X = frame[FEATURES].copy()
    y = frame["class"].astype(str).str.strip().str.lower().map({"positive": 1, "negative": 0})
    if y.isna().any():
        raise ValueError("Dataset contains an unknown target label.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    preprocessor = ColumnTransformer(
        [
            ("age", StandardScaler(), ["age"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "dataset": "UCI Early Stage Diabetes Risk Prediction",
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
            "Small, single-hospital questionnaire dataset from Bangladesh; performance does not "
            "establish clinical validity or generalizability."
        ),
    }
    joblib.dump(pipeline, output_dir / "diabetes_risk_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {output_dir / 'diabetes_risk_model.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the experimental diabetes screening model")
    parser.add_argument("--output", type=Path, default=Path("data/ml"))
    args = parser.parse_args()
    train(args.output)
