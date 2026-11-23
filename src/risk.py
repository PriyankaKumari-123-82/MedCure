from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class RiskResult:
    screening_result: str
    model_score: float


class DiabetesRiskModel:
    def __init__(self, model_dir: str | Path = "data/ml") -> None:
        directory = Path(model_dir)
        self.pipeline = joblib.load(directory / "diabetes_risk_model.joblib")
        self.metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))

    def predict(self, patient: dict) -> RiskResult:
        probability = float(self.pipeline.predict_proba(pd.DataFrame([patient]))[0, 1])
        return RiskResult(
            screening_result="Screen-positive" if probability >= 0.5 else "Screen-negative",
            model_score=probability,
        )


class HeartRiskModel:
    def __init__(self, model_dir: str | Path = "data/ml") -> None:
        directory = Path(model_dir)
        self.pipeline = joblib.load(directory / "heart_risk_model.joblib")
        self.metrics = json.loads((directory / "heart_metrics.json").read_text(encoding="utf-8"))

    def predict(self, patient: dict) -> RiskResult:
        probability = float(self.pipeline.predict_proba(pd.DataFrame([patient]))[0, 1])
        return RiskResult(
            screening_result="Screen-positive" if probability >= 0.5 else "Screen-negative",
            model_score=probability,
        )


def patient_summary(patient: dict) -> str:
    present = [key.replace("_", " ") for key, value in patient.items() if value == "Yes"]
    symptoms = ", ".join(present) if present else "none of the listed symptoms"
    return f"Age {patient['age']}; gender {patient['gender']}; reported factors: {symptoms}."
