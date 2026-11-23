from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag import RAGEngine, UNAVAILABLE, cited_passages
from src.risk import DiabetesRiskModel, HeartRiskModel, patient_summary

load_dotenv()
st.set_page_config(page_title="MedCure", page_icon="🩺", layout="centered")
st.title("MedCure Knowledge Assistant")
st.caption("ML screening plus answers grounded in public WHO, CDC, and NIH documents.")
st.warning(
    "Educational demonstration only. The screening model is not clinically validated, does not "
    "diagnose diabetes, and must not be used to make treatment decisions."
)

index_dir = Path(os.environ.get("RAG_INDEX_DIR", "data/index"))
model_dir = Path(os.environ.get("RISK_MODEL_DIR", "data/ml"))
api_key = os.environ.get("OPENAI_API_KEY", "").strip()
rag_ready = (index_dir / "vectors.faiss").exists() and api_key not in {"", "replace-me"}
risk_ready = (model_dir / "diabetes_risk_model.joblib").exists()
heart_ready = (model_dir / "heart_risk_model.joblib").exists()


@st.cache_resource
def load_engine(path: str) -> RAGEngine:
    return RAGEngine(path)


@st.cache_resource
def load_risk_model(path: str) -> DiabetesRiskModel:
    return DiabetesRiskModel(path)


@st.cache_resource
def load_heart_model(path: str) -> HeartRiskModel:
    return HeartRiskModel(path)


def show_sources(answer: str, passages: list) -> None:
    used_sources = cited_passages(answer, passages)
    if used_sources and answer != UNAVAILABLE:
        st.subheader("Sources used")
        for passage in used_sources:
            with st.expander(f"[{passage.label}] {passage.title} · relevance {passage.score:.2f}"):
                st.write(passage.text)
                st.markdown(f"[{passage.publisher} — original document]({passage.url})")


knowledge_tab, risk_tab, heart_tab = st.tabs(
    ["Ask the knowledge base", "Diabetes risk screening", "Heart risk screening"]
)

with knowledge_tab:
    if not rag_ready:
        st.error("Build the index with `python -m src.ingest` and configure a valid API key.")
    else:
        engine = load_engine(str(index_dir))
        question = st.text_area(
            "Ask a healthcare question",
            placeholder="What lifestyle changes can help lower high blood pressure?",
            height=100,
        )
        if st.button("Search knowledge base", type="primary", disabled=not question.strip()):
            try:
                with st.spinner("Retrieving evidence and preparing an answer…"):
                    passages = engine.retrieve(question.strip())
                    answer = engine.answer(question.strip(), passages)
                st.subheader("Answer")
                st.write(answer)
                show_sources(answer, passages)
            except Exception:
                st.error("The knowledge service is temporarily unavailable. Please try again.")

with risk_tab:
    st.subheader("Early-stage diabetes symptom screening")
    st.caption("Do not enter names, contact details, record numbers, or other identifying data.")
    if not risk_ready:
        st.error("Risk model not trained. Run `python -m src.train_risk_model` first.")
    else:
        symptom_fields = {
            "polyuria": "Frequent urination (polyuria)",
            "polydipsia": "Excessive thirst (polydipsia)",
            "sudden_weight_loss": "Sudden weight loss",
            "weakness": "Weakness",
            "polyphagia": "Excessive hunger (polyphagia)",
            "genital_thrush": "Genital thrush",
            "visual_blurring": "Blurred vision",
            "itching": "Itching",
            "irritability": "Irritability",
            "delayed_healing": "Delayed wound healing",
            "partial_paresis": "Partial muscle weakness (paresis)",
            "muscle_stiffness": "Muscle stiffness",
            "alopecia": "Hair loss (alopecia)",
            "obesity": "Obesity",
        }
        with st.form("diabetes-risk-form"):
            age = st.number_input("Age", min_value=16, max_value=100, value=40)
            gender = st.selectbox("Gender (dataset category)", ["Male", "Female"])
            st.markdown("Reported symptoms or factors")
            selections = {
                field: "Yes" if st.checkbox(label, key=f"risk-{field}") else "No"
                for field, label in symptom_fields.items()
            }
            submitted = st.form_submit_button("Estimate screening score", type="primary")

        if submitted:
            patient = {"age": int(age), "gender": gender, **selections}
            model = load_risk_model(str(model_dir))
            result = model.predict(patient)
            st.subheader("Experimental ML result")
            st.metric(result.screening_result, f"Model score: {result.model_score:.1%}")
            st.caption(
                "This is the model's classification score for its training data, not an individual "
                "probability of having or developing diabetes."
            )

            with st.expander("Model provenance and held-out evaluation"):
                metrics = model.metrics
                st.write(
                    f"Dataset: [{metrics['dataset']}]({metrics['dataset_url']}) · "
                    f"{metrics['instances']} records · {metrics['license']}"
                )
                st.write(
                    f"Held-out ROC AUC: {metrics['roc_auc']:.3f}; accuracy: "
                    f"{metrics['accuracy']:.3f}; recall: {metrics['recall']:.3f}."
                )
                st.warning(metrics["limitations"])

            if rag_ready:
                engine = load_engine(str(index_dir))
                query = (
                    "diabetes symptoms risk factors healthy lifestyle changes and when to seek "
                    "assessment from a healthcare professional"
                )
                explanation_request = (
                    "Provide a document-grounded educational explanation for this anonymous "
                    f"screening profile: {patient_summary(patient)} The separate experimental ML "
                    f"result was {result.screening_result} with score {result.model_score:.1%}. "
                    "Do not validate or reinterpret the ML score. Explain only relevant documented "
                    "symptoms, general lifestyle information, and when professional assessment is "
                    "appropriate. State clearly that only clinical testing can diagnose diabetes."
                )
                try:
                    with st.spinner("Retrieving sources for the explanation…"):
                        passages = engine.retrieve(query, top_k=6)
                        explanation = engine.answer(explanation_request, passages)
                    st.subheader("RAG-based educational explanation")
                    st.write(explanation)
                    show_sources(explanation, passages)
                except Exception:
                    st.error(
                        "The ML result is available, but the cited RAG explanation service is "
                        "temporarily unavailable. Please try again."
                    )
            else:
                st.info(
                    "The ML score is available, but the RAG explanation requires a built index "
                    "and a valid OPENAI_API_KEY."
                )

with heart_tab:
    st.subheader("Heart-disease model demonstration")
    st.caption(
        "This form includes specialist test results and is not suitable for self-diagnosis. "
        "Use measured values only; do not enter identifying information."
    )
    if not heart_ready:
        st.error("Heart model not trained. Run `python -m src.train_heart_model` first.")
    else:
        with st.form("heart-risk-form"):
            age_h = st.number_input("Age", min_value=18, max_value=100, value=55, key="heart-age")
            sex = st.selectbox("Sex (dataset category)", ["Female", "Male"], key="heart-sex")
            cp = st.selectbox(
                "Chest pain category",
                [
                    "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic",
                ],
            )
            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)", min_value=70, max_value=250, value=130
            )
            chol = st.number_input(
                "Serum cholesterol (mg/dL)", min_value=80, max_value=700, value=240
            )
            fbs = st.checkbox("Fasting blood sugar above 120 mg/dL")
            restecg = st.selectbox(
                "Resting ECG result",
                ["Normal", "ST-T wave abnormality", "Probable/definite LV hypertrophy"],
            )
            thalach = st.number_input(
                "Maximum heart rate achieved", min_value=50, max_value=250, value=150
            )
            exang = st.checkbox("Exercise-induced angina")
            oldpeak = st.number_input(
                "ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0
            )
            slope = st.selectbox(
                "Peak exercise ST-segment slope", ["Upsloping", "Flat", "Downsloping"]
            )
            ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3])
            thal = st.selectbox("Thallium result", ["Normal", "Fixed defect", "Reversible defect"])
            heart_submitted = st.form_submit_button("Estimate heart screening score", type="primary")

        if heart_submitted:
            cp_codes = {
                "Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3,
                "Asymptomatic": 4,
            }
            ecg_codes = {"Normal": 0, "ST-T wave abnormality": 1, "Probable/definite LV hypertrophy": 2}
            slope_codes = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
            thal_codes = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}
            heart_patient = {
                "age": int(age_h), "sex": int(sex == "Male"), "cp": cp_codes[cp],
                "trestbps": int(trestbps), "chol": int(chol), "fbs": int(fbs),
                "restecg": ecg_codes[restecg], "thalach": int(thalach), "exang": int(exang),
                "oldpeak": float(oldpeak), "slope": slope_codes[slope], "ca": int(ca),
                "thal": thal_codes[thal],
            }
            heart_model = load_heart_model(str(model_dir))
            heart_result = heart_model.predict(heart_patient)
            st.subheader("Experimental ML result")
            st.metric(heart_result.screening_result, f"Model score: {heart_result.model_score:.1%}")
            st.caption(
                "This score reflects the historical training dataset; it is not a personal disease "
                "probability and cannot confirm or exclude heart disease."
            )
            with st.expander("Model provenance and held-out evaluation"):
                metrics = heart_model.metrics
                st.write(
                    f"Dataset: [{metrics['dataset']}]({metrics['dataset_url']}) · "
                    f"{metrics['instances']} records · {metrics['license']}"
                )
                st.write(
                    f"Held-out ROC AUC: {metrics['roc_auc']:.3f}; accuracy: "
                    f"{metrics['accuracy']:.3f}; recall: {metrics['recall']:.3f}."
                )
                st.warning(metrics["limitations"])
                st.info(metrics["notebook_note"])

            if rag_ready:
                engine = load_engine(str(index_dir))
                query = (
                    "heart disease and hypertension risk factors symptoms healthy lifestyle "
                    "changes and when to seek medical care"
                )
                request = (
                    f"The separate experimental heart model returned {heart_result.screening_result} "
                    f"with score {heart_result.model_score:.1%} for an anonymous age-{int(age_h)} "
                    "profile. Do not validate or reinterpret that score. Using only the documents, "
                    "explain relevant general heart-health risk factors, lifestyle information, and "
                    "when professional assessment is appropriate. State that the ML result is not a "
                    "diagnosis."
                )
                try:
                    with st.spinner("Retrieving sources for the explanation…"):
                        passages = engine.retrieve(query, top_k=6)
                        explanation = engine.answer(request, passages)
                    st.subheader("RAG-based educational explanation")
                    st.write(explanation)
                    show_sources(explanation, passages)
                except Exception:
                    st.error(
                        "The ML result is available, but the cited RAG explanation service is "
                        "temporarily unavailable. Please try again."
                    )
            else:
                st.info(
                    "The ML score is available, but the RAG explanation requires a built index "
                    "and a valid OPENAI_API_KEY."
                )
