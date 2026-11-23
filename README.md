# MedCure RAG

A small retrieval-augmented generation system grounded in eight public healthcare documents from WHO, CDC, and NIH. It downloads and cleans the documents, chunks them, creates embeddings with OpenAI, stores them locally in FAISS, retrieves relevant passages, and asks an OpenAI model to answer with evidence citations.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`, then build the knowledge base (the embedding call uses the API) and run the UI:

```bash
python -m src.ingest
python -m src.train_risk_model
python -m src.train_heart_model
streamlit run app.py
```

The first ingestion downloads eight official documents. Later runs reuse `data/raw`, so a failed run can be resumed without downloading completed documents again. Use `python -m src.ingest --refresh` only when you want fresh copies. Generated raw documents and index files are kept under `data/` and ignored by Git.

## Grounding and citations

- Retrieval uses normalized `text-embedding-3-small` vectors and cosine-equivalent FAISS inner-product search.
- Only passages above the configured relevance threshold are sent to the LLM.
- The prompt forbids outside knowledge and requires an inline passage citation for each factual claim.
- The UI exposes the exact retrieved text and original URL behind every citation.
- When no passage clears the threshold, or the evidence cannot support an answer, the assistant says the answer is unavailable in the provided documents.

## Project structure

```text
app.py             Streamlit question-answer interface
sources.json       Eight curated public healthcare sources
src/ingest.py      Download, clean, chunk, embed, and index pipeline
src/rag.py         FAISS retrieval, grounding prompt, and answer generation
```

This project provides general educational information only and is not a medical device or a substitute for professional medical advice.

## Experimental diabetes screening model

The optional risk tab trains a logistic-regression classifier on the [UCI Early Stage Diabetes Risk Prediction dataset](https://doi.org/10.24432/C5VG8H) (CC BY 4.0). Its held-out metrics and limitations are shown in the app. The dataset has only 520 questionnaire records from one hospital in Bangladesh, so the score is not clinically validated or broadly generalizable. It is displayed as an experimental screening result—not a diagnosis or a personal disease probability. Patient symptom fields are used for prediction; the anonymous summary is sent to the configured LLM only when producing the cited RAG explanation.

The heart-disease tab adapts the preprocessing, evaluation, saved-model, and prediction structure from the supplied `heart-disease-prediction.ipynb`. It deliberately does not reuse that notebook's unavailable synthetic Kaggle data or its perfect synthetic scores. Instead it trains logistic regression on the [UCI Heart Disease Cleveland dataset](https://doi.org/10.24432/C52P4X) (303 records, CC BY 4.0). Several inputs are specialist test results, and the model is an educational demonstration—not a clinical screening instrument.

##Screnshots
<img width="1768" height="1036" alt="Screenshot From 2026-11-23 14-33-36" src="https://github.com/user-attachments/assets/e7076d05-e586-4517-884b-33201112cb23" />

<img width="1768" height="1036" alt="image" src="https://github.com/user-attachments/assets/9cd29b35-f26b-4ca4-a058-ad64c4b6d82b" />
