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

## Screenshots
<img width="1768" height="1036" alt="Screenshot From 2026-11-23 14-33-36" src="https://github.com/user-attachments/assets/66028e81-b69c-4185-a5d3-5b6424ddc83d" />
<img width="1768" height="1036" alt="Screenshot From 2026-11-23 14-33-48" src="https://github.com/user-attachments/assets/18a9141f-0fc2-4028-acff-a443bc1e1284" />
<img width="1768" height="1036" alt="Screenshot From 2026-11-23 14-35-02" src="https://github.com/user-attachments/assets/22b899c1-8cd5-4e56-a421-70ae2df2bc37" />
<img width="1768" height="1036" alt="Screenshot From 2026-11-23 14-35-15" src="https://github.com/user-attachments/assets/d2ee3132-e82c-4469-bbaa-d83e9470a715" />
<img width="1768" height="1036" alt="Screenshot From 2026-11-23 14-35-24" src="https://github.com/user-attachments/assets/8ea8790b-df23-4766-a57c-a0aa581fb622" />
<img width="1768" height="1036" alt="Screenshot From 2026-11-23 14-35-43" src="https://github.com/user-attachments/assets/ee976ce2-bf00-42b8-bf63-17422ae6e0f1" />
<img width="1768" height="1036" alt="Screenshot From 2026-11-23 14-35-55" src="https://github.com/user-attachments/assets/ab5518b3-cbe0-43d9-8f91-fff0f40d0455" />
<img width="1768" height="1036" alt="Screenshot From 2026-11-23 14-36-09" src="https://github.com/user-attachments/assets/5ebecf9f-9b35-4aa9-9f66-fa7c13b5b81c" />
<img width="1768" height="1036" alt="image" src="https://github.com/user-attachments/assets/9cd29b35-f26b-4ca4-a058-ad64c4b6d82b" />
