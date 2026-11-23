from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

UNAVAILABLE = "I couldn't find enough information in the provided healthcare documents to answer that question."

SYSTEM_PROMPT = """You are MedCure, a careful healthcare knowledge-base assistant.

Use ONLY the supplied evidence passages. Do not use prior knowledge and do not invent facts,
diagnoses, treatments, dosages, or citations. If the passages do not directly support a useful
answer, reply exactly with: {unavailable}

When evidence is sufficient:
1. Answer the user's question directly in plain language.
2. Cite every factual claim using one or more passage labels, for example [S1] or [S1, S3].
3. Give general educational information, not a diagnosis or individualized treatment plan.
4. If the user asks what they personally should do, explain the documented options and advise
   consultation with a qualified healthcare professional. For possible emergencies, advise local
   emergency services without attempting to diagnose.
5. Do not add a bibliography; the application builds the source list from your inline citations.
"""


@dataclass
class Passage:
    label: str
    score: float
    chunk_id: str
    source_id: str
    title: str
    publisher: str
    url: str
    text: str


class RAGEngine:
    def __init__(self, index_dir: str | Path = "data/index") -> None:
        directory = Path(index_dir)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(directory / "vectors.faiss"))
        self.embedding_model = manifest["embedding_model"]
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def retrieve(self, question: str, top_k: int = 5, min_score: float = 0.28) -> list[Passage]:
        embedding = self.client.embeddings.create(model=self.embedding_model, input=[question])
        vector = np.asarray([embedding.data[0].embedding], dtype="float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(top_k * 2, len(self.chunks)))
        results: list[Passage] = []
        seen_text: set[str] = set()
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or float(score) < min_score:
                continue
            item = self.chunks[int(index)]
            fingerprint = item["text"][:180]
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            results.append(
                Passage(label=f"S{len(results) + 1}", score=float(score), **item)
            )
            if len(results) == top_k:
                break
        return results

    def answer(self, question: str, passages: list[Passage]) -> str:
        if not passages:
            return UNAVAILABLE
        evidence = "\n\n".join(
            f"[{p.label}] {p.title} — {p.publisher}\nURL: {p.url}\nPASSAGE:\n{p.text}"
            for p in passages
        )
        prompt = f"USER QUESTION:\n{question}\n\nEVIDENCE PASSAGES:\n{evidence}"
        response = self.client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(unavailable=UNAVAILABLE)},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content or UNAVAILABLE
        allowed = {p.label for p in passages}
        cited = set(re.findall(r"\bS\d+\b", answer))
        # Reject ungrounded prose or labels that were not supplied to the model.
        if answer != UNAVAILABLE and (not cited or not cited.issubset(allowed)):
            return UNAVAILABLE
        return answer


def cited_passages(answer: str, passages: list[Passage]) -> list[Passage]:
    """Return only sources whose evidence label appears in the generated answer."""
    labels = set(re.findall(r"\bS\d+\b", answer))
    return [passage for passage in passages if passage.label in labels]
