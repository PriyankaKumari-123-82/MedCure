from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "MedCure-RAG/1.0 (educational knowledge-base indexer)"


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    publisher: str
    url: str
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return ""
    blocks = main.find_all(["h1", "h2", "h3", "p", "li"])
    return clean_text("\n".join(b.get_text(" ", strip=True) for b in blocks))


def extract_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return clean_text("\n\n".join(page.extract_text() or "" for page in reader.pages))


def _extract_file(path: Path) -> str:
    content = path.read_bytes()
    return extract_pdf(content) if path.suffix.lower() == ".pdf" else extract_html(content)


def _http_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=2,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def download(
    source: dict,
    raw_dir: Path,
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> str:
    cached = [raw_dir / f'{source["id"]}.pdf', raw_dir / f'{source["id"]}.html']
    if not refresh:
        for path in cached:
            if path.exists() and path.stat().st_size > 0:
                text = _extract_file(path)
                if len(text) >= 200:
                    return text

    client = session or _http_session()
    response = client.get(source["url"], timeout=(10, 30))
    response.raise_for_status()
    is_pdf = "pdf" in response.headers.get("content-type", "").lower() or source[
        "url"
    ].lower().endswith(".pdf")
    suffix = ".pdf" if is_pdf else ".html"
    destination = raw_dir / f'{source["id"]}{suffix}'
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    text = extract_pdf(response.content) if is_pdf else extract_html(response.content)
    if len(text) < 200:
        raise ValueError(f'Not enough readable content in {source["id"]}')
    return text


def split_text(text: str, source: dict, size: int = 900, overlap: int = 140) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= size:
            current = f"{current}\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        tail = current[-overlap:] if current else ""
        current = f"{tail}\n{paragraph}".strip()
        while len(current) > size * 2:
            chunks.append(current[:size])
            current = current[size - overlap :]
    if current:
        chunks.append(current)

    output = []
    for i, body in enumerate(chunks, 1):
        digest = hashlib.sha1(body.encode()).hexdigest()[:8]
        output.append(
            Chunk(
                chunk_id=f'{source["id"]}-{i:03d}-{digest}',
                source_id=source["id"],
                title=source["title"],
                publisher=source["publisher"],
                url=source["url"],
                text=body,
            )
        )
    return output


def embed_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    client = OpenAI(timeout=60, max_retries=2)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        print(f"Embedding chunks {start + 1}-{end} of {len(texts)}...", flush=True)
        response = client.embeddings.create(model=model_name, input=texts[start : start + batch_size])
        vectors.extend(item.embedding for item in response.data)
    array = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array


def build_index(
    sources_file: Path, output_dir: Path, model_name: str, *, refresh: bool = False
) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "replace-me":
        raise RuntimeError(
            "OPENAI_API_KEY is missing or still set to 'replace-me'. Add a valid key to .env."
        )
    sources = json.loads(sources_file.read_text(encoding="utf-8"))
    if not 5 <= len(sources) <= 10:
        raise ValueError("The knowledge base must contain 5 to 10 documents.")
    raw_dir = output_dir.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Chunk] = []
    failures: list[str] = []
    session = _http_session()
    for source in sources:
        print(f'Loading {source["id"]}...', flush=True)
        try:
            text = download(source, raw_dir, refresh=refresh, session=session)
            source_chunks = split_text(text, source)
            chunks.extend(source_chunks)
            print(f'Loaded {source["id"]} ({len(source_chunks)} chunks)', flush=True)
        except Exception as exc:
            failures.append(f'{source["id"]}: {exc}')
    if failures:
        raise RuntimeError("Document ingestion failed:\n" + "\n".join(failures))

    vectors = embed_texts([c.text for c in chunks], model_name)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.asarray(vectors))
    faiss.write_index(index, str(output_dir / "vectors.faiss"))
    (output_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {"embedding_model": model_name, "documents": len(sources), "chunks": len(chunks)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Indexed {len(chunks)} chunks from {len(sources)} documents in {output_dir}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build the MedCure FAISS knowledge base")
    parser.add_argument("--sources", default="sources.json", type=Path)
    parser.add_argument("--output", default="data/index", type=Path)
    parser.add_argument(
        "--model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Download documents again instead of using data/raw"
    )
    args = parser.parse_args()
    build_index(args.sources, args.output, args.model, refresh=args.refresh)
