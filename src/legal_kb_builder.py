"""
legal_kb_builder.py
Builds/loads the legal knowledge base vector store from statute text files,
using OpenAI embeddings + Chroma directly.

Expected input: JSON files in data/legal_corpus/, one entry per section:
[
  {
    "act": "Bharatiya Nyaya Sanhita, 2023",
    "section": "103",
    "old_equivalent": "IPC Section 302 (Murder)",
    "title": "Punishment for murder",
    "text": "Whoever commits murder shall be punished with death or imprisonment for life...",
    "amendment_note": "BNS 2023 replaced IPC w.e.f. 1 July 2024.",
    "related_case_law": ["Virsa Singh v. State of Punjab (1958)"]
  }
]

Populate this yourself from verified official sources — this pipeline does
not scrape or fabricate law text.
"""

import json
import glob
import chromadb
from dotenv import load_dotenv
from src.llm_client import LLMClient

load_dotenv()  # reads OPENAI_API_KEY from .env when run as a standalone script

CORPUS_DIR = "data/legal_corpus"
VECTORSTORE_DIR = "data/vectorstore"
COLLECTION_NAME = "legal_corpus"
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def load_corpus_entries() -> list[dict]:
    entries = []
    for filepath in glob.glob(f"{CORPUS_DIR}/*.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            content = (
                f"Act: {item.get('act')}\n"
                f"Section: {item.get('section')}\n"
                f"Title: {item.get('title')}\n"
                f"Text: {item.get('text')}\n"
                f"Amendment note: {item.get('amendment_note', 'N/A')}\n"
                f"Related case law: {', '.join(item.get('related_case_law', []))}"
            )
            entries.append({
                "content": content,
                "metadata": {
                    "act": item.get("act") or "",
                    "section": item.get("section") or "",
                    "old_equivalent": item.get("old_equivalent") or "",
                    "source_file": filepath,
                },
            })
    return entries


def build_vectorstore():
    entries = load_corpus_entries()
    if not entries:
        raise ValueError(f"No corpus files found in {CORPUS_DIR}. Add bare-act JSON files first.")

    llm = LLMClient()
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    doc_ids, doc_texts, doc_metadatas = [], [], []
    for i, entry in enumerate(entries):
        chunks = _chunk_text(entry["content"])
        for j, chunk in enumerate(chunks):
            doc_ids.append(f"doc{i}_chunk{j}")
            doc_texts.append(chunk)
            doc_metadatas.append(entry["metadata"])

    BATCH = 100
    for start in range(0, len(doc_texts), BATCH):
        batch_texts = doc_texts[start:start + BATCH]
        batch_ids = doc_ids[start:start + BATCH]
        batch_meta = doc_metadatas[start:start + BATCH]
        embeddings = llm.embed(batch_texts, model=EMBEDDING_MODEL)
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    print(f"Indexed {len(doc_texts)} chunks from {len(entries)} legal provisions.")
    return collection


def load_vectorstore():
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    return client.get_collection(COLLECTION_NAME)


if __name__ == "__main__":
    build_vectorstore()
