"""
rag_engine.py
Retrieval layer: given case facts / sections cited, pull the most relevant
statute provisions + amendment notes + case law from the vector store.
"""

from src.legal_kb_builder import load_vectorstore, EMBEDDING_MODEL
from src.llm_client import LLMClient


class LegalRAGEngine:
    def __init__(self, llm_client: LLMClient = None):
        self.collection = load_vectorstore()
        self.llm = llm_client or LLMClient()

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.llm.embed([query], model=EMBEDDING_MODEL)[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        out = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"content": doc, "metadata": meta, "relevance_score": dist})
        return out

    def retrieve_for_sections(self, sections_cited: list[str], k_per_section: int = 3) -> dict:
        results = {}
        for section in sections_cited:
            results[section] = self.retrieve(section, k=k_per_section)
        return results

    def retrieve_for_allegation(self, allegation_summary: str, k: int = 8) -> list[dict]:
        return self.retrieve(allegation_summary, k=k)
