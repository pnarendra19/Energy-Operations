"""RAG pipeline: ingest -> chunk -> embed/index -> retrieve -> gate -> generate -> verify."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from . import config
from .generation import extractive_answer, llm_answer
from .guardrails import check_answer, retrieval_gate
from .ingest import Chunk, Document, build_chunks, load_documents, parse_document
from .retrieval import Retriever

ABSTAIN_TEXT = ("I could not find this in the provided SOPs, so I won't guess. "
                "Please check with your Shift In-charge or the relevant OEM/HSE documentation.")


class RAGPipeline:
    def __init__(self, docs: list[Document] | None = None):
        self._lock = threading.Lock()
        self.docs: list[Document] = docs if docs is not None else load_documents()
        self._reindex()

    def _reindex(self):
        self.chunks: list[Chunk] = build_chunks(self.docs)
        self.retriever = Retriever(self.chunks)

    # ---- document management -------------------------------------------------
    def add_document(self, raw: str, filename: str) -> Document:
        doc = parse_document(raw, filename)
        with self._lock:
            self.docs = [d for d in self.docs if d.doc_id != doc.doc_id] + [doc]
            self._reindex()
        return doc

    def list_documents(self) -> list[dict]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        return [{"doc_id": d.doc_id, "title": d.title, "revision": d.revision, "source": d.source,
                 "sections": len(d.sections), "chunks": counts.get(d.doc_id, 0)} for d in self.docs]

    # ---- question answering --------------------------------------------------
    def ask(self, question: str, top_k: int | None = None) -> dict:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=top_k or config.TOP_K)
        gate = retrieval_gate(question, hits)
        base = {"question": question, "evidence": [h.to_dict() for h in hits], "gate": gate}

        if not gate["passed"]:
            return {**base, "answer": ABSTAIN_TEXT, "abstained": True, "mode": "abstain", "citations": [],
                    "checks": {"verdict": "abstained", "groundedness": None, "citations_valid": True,
                               "sentences": [], "note": gate["reason"]},
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

        mode, fallback, answer = "extractive", False, None
        llm_text = llm_answer(question, hits)
        if llm_text:
            checks = check_answer(llm_text, hits)
            if checks["verdict"] == "grounded":
                mode, answer = "llm", llm_text
            else:
                fallback = True  # LLM text failed verification -> never show it
        if answer is None:
            answer = extractive_answer(question, hits, self.retriever)
            checks = check_answer(answer, hits)

        cited = sorted({n for s in checks["sentences"] for n in s["cited"] if 1 <= n <= len(hits)})
        citations = [{"n": n, **hits[n - 1].to_dict()} for n in cited]
        return {**base, "answer": answer, "abstained": False, "mode": mode, "fallback_used": fallback,
                "citations": citations, "checks": checks,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
