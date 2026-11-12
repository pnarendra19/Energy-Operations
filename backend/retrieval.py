"""Embeddings + retrieval.

Embedding choice (offline, reproducible, no API key needed):
  * sparse  : TF-IDF over stemmed unigrams+bigrams  -> exact-term matching (codes, thresholds, units)
  * dense   : LSA (TruncatedSVD of the TF-IDF matrix) -> a low-dimensional semantic embedding that
              links related wording ("climb" ~ "height", "lock" ~ "isolation")
  * hybrid  : alpha * cosine(dense) + (1 - alpha) * cosine(sparse)

The class is deliberately small so a neural embedder (e.g. a sentence-transformer or an API
embedding model) can replace `HybridEmbedder` without touching the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config
from .ingest import Chunk
from .textutils import analyzer, content_tokens, expand_query, expansion_groups


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float
    sparse: float
    rank: int

    def to_dict(self) -> dict:
        d = self.chunk.to_dict()
        d.update(score=round(self.score, 4), dense_score=round(self.dense, 4),
                 sparse_score=round(self.sparse, 4), rank=self.rank)
        return d


class HybridEmbedder:
    def __init__(self, dims: int | None = None):
        self.dims = dims or config.LSA_DIMS
        self.tfidf = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, norm="l2")
        self.svd: TruncatedSVD | None = None

    def fit_transform(self, texts: list[str]):
        sparse = self.tfidf.fit_transform(texts)
        k = max(2, min(self.dims, sparse.shape[0] - 1, sparse.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        dense = normalize(self.svd.fit_transform(sparse))
        return sparse, dense

    def transform(self, texts: list[str]):
        sparse = self.tfidf.transform(texts)
        dense = normalize(self.svd.transform(sparse))
        return sparse, dense


class Retriever:
    def __init__(self, chunks: list[Chunk], alpha: float | None = None, embed_title: bool = True,
                 beta: float | None = None):
        if not chunks:
            raise ValueError("No chunks to index")
        self.chunks = chunks
        self.alpha = config.HYBRID_ALPHA if alpha is None else alpha
        self.beta = config.DOC_PRIOR if beta is None else beta
        self.entity_penalty = config.ENTITY_PENALTY
        # Document-level index: whole-SOP TF-IDF. It tells us *which equipment/SOP* the question is about
        # (e.g. "gearbox" -> SOP-MNT-101) so a chunk with the right words from the wrong SOP is down-weighted.
        self.doc_ids = sorted({c.doc_id for c in chunks})
        doc_text = {d: "" for d in self.doc_ids}
        for c in chunks:
            doc_text[c.doc_id] += f" {c.title}. {c.section}. {c.text}" if doc_text[c.doc_id] else f"{c.title}. {c.title}. {c.section}. {c.text}"
        self.doc_vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, norm="l2")
        self.doc_matrix = self.doc_vec.fit_transform([doc_text[d] for d in self.doc_ids])
        self._doc_index = {d: i for i, d in enumerate(self.doc_ids)}
        self._doc_tokens = {d: set(content_tokens(t)) for d, t in doc_text.items()}
        self._chunk_doc_idx = np.array([self._doc_index[c.doc_id] for c in chunks])
        self.embedder = HybridEmbedder()
        texts = [c.embed_text if embed_title else f"{c.section}. {c.text}" for c in chunks]
        self.sparse, self.dense = self.embedder.fit_transform(texts)

    def search(self, query: str, k: int | None = None, alpha: float | None = None,
               beta: float | None = None) -> list[Hit]:
        k = k or config.TOP_K
        a = self.alpha if alpha is None else alpha
        b = self.beta if beta is None else beta
        q = expand_query(query)
        qs, qd = self.embedder.transform([q])
        sparse_sim = (self.sparse @ qs.T).toarray().ravel()
        dense_sim = np.clip(self.dense @ qd.ravel(), 0, 1)
        doc_sim = (self.doc_matrix @ self.doc_vec.transform([q]).T).toarray().ravel()[self._chunk_doc_idx]
        score = (1 - b) * (a * dense_sim + (1 - a) * sparse_sim) + b * doc_sim
        # Entity check: an SOP that never mentions a key term of the question (e.g. "gearbox") is probably
        # the wrong SOP, even if it shares generic words (oil, temperature, trip). Scale its chunks down.
        groups = expansion_groups(query)
        if groups and self.entity_penalty:
            cov = np.array([sum(bool(g & self._doc_tokens[d]) for g in groups) / len(groups) for d in self.doc_ids])
            score = score * (1 - self.entity_penalty + self.entity_penalty * cov[self._chunk_doc_idx])
        order = np.argsort(-score)[:k]
        return [Hit(self.chunks[i], float(score[i]), float(dense_sim[i]), float(sparse_sim[i]), r + 1)
                for r, i in enumerate(order)]

    def idf(self, stem_token: str) -> float:
        j = self.embedder.tfidf.vocabulary_.get(stem_token)
        return float(self.embedder.tfidf.idf_[j]) if j is not None else 1.0
