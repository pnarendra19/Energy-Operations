"""Answer generation.

Two interchangeable generators, both producing text with [n] citations:
  * extractive (default): selects the most relevant sentences from the retrieved chunks and cites them.
    Runs offline, is deterministic, and cannot invent facts - ideal for a safety-critical SOP assistant.
  * LLM (optional): Anthropic Messages API, prompted to answer ONLY from numbered context.
    Enabled with LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY. Always passes through the same guardrails
    and falls back to the extractive answer if the check fails or the request errors.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from . import config
from .retrieval import Hit, Retriever
from .textutils import content_tokens, expansion_groups

PROCEDURAL = re.compile(r"\b(how do|how to|how should|steps?|procedure|what should (?:i|we) do|what must|what happens)\b", re.I)

SYSTEM_PROMPT = (
    "You are an operations assistant for renewable-energy SOPs. Answer ONLY from the numbered context below. "
    "Cite the supporting context number in square brackets after every sentence, e.g. [1]. "
    "Copy numbers, units and thresholds exactly as written. "
    "If the context does not contain the answer, reply exactly: INSUFFICIENT_CONTEXT. "
    "Be concise (max 4 sentences). Do not add safety advice that is not in the context."
)


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i + 1}] ({h.chunk.doc_id} - {h.chunk.section})\n{h.chunk.text}" for i, h in enumerate(hits))


def extractive_answer(question: str, hits: list[Hit], retriever: Retriever) -> str:
    """Pick the best-matching sentence, then keep its immediate neighbours.

    SOP sentences usually split a rule across two lines ("Test earth resistance once a year. The value must
    be below 1 ohm."), so a +/-1 sentence window recovers the value that the topic sentence points to.
    Procedural questions ("how do I...") return the whole top-ranked section as ordered steps.
    """
    groups = expansion_groups(question)
    total_w = sum(max(retriever.idf(next(iter(g))), 1.0) for g in groups) or 1.0
    if not hits:
        return ""
    # Answer from the top-ranked SOP only. Mixing in another SOP's numbers is riskier than missing a cross-SOP detail.
    # (Guards against "right words, wrong equipment", e.g. transformer limits for a gearbox question.)
    lead = hits[0]
    top3 = [h for h in hits[:4] if h.chunk.doc_id == lead.chunk.doc_id][:4]

    if PROCEDURAL.search(question):
        h = top3[0]
        return " ".join(f"{u.rstrip('.!? ')} [{h.rank}]." for u in h.chunk.units[:6])

    cands = []  # (score, hit, position)
    for h in top3:
        head = set(content_tokens(h.chunk.section))
        for pos, unit in enumerate(h.chunk.units):
            toks = set(content_tokens(unit))
            matched = sum(max(retriever.idf(next(iter(g))), 1.0) * (1.0 if g & toks else 0.5 if g & head else 0.0)
                          for g in groups)
            cands.append(((matched / total_w) * (0.5 + 0.5 * h.score / lead.score) + 0.25 * h.score, h, pos))
    cands.sort(key=lambda c: -c[0])
    best_score, best_hit, best_pos = cands[0]

    picked: list[tuple[int, int, str]] = []  # (rank, position, sentence)
    units = best_hit.chunk.units
    for pos in range(max(0, best_pos - 1), min(len(units), best_pos + 2)):
        picked.append((best_hit.rank, pos, units[pos]))
    # a strong second chunk of the same SOP is added as extra evidence
    for score, h, pos in cands[1:]:
        if h.rank != best_hit.rank and score >= 0.85 * best_score and len(picked) < 4:
            picked.append((h.rank, pos, h.chunk.units[pos]))
            break
    picked.sort(key=lambda x: (x[0], x[1]))
    return " ".join(f"{sent.rstrip('.!? ')} [{rank}]." for rank, _, sent in picked)


def llm_answer(question: str, hits: list[Hit]) -> str | None:
    """Return the LLM answer, or None when disabled / failed (caller then falls back)."""
    if config.LLM_PROVIDER != "anthropic" or not config.ANTHROPIC_API_KEY:
        return None
    body = json.dumps({
        "model": config.LLM_MODEL, "max_tokens": 400, "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"content-type": "application/json", "x-api-key": config.ANTHROPIC_API_KEY,
                 "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_S) as resp:
            data = json.load(resp)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        return None if not text or "INSUFFICIENT_CONTEXT" in text else text
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
