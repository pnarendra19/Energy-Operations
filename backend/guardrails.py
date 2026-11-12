"""Hallucination checks. These run on *every* answer (extractive or LLM-written).

Checks, in order of strictness:
  1. Retrieval gate   - abstain when the retrieved evidence is weak or does not cover the question.
  2. Citation check   - every answer sentence must carry a [n] that points to a retrieved chunk.
  3. Support check    - share of a sentence's content words found in the cited chunk(s).
  4. Number check     - every number in an answer must appear in the cited evidence
                        (thresholds, temperatures and timings are exactly what an LLM would invent).

Limits (documented in the README): this is lexical grounding, not entailment. It catches invented
facts and altered numbers well, but can miss a sentence that reuses real words with a flipped meaning
("do" -> "do not"). A production system would add an NLI / LLM-judge pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from . import config
from .retrieval import Hit
from .textutils import content_tokens, expansion_groups

CITE_RE = re.compile(r"\[(\d+)\]")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[int]
    support: float
    unsupported_numbers: list[str]
    status: str  # supported | unsupported | uncited | invalid_citation

    def to_dict(self):
        d = asdict(self)
        d["support"] = round(self.support, 3)
        return d


def split_answer(text: str) -> list[str]:
    text = re.sub(r"\.\s*((?:\[\d+\]\s*)+)", lambda m: " " + m.group(1).strip() + ".", text)  # ". [1]" -> " [1]."
    parts: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line.strip())
        if line:
            parts.extend(p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", line) if p.strip())
    return parts


def check_answer(answer: str, hits: list[Hit]) -> dict:
    by_n = {i + 1: h for i, h in enumerate(hits)}
    all_text = " ".join(h.chunk.text for h in hits)
    checks: list[SentenceCheck] = []
    for sent in split_answer(answer):
        cited = [int(n) for n in CITE_RE.findall(sent)]
        body = CITE_RE.sub("", sent)
        valid = [n for n in cited if n in by_n]
        if not cited:
            status, evidence = "uncited", all_text
        elif len(valid) != len(cited):
            status, evidence = "invalid_citation", all_text
        else:
            status, evidence = "supported", " ".join(by_n[n].chunk.text for n in valid)
        ev_tokens = set(content_tokens(evidence))
        sent_tokens = content_tokens(body)
        support = (sum(t in ev_tokens for t in sent_tokens) / len(sent_tokens)) if sent_tokens else 1.0
        ev_numbers = set(NUM_RE.findall(evidence))
        bad_nums = sorted({n for n in NUM_RE.findall(body) if n not in ev_numbers})
        if status == "supported" and (support < config.SUPPORT_THRESHOLD or bad_nums):
            status = "unsupported"
        checks.append(SentenceCheck(sent, cited, support, bad_nums, status))

    n = len(checks) or 1
    ok = sum(c.status == "supported" for c in checks)
    groundedness = ok / n
    if checks and ok == len(checks):
        verdict = "grounded"
    elif groundedness >= 0.5:
        verdict = "partially_supported"
    else:
        verdict = "unsupported"
    return {"verdict": verdict, "groundedness": round(groundedness, 3),
            "citations_valid": all(c.status != "invalid_citation" for c in checks),
            "sentences": [c.to_dict() for c in checks]}


def retrieval_gate(question: str, hits: list[Hit]) -> dict:
    """Decide whether the evidence is good enough to answer at all."""
    top = hits[0].score if hits else 0.0
    ctx = set()
    for h in hits[:3]:
        ctx.update(content_tokens(h.chunk.embed_text))
    groups = expansion_groups(question)
    covered = sum(bool(g & ctx) for g in groups)
    coverage = covered / len(groups) if groups else 0.0
    passed = top >= config.MIN_RETRIEVAL_SCORE and coverage >= config.MIN_QUERY_COVERAGE
    reason = ""
    if not passed:
        reason = ("Top retrieval score %.2f is below %.2f" % (top, config.MIN_RETRIEVAL_SCORE)
                  if top < config.MIN_RETRIEVAL_SCORE else
                  "Only %.0f%% of the question's key terms appear in the retrieved SOP text" % (coverage * 100))
    return {"passed": passed, "top_score": round(top, 3), "query_coverage": round(coverage, 3), "reason": reason}
