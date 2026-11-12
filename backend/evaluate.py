"""Offline evaluation. Run:  python -m backend.evaluate   (writes docs/eval_results.json + .md)

Metrics
  Retrieval  : Recall@1 / Recall@k (does a top-ranked chunk come from the gold SOP?), MRR,
               Fact-recall@k (do ALL gold facts appear in the retrieved text?)
  Answer     : Fact-recall (share of gold facts present in the answer), Exact-answer rate (all facts)
  Safety     : Abstention rate on out-of-scope questions, false-abstain rate on in-scope ones,
               Groundedness rate and citation validity of produced answers
  Checker    : Detection rate of injected hallucinations (fabricated sentence, altered number, removed citations)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import config
from .guardrails import NUM_RE, check_answer
from .ingest import build_chunks, load_documents
from .rag import RAGPipeline
from .retrieval import Retriever


def load_eval(path: Path | None = None) -> list[dict]:
    return json.loads(Path(path or config.EVAL_FILE).read_text(encoding="utf-8"))


def _has(text: str, group: list[str]) -> bool:
    t = text.lower()
    return any(alt.lower() in t for alt in group)


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


def retrieval_metrics(retriever: Retriever, items: list[dict], k: int, alpha: float | None = None,
                      beta: float | None = None) -> dict:
    ranks, fact_hits = [], []
    for it in (i for i in items if i["gold_doc"]):
        hits = retriever.search(it["question"], k=k, alpha=alpha, beta=beta)
        rank = next((h.rank for h in hits if h.chunk.doc_id == it["gold_doc"]), None)
        ranks.append(rank)
        ctx = " ".join(h.chunk.text for h in hits)
        fact_hits.append(all(_has(ctx, g) for g in it["facts"]))
    return {"recall@1": _mean(r == 1 for r in ranks),
            "recall@3": _mean(r is not None and r <= 3 for r in ranks),
            f"recall@{k}": _mean(r is not None for r in ranks),
            "mrr": _mean((1 / r) if r else 0 for r in ranks),
            f"fact_recall@{k}": _mean(fact_hits)}


def ablation(items: list[dict], k: int) -> list[dict]:
    chunks = build_chunks(load_documents())
    rows = []
    for title in (True, False):
        r = Retriever(chunks, embed_title=title)
        for name, a, b, ent in (("sparse only (TF-IDF)", 0.0, 0.0, 0.0), ("dense only (LSA)", 1.0, 0.0, 0.0),
                                ("hybrid a=0.5", 0.5, 0.0, 0.0), ("hybrid + doc prior 0.25", 0.5, 0.25, 0.0),
                                ("hybrid + doc prior + key-term penalty (default)", 0.5, 0.25, 1.0)):
            r.entity_penalty = ent
            m = retrieval_metrics(r, items, k, alpha=a, beta=b)
            rows.append({"retriever": name, "title_in_embedding": title, **m})
    return rows


def hallucination_stress_test(pipeline: RAGPipeline, items: list[dict]) -> dict:
    fabricated = " The maximum permitted working temperature is 140 degrees C and must be signed off by the Regional Director [1]."
    n = det_fab = det_num = n_num = det_cite = 0
    for it in items:
        if not it["gold_doc"]:
            continue
        res = pipeline.ask(it["question"])
        if res["abstained"]:
            continue
        hits = pipeline.retriever.search(it["question"], k=config.TOP_K)
        ans = res["answer"]
        n += 1
        det_fab += check_answer(ans + fabricated, hits)["verdict"] != "grounded"
        # alter the first number in the answer (e.g. 85 -> 92)
        m = NUM_RE.search(re.sub(r"\[\d+\]", "", ans))
        if m:
            n_num += 1
            new = str(round(float(m.group()) + 7, 1)).rstrip("0").rstrip(".")
            mutated = re.sub(r"\[\d+\]", lambda x: x.group(), ans)
            mutated = mutated[:mutated.find(m.group())] + new + mutated[mutated.find(m.group()) + len(m.group()):]
            det_num += check_answer(mutated, hits)["verdict"] != "grounded"
        det_cite += check_answer(re.sub(r"\s*\[\d+\]", "", ans), hits)["verdict"] != "grounded"
    return {"answers_tested": n,
            "fabricated_sentence_detected": _mean([det_fab / n]) if n else None,
            "altered_number_detected": _mean([det_num / n_num]) if n_num else None,
            "missing_citations_detected": _mean([det_cite / n]) if n else None}


def run_evaluation(pipeline: RAGPipeline | None = None, with_ablation: bool = True) -> dict:
    pipeline = pipeline or RAGPipeline()
    items, k = load_eval(), config.TOP_K
    per_q, src_ok, in_scope_fact_rec, exact, abst_out, false_abst, grounded, cites_ok = [], [], [], [], [], [], [], []
    for it in items:
        res = pipeline.ask(it["question"], k)
        row = {"id": it["id"], "question": it["question"], "in_scope": bool(it["gold_doc"]),
               "gold_doc": it["gold_doc"], "abstained": res["abstained"], "top_doc": res["evidence"][0]["doc_id"],
               "top_score": res["gate"]["top_score"], "coverage": res["gate"]["query_coverage"],
               "verdict": res["checks"]["verdict"], "answer": res["answer"]}
        if it["gold_doc"]:
            false_abst.append(res["abstained"])
            if not res["abstained"]:
                found = [_has(res["answer"], g) for g in it["facts"]]
                cited_docs = {c["doc_id"] for c in res["citations"]}
                row["source_correct"] = cited_docs == {it["gold_doc"]}
                src_ok.append(row["source_correct"])
                row["fact_recall"] = round(sum(found) / len(found), 3)
                in_scope_fact_rec.append(row["fact_recall"])
                exact.append(all(found))
            else:
                in_scope_fact_rec.append(0.0)
                exact.append(False)
                src_ok.append(False)
        else:
            abst_out.append(res["abstained"])
        if not res["abstained"]:
            grounded.append(res["checks"]["verdict"] == "grounded")
            cites_ok.append(res["checks"]["citations_valid"])
        per_q.append(row)

    out = {
        "config": {"top_k": k, "chunk_max_words": config.CHUNK_MAX_WORDS, "alpha": config.HYBRID_ALPHA,
                   "min_retrieval_score": config.MIN_RETRIEVAL_SCORE, "min_query_coverage": config.MIN_QUERY_COVERAGE,
                   "generator": "llm+fallback" if config.LLM_PROVIDER == "anthropic" else "extractive (offline)"},
        "dataset": {"documents": len(pipeline.docs), "chunks": len(pipeline.chunks),
                    "questions": len(items), "in_scope": sum(bool(i["gold_doc"]) for i in items),
                    "out_of_scope": sum(not i["gold_doc"] for i in items)},
        "retrieval": retrieval_metrics(pipeline.retriever, items, k),
        "answer": {"fact_recall": _mean(in_scope_fact_rec), "exact_answer_rate": _mean(exact),
                   "cites_only_correct_sop_rate": _mean(src_ok)},
        "safety": {"abstention_on_out_of_scope": _mean(abst_out), "false_abstain_rate": _mean(false_abst),
                   "groundedness_rate": _mean(grounded), "citation_validity_rate": _mean(cites_ok)},
        "hallucination_checker": hallucination_stress_test(pipeline, items),
        "per_question": per_q,
    }
    if with_ablation:
        out["ablation"] = ablation(items, k)
    return out


def to_markdown(r: dict) -> str:
    L = ["# Evaluation results", "", f"Generator: **{r['config']['generator']}** | top_k={r['config']['top_k']} | "
         f"alpha={r['config']['alpha']} | chunk<= {r['config']['chunk_max_words']} words", "",
         f"Dataset: {r['dataset']['documents']} SOPs, {r['dataset']['chunks']} chunks, {r['dataset']['questions']} questions "
         f"({r['dataset']['in_scope']} in-scope, {r['dataset']['out_of_scope']} out-of-scope)", "",
         "| Metric | Value |", "|---|---:|"]
    for sec in ("retrieval", "answer", "safety", "hallucination_checker"):
        for k, v in r[sec].items():
            L.append(f"| {sec}: {k} | {v} |")
    if "ablation" in r:
        L += ["", "## Retriever ablation", "", "| Retriever | Title in embedding | Recall@1 | MRR | Fact-recall@k |", "|---|---|---:|---:|---:|"]
        fk = [k for k in r["ablation"][0] if k.startswith("fact_recall")][0]
        for a in r["ablation"]:
            L.append(f"| {a['retriever']} | {a['title_in_embedding']} | {a['recall@1']} | {a['mrr']} | {a[fk]} |")
    L += ["", "## Per-question results", "", "| ID | In scope | Abstained | Top doc | Score | Cover | Verdict | Fact recall | Correct SOP cited |", "|---|---|---|---|---:|---:|---|---:|---|"]
    for q in r["per_question"]:
        L.append(f"| {q['id']} | {q['in_scope']} | {q['abstained']} | {q['top_doc']} | {q['top_score']} | {q['coverage']} | {q['verdict']} | {q.get('fact_recall', '-')} | {q.get('source_correct', '-')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    res = run_evaluation()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "eval_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (config.RESULTS_DIR / "eval_results.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
