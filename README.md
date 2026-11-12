# Energy Operations Document RAG Assistant

Answers operational questions about renewable-energy **safety, maintenance and operations SOPs**, with
**citations, retrieval evidence and hallucination checks**. Built for the Greenko AI/ML case study
(Python API + HTML frontend + Railway deployment).

> The 8 bundled SOPs in `data/documents/` are **fictional dummy data written for this exercise**. All numbers
> (temperatures, wind speeds, timings) are illustrative and must never be used as real engineering guidance.

## 1. Problem, user, success
* **Problem:** field and control-room staff need exact values and steps (isolation waits, wind limits, alarm
  thresholds, reporting deadlines) buried in long SOPs. A wrong number is a safety risk, so an assistant must
  **ground every statement in a cited SOP and refuse when the SOPs do not say**.
* **User:** an operations / maintenance engineer or shift in-charge.
* **Input:** a natural-language question. **Output:** answer with `[n]` citations, the retrieved chunks and
  scores, and a verdict from the hallucination checks (`grounded`, `partially_supported`, `unsupported`, `abstained`).
* **Success:** right SOP retrieved, right value in the answer, only the right SOP cited, refusal on out-of-scope questions.

## 2. Architecture
```
SOP .md files -> parse (title/sections) -> section-aware chunks (sentence units, <=80 words, 1-sentence overlap)
   -> embeddings: TF-IDF (sparse) + LSA/SVD (dense)  +  document-level index
   -> hybrid retrieval  -> confidence gate (abstain?) -> generator -> hallucination checks -> API -> HTML UI
```
| Step | File | Design choice and why |
|---|---|---|
| Chunking | `backend/ingest.py` | Split by heading, then by sentence/list item, pack whole sentences. SOP facts live in one step or threshold; fixed windows would split "alarm 75 / trip 85". Heading text is embedded with the chunk. |
| Embeddings | `backend/retrieval.py` | **Offline and reproducible** (no API key): TF-IDF (stemmed uni+bigrams) and LSA (dense, 64-d). Swappable for a neural embedder via `HybridEmbedder`. |
| Retrieval | `backend/retrieval.py` | `score = (1-b)*(a*dense + (1-a)*sparse) + b*doc_score`, then multiplied by the SOP's **key-term coverage**. The last term stops "right words, wrong equipment" (see section 4). Domain glossary expands LOTO, BESS, DGA, PTW etc. (`textutils.py`). |
| Gate | `backend/guardrails.py` | Abstain if top score is low **or** fewer than 60% of the question's key terms appear in the retrieved text. |
| Generation | `backend/generation.py` | Default **extractive** generator (best sentence +/-1 neighbour; whole section for "how do I" questions; top-ranked SOP only). Optional **LLM** (`LLM_PROVIDER=anthropic`) answers only from numbered context. |
| Hallucination checks | `backend/guardrails.py` | Per sentence: must cite a valid `[n]`; content-word support >= 0.70 against cited chunk; **every number must appear in the cited evidence**. An LLM answer that fails is discarded and replaced by the extractive answer. |
| API | `backend/app.py` | FastAPI: `/health`, `POST /api/ask`, `GET/POST /api/documents`, `GET /api/evaluate`, `/` (UI). |
| UI | `frontend/index.html` | Plain HTML/CSS/JS. Clickable citation chips, evidence panel with dense/sparse scores, per-sentence check table, evaluation button. |

**What is RAG vs agent vs plain LLM here:** this is **RAG** (documents are retrieved, then used to answer).
It is **not an agent** (no tools or multi-step decisions). By default there is **no LLM call at all**.

## 3. Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.12 recommended (see Railway notes)
pip install -r requirements.txt
uvicorn backend.app:app --reload                        # open http://localhost:8000
python -m backend.evaluate                              # writes docs/eval_results.{json,md}
pip install pytest httpx && pytest -q                   # 8 tests
```
Optional LLM mode (never hard-code keys): `export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... [LLM_MODEL=...]`.
Tunables (env vars) are listed in `backend/config.py`.

## 4. Evaluation (offline extractive mode, 30 questions)
Questions: 24 in-scope (`Q01-Q24`) and 6 out-of-scope (`U01-U06`) in `data/eval_questions.json`.

| Metric | Result |
|---|---:|
| Retrieval Recall@1 / MRR (right SOP) | 1.00 / 1.00 |
| Fact present in retrieved text (all gold facts) | 1.00 |
| Answer fact-recall | 0.98 |
| Exact answers (all gold facts present) | 0.96 (23/24) |
| Answers cite only the correct SOP | 1.00 |
| Abstains on out-of-scope | 6/6 |
| False abstain on in-scope | 0/24 |
| Injected fabricated sentence / altered number / removed citation flagged (24 answers tested) | 100% / 100% / 100% |

**Read these numbers with care (important limitations of the evaluation itself):**
* The questions were **written by the developer alongside the SOPs, and the pipeline was tuned while looking at them**. These are *in-sample* scores on a tiny corpus (42 chunks) and are optimistic. A held-out set written by someone else would score lower.
* **Groundedness 100% is nearly built in**: the extractive generator copies SOP sentences, so the checker has little to catch. The meaningful evidence is the **stress test** (fabricated sentence / changed number / missing citation are all flagged), and the checker's value is mainly for the LLM path.
* The ablation in `docs/eval_results.md` shows TF-IDF, LSA and hybrid are **indistinguishable on 42 chunks**; I do not claim hybrid is better, only that it is harmless here. Only the key-term penalty changed results (Recall@1 0.958 -> 1.0).

**A real failure found and fixed:** for "At what oil temperature does the gearbox trip?" the first version
answered with the **transformer's** limits (alarm 85 C / trip 95 C), marked "grounded", because those words really are in an SOP.
My fact check initially passed it by coincidence ("85 degrees" appears in both SOPs). Fixes: (1) stricter gold facts and a "cites only the correct SOP" metric,
(2) the key-term penalty (the transformer SOP never mentions "gearbox"), (3) the generator uses only the top-ranked SOP.
This class of error (correct sentence, wrong equipment) is the main residual risk of the approach.

**Known remaining failures:** Q23 ("how is a near miss classified **and** when is it logged") returns the severity
definition but not the reporting deadline (multi-part questions across sections). `U03` (torque for *gearbox* bolts) is only refused because the
coverage gate sees "mounting/bolts" missing; the inverter SOP's "no torque values here" sentence is a close call.

## 5. Assumptions and limitations
* Plain-text/Markdown SOPs only (no PDF/table/figure parsing, no OCR). Uploaded SOPs are held **in memory** and vanish on restart.
* Lexical grounding check, not entailment: it can miss a sentence that reuses real words with flipped meaning ("do" vs "do not"). Production would add an NLI or LLM-judge pass.
* The abstention thresholds (0.60 coverage, 0.20 score) were set on this small set; one in-scope question sits exactly at 0.60. Top retrieval score alone does not separate in/out-of-scope here; term coverage does the work.
* Cross-SOP questions are answered from the top SOP only (a deliberate safety trade-off).
* The optional LLM path is implemented but **was not run against a live API key** in development; only the fallback logic path is exercised by design. Test it before relying on it.
* No auth, rate limiting, audit log, or document versioning; revision numbers are shown but not enforced.

## 6. Production next steps
Neural embeddings plus a reranker; held-out eval set from real operator questions; NLI/LLM-judge groundedness;
persistent vector store and document versioning with approval workflow; PDF ingestion; auth, logging and feedback capture; red-team tests for prompt injection inside documents.

## 7. Deploy to Railway
Files already included: `Procfile` (`uvicorn backend.app:app --host 0.0.0.0 --port $PORT`), `requirements.txt`, `.python-version` (3.12), `railway.json` (health check `/health`).
1. Push this folder to a GitHub repo (`.env` and `.venv` are git-ignored).
2. Railway -> **New Project -> Deploy from GitHub repo** -> select the repo.
3. Service -> **Settings -> Networking -> Generate Domain**. Open the URL; `/health` should return `{"status":"ok",...}`.
4. (Optional) Variables: `LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `LLM_MODEL`.
CLI alternative: `railway login && railway init && railway up && railway domain`.

**Python version matters:** Railpack defaults to Python 3.13, but the pinned `numpy==1.26.4` / `scikit-learn==1.5.1` have no 3.13 wheels.
Keep `.python-version` = `3.12`; if the build still picks 3.13, set the service variable `RAILPACK_PYTHON_VERSION=3.12`.
**Deployment status: not yet deployed.** Add your live URL here after deploying, with a screenshot of `/health` and the UI as evidence.

## 8. Presenting this (maps to the Presentation Guide)
* **Problem (1 min):** wrong numbers in safety SOPs are dangerous, so answers must be cited and the system must refuse when unsure.
* **Flow:** SOPs -> chunk -> embed -> retrieve -> gate -> generate -> verify -> UI. Open `ingest.py`, `retrieval.py`, `guardrails.py`, `evaluate.py` only.
* **Model justification:** TF-IDF + LSA because the corpus is tiny, offline, explainable; baseline = sparse-only. Key settings: chunk size (80 words: precision vs context), alpha (dense vs sparse mix), coverage threshold (answer vs abstain).
* **Validation:** show the metrics table and say they are in-sample. Show one failure (the gearbox/transformer case) and how it was fixed.
* **GenAI part:** be precise: default has no LLM; RAG yes; agent no; LLM optional and guarded.
* **Likely questions:** what if the key-term penalty were removed (Recall@1 drops to 0.958 and the wrong-equipment answer returns)? What is calculated vs generated? How do you know it works on new data (you do not yet: no held-out set)? Where did AI tools help and how did you verify (tests, stress test, manual review of failures)?
#   E n e r g y - O p e r a t i o n s  
 