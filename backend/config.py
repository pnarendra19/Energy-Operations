"""Central configuration. Every value can be overridden with an environment variable,
so nothing secret or deployment-specific is hard-coded (see submission_instructions.md)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "data" / "documents"))
EVAL_FILE = Path(os.getenv("EVAL_FILE", BASE_DIR / "data" / "eval_questions.json"))
FRONTEND_DIR = BASE_DIR / "frontend"
RESULTS_DIR = BASE_DIR / "docs"

# --- chunking -------------------------------------------------------------
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "80"))   # soft cap per chunk
CHUNK_OVERLAP_UNITS = int(os.getenv("CHUNK_OVERLAP_UNITS", "1"))  # sentences repeated between chunks

# --- retrieval ------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))      # weight of dense (LSA) vs sparse (TF-IDF)
LSA_DIMS = int(os.getenv("LSA_DIMS", "64"))
ENTITY_PENALTY = float(os.getenv("ENTITY_PENALTY", "1.0"))            # 0 disables the missing-key-term penalty
DOC_PRIOR = float(os.getenv("DOC_PRIOR", "0.25"))                  # weight of the document-level (which SOP?) score
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.20"))  # below this -> abstain
MIN_QUERY_COVERAGE = float(os.getenv("MIN_QUERY_COVERAGE", "0.60"))    # share of question terms found in context

# --- generation / guardrails ---------------------------------------------
# Default is a fully offline extractive generator. Set LLM_PROVIDER=anthropic and
# ANTHROPIC_API_KEY to let an LLM write the answer (guardrails still apply).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "20"))
SUPPORT_THRESHOLD = float(os.getenv("SUPPORT_THRESHOLD", "0.70"))  # content-word recall for a "supported" sentence

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024)))
