"""Shared text normalisation: tokenising, light stemming, domain synonym expansion."""

"""This code for clean and normailzes text by converting words i"""



import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep codes like "E-204" and numbers like "18" or "49.5" as single tokens: in SOPs they carry the answer.
TOKEN_RE = re.compile(r"[a-z]+-\d+|\d+(?:\.\d+)?|[a-z]+")
STOP = (set(ENGLISH_STOP_WORDS) - {"not", "no"}) | {"does", "did", "doing", "done", "got", "gets"}
# Words that appear in almost every question but say nothing about *which* SOP is relevant.
FILLER = {"procedure", "process", "need", "needed", "required", "require", "allowed", "take", "make",
          "happen", "happens", "use", "used", "work", "done", "long", "many", "much", "far",
          "kind", "type", "tell", "give", "get", "safe", "safely", "away", "dead"}

# Domain vocabulary: abbreviations and near-synonyms used by operators but not always by the SOP text.
SYNONYMS = {
    "loto": "lockout tagout isolation lock",
    "lockout": "loto isolation lock", "tagout": "loto isolation lock",
    "ptw": "permit work",
    "wtg": "wind turbine generator",
    "bess": "battery energy storage system container",
    "battery": "bess cell container",
    "pv": "solar photovoltaic module",
    "dga": "dissolved gas analysis",
    "hse": "health safety",
    "imd": "india meteorological department cyclone",
    "regulator": "authorities regulators compliance",
    "inverter": "converter", "converter": "inverter",
    "climb": "height tower", "climbing": "height tower",
    "height": "climb tower",
    "megger": "insulation resistance",
    "fire": "thermal event",
    "classified": "severity level", "classification": "severity level", "categorised": "severity level",
    "lab": "laboratory analysis", "sample": "sampling",
    "often": "frequency interval", "frequently": "frequency interval", "frequent": "frequency interval",
    "frequency": "interval",
}


def _raw(text: str) -> list[str]:
    """Lowercase word tokens; remove possessives and drop one-letter words."""
    text = re.sub(r"[\u2019']s\b", "", text.lower())
    return [t for t in TOKEN_RE.findall(text) if len(t) > 1 or t.isdigit()]


def stem(w: str) -> str:
    """Very light suffix stripper (no external NLP dependency). Good enough to align
    climb/climbing, isolate/isolation/isolated, inspect/inspection."""
    if w.isdigit() or "-" in w or "." in w or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 4:
        w = w[:-1]
    if w.endswith("ied") and len(w) > 5:
        w = w[:-3] + "y"
    else:
        for suf in ("ing", "ed", "ion"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":  # logged -> logg -> log
                    w = w[:-1]
                break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    toks = _raw(text)
    return [stem(t) for t in toks if keep_stop or t not in STOP]


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in _raw(text) if t not in STOP and t not in FILLER]


def analyzer(text: str) -> list[str]:
    """TF-IDF analyzer: stemmed unigrams plus bigrams of adjacent content tokens."""
    toks = tokens(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def expand_query(q: str) -> str:
    extra = []
    for t in _raw(q):
        if t in SYNONYMS:
            extra.append(SYNONYMS[t])
    return q + (" " + " ".join(extra) if extra else "")


def expansion_groups(q: str) -> list[set[str]]:
    """One set of acceptable stems per content word in the question (word itself + its synonyms).
    Used to measure how much of the question the retrieved context actually covers."""
    groups = []
    for t in _raw(q):
        if t in STOP or t in FILLER:
            continue
        g = {stem(t)}
        for s in SYNONYMS.get(t, "").split():
            g.add(stem(s))
        groups.append(g)
    return groups
