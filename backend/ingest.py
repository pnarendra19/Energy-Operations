"""Document loading and section-aware chunking.

Why section-aware? SOPs are short, structured documents where one numbered step or one
threshold is the answer. Fixed-size character windows would split "alarm at 75 C / trip at 85 C"
across chunks. We therefore: (1) split by markdown headings, (2) split sections into sentence/list-item
"units", (3) pack whole units into chunks up to CHUNK_MAX_WORDS with a small overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+")
_TITLE_RE = re.compile(r"^#\s+(?P<id>[A-Z]+-[A-Z]+-\d+)\s*:\s*(?P<title>.+)$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    units: list[str]
    index: int
    text: str = field(init=False)

    def __post_init__(self):
        self.text = " ".join(self.units)

    @property
    def embed_text(self) -> str:
        # Heading context is embedded with the body so "Weather Limits" helps match "wind speed" queries.
        return f"{self.title}. {self.section}. {self.text}"

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "title": self.title,
                "section": self.section, "text": self.text}


@dataclass
class Document:
    doc_id: str
    title: str
    revision: str
    source: str
    sections: list[tuple[str, str]]  # (heading, body)


def parse_document(raw: str, source: str) -> Document:
    lines = raw.strip().splitlines()
    doc_id, title = Path(source).stem.upper().replace("_", "-"), Path(source).stem
    m = _TITLE_RE.match(lines[0]) if lines else None
    if m:
        doc_id, title = m.group("id"), m.group("title").strip()
    elif lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
    revision = ""
    sections: list[tuple[str, str]] = []
    heading, buf = "Overview", []
    for line in lines[1:]:
        if line.startswith("## "):
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        elif not sections and not buf and "|" in line and not revision:
            revision = line.strip()  # metadata line under the title
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return Document(doc_id, title, revision, source, sections)


def split_units(body: str) -> list[str]:
    """Split a section into sentence / list-item units (list numbering is stripped)."""
    units: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _LIST_ITEM.sub("", line)
        units.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
    return units


def chunk_document(doc: Document, max_words: int | None = None, overlap: int | None = None) -> list[Chunk]:
    max_words = max_words or config.CHUNK_MAX_WORDS
    overlap = config.CHUNK_OVERLAP_UNITS if overlap is None else overlap
    chunks: list[Chunk] = []
    for s_idx, (heading, body) in enumerate(doc.sections, start=1):
        units, cur, words, c_idx = split_units(body), [], 0, 1
        i = 0
        while i < len(units):
            u = units[i]
            n = len(u.split())
            if cur and words + n > max_words:
                chunks.append(Chunk(f"{doc.doc_id}#s{s_idx}c{c_idx}", doc.doc_id, doc.title, heading, cur, len(chunks)))
                c_idx += 1
                cur = cur[-overlap:] if overlap else []
                words = sum(len(x.split()) for x in cur)
                if cur and words + n > max_words:  # overlap would stall progress
                    cur, words = [], 0
            cur.append(u)
            words += n
            i += 1
        if cur:
            chunks.append(Chunk(f"{doc.doc_id}#s{s_idx}c{c_idx}", doc.doc_id, doc.title, heading, cur, len(chunks)))
    return chunks


def load_documents(directory: Path | None = None) -> list[Document]:
    directory = Path(directory or config.DOCS_DIR)
    docs = []
    for p in sorted(directory.glob("*.md")) + sorted(directory.glob("*.txt")):
        docs.append(parse_document(p.read_text(encoding="utf-8"), p.name))
    return docs


def build_chunks(docs: list[Document]) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        for c in chunk_document(d):
            c.index = len(out)
            out.append(c)
    return out
