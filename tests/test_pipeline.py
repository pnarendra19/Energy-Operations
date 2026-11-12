import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.guardrails import check_answer
from backend.ingest import build_chunks, load_documents
from backend.rag import RAGPipeline


@pytest.fixture(scope="module")
def pipe():
    return RAGPipeline()


def test_chunks_are_whole_sentences_and_cited(pipe):
    assert len(pipe.docs) == 8 and len(pipe.chunks) > 30
    for c in pipe.chunks:
        assert c.text and all(u.strip() for u in c.units)
        assert c.chunk_id.startswith(c.doc_id)


def test_answers_with_correct_value_and_source(pipe):
    r = pipe.ask("At what wind speed do turbines cut out automatically?")
    assert not r["abstained"] and "25 m/s" in r["answer"]
    assert {c["doc_id"] for c in r["citations"]} == {"SOP-OPS-202"}


def test_wrong_equipment_not_mixed_in(pipe):
    r = pipe.ask("At what oil temperature does the gearbox trip?")
    assert {c["doc_id"] for c in r["citations"]} == {"SOP-MNT-101"}
    assert "95 degrees" not in r["answer"]  # transformer limit must not leak in


@pytest.mark.parametrize("q", ["What is the CEO's annual salary?", "How do I change the engine oil in my car?"])
def test_abstains_out_of_scope(pipe, q):
    r = pipe.ask(q)
    assert r["abstained"] and r["citations"] == []


def test_guardrail_flags_fabricated_sentence_and_altered_number(pipe):
    hits = pipe.retriever.search("At what wind speed do turbines cut out automatically?")
    good = pipe.ask("At what wind speed do turbines cut out automatically?")["answer"]
    assert check_answer(good, hits)["verdict"] == "grounded"
    assert check_answer(good + " Turbines must be inspected every 3 days by the CEO [1].", hits)["verdict"] != "grounded"
    assert check_answer(good.replace("25", "32"), hits)["verdict"] != "grounded"
    assert check_answer("Turbines cut out at 25 m/s.", hits)["verdict"] != "grounded"  # no citation


def test_invalid_citation_detected(pipe):
    hits = pipe.retriever.search("cut out wind speed", k=2)
    assert not check_answer("Turbines cut out at 25 m/s [9].", hits)["citations_valid"]


def test_api_endpoints():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/").status_code == 200
        assert len(c.get("/api/documents").json()) == 8
        r = c.post("/api/ask", json={"question": "When should PV modules be cleaned?"}).json()
        assert "4 percent" in r["answer"] and r["checks"]["verdict"] == "grounded"
        assert c.post("/api/ask", json={"question": "hi"}).status_code == 422
        assert c.post("/api/documents", files={"file": ("x.exe", b"abc", "application/octet-stream")}).status_code == 400
