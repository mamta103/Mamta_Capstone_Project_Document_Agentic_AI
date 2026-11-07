import pytest
from fastapi.testclient import TestClient

from app.backend.main import app, get_service
from app.backend.rag_service import IngestReport, RagAnswer
from app.core.chroma_store import SearchResult

SOURCE = SearchResult(text="Users authenticate via POST /api/v1/session.", metadata={"source": "guide.txt"}, score=0.56)


class FakeStore:
    def __init__(self):
        self.sources = ["guide.txt"]

    def list_sources(self):
        return list(self.sources)

    def count(self):
        return len(self.sources)

    def reset(self):
        self.sources = []


class FakeService:
    def __init__(self):
        self.store = FakeStore()
        self.ingested = []

    def ingest(self, files):
        self.ingested.extend(files)
        if files[0][0].endswith(".exe"):
            return IngestReport(chunk_counts={}, errors=[f"{files[0][0]}: Unsupported file type '.exe'."])
        return IngestReport(chunk_counts={name: 2 for name, _ in files}, errors=[])

    def ask(self, question):
        return RagAnswer(status="answered", answer="Call POST /api/v1/session [1].", sources=[SOURCE],
                         retrieved=[SOURCE], cited=True, validated=True, attempts=1, trace=["Planner: retrieve"])


@pytest.fixture
def client():
    service = FakeService()
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app), service
    app.dependency_overrides.clear()


def test_health(client):
    c, _ = client
    response = c.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_indexes_files(client):
    c, service = client
    response = c.post("/upload", files=[("files", ("guide.txt", b"hello world", "text/plain"))])
    assert response.status_code == 200
    assert response.json() == {"chunks": {"guide.txt": 2}, "errors": []}
    assert service.ingested == [("guide.txt", b"hello world")]


def test_upload_unsupported_returns_422(client):
    c, _ = client
    response = c.post("/upload", files=[("files", ("bad.exe", b"MZ", "application/octet-stream"))])
    assert response.status_code == 422
    assert "Unsupported" in response.json()["detail"]["errors"][0]


def test_upload_oversized_returns_413(client, monkeypatch):
    import dataclasses
    import app.backend.main as api_main
    monkeypatch.setattr(api_main, "settings", dataclasses.replace(api_main.settings, max_file_mb=1))
    c, _ = client
    response = c.post("/upload", files=[("files", ("big.txt", b"a" * (1024 * 1024 + 1), "text/plain"))])
    assert response.status_code == 413


def test_ask_returns_answer_and_sources(client):
    c, _ = client
    response = c.post("/ask", json={"question": "How do users log in?"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "answered"
    assert body["sources"][0]["source"] == "guide.txt"
    assert body["validated"] is True


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_rejects_empty_question(client, question):
    c, _ = client
    assert c.post("/ask", json={"question": question}).status_code in (400, 422)


def test_documents_and_reset(client):
    c, _ = client
    assert c.get("/documents").json() == {"documents": ["guide.txt"], "chunks": 1}
    assert c.post("/reset").json()["status"] == "ok"
    assert c.get("/documents").json()["documents"] == []
