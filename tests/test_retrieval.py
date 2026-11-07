import math
import zlib

import pytest

from app.core.ingestion.models import DocumentSection
from app.core.text_chunker import Chunk, chunk_sections
from app.core.retriever import InvalidQuestionError, Retriever
from app.backend.rag_service import RagService
from app.core.chroma_store import ChromaStore, SearchResult

from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_documents"


class FakeEmbedder:
    """Deterministic bag-of-words embedder: fast, offline, no model download."""

    DIM = 64

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        vec = [0.0] * self.DIM
        for word in text.lower().split():
            vec[zlib.crc32(word.encode()) % self.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def make_chunk(source, index, text, **extra):
    chunk_id = f"{source}:{index}"
    meta = {"source": source, "file_type": "txt", "chunk_index": index, "chunk_id": chunk_id, **extra}
    return Chunk(id=chunk_id, text=text, metadata=meta)


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_dir=str(tmp_path / "db"))


@pytest.fixture
def fake():
    return FakeEmbedder()


def add(store, embedder, chunks):
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))


# ---------- ChromaStore ----------

def test_add_and_metadata_roundtrip(store, fake):
    add(store, fake, [make_chunk("a.txt", 0, "alpha beta"), make_chunk("a.txt", 1, "gamma delta")])
    assert store.count() == 2
    results = store.query(fake.embed_query("alpha beta"), top_k=1)
    assert results[0].text == "alpha beta"
    assert results[0].metadata["chunk_id"] == "a.txt:0"
    assert results[0].metadata["source"] == "a.txt"


def test_readding_source_replaces_chunks(store, fake):
    add(store, fake, [make_chunk("a.txt", i, f"text {i}") for i in range(3)])
    add(store, fake, [make_chunk("a.txt", 0, "new shorter version")])
    assert store.count() == 1
    assert store.query(fake.embed_query("anything"), top_k=5)[0].text == "new shorter version"


def test_delete_source_only_removes_that_source(store, fake):
    add(store, fake, [make_chunk("a.txt", 0, "alpha")])
    add(store, fake, [make_chunk("b.txt", 0, "beta")])
    store.delete_source("a.txt")
    assert store.list_sources() == ["b.txt"]


def test_query_ordered_and_limited(store, fake):
    add(store, fake, [
        make_chunk("a.txt", 0, "port configuration port"),
        make_chunk("a.txt", 1, "port settings"),
        make_chunk("a.txt", 2, "unrelated banana text"),
    ])
    results = store.query(fake.embed_query("port configuration"), top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].text == "port configuration port"


def test_persistence_across_instances(tmp_path, fake):
    path = str(tmp_path / "db")
    add(ChromaStore(persist_dir=path), fake, [make_chunk("a.txt", 0, "persisted text")])
    assert ChromaStore(persist_dir=path).count() == 1


def test_reset_clears_everything(store, fake):
    add(store, fake, [make_chunk("a.txt", 0, "alpha")])
    store.reset()
    assert store.count() == 0
    assert store.list_sources() == []


def test_add_length_mismatch_rejected(store):
    with pytest.raises(ValueError):
        store.add_chunks([make_chunk("a.txt", 0, "x")], [])


def test_batching_more_than_batch_size(store, fake):
    chunks = [make_chunk("big.txt", i, f"row {i} value {i}") for i in range(1200)]
    add(store, fake, chunks)
    assert store.count() == 1200


# ---------- SearchResult labels ----------

@pytest.mark.parametrize("meta, expected", [
    ({"source": "a.pdf", "page": 3}, "a.pdf, page 3"),
    ({"source": "b.csv", "row_start": 2, "row_end": 7}, "b.csv, rows 2-7"),
    ({"source": "c.xlsx", "sheet": "Config", "row_start": 4, "row_end": 4}, "c.xlsx, sheet Config, row 4"),
    ({"source": "d.txt"}, "d.txt"),
])
def test_source_label(meta, expected):
    assert SearchResult(text="", metadata=meta, score=1.0).source_label == expected


# ---------- Retriever ----------

def test_retriever_empty_store_returns_empty(store, fake):
    assert Retriever(fake, store).search("anything") == []


def test_retriever_filters_by_min_score(store, fake):
    add(store, fake, [make_chunk("a.txt", 0, "port configuration"), make_chunk("a.txt", 1, "banana apple")])
    results = Retriever(fake, store).search("port configuration", top_k=5, min_score=0.5)
    assert [r.text for r in results] == ["port configuration"]


@pytest.mark.parametrize("question", ["", "   ", None, "x" * 1001])
def test_retriever_rejects_invalid_questions(store, fake, question):
    with pytest.raises(InvalidQuestionError):
        Retriever(fake, store).search(question)


# ---------- Real embedding model ----------

@pytest.fixture(scope="module")
def real_embedder():
    from app.core.embedding_service import EmbeddingService
    return EmbeddingService()


def test_real_embedding_shape_and_norm(real_embedder):
    vectors = real_embedder.embed_documents(["hello world", "second text"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert math.isclose(math.sqrt(sum(v * v for v in vectors[0])), 1.0, abs_tol=1e-3)


def test_real_embedding_semantic_similarity(real_embedder):
    q, auth, port = real_embedder.embed_documents([
        "How do users log in?",
        "Users authenticate by calling POST /api/v1/session with a username and password.",
        "The default HTTP port is 8080.",
    ])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(q, auth) > dot(q, port)


@pytest.fixture(scope="module")
def sample_service(real_embedder, tmp_path_factory):
    service = RagService(embedder=real_embedder, store=ChromaStore(persist_dir=str(tmp_path_factory.mktemp("db"))))
    report = service.ingest([(p.name, p.read_bytes()) for p in SAMPLES.glob("sample_*")])
    assert report.errors == []
    service.expected_count = sum(report.chunk_counts.values())
    return service


def test_ingest_report_counts(sample_service):
    assert sample_service.store.list_sources() == [
        "sample_api_endpoints.csv", "sample_architecture.pdf", "sample_config_reference.xlsx", "sample_install_guide.txt",
    ]
    assert sample_service.expected_count > 0
    assert sample_service.store.count() == sample_service.expected_count


def test_reingest_does_not_duplicate(sample_service):
    files = [(p.name, p.read_bytes()) for p in SAMPLES.glob("sample_*")]
    sample_service.ingest(files)
    assert sample_service.store.count() == sample_service.expected_count


def test_e2e_session_endpoint_question(sample_service):
    results = sample_service.search("Which API endpoint is used to create a session?")
    assert results
    top_texts = " ".join(r.text for r in results[:2])
    assert "/api/v1/session" in top_texts


def test_e2e_service_fails_to_start_question(sample_service):
    results = sample_service.search("What should I check if the service fails to start?")
    assert results
    assert any("fails to start" in r.text for r in results[:2])


def test_e2e_unrelated_question_scores_lower(sample_service):
    relevant = sample_service.retriever.search("How do users authenticate?", min_score=0.0)
    unrelated = sample_service.retriever.search("What is the capital of France?", min_score=0.0)
    assert relevant[0].score > unrelated[0].score
