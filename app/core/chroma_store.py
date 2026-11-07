from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.text_chunker import Chunk

COLLECTION_NAME = "documents"
BATCH_SIZE = 500


@dataclass(frozen=True)
class SearchResult:
    text: str
    metadata: dict
    score: float

    @property
    def source_label(self) -> str:
        m = self.metadata
        label = m["source"]
        if "page" in m:
            label += f", page {m['page']}"
        if "sheet" in m:
            label += f", sheet {m['sheet']}"
        if "row_start" in m:
            rows = m["row_start"], m["row_end"]
            label += f", row {rows[0]}" if rows[0] == rows[1] else f", rows {rows[0]}-{rows[1]}"
        return label


class ChromaStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str = COLLECTION_NAME):
        self._client = chromadb.PersistentClient(
            path=persist_dir or settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._name = collection_name
        self._collection = self._open_collection()

    def _open_collection(self):
        # Embeddings are supplied by EmbeddingService, so Chroma's built-in embedder is disabled.
        return self._client.get_or_create_collection(
            self._name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for source in {c.metadata["source"] for c in chunks}:
            self.delete_source(source)
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            self._collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings[start:start + BATCH_SIZE],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def query(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(text=doc, metadata=meta, score=1.0 - dist)
            for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]

    def list_sources(self) -> list[str]:
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metadatas})

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._open_collection()
