from app.core.config import MAX_QUESTION_CHARS, settings  # noqa: F401  (re-exported)
from app.core.embedding_service import EmbeddingService
from app.core.chroma_store import ChromaStore, SearchResult


class InvalidQuestionError(ValueError):
    pass


class Retriever:
    def __init__(self, embedder: EmbeddingService, store: ChromaStore):
        self._embedder = embedder
        self._store = store

    def search(self, question: str, top_k: int | None = None, min_score: float | None = None) -> list[SearchResult]:
        question = (question or "").strip()
        if not question:
            raise InvalidQuestionError("Please enter a question.")
        if len(question) > MAX_QUESTION_CHARS:
            raise InvalidQuestionError(f"Question is too long (max {MAX_QUESTION_CHARS} characters).")

        top_k = top_k or settings.top_k
        min_score = settings.min_score if min_score is None else min_score
        results = self._store.query(self._embedder.embed_query(question), top_k)
        return [r for r in results if r.score >= min_score]
