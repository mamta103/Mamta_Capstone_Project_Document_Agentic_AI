from collections import Counter
from dataclasses import dataclass, field

from app.agents.workflow import build_workflow, run_workflow
from app.core.embedding_service import EmbeddingService
from app.core.ingestion.document_loader import load_documents
from app.core.text_chunker import chunk_sections
from app.core.retriever import MAX_QUESTION_CHARS, Retriever
from app.core.llm_client import LLMClient, LLMError
from app.core.chroma_store import ChromaStore, SearchResult


@dataclass
class IngestReport:
    chunk_counts: dict[str, int]
    errors: list[str]


@dataclass
class RagAnswer:
    status: str  # "answered" | "insufficient_evidence" | "responded" | "blocked" | "error"
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    retrieved: list[SearchResult] = field(default_factory=list)
    cited: bool = False
    validated: bool = False
    attempts: int = 0
    trace: list[str] = field(default_factory=list)


class RagService:
    def __init__(
        self,
        embedder: EmbeddingService | None = None,
        store: ChromaStore | None = None,
        llm: LLMClient | None = None,
    ):
        self.embedder = embedder or EmbeddingService()
        self.store = store or ChromaStore()
        self.retriever = Retriever(self.embedder, self.store)
        self._llm = llm
        self._workflow = None

    def _get_workflow(self):
        # LLM client created lazily so ingestion and search work without an API key.
        if self._workflow is None:
            if self._llm is None:
                self._llm = LLMClient()
            self._workflow = build_workflow(self._llm, self.retriever, self.store)
        return self._workflow

    def ingest(self, files: list[tuple[str, bytes]]) -> IngestReport:
        result = load_documents(files)
        chunks = chunk_sections(result.sections)
        if chunks:
            embeddings = self.embedder.embed_documents([c.text for c in chunks])
            self.store.add_chunks(chunks, embeddings)
        return IngestReport(
            chunk_counts=dict(Counter(c.metadata["source"] for c in chunks)),
            errors=result.errors,
        )

    def search(self, question: str) -> list[SearchResult]:
        return self.retriever.search(question)

    def ask(self, question: str) -> RagAnswer:
        question = (question or "").strip()
        if not question:
            return RagAnswer(status="error", answer="Please enter a question.")
        if len(question) > MAX_QUESTION_CHARS:
            return RagAnswer(status="error", answer=f"Question is too long (max {MAX_QUESTION_CHARS} characters).")

        try:
            workflow = self._get_workflow()
        except LLMError as exc:
            return RagAnswer(status="error", answer=str(exc))

        state = run_workflow(workflow, question)
        status = state.get("status", "error")
        retrieved = state.get("retrieved", [])
        common = dict(retrieved=retrieved, attempts=state.get("attempts", 0), trace=state.get("trace", []))

        if status == "responded":
            return RagAnswer(status=status, answer=state["reply"], **common)
        if status == "answered":
            cited = state.get("cited", [])
            sources = [retrieved[i - 1] for i in cited] if cited else retrieved
            return RagAnswer(
                status=status, answer=state["answer"], sources=sources,
                cited=bool(cited), validated=state.get("validated", False), **common,
            )
        return RagAnswer(status=status, answer=state.get("answer", "Something went wrong."), **common)
