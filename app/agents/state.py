import operator
from typing import Annotated, TypedDict

from app.core.chroma_store import SearchResult


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    route: str  # "retrieve" | "respond" | "end"
    reply: str
    retrieved: list[SearchResult]
    answer: str
    cited: list[int]
    validation: dict
    feedback: str
    attempts: int
    status: str  # "answered" | "insufficient_evidence" | "responded" | "blocked" | "error"
    validated: bool
    # operator.add reducer: each node returns only its new trace lines and LangGraph appends them.
    trace: Annotated[list[str], operator.add]
