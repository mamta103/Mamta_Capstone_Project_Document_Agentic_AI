import dataclasses
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app.core.config as settings_module
from app.backend.rag_service import IngestReport, RagAnswer
from app.core.chroma_store import SearchResult
from app.core.retriever import MAX_QUESTION_CHARS

APP = str(Path(__file__).resolve().parent.parent / "app" / "frontend" / "streamlit_app.py")

OTHER = SearchResult(text="The default port is 8080.", metadata={"source": "guide.txt"}, score=0.3)
SOURCE = SearchResult(text="Users authenticate via POST /api/v1/session.", metadata={"source": "guide.txt", "page": 2}, score=0.56)
ANSWERED = RagAnswer(
    status="answered", answer="Users call POST /api/v1/session [2].", sources=[SOURCE], retrieved=[OTHER, SOURCE],
    cited=True, validated=True, attempts=1,
    trace=["Planner: retrieve", "Retrieve (attempt 1): 1 chunk(s)", "Decision: answer accepted (validated)"],
)


class FakeStore:
    def __init__(self, sources):
        self.sources = list(sources)
        self.reset_calls = 0

    def list_sources(self):
        return list(self.sources)

    def reset(self):
        self.reset_calls += 1
        self.sources = []


class FakeService:
    """Stands in for RagService so UI tests need no model, database, or LLM."""

    instance = None
    answer = None
    ask_error = None
    ingest_error = None
    ingest_errors = ()
    sources = ()

    def __init__(self):
        self.store = FakeStore(FakeService.sources)
        self.questions = []
        self.ingested = []
        FakeService.instance = self

    def ingest(self, files):
        if FakeService.ingest_error:
            raise FakeService.ingest_error
        names = [name for name, _ in files]
        self.ingested.extend(names)
        self.store.sources.extend(names)
        return IngestReport(chunk_counts={name: 1 for name in names}, errors=list(FakeService.ingest_errors))

    def ask(self, question):
        self.questions.append(question)
        if FakeService.ask_error:
            raise FakeService.ask_error
        return FakeService.answer


@pytest.fixture
def run_app(monkeypatch):
    def run(answer=None, ask_error=None, sources=("guide.txt",), api_key="test-key", ingest_error=None, ingest_errors=()):
        st.cache_resource.clear()
        FakeService.answer, FakeService.ask_error, FakeService.sources = answer, ask_error, sources
        FakeService.ingest_error, FakeService.ingest_errors = ingest_error, ingest_errors
        monkeypatch.setattr("app.backend.rag_service.RagService", FakeService)
        monkeypatch.setattr("app.core.config.settings", dataclasses.replace(settings_module.settings, groq_api_key=api_key))
        return AppTest.from_file(APP, default_timeout=30).run()
    yield run
    st.cache_resource.clear()


def ask(at, question):
    at.chat_input(key="question").set_value(question)
    return at.run()


def texts(elements):
    return [e.value for e in elements]


def labels(at):
    return [e.label for e in at.expander]


# ---------- First screen ----------

def test_first_screen_guides_new_user(run_app):
    at = run_app(sources=())
    assert not at.exception
    assert texts(at.title) == ["Document AI Assistant"]
    assert labels(at) == ["Your documents (0)"]
    assert at.expander[0].proto.expanded
    assert any("Add a document above" in m for m in texts(at.markdown))
    question_box = at.chat_input(key="question")
    assert question_box.disabled
    assert question_box.placeholder == "Add a document first"


def test_documents_collapse_once_added(run_app):
    at = run_app(sources=("a.pdf", "b.csv"))
    assert labels(at) == ["Your documents (2)"]
    assert not at.expander[0].proto.expanded
    assert {"- a.pdf", "- b.csv"} <= set(texts(at.markdown))
    assert not at.chat_input(key="question").disabled


def test_technical_panels_are_gone(run_app):
    at = run_app()
    assert not at.sidebar.children
    assert not at.metric
    assert not {"Retrieval debug", "Activity log"} & set(labels(at))


def test_question_input_has_length_limit(run_app):
    assert run_app().chat_input(key="question").max_chars == MAX_QUESTION_CHARS


def test_missing_api_key_warning_shown(run_app):
    assert any("GROQ_API_KEY is missing" in w for w in texts(run_app(api_key="").warning))


def test_no_key_warning_when_key_present(run_app):
    assert not any("GROQ_API_KEY" in w for w in texts(run_app().warning))


# ---------- Adding and removing documents ----------

def test_sample_button_adds_real_sample_files(run_app):
    at = run_app(sources=())
    at.button(key="use_samples").click().run()
    assert not at.exception
    assert sorted(FakeService.instance.ingested) == [
        "sample_api_endpoints.csv", "sample_architecture.pdf", "sample_config_reference.xlsx", "sample_install_guide.txt",
    ]
    assert labels(at) == ["Your documents (4)"]
    assert any("Ready to answer questions" in s for s in texts(at.success))
    assert not at.chat_input(key="question").disabled


def test_rejected_file_message_is_shown_once(run_app):
    at = run_app(sources=(), ingest_errors=("bad.pdf: File content is not a valid PDF.",))
    at.button(key="use_samples").click().run()
    assert "bad.pdf: File content is not a valid PDF." in texts(at.error)
    at.run()
    assert not at.error


def test_indexing_failure_is_friendly(run_app):
    at = run_app(sources=(), ingest_error=RuntimeError("secret internal detail"))
    at.button(key="use_samples").click().run()
    assert not at.exception
    assert any("could not be read" in e for e in texts(at.error))
    assert not any("secret internal detail" in e for e in texts(at.error))


def test_remove_all_documents_clears_store_and_chat(run_app):
    at = ask(run_app(answer=ANSWERED, sources=("a.pdf",)), "How do users log in?")
    at.button(key="remove_documents").click().run()
    assert FakeService.instance.store.reset_calls == 1
    assert labels(at) == ["Your documents (0)"]
    assert not any("POST /api/v1/session" in m for m in texts(at.markdown))
    assert at.chat_input(key="question").disabled


# ---------- Answers ----------

def test_answer_shows_sources_with_citation_numbers_and_steps(run_app):
    at = ask(run_app(answer=ANSWERED), "How do users log in?")
    assert not at.exception
    assert FakeService.instance.questions == ["How do users log in?"]
    shown = texts(at.markdown)
    assert "How do users log in?" in shown
    assert ANSWERED.answer in shown
    assert "**[2] guide.txt, page 2**" in shown
    assert SOURCE.text in texts(at.text)
    assert {"Sources (1)", "How I got this answer"} <= set(labels(at))
    assert "3. Decision: answer accepted (validated)" in shown
    assert not at.warning


def test_chat_keeps_earlier_questions_until_new_chat(run_app):
    at = ask(run_app(answer=ANSWERED), "First question?")
    at = ask(at, "Second question?")
    assert {"First question?", "Second question?"} <= set(texts(at.markdown))
    at.button(key="new_chat").click().run()
    assert not {"First question?", "Second question?"} & set(texts(at.markdown))


def test_answer_is_not_asked_again_on_rerun(run_app):
    at = ask(run_app(answer=ANSWERED), "How do users log in?")
    at.run()
    assert FakeService.instance.questions == ["How do users log in?"]


def test_unvalidated_answer_shows_warning(run_app):
    at = ask(run_app(answer=dataclasses.replace(ANSWERED, validated=False)), "q")
    assert any("couldn't double-check" in w for w in texts(at.warning))


def test_responded_shows_reply(run_app):
    reply = RagAnswer(status="responded", answer="Hello! Ask me about your documents.", trace=["Planner: respond"])
    at = ask(run_app(answer=reply), "Hello")
    assert reply.answer in texts(at.markdown)
    assert not at.error and not at.info


def test_insufficient_evidence_shows_info(run_app):
    answer = RagAnswer(status="insufficient_evidence", answer="The uploaded documents do not contain enough information.")
    at = ask(run_app(answer=answer), "What is the capital of France?")
    assert answer.answer in texts(at.info)


def test_blocked_answer_is_shown_as_warning(run_app):
    answer = RagAnswer(status="blocked", answer="Blocked by policy.", trace=["Safety (input): BLOCKED - weapons"])
    at = ask(run_app(answer=answer), "bad question")
    assert not at.exception
    assert "Blocked by policy." in texts(at.warning)
    assert not at.error


def test_error_status_shows_error(run_app):
    answer = RagAnswer(status="error", answer="LLM rate limit reached. Please wait a moment and try again.")
    at = ask(run_app(answer=answer), "q")
    assert answer.answer in texts(at.error)


def test_unexpected_exception_is_friendly(run_app):
    at = ask(run_app(ask_error=RuntimeError("secret internal detail")), "q")
    assert not at.exception
    errors = texts(at.error)
    assert any("Something went wrong" in e for e in errors)
    assert not any("secret internal detail" in e for e in errors)
