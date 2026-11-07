import logging
import sys
from pathlib import Path

import streamlit as st

# `streamlit run` puts only this file's folder on the import path, so add the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Hosted deployments have no .env file. Loading secrets exports root-level keys to environment
# variables, so this must run before app.core.config reads them.
st.secrets.load_if_toml_exists()

# Only light imports here. The AI libraries load inside get_service(), so the page draws first
# and shows a loading spinner instead of a blank screen.
from app.core.config import MAX_QUESTION_CHARS, settings  # noqa: E402

logger = logging.getLogger(__name__)

SAMPLE_FOLDER = PROJECT_ROOT / "data" / "sample_documents"

st.set_page_config(page_title="Document AI Assistant", layout="centered")

# Streamlit's default top space and title size use up too much of a phone screen.
st.markdown(
    """<style>
    .block-container { padding-top: 2rem; }
    h1 { font-size: 1.8rem !important; padding-bottom: 0 !important; }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Getting things ready...")
def get_service():
    from app.backend.rag_service import RagService  # takes ~20s: torch, embeddings, ChromaDB

    return RagService()


def add_documents(files: list[tuple[str, bytes]]) -> None:
    with st.spinner("Reading your documents..."):
        try:
            report = service.ingest(files)
        except Exception:
            logger.exception("Indexing failed")
            st.session_state["notices"].append(("error", "Sorry, those documents could not be read. Please try again."))
            return
    if report.chunk_counts:
        names = ", ".join(report.chunk_counts)
        st.session_state["notices"].append(("success", f"Ready to answer questions about: {names}"))
    st.session_state["notices"].extend(("error", error) for error in report.errors)


def render_sources(result) -> None:
    if not result.sources:
        return
    with st.expander(f"Sources ({len(result.sources)})"):
        for source in result.sources:
            # Same numbers as the [n] citations in the answer.
            number = result.retrieved.index(source) + 1
            st.markdown(f"**[{number}] {source.source_label}**")
            st.text(source.text)


def render_answer(result) -> None:
    if result.status == "answered":
        if not result.validated:
            st.warning("I couldn't double-check this answer against your documents, so please verify it.")
        st.markdown(result.answer)
        render_sources(result)
    elif result.status == "responded":
        st.markdown(result.answer)
    elif result.status == "insufficient_evidence":
        st.info(result.answer)
    elif result.status == "blocked":
        st.warning(result.answer)
    else:
        st.error(result.answer)

    if result.trace:
        with st.expander("How I got this answer"):
            for number, step in enumerate(result.trace, start=1):
                st.markdown(f"{number}. {step}")


service = get_service()
st.session_state.setdefault("chat", [])
st.session_state.setdefault("notices", [])
st.session_state.setdefault("indexed_ids", set())
st.session_state.setdefault("upload_round", 0)  # A new widget key is the only way to empty the uploader.

st.title("Document AI Assistant")
st.caption("Add your documents, then ask anything about them.")

if not settings.groq_api_key:
    st.warning("Answers are switched off because GROQ_API_KEY is missing. Add it to your .env file, then restart the app.")

# New uploads are read before the page is drawn, so the document list below is already up to date.
uploader_key = f"uploader_{st.session_state['upload_round']}"
new_uploads = [f for f in st.session_state.get(uploader_key) or [] if f.file_id not in st.session_state["indexed_ids"]]
if new_uploads:
    add_documents([(f.name, f.getvalue()) for f in new_uploads])
    # Failed files are included so that the same error doesn't repeat on every interaction.
    st.session_state["indexed_ids"].update(f.file_id for f in new_uploads)

sources = service.store.list_sources()

with st.expander(f"Your documents ({len(sources)})", expanded=not sources):
    st.file_uploader(
        f"Add up to {settings.max_files} files",
        type=["pdf", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
        key=uploader_key,
    )
    if st.button("Try with sample files", key="use_samples", use_container_width=True):
        add_documents([(p.name, p.read_bytes()) for p in sorted(SAMPLE_FOLDER.glob("sample_*"))])
        st.rerun()
    for source in sources:
        st.markdown(f"- {source}")
    if sources and st.button("Remove all documents", key="remove_documents", use_container_width=True):
        service.store.reset()
        st.session_state["chat"] = []
        st.session_state["upload_round"] += 1
        st.rerun()

for kind, message in st.session_state["notices"]:
    (st.success if kind == "success" else st.error)(message)
st.session_state["notices"] = []

if not st.session_state["chat"]:
    with st.chat_message("assistant"):
        if sources:
            st.markdown("Ask me a question about your documents, and I'll show where the answer came from. "
                        "I answer each question on its own, so please include all the details you need.")
        else:
            st.markdown("Hi! Add a document above, or try the sample files, to get started.")

for question, result in st.session_state["chat"]:
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        render_answer(result)

if st.session_state["chat"] and st.button("Start a new chat", key="new_chat", use_container_width=True):
    st.session_state["chat"] = []
    st.rerun()

question = st.chat_input(
    "Ask about your documents" if sources else "Add a document first",
    max_chars=MAX_QUESTION_CHARS,
    disabled=not sources,
    key="question",
)

if question:
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Looking through your documents..."):
                result = service.ask(question)
        except Exception:
            logger.exception("Question answering failed")
            st.error("Something went wrong while answering. Please try again.")
        else:
            st.session_state["chat"].append((question, result))
            # Rerun so the new answer is drawn with the chat above, and the "new chat" button appears.
            st.rerun()
