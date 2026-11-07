import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Lives here (not in retriever) so the UI can read it without importing the embedding model.
MAX_QUESTION_CHARS = 1000

# The value shipped in .env.example. Treated as "no key" so a fresh copy shows the missing-key
# warning and skips the live tests instead of failing them.
PLACEHOLDER_API_KEY = "your-groq-api-key-here"


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    groq_api_key: str
    llm_model: str
    moderation_model: str
    embedding_model: str
    chroma_dir: str
    chunk_size: int
    chunk_overlap: int
    max_file_mb: int
    max_files: int
    top_k: int
    min_score: float


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _score_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def _api_key_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    return "" if value == PLACEHOLDER_API_KEY else value


def load_settings() -> Settings:
    s = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "groq"),
        groq_api_key=_api_key_env("GROQ_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        moderation_model=os.getenv("MODERATION_MODEL", "openai/gpt-oss-20b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chroma_dir=os.getenv("CHROMA_DIR", "./vectordb"),
        chunk_size=_int_env("CHUNK_SIZE", 500),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 80),
        max_file_mb=_int_env("MAX_FILE_MB", 20),
        max_files=_int_env("MAX_FILES", 10),
        top_k=_int_env("TOP_K", 5),
        min_score=_score_env("MIN_SCORE", 0.2),
    )
    if s.chunk_overlap >= s.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    return s


settings = load_settings()
