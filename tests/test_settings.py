import pytest

from app.core.config import load_settings

ENV_KEYS = [
    "LLM_PROVIDER", "GROQ_API_KEY", "LLM_MODEL", "MODERATION_MODEL", "EMBEDDING_MODEL", "CHROMA_DIR",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "MAX_FILE_MB", "MAX_FILES", "TOP_K", "MIN_SCORE",
]


@pytest.fixture
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_defaults(clean_env):
    s = load_settings()
    assert s.llm_provider == "groq"
    assert s.llm_model == "openai/gpt-oss-120b"
    assert s.moderation_model == "openai/gpt-oss-20b"
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.chroma_dir == "./vectordb"
    assert s.chunk_size == 500
    assert s.chunk_overlap == 80
    assert s.max_file_mb == 20
    assert s.max_files == 10
    assert s.top_k == 5
    assert s.min_score == 0.2
    assert s.groq_api_key == ""


def test_placeholder_api_key_counts_as_missing(clean_env):
    clean_env.setenv("GROQ_API_KEY", "your-groq-api-key-here")
    assert load_settings().groq_api_key == ""


def test_env_override(clean_env):
    clean_env.setenv("CHUNK_SIZE", "600")
    clean_env.setenv("MAX_FILES", "3")
    s = load_settings()
    assert s.chunk_size == 600
    assert s.max_files == 3


def test_non_integer_rejected(clean_env):
    clean_env.setenv("CHUNK_SIZE", "abc")
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        load_settings()


def test_non_positive_rejected(clean_env):
    clean_env.setenv("MAX_FILE_MB", "0")
    with pytest.raises(ValueError, match="MAX_FILE_MB"):
        load_settings()


def test_overlap_must_be_smaller_than_chunk(clean_env):
    clean_env.setenv("CHUNK_SIZE", "100")
    clean_env.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        load_settings()


@pytest.mark.parametrize("raw", ["abc", "1.5", "-0.1"])
def test_invalid_min_score_rejected(clean_env, raw):
    clean_env.setenv("MIN_SCORE", raw)
    with pytest.raises(ValueError, match="MIN_SCORE"):
        load_settings()
