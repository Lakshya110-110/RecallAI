import json
from pathlib import Path
import tempfile
import uuid

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app import document_processing, main
from app.document_processing import (
    KeywordExtractionError,
    _parse_and_validate_five_keywords,
    get_nvidia_config,
    sanitize_nemotron_json_text,
)
from app.main import (
    C4AssessmentResult,
    Lesson,
    Submission,
    User,
    build_db_engine,
    get_cors_origins,
    normalize_database_url,
)
from migrate_sqlite_to_postgres import read_sqlite_snapshot, write_snapshot_to_target

# Preserve original functions before conftest autouse fixture monkeypatches them
_REAL_CALL_NEMOTRON_EXTRACTION = document_processing.call_nemotron_keyword_extraction
_REAL_NVIDIA_C4_ASSESS = main.nvidia_c4_assess

client = TestClient(main.app)


def test_postgres_url_normalization_and_engine_creation():
    """Verify postgres:// and postgresql:// (including Neon pooled URLs) normalize to postgresql+psycopg:// and create valid dialect engines."""
    assert normalize_database_url("postgres://user:pass@localhost:5432/recallai") == (
        "postgresql+psycopg://user:pass@localhost:5432/recallai"
    )
    assert normalize_database_url("postgresql://user:pass@localhost:5432/recallai") == (
        "postgresql+psycopg://user:pass@localhost:5432/recallai"
    )
    neon_pooled = "postgresql://neondb_owner:secret@ep-cool-darkness-123456-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
    assert normalize_database_url(neon_pooled) == (
        "postgresql+psycopg://neondb_owner:secret@ep-cool-darkness-123456-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
    )
    assert normalize_database_url("sqlite:///./recallai.db") == "sqlite:///./recallai.db"

    pg_engine = build_db_engine(neon_pooled)
    try:
        assert pg_engine.dialect.name == "postgresql"
        assert pg_engine.dialect.driver == "psycopg"
        assert pg_engine.pool._pre_ping is True
    finally:
        pg_engine.dispose()

    sqlite_engine = build_db_engine("sqlite:///:memory:")
    try:
        assert sqlite_engine.dialect.name == "sqlite"
    finally:
        sqlite_engine.dispose()


def test_cors_origins_env_driven_and_preserves_localhost(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://recallai-demo.vercel.app, https://custom.college.edu/")
    monkeypatch.setenv("FRONTEND_URL", "https://preview-recallai.vercel.app/")
    origins = get_cors_origins()
    assert "http://localhost:5173" in origins
    assert "http://127.0.0.1:5173" in origins
    assert "https://recallai-demo.vercel.app" in origins
    assert "https://custom.college.edu" in origins
    assert "https://preview-recallai.vercel.app" in origins


def test_health_endpoint_reports_dialect_and_nvidia_status(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-valid-test-token")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["service"] == "RecallAI"
    assert body["subject"] == "Business Forecasting"
    assert body["database_dialect"] == "sqlite"
    assert body["nvidia_configured"] is True
    assert body["nvidia_model"] == "nvidia/nemotron-3-ultra-550b-a55b"


def test_nvidia_config_validation_rejects_missing_or_placeholder(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "your-nvidia-api-key-here")
    monkeypatch.setenv("NVIDIA_MODEL", "")
    with pytest.raises(KeywordExtractionError) as exc_info:
        get_nvidia_config(require=True)
    assert "NVIDIA_API_KEY" in str(exc_info.value)
    assert "NVIDIA_MODEL" in str(exc_info.value)


def test_sanitize_nemotron_json_handles_fences_and_reasoning_blocks():
    # 1. <think> block + markdown ```json fence
    raw_with_think_and_fence = (
        "<think>\nLet me analyze the Business Forecasting document carefully.\n"
        "I should extract 5 concepts.\n</think>\n"
        "```json\n"
        '{"keywords": ["Seasonality index", "Trend projection", "Moving average", "Forecast error", "Causal modeling"]}\n'
        "```"
    )
    kws = _parse_and_validate_five_keywords(raw_with_think_and_fence)
    assert len(kws) == 5
    assert kws[0]["raw_keyword"] == "Seasonality index"

    # 2. <reasoning> block + surrounding explanatory text
    raw_with_prose = (
        "<reasoning>Evaluating student reflection against pool.</reasoning>\n"
        "Here is the JSON assessment result:\n"
        '{"concept_results": [{"concept": "Trend projection", "status": "recalled", "evidence": "", "explanation": "ok"}], '
        '"feedback": "Well done."}\n'
        "Hope this helps!"
    )
    cleaned = sanitize_nemotron_json_text(raw_with_prose)
    parsed = main.NvidiaC4MatchResponse.model_validate_json(cleaned)
    assert len(parsed.concept_results) == 1
    assert parsed.feedback == "Well done."


def test_nvidia_timeout_and_bounded_retry_recovery(monkeypatch):
    """Verify bounded retries recover from transient timeout/API errors and fail cleanly when exhausted."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    monkeypatch.setenv("NVIDIA_MAX_RETRIES", "2")
    monkeypatch.setenv("NVIDIA_TIMEOUT_SECONDS", "15.0")

    # Avoid real sleep delay in unit test
    monkeypatch.setattr("time.sleep", lambda s: None)

    attempts = {"count": 0}

    class FakeChoice:
        def __init__(self, content: str):
            self.message = type("Msg", (), {"content": content, "reasoning_content": None})()

    class FakeResponse:
        def __init__(self, content: str):
            self.choices = [FakeChoice(content)]

    class FakeChatCompletions:
        def create(self, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise TimeoutError("Simulated transient NVIDIA API timeout on attempt 1")
            return FakeResponse(
                "<think>Reasoning about concepts</think>\n```json\n"
                '{"keywords": ["Autocorrelation", "Stationarity", "Exponential smoothing", "Prediction intervals", "Holdout validation"]}\n```'
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 15.0
            self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()

    monkeypatch.setattr(document_processing, "call_nemotron_keyword_extraction", _REAL_CALL_NEMOTRON_EXTRACTION)
    monkeypatch.setattr(document_processing, "OpenAI", FakeOpenAI)
    extracted = document_processing.extract_five_keywords_with_retries(
        "Document text about autocorrelation, stationarity, smoothing, intervals, and validation.",
        max_attempts=1,
    )
    assert attempts["count"] == 2
    assert len(extracted) == 5


def test_reversible_migration_and_restart_persistence():
    """
    Verify that read_sqlite_snapshot and write_snapshot_to_target:
    - Read source SQLite in read-only mode without modifying it.
    - Migrate all users, lessons, contributor assignments, documents, keywords, pools, submissions, and C4 results.
    - Preserve exact IDs and relationships across engine disposal/restart.
    """
    project_root = Path(__file__).resolve().parents[2]
    orig_db = project_root / "recallai.db"
    if not orig_db.exists():
        pytest.skip("recallai.db not present")

    orig_mtime = orig_db.stat().st_mtime_ns
    orig_size = orig_db.stat().st_size

    snapshot = read_sqlite_snapshot(orig_db)
    assert len(snapshot["users"]) >= 2
    assert len(snapshot["lessons"]) >= 1

    with tempfile.TemporaryDirectory() as tmpdir:
        migrated_db_path = Path(tmpdir) / "migrated_target.db"
        target_url = f"sqlite:///{migrated_db_path}"

        counts = write_snapshot_to_target(snapshot, target_url, dry_run=False)
        assert counts["users"] == len(snapshot["users"])
        assert counts["lessons"] == len(snapshot["lessons"])
        assert counts["submissions"] == len(snapshot["submissions"])

        # Simulate backend restart by creating a brand new engine & session on the migrated DB
        restarted_engine = build_db_engine(target_url)
        try:
            RestartSession = sessionmaker(bind=restarted_engine, expire_on_commit=False)
            with RestartSession() as session:
                migrated_users = session.scalars(select(User).order_by(User.id.asc())).all()
                assert [u.id for u in migrated_users] == [u["id"] for u in snapshot["users"]]
                assert [u.email for u in migrated_users] == [u["email"] for u in snapshot["users"]]

                migrated_lessons = session.scalars(select(Lesson).order_by(Lesson.id.asc())).all()
                assert [l.id for l in migrated_lessons] == [l["id"] for l in snapshot["lessons"]]
        finally:
            restarted_engine.dispose()

    # Verify original recallai.db was untouched
    assert orig_db.stat().st_mtime_ns == orig_mtime
    assert orig_db.stat().st_size == orig_size
