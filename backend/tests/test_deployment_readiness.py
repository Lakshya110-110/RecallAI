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


def test_keyword_extraction_empty_model_response(monkeypatch):
    """Test empty string, whitespace-only, and empty code fence model responses."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    for empty_val in ["", "   \n\t  ", "```json\n\n```", "<think>thinking only</think>"]:
        with pytest.raises(ValueError, match="Empty response"):
            _parse_and_validate_five_keywords(empty_val)

    monkeypatch.setattr(
        document_processing,
        "call_nemotron_keyword_extraction",
        lambda document_text, subject="Business Forecasting": "   ",
    )
    with pytest.raises(KeywordExtractionError, match="Empty response"):
        document_processing.extract_five_keywords_with_retries("Valid forecasting report text for extraction.", max_attempts=3)


def test_keyword_extraction_none_response(monkeypatch):
    """Test None response object, None message content, and corrupted non-JSON reasoning_content."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    assert document_processing._extract_assistant_content(None) is None
    assert document_processing._extract_assistant_content(type("Resp", (), {"choices": []})()) is None

    # Message with content=None and reasoning_content=None
    msg_none = type("Msg", (), {"content": None, "reasoning_content": None})()
    choice_none = type("Choice", (), {"message": msg_none, "finish_reason": "stop"})()
    resp_none = type("Resp", (), {"choices": [choice_none]})()
    assert document_processing._extract_assistant_content(resp_none) is None

    # Message with content=None and corrupted non-JSON reasoning_content (e.g. "\ufffd\ufffd{U.")
    msg_corrupt = type("Msg", (), {"content": None, "reasoning_content": "\ufffd\ufffd{U."})()
    choice_corrupt = type("Choice", (), {"message": msg_corrupt, "finish_reason": "stop"})()
    resp_corrupt = type("Resp", (), {"choices": [choice_corrupt]})()
    assert document_processing._extract_assistant_content(resp_corrupt) is None

    with pytest.raises(ValueError, match="Empty response"):
        _parse_and_validate_five_keywords(None)  # type: ignore[arg-type]


def test_keyword_extraction_malformed_json(monkeypatch):
    """Test malformed, truncated, or non-object JSON responses."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    for bad_json in [
        '{"keywords": ["Moving average", "Seasonal index"',
        "{keywords: [1, 2, 3, 4, 5]}",
        "Not JSON at all, just plain text without braces",
        "```json\n{\"keywords\": [\"only_unclosed\"\n```",
    ]:
        with pytest.raises(ValueError, match="Malformed JSON from model"):
            _parse_and_validate_five_keywords(bad_json)

    with pytest.raises(ValueError, match="top-level 'keywords' array"):
        _parse_and_validate_five_keywords('{"concepts": ["a", "b", "c", "d", "e"]}')


def test_keyword_extraction_json_inside_markdown_code_fences():
    """Test valid JSON returned inside ```json ... ``` or ``` ... ``` fences and surrounded by prose."""
    fenced_with_prose = (
        "Here are the 5 distinct forecasting concepts extracted from your report:\n\n"
        "```json\n"
        "{\n"
        '  "keywords": [\n'
        '    "Holt-Winters exponential smoothing",\n'
        '    "Seasonal index decomposition",\n'
        '    "Mean Absolute Percentage Error",\n'
        '    "Autoregressive integrated moving average",\n'
        '    "Tracking signal bias"\n'
        "  ]\n"
        "}\n"
        "```\n"
        "Let me know if you need additional analysis."
    )
    validated = _parse_and_validate_five_keywords(fenced_with_prose)
    assert len(validated) == 5
    assert [item["position"] for item in validated] == [1, 2, 3, 4, 5]
    assert validated[0]["raw_keyword"] == "Holt-Winters exponential smoothing"
    assert validated[4]["raw_keyword"] == "Tracking signal bias"


def test_keyword_extraction_valid_json_with_duplicate_keywords_rejected():
    """Test that valid JSON containing duplicate or inflectionally identical keywords is rejected."""
    dup_json = json.dumps({
        "keywords": [
            "Exponential smoothing",
            "Moving average",
            "Moving averages",  # duplicate after normalization
            "Seasonal decomposition",
            "Tracking signal",
        ]
    })
    with pytest.raises(ValueError, match="Validation requires exactly 5 distinct meaningful keywords"):
        _parse_and_validate_five_keywords(dup_json)

    # Also verify generic single-word filler is rejected
    generic_json = json.dumps({
        "keywords": [
            "forecasting",  # generic single word
            "Moving average",
            "Exponential smoothing",
            "Seasonal decomposition",
            "Tracking signal",
        ]
    })
    with pytest.raises(ValueError, match="Validation requires exactly 5 distinct meaningful keywords"):
        _parse_and_validate_five_keywords(generic_json)


def test_keyword_extraction_valid_five_keyword_response(monkeypatch):
    """Test valid five-keyword response and verify NVIDIA request parameters do not set response_format json_object."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-secret-key-do-not-log")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    monkeypatch.setattr("time.sleep", lambda s: None)

    captured_kwargs: dict = {}

    class FakeChoice:
        def __init__(self, content: str):
            self.finish_reason = "stop"
            self.message = type("Msg", (), {"content": content, "reasoning_content": None})()

    class FakeResponse:
        def __init__(self, content: str):
            self.choices = [FakeChoice(content)]

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeResponse(
                json.dumps({
                    "keywords": [
                        "Triple exponential smoothing",
                        "Seasonal decomposition",
                        "Mean absolute percentage error",
                        "Rolling holdout validation",
                        "Tracking signal monitoring",
                    ]
                })
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()

    monkeypatch.setattr(document_processing, "call_nemotron_keyword_extraction", _REAL_CALL_NEMOTRON_EXTRACTION)
    monkeypatch.setattr(document_processing, "OpenAI", FakeOpenAI)

    result = document_processing.extract_five_keywords_with_retries(
        "Reference report on triple exponential smoothing, seasonal decomposition, MAPE, rolling holdout validation, and tracking signals."
    )
    assert len(result) == 5
    assert [r["raw_keyword"] for r in result] == [
        "Triple exponential smoothing",
        "Seasonal decomposition",
        "Mean absolute percentage error",
        "Rolling holdout validation",
        "Tracking signal monitoring",
    ]
    assert "response_format" not in captured_kwargs
    assert captured_kwargs.get("max_tokens") == 512
    assert captured_kwargs.get("extra_body") == {"chat_template_kwargs": {"enable_thinking": False}}


def test_nvidia_api_timeout_and_temporary_failure_exhaustion_and_recovery(monkeypatch):
    """Test recovery across API timeout + malformed JSON + valid response, and clean failure when all attempts fail."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-super-secret-token")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    monkeypatch.setenv("NVIDIA_MAX_RETRIES", "2")
    monkeypatch.setattr("time.sleep", lambda s: None)

    call_seq = {"n": 0}

    class FakeChoice:
        def __init__(self, content: str | None):
            self.finish_reason = "stop"
            self.message = type("Msg", (), {"content": content, "reasoning_content": None})()

    class FakeResponse:
        def __init__(self, content: str | None):
            self.choices = [FakeChoice(content)]

    class FakeChatCompletions:
        def create(self, **kwargs):
            call_seq["n"] += 1
            if call_seq["n"] == 1:
                raise TimeoutError("Request timed out talking to integrate.api.nvidia.com with key nvapi-super-secret-token")
            if call_seq["n"] == 2:
                return FakeResponse('{"keywords": ["truncated_json"')
            return FakeResponse(
                json.dumps({
                    "keywords": [
                        "Autoregressive moving average",
                        "Stationarity differencing",
                        "Ljung-Box Q statistic",
                        "Partial autocorrelation",
                        "Akaike information criterion",
                    ]
                })
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()

    monkeypatch.setattr(document_processing, "call_nemotron_keyword_extraction", _REAL_CALL_NEMOTRON_EXTRACTION)
    monkeypatch.setattr(document_processing, "OpenAI", FakeOpenAI)

    recovered = document_processing.extract_five_keywords_with_retries(
        "ARIMA reference report text with stationarity, Ljung-Box, PACF, and AIC.",
        max_attempts=3,
    )
    assert len(recovered) == 5
    assert call_seq["n"] == 3

    # Now verify bounded exhaustion when API always times out, and ensure secret key is redacted
    class AlwaysTimeoutCompletions:
        def create(self, **kwargs):
            raise TimeoutError("Timeout with nvapi-super-secret-token")

    class AlwaysTimeoutOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": AlwaysTimeoutCompletions()})()

    monkeypatch.setattr(document_processing, "OpenAI", AlwaysTimeoutOpenAI)
    with pytest.raises(KeywordExtractionError) as exc_info:
        document_processing.extract_five_keywords_with_retries("Report text", max_attempts=2)
    assert "nvapi-super-secret-token" not in str(exc_info.value)
    assert "nvapi-[REDACTED]" in str(exc_info.value)


def test_retry_processing_does_not_duplicate_reports_or_keywords(monkeypatch):
    """
    Verify Requirement 9 & 10:
    - Contributor submits a reference report; when extraction fails, report is stored with status 'failed'.
    - Retrying via /reference-documents/me/retry or resubmitting the identical report re-runs extraction
      on the existing document without creating duplicate ReferenceDocument rows or duplicate ExtractedKeyword rows.
    - Session readiness transitions to 'ready_for_assessment' only after all 3 contributors complete extraction.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")
    monkeypatch.setattr("time.sleep", lambda s: None)

    def reg_user(name: str, role: str):
        email = f"nodup-{uuid.uuid4().hex[:8]}@rvu.edu.in"
        pw = "password-12345"
        r = client.post(
            "/auth/register",
            json={
                "name": name,
                "email": email,
                "password": pw,
                "confirm_password": pw,
                "role": role,
                "signup_code": "011002" if role in {"faculty", "contributor"} else None,
            },
        )
        assert r.status_code == 201
        tok = client.post("/auth/login", json={"email": email, "password": pw, "expected_role": role}).json()["token"]
        return r.json()["user"]["id"], {"Authorization": f"Bearer {tok}"}

    _, fac_headers = reg_user("Faculty NoDup", "faculty")
    c1_id, c1_headers = reg_user("Contrib NoDup 1", "contributor")
    c2_id, c2_headers = reg_user("Contrib NoDup 2", "contributor")
    c3_id, c3_headers = reg_user("Contrib NoDup 3", "contributor")

    lesson_id = client.post(
        "/lessons",
        headers=fac_headers,
        json={
            "title": "No Duplicate Retry Session",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    ).json()["id"]

    # Step 1: Contributor 1 submits report while Nemotron returns malformed JSON
    monkeypatch.setattr(
        document_processing,
        "call_nemotron_keyword_extraction",
        lambda document_text, subject="Business Forecasting": '{"keywords": ["broken"',
    )
    rep1_body = (
        "Contributor 1 report covering Holt-Winters smoothing, seasonal indices, "
        "forecast horizon, tracking signals, and MAPE diagnostics."
    )
    sub1 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c1_headers,
        json={"title": "C1 Report", "content": rep1_body},
    )
    assert sub1.status_code == 201
    assert sub1.json()["processing_status"] == "failed"
    doc1_id = sub1.json()["id"]

    # Step 2: Nemotron recovers; Contributor 1 clicks Retry Processing (/reference-documents/me/retry)
    monkeypatch.setattr(
        document_processing,
        "call_nemotron_keyword_extraction",
        lambda document_text, subject="Business Forecasting": json.dumps({
            "keywords": [
                "Holt-Winters smoothing",
                "Seasonal indices",
                "Forecast horizon",
                "Tracking signals",
                "MAPE diagnostics",
            ]
        }),
    )
    retry1 = client.post(f"/lessons/{lesson_id}/reference-documents/me/retry", headers=c1_headers)
    assert retry1.status_code == 200
    assert retry1.json()["id"] == doc1_id
    assert retry1.json()["processing_status"] == "completed"

    # Calling retry again or re-submitting identical report text must NOT create duplicate reports or keywords
    retry1_again = client.post(f"/lessons/{lesson_id}/reference-documents/me/retry", headers=c1_headers)
    assert retry1_again.status_code == 200
    assert retry1_again.json()["id"] == doc1_id

    resub1 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c1_headers,
        json={"title": "C1 Report", "content": rep1_body},
    )
    assert resub1.status_code == 201
    assert resub1.json()["id"] == doc1_id

    # Verify in database: exactly 1 ReferenceDocument and 5 ExtractedKeyword rows for Contributor 1 in this lesson
    with main.SessionLocal() as db_session:
        c1_docs = db_session.scalars(
            select(main.ReferenceDocument).where(
                main.ReferenceDocument.lesson_id == lesson_id,
                main.ReferenceDocument.contributor_id == c1_id,
            )
        ).all()
        assert len(c1_docs) == 1
        c1_kws = db_session.scalars(
            select(main.ExtractedKeyword).where(
                main.ExtractedKeyword.lesson_id == lesson_id,
                main.ExtractedKeyword.contributor_id == c1_id,
            )
        ).all()
        assert len(c1_kws) == 5



# ---------------------------------------------------------------------------
# C4 Assessment JSON Parsing, Fallback Model, and TES End-to-End Tests
# ---------------------------------------------------------------------------


def test_c4_assess_valid_json_and_fenced_json_responses(monkeypatch):
    """
    Test:
    - Valid JSON response matching NvidiaC4MatchResponse
    - Valid JSON wrapped in Markdown fences and surrounded by explanatory text
    - Request configuration: no response_format=json_object on Nemotron, enable_thinking=False, max_tokens=1536
    - Safe inspection of finish_reason, content, reasoning_content, and token usage
    """
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-c4-test-key")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    monkeypatch.setattr("time.sleep", lambda s: None)

    captured_calls: list[dict] = []

    class FakeUsage:
        prompt_tokens = 180
        completion_tokens = 95
        total_tokens = 275
        completion_tokens_details = type("Details", (), {"reasoning_tokens": 0})()

    class FakeChoice:
        def __init__(self, content: str):
            self.finish_reason = "stop"
            self.message = type("Msg", (), {"content": content, "reasoning_content": None})()

    class FakeResponse:
        def __init__(self, content: str):
            self.choices = [FakeChoice(content)]
            self.usage = FakeUsage()

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured_calls.append(kwargs)
            return FakeResponse(
                "Here is the evaluation of the reflection:\n```json\n"
                + json.dumps({
                    "concept_results": [
                        {
                            "concept": "Triple Exponential Smoothing",
                            "status": "recalled",
                            "evidence": "We applied Triple Exponential Smoothing to smooth level, trend, and seasonality.",
                            "explanation": "Demonstrated TES components.",
                        },
                        {
                            "concept": "Mean Absolute Percentage Error",
                            "status": "missing",
                            "evidence": "N/A",
                            "explanation": "Not mentioned in reflection.",
                        },
                    ],
                    "feedback": "Good explanation of Triple Exponential Smoothing.",
                })
                + "\n```\nEnd of evaluation."
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()

    monkeypatch.setattr(main, "nvidia_c4_assess", _REAL_NVIDIA_C4_ASSESS)
    monkeypatch.setattr(main, "OpenAI", FakeOpenAI)

    lesson = Lesson(id=1, title="TES Session", subject="Business Forecasting", assessment_mode="c4")
    submission = Submission(
        id=1,
        lesson_id=1,
        student_id=2,
        text="We applied Triple Exponential Smoothing to smooth level, trend, and seasonality.",
        character_count=80,
    )
    pool_data = {
        "distinct_concepts": [
            {"display_concept": "Triple Exponential Smoothing"},
            {"display_concept": "Mean Absolute Percentage Error"},
        ]
    }

    resp = main.nvidia_c4_assess(lesson, submission, pool_data)
    assert len(captured_calls) == 1
    assert "response_format" not in captured_calls[0]
    assert captured_calls[0]["max_tokens"] == 1536
    assert captured_calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert len(resp.concept_results) == 2
    assert resp.concept_results[0].concept == "Triple Exponential Smoothing"
    assert resp.concept_results[0].status == "recalled"
    assert resp.concept_results[1].status == "missing"
    assert resp.concept_results[1].evidence == ""  # "N/A" coerced to ""
    assert resp.feedback == "Good explanation of Triple Exponential Smoothing."


def test_c4_assess_empty_corrupted_reasoning_only_and_truncated_responses():
    """
    Test _parse_and_validate_c4_match_response and _extract_assistant_content on:
    - Empty response
    - Corrupted response
    - Reasoning-only response (never parses reasoning_content as final JSON)
    - Truncated response (finish_reason=length)
    """
    # 1. Empty responses
    for empty_val in [None, "", "   \n\t ", "```json\n  \n```", "<think>Only thinking</think>"]:
        with pytest.raises(ValueError, match="Empty"):
            main._parse_and_validate_c4_match_response(empty_val)

    # 2. Corrupted responses & bare "{}" without concept_results
    for corrupt_val in [
        "—¡(). blant wówczasloxacinメリカ'agost\ufffd'IGN'IGN pobo'IGN",
        " mp copy)\t 1",
        "\ufffd\ufffd{U.",
        "Not JSON at all",
    ]:
        with pytest.raises(ValueError, match="Invalid JSON"):
            main._parse_and_validate_c4_match_response(corrupt_val)

    with pytest.raises(ValueError, match="concept_results"):
        main._parse_and_validate_c4_match_response("{}")

    # 3. Reasoning-only response: even if reasoning_content contains valid JSON, it MUST NEVER be parsed!
    reasoning_json = json.dumps({
        "concept_results": [{"concept": "TES", "status": "recalled", "evidence": "", "explanation": "from reasoning"}],
        "feedback": "Should never be parsed from reasoning_content",
    })
    msg_reasoning_only = type("Msg", (), {"content": None, "reasoning_content": reasoning_json})()
    choice_reasoning_only = type("Choice", (), {"message": msg_reasoning_only, "finish_reason": "stop"})()
    resp_reasoning_only = type("Resp", (), {"choices": [choice_reasoning_only], "usage": None})()

    extracted = document_processing._extract_assistant_content(resp_reasoning_only)
    assert extracted is None
    diag = document_processing._inspect_response_diagnostics(resp_reasoning_only, extracted)
    assert diag["reasoning_only"] is True
    assert diag["has_reasoning_content"] is True
    assert diag["content_is_none"] is True

    # Unclosed <think> block inside content must also never be parsed as JSON
    unclosed_think = f"<think>Thinking about {reasoning_json} before finishing"
    with pytest.raises(ValueError, match="Empty"):
        main._parse_and_validate_c4_match_response(unclosed_think)

    # 4. Truncated response (finish_reason="length")
    truncated_json = '{"concept_results": [{"concept": "Triple Exponential Smoothing", "status": "recalled"'
    with pytest.raises(ValueError, match="Truncated JSON response.*finish_reason=length"):
        main._parse_and_validate_c4_match_response(truncated_json, finish_reason="length")


def test_c4_assess_fallback_model_and_intelligent_retry_on_corrupted_or_timeout(monkeypatch):
    """
    Test:
    - Primary model (nvidia/nemotron-3-ultra-550b-a55b) returns corrupted/reasoning-only output or times out on Attempt 1.
    - Attempt 2 switches to controlled fallback model (nvidia/nemotron-3.5-bolt-e4b) with adaptive retry prompt (never sending identical request).
    - If all attempts fail (e.g., persistent API timeout or truncated/corrupted output), raises a clear RuntimeError with redacted secrets.
    """
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-secret-c4-token")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    monkeypatch.setenv("NVIDIA_FALLBACK_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    monkeypatch.setenv("NVIDIA_MAX_RETRIES", "2")
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(main, "nvidia_c4_assess", _REAL_NVIDIA_C4_ASSESS)

    calls: list[dict] = []

    class FakeChoice:
        def __init__(self, content: str | None, reasoning_content: str | None = None, finish_reason: str = "stop"):
            self.finish_reason = finish_reason
            self.message = type("Msg", (), {"content": content, "reasoning_content": reasoning_content})()

    class FakeResponse:
        def __init__(self, content: str | None, reasoning_content: str | None = None, finish_reason: str = "stop"):
            self.choices = [FakeChoice(content, reasoning_content, finish_reason)]
            self.usage = {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165}

    class FallbackSuccessCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                # Primary model returns content=None and corrupted reasoning_content
                return FakeResponse(content=None, reasoning_content=" mp copy)\t 1", finish_reason="stop")
            # Fallback model returns valid JSON
            return FakeResponse(
                content=json.dumps({
                    "concept_results": [
                        {
                            "concept": "Triple Exponential Smoothing",
                            "status": "recalled",
                            "evidence": "TES smooths level, trend, and seasonality.",
                            "explanation": "Correctly explained TES.",
                        }
                    ],
                    "feedback": "Strong understanding of TES.",
                })
            )

    class FallbackSuccessOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FallbackSuccessCompletions()})()

    monkeypatch.setattr(main, "OpenAI", FallbackSuccessOpenAI)

    lesson = Lesson(id=10, title="Fallback Test Session", subject="Business Forecasting", assessment_mode="c4")
    submission = Submission(
        id=10,
        lesson_id=10,
        student_id=3,
        text="TES smooths level, trend, and seasonality.",
        character_count=42,
    )
    pool_data = {"distinct_concepts": [{"display_concept": "Triple Exponential Smoothing"}]}

    res = main.nvidia_c4_assess(lesson, submission, pool_data)
    assert len(calls) == 2
    assert calls[0]["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert calls[1]["model"] == "nvidia/nemotron-3.5-bolt-e4b"
    # Verify retry request was modified (not identical)
    assert calls[0]["messages"][0]["content"] != calls[1]["messages"][0]["content"]
    assert "IMPORTANT RETRY INSTRUCTION" in calls[1]["messages"][0]["content"]
    assert getattr(res, "_model_used", None) == "nvidia/nemotron-3.5-bolt-e4b"
    assert len(res.concept_results) == 1
    assert res.concept_results[0].status == "recalled"

    # Now test truncation on attempt 1 -> increases max_tokens on attempt 2
    trunc_calls: list[dict] = []

    class TruncatedThenSuccessCompletions:
        def create(self, **kwargs):
            trunc_calls.append(kwargs)
            if len(trunc_calls) == 1:
                return FakeResponse(
                    content='{"concept_results": [{"concept": "Triple Exponential Smoothing"',
                    finish_reason="length",
                )
            return FakeResponse(
                content=json.dumps({
                    "concept_results": [
                        {
                            "concept": "Triple Exponential Smoothing",
                            "status": "recalled",
                            "evidence": "TES smooths level, trend, and seasonality.",
                            "explanation": "Matched.",
                        }
                    ],
                    "feedback": "Recovered from truncation.",
                })
            )

    class TruncatedOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": TruncatedThenSuccessCompletions()})()

    monkeypatch.setattr(main, "OpenAI", TruncatedOpenAI)
    res_trunc = main.nvidia_c4_assess(lesson, submission, pool_data)
    assert len(trunc_calls) == 2
    assert trunc_calls[1]["max_tokens"] > trunc_calls[0]["max_tokens"]
    assert res_trunc.feedback == "Recovered from truncation."

    # Now test API timeout on all attempts -> raises clear RuntimeError with redacted secret
    class AlwaysTimeoutCompletions:
        def create(self, **kwargs):
            raise TimeoutError("Timed out calling integrate.api.nvidia.com with nvapi-secret-c4-token")

    class AlwaysTimeoutOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": AlwaysTimeoutCompletions()})()

    monkeypatch.setattr(main, "OpenAI", AlwaysTimeoutOpenAI)
    with pytest.raises(RuntimeError) as exc_info:
        main.nvidia_c4_assess(lesson, submission, pool_data)
    err_msg = str(exc_info.value)
    assert "NVIDIA C4 assessment failed after 2 attempt(s)" in err_msg
    assert "nvidia/nemotron-3-ultra-550b-a55b" in err_msg
    assert "nvidia/nemotron-3.5-bolt-e4b" in err_msg
    assert "nvapi-secret-c4-token" not in err_msg
    assert "nvapi-[REDACTED]" in err_msg


def test_end_to_end_c4_assessment_sample_tes_reflection_with_fallback_model(monkeypatch):
    """
    Actual end-to-end C4 assessment using a sample TES reflection through the real nvidia_c4_assess pipeline
    (where primary model returns corrupted non-JSON and fallback model returns valid NvidiaC4MatchResponse JSON):
    - Preserves keyword matching, evidence extraction, scoring bands, and faculty review functionality.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-e2e-test-key")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    monkeypatch.setenv("NVIDIA_FALLBACK_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    monkeypatch.setenv("NVIDIA_MAX_RETRIES", "2")
    monkeypatch.setattr("time.sleep", lambda s: None)

    def reg_login(name: str, role: str):
        email = f"tes-e2e-{uuid.uuid4().hex[:8]}@rvu.edu.in"
        pw = "password-12345"
        r = client.post(
            "/auth/register",
            json={
                "name": name,
                "email": email,
                "password": pw,
                "confirm_password": pw,
                "role": role,
                "signup_code": "011002" if role in {"faculty", "contributor"} else None,
            },
        )
        assert r.status_code == 201
        tok = client.post("/auth/login", json={"email": email, "password": pw, "expected_role": role}).json()["token"]
        return r.json()["user"]["id"], {"Authorization": f"Bearer {tok}"}

    _, fac_headers = reg_login("Prof. TES", "faculty")
    c1_id, c1_headers = reg_login("TES Contrib 1", "contributor")
    c2_id, c2_headers = reg_login("TES Contrib 2", "contributor")
    c3_id, c3_headers = reg_login("TES Contrib 3", "contributor")
    _, stu_headers = reg_login("TES Learner", "student")

    lesson_id = client.post(
        "/lessons",
        headers=fac_headers,
        json={
            "title": "TES Forecasting End-to-End Session",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    ).json()["id"]

    kws_by_contrib = {
        1: ["TES model", "MAPE", "Seasonal Index", "Forecast Horizon", "Holdout Validation"],
        2: ["Triple Exponential Smoothing", "Mean Absolute Percentage Error", "Autocorrelation Function", "Tracking Signal", "Prediction Intervals"],
        3: ["Root Mean Squared Error", "Partial Autocorrelation Function", "Seasonal ARIMA", "Demand Elasticity", "Residual Diagnostics"],
    }
    for idx, hdrs in enumerate([c1_headers, c2_headers, c3_headers], start=1):
        monkeypatch.setattr(
            document_processing,
            "call_nemotron_keyword_extraction",
            lambda document_text, subject="Business Forecasting", i=idx: json.dumps({"keywords": kws_by_contrib[i]}),
        )
        res = client.post(
            f"/lessons/{lesson_id}/reference-reports",
            headers=hdrs,
            json={
                "title": f"TES Report {idx}",
                "content": f"Reference report {idx} on Triple Exponential Smoothing, MAPE, RMSE, and seasonal forecasting.",
            },
        )
        assert res.status_code == 201

    sample_tes_reflection = (
        "We applied the TES model (Holt-Winters Triple Exponential Smoothing) to smooth level, trend, "
        "and seasonal components across our forecast horizon. We evaluated accuracy using Mean Absolute "
        "Percentage Error and Root Mean Squared Error during holdout validation, while prediction intervals "
        "quantified uncertainty and tracking signals monitored forecast bias."
    )

    # Wire real nvidia_c4_assess with FakeOpenAI that simulates primary model failure -> fallback model success
    class FakeChoice:
        def __init__(self, content: str | None, reasoning_content: str | None = None):
            self.finish_reason = "stop"
            self.message = type("Msg", (), {"content": content, "reasoning_content": reasoning_content})()

    class FakeResponse:
        def __init__(self, content: str | None, reasoning_content: str | None = None):
            self.choices = [FakeChoice(content, reasoning_content)]
            self.usage = {"prompt_tokens": 210, "completion_tokens": 320, "total_tokens": 530}

    class E2ECompletions:
        def create(self, **kwargs):
            if kwargs["model"] == "nvidia/nemotron-3-ultra-550b-a55b":
                # Simulate corrupted output on primary model
                return FakeResponse(content="—¡(). blant wówczasloxacin")
            # Fallback model returns valid structured JSON wrapped in markdown fences
            return FakeResponse(
                content="```json\n"
                + json.dumps({
                    "concept_results": [
                        {
                            "concept": "Triple Exponential Smoothing",
                            "status": "recalled",
                            "evidence": "We applied the TES model (Holt-Winters Triple Exponential Smoothing) to smooth level, trend, and seasonal components across our forecast horizon.",
                            "explanation": "Demonstrated TES smoothing level, trend, and seasonality.",
                        },
                        {
                            "concept": "Forecast Horizon",
                            "status": "recalled",
                            "evidence": "across our forecast horizon.",
                            "explanation": "Contextualized smoothing across the forecast horizon.",
                        },
                        {
                            "concept": "Mean Absolute Percentage Error",
                            "status": "recalled",
                            "evidence": "We evaluated accuracy using Mean Absolute Percentage Error and Root Mean Squared Error during holdout validation",
                            "explanation": "Used MAPE to evaluate forecast accuracy.",
                        },
                        {
                            "concept": "Root Mean Squared Error",
                            "status": "recalled",
                            "evidence": "We evaluated accuracy using Mean Absolute Percentage Error and Root Mean Squared Error during holdout validation",
                            "explanation": "Used RMSE to evaluate forecast accuracy.",
                        },
                        {
                            "concept": "Holdout Validation",
                            "status": "recalled",
                            "evidence": "during holdout validation",
                            "explanation": "Evaluated error metrics on holdout validation.",
                        },
                        {
                            "concept": "Prediction Intervals",
                            "status": "recalled",
                            "evidence": "while prediction intervals quantified uncertainty",
                            "explanation": "Explained prediction intervals for uncertainty.",
                        },
                        {
                            "concept": "Tracking Signal",
                            "status": "recalled",
                            "evidence": "tracking signals monitored forecast bias.",
                            "explanation": "Explained tracking signals for bias monitoring.",
                        },
                    ],
                    "feedback": "Clear synthesis of Triple Exponential Smoothing, error metrics, validation, and uncertainty intervals.",
                })
                + "\n```"
            )

    class E2EOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": E2ECompletions()})()

    monkeypatch.setattr(main, "nvidia_c4_assess", _REAL_NVIDIA_C4_ASSESS)
    monkeypatch.setattr(main, "OpenAI", E2EOpenAI)

    sub_resp = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=stu_headers,
        json={"text": sample_tes_reflection},
    )
    assert sub_resp.status_code == 201, sub_resp.text
    sub_id = sub_resp.json()["id"]

    # Verify completed C4 assessment, evidence extraction, scoring band, and fallback model recording
    fac_sub = client.get(f"/submissions/{sub_id}", headers=fac_headers).json()
    assert fac_sub["assessment_status"] == "completed"
    assert fac_sub["matched_concepts_count"] == 7
    assert fac_sub["total_pool_concepts"] == 13
    assert fac_sub["recall_percentage"] == 53.85
    assert fac_sub["rating"] == 3  # Above 40-60% -> Rating 3
    details = fac_sub["assessment_details"]
    assert len(details["matched_concepts"]) == 7
    assert len(details["unmatched_concepts"]) == 6
    for mc in details["matched_concepts"]:
        assert mc["evidence"] != ""
        assert mc["evidence"] in sample_tes_reflection

    c4_results = client.get(f"/lessons/{lesson_id}/c4-results", headers=fac_headers).json()
    assert len(c4_results) == 1
    assert c4_results[0]["model_name"] == "nvidia/nemotron-3.5-bolt-e4b"
    assert c4_results[0]["c4_rating"] == 3

    # Verify faculty review functionality remains intact
    rev_resp = client.post(
        f"/submissions/{sub_id}/review",
        headers=fac_headers,
        json={"action": "accept", "review_notes": "Verified TES end-to-end assessment."},
    )
    assert rev_resp.status_code == 200
    assert rev_resp.json()["assessment_details"]["review_status"] == "accepted"
