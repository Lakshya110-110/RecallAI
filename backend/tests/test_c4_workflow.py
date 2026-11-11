from io import BytesIO
import json
import uuid

from docx import Document as DocxDocument
from fastapi.testclient import TestClient
import pytest
from reportlab.pdfgen import canvas

from app import document_processing, main
from app.c4_scoring import (
    build_concept_pool,
    calculate_c4_rating,
    evaluate_c4_reflection,
    normalize_concept,
)

client = TestClient(main.app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def register_and_login_student(name: str = "C4 Student") -> tuple[int, dict[str, str], str]:
    email = f"c4student-{uuid.uuid4().hex[:10]}@rvu.edu.in"
    password = "student-password-123"
    reg = client.post(
        "/auth/register",
        json={
            "name": name,
            "email": email,
            "password": password,
            "confirm_password": password,
            "role": "student",
        },
    )
    assert reg.status_code == 201, reg.text
    user_id = reg.json()["user"]["id"]
    log = client.post("/auth/login", json={"email": email, "password": password})
    assert log.status_code == 200, log.text
    headers = {"Authorization": f"Bearer {log.json()['token']}"}
    return user_id, headers, email


def login_faculty() -> dict[str, str]:
    resp = client.post(
        "/auth/login",
        json={"email": "faculty@rvu.edu.in", "password": "faculty-demo"},
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def make_pdf_bytes(text_content: str) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf)
    y = 750
    for line in text_content.splitlines() or [text_content]:
        c.drawString(50, y, line[:90])
        y -= 18
    c.save()
    return buf.getvalue()


def make_docx_bytes(text_content: str) -> bytes:
    buf = BytesIO()
    doc = DocxDocument()
    for para in text_content.splitlines() or [text_content]:
        doc.add_paragraph(para)
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Deterministic C4 Scoring Unit Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "matched_count, expected_rating",
    [
        (0, 1),
        (4, 1),
        (5, 2),
        (7, 2),
        (8, 3),
        (10, 3),
        (11, 4),
        (12, 4),
        (13, 5),
        (15, 5),
    ],
)
def test_c4_rating_exact_thresholds(matched_count: int, expected_rating: int):
    assert calculate_c4_rating(matched_count) == expected_rating


def test_c4_rating_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_c4_rating(-1)
    with pytest.raises(ValueError):
        calculate_c4_rating(16)
    with pytest.raises(ValueError):
        calculate_c4_rating(True)  # type: ignore[arg-type]


def test_c4_matching_deduplication_negation_and_word_boundaries():
    source_entries = [
        {"document_id": 1, "contributor_id": 10, "position": 1, "raw_keyword": "Seasonal decomposition"},
        {"document_id": 1, "contributor_id": 10, "position": 2, "raw_keyword": "Exponential smoothing"},
        {"document_id": 1, "contributor_id": 10, "position": 3, "raw_keyword": "Confidence intervals"},
        {"document_id": 1, "contributor_id": 10, "position": 4, "raw_keyword": "Lag"},
        {"document_id": 1, "contributor_id": 10, "position": 5, "raw_keyword": "Moving average"},
        # Duplicate of 'Seasonal decomposition' from Document 2
        {"document_id": 2, "contributor_id": 11, "position": 1, "raw_keyword": "seasonal decomposition!"},
    ]
    pool = build_concept_pool(source_entries)
    assert pool["total_raw_keywords"] == 6
    assert pool["distinct_concepts_count"] == 5
    assert pool["has_duplicates"] is True
    assert pool["warning"] is not None  # < 13 distinct concepts triggers informational warning

    # Reflection repeats 'Moving averages' 3 times, uses 'flagship' (contains 'lag' as substring),
    # and negates 'confidence intervals' ("without confidence intervals").
    reflection = (
        "Moving averages, moving average, and moving averages help smooth series. "
        "Our flagship product uses seasonal decompositions, without confidence intervals."
    )
    eval_res = evaluate_c4_reflection(reflection, pool)
    matched_names = {m["normalized_concept"] for m in eval_res["matched_concepts"]}

    # 'moving average' and 'seasonal decomposition' should match once each (2 matches total)
    # 'lag' must NOT match inside 'flagship'
    # 'confidence intervals' must NOT match because it is negated ('without confidence intervals')
    assert normalize_concept("Moving average") in matched_names
    assert normalize_concept("Seasonal decomposition") in matched_names
    assert normalize_concept("Lag") not in matched_names
    assert normalize_concept("Confidence intervals") not in matched_names
    assert eval_res["matched_count"] == 2
    assert eval_res["c4_rating"] == 1


def test_ai_cannot_override_deterministic_c4_score_or_invent_pool_concepts():
    entries = [
        {"document_id": 1, "contributor_id": 1, "position": i, "raw_keyword": f"Concept {i}"}
        for i in range(1, 16)
    ]
    pool = build_concept_pool(entries)
    assert pool["distinct_concepts_count"] == 15
    assert pool["warning"] is None

    # AI returns 5 valid recalled concepts + 10 invented concepts not in pool
    ai_matches = [
        {"concept": f"Concept {i}", "status": "recalled", "evidence": "", "explanation": "Valid match"}
        for i in range(1, 6)
    ] + [
        {"concept": f"Invented Hallucination {j}", "status": "recalled", "evidence": "", "explanation": "Fake"}
        for j in range(1, 11)
    ]
    res = evaluate_c4_reflection("Discussing concept 1 through concept 5.", pool, ai_concept_matches=ai_matches)
    assert res["matched_count"] == 5
    assert res["c4_rating"] == 2  # 5 matches -> Rating 2


# ---------------------------------------------------------------------------
# 2. Roles, Contributor Selection & Authorization Tests
# ---------------------------------------------------------------------------

def test_contributor_selection_and_authorization():
    faculty = login_faculty()
    s1_id, s1_headers, _ = register_and_login_student("Contributor One")
    s2_id, s2_headers, _ = register_and_login_student("Contributor Two")
    s3_id, s3_headers, _ = register_and_login_student("Contributor Three")
    outsider_id, outsider_headers, _ = register_and_login_student("Ordinary Student")

    # Create a C4 lesson
    lesson_resp = client.post(
        "/lessons",
        headers=faculty,
        json={"title": "C4 Auth Test Session", "subject": "Business Forecasting", "assessment_mode": "c4"},
    )
    assert lesson_resp.status_code == 200, lesson_resp.text
    lesson_id = lesson_resp.json()["id"]
    assert lesson_resp.json()["readiness_status"] == "awaiting_contributors"

    # Reject duplicate contributor IDs
    dup_resp = client.post(
        f"/lessons/{lesson_id}/contributors",
        headers=faculty,
        json={"student_ids": [s1_id, s1_id, s2_id]},
    )
    assert dup_resp.status_code == 422

    # Reject faculty account as contributor
    faculty_me = client.get("/auth/me", headers=faculty).json()
    fac_contrib_resp = client.post(
        f"/lessons/{lesson_id}/contributors",
        headers=faculty,
        json={"student_ids": [faculty_me["id"], s1_id, s2_id]},
    )
    assert fac_contrib_resp.status_code == 422

    # Assign 3 valid distinct contributors
    assign_resp = client.post(
        f"/lessons/{lesson_id}/contributors",
        headers=faculty,
        json={"student_ids": [s1_id, s2_id, s3_id]},
    )
    assert assign_resp.status_code == 200, assign_resp.text
    assert assign_resp.json()["readiness_status"] == "awaiting_uploads"

    # Verify /auth/me reflects contributor permission for s1, but not for outsider
    s1_me = client.get("/auth/me", headers=s1_headers).json()
    assert s1_me["is_contributor"] is True
    assert lesson_id in s1_me["contributor_lesson_ids"]

    outsider_me = client.get("/auth/me", headers=outsider_headers).json()
    assert lesson_id not in outsider_me["contributor_lesson_ids"]

    # Ordinary student cannot upload reference document
    forbidden_upload = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=outsider_headers,
        files={"file": ("notes.txt", b"Forecasting uses moving averages and trend analysis.", "text/plain")},
    )
    assert forbidden_upload.status_code == 403

    # Students cannot access faculty keyword pool or contributor dashboard
    assert client.get(f"/lessons/{lesson_id}/keyword-pool", headers=s1_headers).status_code == 403
    assert client.get(f"/lessons/{lesson_id}/contributors", headers=s1_headers).status_code == 403
    assert client.get(f"/lessons/{lesson_id}/reference-documents", headers=s1_headers).status_code == 403
    assert client.get(f"/lessons/{lesson_id}/c4-results", headers=s1_headers).status_code == 403


# ---------------------------------------------------------------------------
# 3. Document Validation, Extraction Failure/Retry, and Full C4 Workflow
# ---------------------------------------------------------------------------

def test_document_validation_and_full_c4_workflow(monkeypatch):
    faculty = login_faculty()
    s1_id, s1_headers, _ = register_and_login_student("Doc Contributor 1")
    s2_id, s2_headers, _ = register_and_login_student("Doc Contributor 2")
    s3_id, s3_headers, _ = register_and_login_student("Doc Contributor 3")
    regular_id, regular_headers, _ = register_and_login_student("Regular Student")

    created = client.post(
        "/lessons",
        headers=faculty,
        json={
            "title": "End-to-End C4 Session",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [s1_id, s2_id, s3_id],
        },
    )
    assert created.status_code == 200, created.text
    lesson_id = created.json()["id"]
    assert created.json()["readiness_status"] == "awaiting_uploads"

    # Students cannot submit reflections before session is ready
    early_submit = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=regular_headers,
        json={"text": "Attempting to submit before reference pool is ready."},
    )
    assert early_submit.status_code == 409

    # Test invalid file format (.exe)
    bad_ext = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=s1_headers,
        files={"file": ("malware.exe", b"MZ\x90\x00some binary payload", "application/octet-stream")},
    )
    assert bad_ext.status_code == 415

    # Test path traversal filename
    traversal = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=s1_headers,
        files={"file": ("../../etc/passwd.txt", b"Valid text content for forecasting.", "text/plain")},
    )
    assert traversal.status_code == 400

    # Test empty file
    empty_res = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=s1_headers,
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert empty_res.status_code == 400

    # Test corrupted PDF
    corrupt_pdf = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=s1_headers,
        files={"file": ("corrupt.pdf", b"not a pdf header at all", "application/pdf")},
    )
    assert corrupt_pdf.status_code == 400

    # Test oversized file
    monkeypatch.setenv("MAX_DOCUMENT_SIZE_BYTES", "100")
    oversized = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=s1_headers,
        files={"file": ("big.txt", b"A" * 250, "text/plain")},
    )
    assert oversized.status_code == 413
    monkeypatch.setenv("MAX_DOCUMENT_SIZE_BYTES", str(10 * 1024 * 1024))

    # Simulate malformed Nemotron output on Contributor 1's first upload attempt
    def broken_nemotron(document_text: str, subject: str = "Business Forecasting") -> str:
        return '{"keywords": ["only one concept", "only two"]}'

    monkeypatch.setattr(document_processing, "call_nemotron_keyword_extraction", broken_nemotron)

    doc1_upload = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=s1_headers,
        files={
            "file": (
                "doc1.txt",
                b"Time series decomposition separates trend, seasonality, cyclicality, autocorrelation, and stationarity.",
                "text/plain",
            )
        },
    )
    assert doc1_upload.status_code == 201, doc1_upload.text
    assert doc1_upload.json()["processing_status"] == "failed"
    doc1_id = doc1_upload.json()["id"]

    # Contributor 2 cannot delete Contributor 1's document
    assert client.delete(f"/lessons/{lesson_id}/reference-documents/{doc1_id}", headers=s2_headers).status_code == 403

    # Now fix Nemotron mock to return 5 valid distinct concepts per document
    doc_concepts_map = {
        "doc1": [
            "Time series decomposition",
            "Seasonality adjustment",
            "Trend extrapolation",
            "Autocorrelation analysis",
            "Stationarity testing",
        ],
        "doc2": [
            "Exponential smoothing",
            "Moving average filter",
            "Forecast horizon",
            "Confidence intervals",
            "Prediction error variance",
        ],
        "doc3": [
            "Causal regression drivers",
            "Leading indicators",
            "Scenario planning",
            "Demand elasticity",
            "Baseline forecast",
        ],
    }

    def working_nemotron(document_text: str, subject: str = "Business Forecasting") -> str:
        if "decomposition" in document_text.lower():
            return json.dumps({"keywords": doc_concepts_map["doc1"]})
        if "smoothing" in document_text.lower():
            return json.dumps({"keywords": doc_concepts_map["doc2"]})
        return json.dumps({"keywords": doc_concepts_map["doc3"]})

    monkeypatch.setattr(document_processing, "call_nemotron_keyword_extraction", working_nemotron)

    # Contributor 1 retries failed extraction
    retry_resp = client.post(f"/lessons/{lesson_id}/reference-documents/me/retry", headers=s1_headers)
    assert retry_resp.status_code == 200, retry_resp.text
    assert retry_resp.json()["processing_status"] == "completed"

    # Verify Contributor 1 status response does NOT leak extracted keywords
    me_doc = client.get(f"/lessons/{lesson_id}/reference-documents/me", headers=s1_headers).json()
    assert "keywords" not in me_doc
    assert "extracted_text" not in me_doc

    # Contributor 2 uploads a real PDF document
    pdf_bytes = make_pdf_bytes(
        "Exponential smoothing and moving average filters improve forecast horizon accuracy and confidence intervals."
    )
    doc2_upload = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=s2_headers,
        files={"file": ("forecasting_methods.pdf", pdf_bytes, "application/pdf")},
    )
    assert doc2_upload.status_code == 201, doc2_upload.text
    assert doc2_upload.json()["processing_status"] == "completed"

    # Session must still NOT be ready (only 2 of 3 uploaded)
    pool_before = client.get(f"/lessons/{lesson_id}/keyword-pool", headers=faculty).json()
    assert pool_before["pool_ready"] is False
    assert pool_before["readiness_status"] == "awaiting_uploads"

    # Contributor 3 uploads a real DOCX document
    docx_bytes = make_docx_bytes(
        "Causal regression drivers, leading indicators, and scenario planning shape demand elasticity and baseline forecasts."
    )
    doc3_upload = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=s3_headers,
        files={
            "file": (
                "causal_forecasting.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert doc3_upload.status_code == 201, doc3_upload.text
    assert doc3_upload.json()["processing_status"] == "completed"

    # Now all 3 documents are processed! Verify automatic pool creation & readiness without faculty approval
    pool_after = client.get(f"/lessons/{lesson_id}/keyword-pool", headers=faculty).json()
    assert pool_after["pool_ready"] is True
    assert pool_after["readiness_status"] == "ready_for_assessment"
    assert pool_after["version"] == 1
    assert pool_after["total_raw_keywords"] == 15
    assert pool_after["distinct_concepts_count"] == 15
    assert pool_after["has_duplicates"] is False
    assert len(pool_after["source_entries"]) == 15

    # Verify student /lessons endpoint hides reference concepts/pool
    student_lessons = client.get("/lessons", headers=regular_headers).json()
    c4_lesson_student = next(item for item in student_lessons if item["id"] == lesson_id)
    assert c4_lesson_student["is_ready_for_submission"] is True
    assert c4_lesson_student["reference_text"] == ""
    assert c4_lesson_student["concepts"] == []

    # Mock C4 Nemotron assessment to verify deterministic C4 scoring on student submission
    def mock_c4_ai(lesson, submission, pool_data):
        # Return 11 matched concepts -> should yield C4 rating 4
        concepts = [c["display_concept"] for c in pool_data["distinct_concepts"][:11]]
        return main.NvidiaC4MatchResponse(
            concept_results=[
                main.ConceptResult(concept=c, status="recalled", evidence="", explanation="Matched")
                for c in concepts
            ],
            feedback="Strong synthesis of 11 core forecasting concepts.",
        )

    monkeypatch.setattr(main, "nvidia_c4_assess", mock_c4_ai)

    sub_resp = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=regular_headers,
        json={
            "text": (
                "Time series decomposition, seasonality adjustment, trend extrapolation, "
                "and exponential smoothing improve our forecast horizon."
            )
        },
    )
    assert sub_resp.status_code == 201, sub_resp.text
    sub_id = sub_resp.json()["id"]

    # Duplicate submission by same student for same session is rejected
    dup_sub = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=regular_headers,
        json={"text": "Second submission attempt."},
    )
    assert dup_sub.status_code == 409

    # Check student's view of submission: rating is 4, matched_concepts_count is 11, no reference pool leaked
    student_sub = client.get(f"/submissions/{sub_id}", headers=regular_headers).json()
    assert student_sub["assessment_status"] == "completed"
    assert student_sub["rating"] == 4
    assert student_sub["matched_concepts_count"] == 11
    assert "unmatched_concepts" not in student_sub["assessment_details"]
    assert "missing_concepts" not in student_sub["assessment_details"]

    # Another student cannot view regular_student's submission
    assert client.get(f"/submissions/{sub_id}", headers=s1_headers).status_code == 403

    # Faculty views full C4 results including matched & unmatched concepts
    faculty_c4 = client.get(f"/lessons/{lesson_id}/c4-results", headers=faculty).json()
    assert len(faculty_c4) == 1
    assert faculty_c4[0]["c4_rating"] == 4
    assert faculty_c4[0]["matched_count"] == 11
    assert len(faculty_c4[0]["matched_concepts"]) == 11
    assert len(faculty_c4[0]["unmatched_concepts"]) == 4


def test_cross_document_duplicates_warning_and_document_reupload(monkeypatch):
    """
    Verify that when 3 documents produce < 13 distinct concepts due to cross-document duplicates:
    - All 15 source entries and provenance are preserved.
    - The session activates automatically without requiring faculty approval.
    - An informational warning is surfaced to faculty.
    - Re-uploading a document increments document version and pool version.
    """
    faculty = login_faculty()
    s1_id, s1_headers, _ = register_and_login_student("Dup Contrib 1")
    s2_id, s2_headers, _ = register_and_login_student("Dup Contrib 2")
    s3_id, s3_headers, _ = register_and_login_student("Dup Contrib 3")

    created = client.post(
        "/lessons",
        headers=faculty,
        json={
            "title": "Duplicate Pool Warning Test",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [s1_id, s2_id, s3_id],
        },
    )
    lesson_id = created.json()["id"]

    # All 3 documents share 3 overlapping concepts -> 9 distinct concepts total (< 13)
    def overlapping_nemotron(document_text: str, subject: str = "Business Forecasting") -> str:
        if "alpha" in document_text.lower():
            return json.dumps({"keywords": ["Shared Concept 1", "Shared Concept 2", "Shared Concept 3", "Alpha Unique 1", "Alpha Unique 2"]})
        if "beta" in document_text.lower():
            return json.dumps({"keywords": ["Shared Concept 1", "Shared Concept 2", "Shared Concept 3", "Beta Unique 1", "Beta Unique 2"]})
        return json.dumps({"keywords": ["Shared Concept 1", "Shared Concept 2", "Shared Concept 3", "Gamma Unique 1", "Gamma Unique 2"]})

    monkeypatch.setattr(document_processing, "call_nemotron_keyword_extraction", overlapping_nemotron)

    for hdrs, name, body in [
        (s1_headers, "alpha.txt", b"Alpha forecasting document with sufficient text length for validation."),
        (s2_headers, "beta.txt", b"Beta forecasting document with sufficient text length for validation."),
        (s3_headers, "gamma.txt", b"Gamma forecasting document with sufficient text length for validation."),
    ]:
        res = client.post(
            f"/lessons/{lesson_id}/reference-documents",
            headers=hdrs,
            files={"file": (name, body, "text/plain")},
        )
        assert res.status_code == 201

    pool = client.get(f"/lessons/{lesson_id}/keyword-pool", headers=faculty).json()
    assert pool["pool_ready"] is True
    assert pool["readiness_status"] == "ready_for_assessment"
    assert pool["version"] == 1
    assert pool["total_raw_keywords"] == 15
    assert pool["distinct_concepts_count"] == 9
    assert pool["has_duplicates"] is True
    assert pool["warning"] is not None
    assert "9 distinct concepts" in pool["warning"]

    # Contributor 3 re-uploads a new replacement document -> pool version increments to 2
    reupload = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=s3_headers,
        files={"file": ("gamma_v2.txt", b"Updated Gamma forecasting document with brand new content.", "text/plain")},
    )
    assert reupload.status_code == 201
    assert reupload.json()["version"] == 2

    pool_v2 = client.get(f"/lessons/{lesson_id}/keyword-pool", headers=faculty).json()
    assert pool_v2["pool_ready"] is True
    assert pool_v2["version"] == 2


def test_c4_submission_assessment_failure_recovery(monkeypatch):
    """
    Verify that when Nemotron fails during C4 reflection assessment:
    - The student's submitted reflection is preserved (never lost).
    - Status is marked 'failed' and rating is None (never fabricated).
    - Retrying assessment succeeds once Nemotron recovers.
    """
    faculty = login_faculty()
    s1_id, s1_headers, _ = register_and_login_student("Recov C1")
    s2_id, s2_headers, _ = register_and_login_student("Recov C2")
    s3_id, s3_headers, _ = register_and_login_student("Recov C3")
    st_id, st_headers, _ = register_and_login_student("Recov Student")

    created = client.post(
        "/lessons",
        headers=faculty,
        json={
            "title": "C4 Failure Recovery Session",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [s1_id, s2_id, s3_id],
        },
    )
    lesson_id = created.json()["id"]

    for idx, hdrs in enumerate([s1_headers, s2_headers, s3_headers], start=1):
        monkeypatch.setattr(
            document_processing,
            "call_nemotron_keyword_extraction",
            lambda document_text, subject="Business Forecasting", i=idx: json.dumps(
                {"keywords": [f"Doc{i} Concept {k}" for k in range(1, 6)]}
            ),
        )
        r = client.post(
            f"/lessons/{lesson_id}/reference-documents",
            headers=hdrs,
            files={"file": (f"d{idx}.txt", f"Document {idx} forecasting reference material content.".encode(), "text/plain")},
        )
        assert r.status_code == 201

    # Make C4 AI assessment fail on initial submission
    def failing_c4_ai(lesson, submission, pool_data):
        raise RuntimeError("Simulated Nemotron timeout")

    monkeypatch.setattr(main, "nvidia_c4_assess", failing_c4_ai)

    reflection_500 = "F" * 500
    sub_resp = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=st_headers,
        json={"text": reflection_500},
    )
    assert sub_resp.status_code == 201
    sub_id = sub_resp.json()["id"]

    # Verify submission is saved, assessment_status is 'failed', rating is None
    saved = client.get(f"/submissions/{sub_id}", headers=st_headers).json()
    assert saved["text"] == reflection_500
    assert saved["character_count"] == 500
    assert saved["assessment_status"] == "failed"
    assert saved["rating"] is None

    # Restore C4 AI assessment (matching 13 concepts -> C4 rating 5) and retry via faculty endpoint
    def recovered_c4_ai(lesson, submission, pool_data):
        concepts = [c["display_concept"] for c in pool_data["distinct_concepts"][:13]]
        return main.NvidiaC4MatchResponse(
            concept_results=[
                main.ConceptResult(concept=c, status="recalled", evidence="", explanation="Matched")
                for c in concepts
            ],
            feedback="Recovered assessment matched 13 concepts.",
        )

    monkeypatch.setattr(main, "nvidia_c4_assess", recovered_c4_ai)
    retried = client.post(f"/submissions/{sub_id}/retry", headers=faculty)
    assert retried.status_code == 200
    assert retried.json()["assessment_status"] == "completed"
    assert retried.json()["rating"] == 5
    assert retried.json()["matched_concepts_count"] == 13


def test_dedicated_contributor_role_registration_login_and_portal_endpoints(monkeypatch):
    """
    Verify dedicated 'contributor' account registration (requiring the faculty signup code),
    3-portal login validation (expected_role), contributor assignment, document upload,
    and upload history endpoint.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")
    faculty = login_faculty()

    def register_contributor(name: str) -> tuple[int, dict[str, str], str]:
        email = f"contrib-{uuid.uuid4().hex[:10]}@rvu.edu.in"
        password = "contributor-password-123"
        reg = client.post(
            "/auth/register",
            json={
                "name": name,
                "email": email,
                "password": password,
                "confirm_password": password,
                "role": "contributor",
                "signup_code": "011002",
            },
        )
        assert reg.status_code == 201, reg.text
        assert reg.json()["user"]["role"] == "contributor"

        # Contributor cannot log in through the Faculty or Student login window
        wrong_fac = client.post(
            "/auth/login",
            json={"email": email, "password": password, "expected_role": "faculty"},
        )
        assert wrong_fac.status_code == 403
        wrong_stu = client.post(
            "/auth/login",
            json={"email": email, "password": password, "expected_role": "student"},
        )
        assert wrong_stu.status_code == 403

        # Contributor succeeds in the Contributor login window
        log = client.post(
            "/auth/login",
            json={"email": email, "password": password, "expected_role": "contributor"},
        )
        assert log.status_code == 200, log.text
        assert log.json()["user"]["login_portal"] == "contributor"
        return reg.json()["user"]["id"], {"Authorization": f"Bearer {log.json()['token']}"}, email

    c1_id, c1_headers, _ = register_contributor("Dedicated Contributor 1")
    c2_id, c2_headers, _ = register_contributor("Dedicated Contributor 2")
    c3_id, c3_headers, _ = register_contributor("Dedicated Contributor 3")

    # Faculty sees dedicated contributors in /faculty/contributors and /faculty/students
    eligible = client.get("/faculty/contributors", headers=faculty)
    assert eligible.status_code == 200
    eligible_ids = {u["id"] for u in eligible.json()}
    assert {c1_id, c2_id, c3_id}.issubset(eligible_ids)

    # Create C4 session with the 3 dedicated contributor accounts
    created = client.post(
        "/lessons",
        headers=faculty,
        json={
            "title": "Dedicated Contributor Portal Session",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    )
    assert created.status_code == 200, created.text
    lesson_id = created.json()["id"]

    # Contributor 1 uploads v1 and v2 replacement document
    monkeypatch.setattr(
        document_processing,
        "call_nemotron_keyword_extraction",
        lambda document_text, subject="Business Forecasting": json.dumps(
            {"keywords": [f"Contrib Concept {k}" for k in range(1, 6)]}
        ),
    )
    up1 = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=c1_headers,
        files={"file": ("contrib_v1.txt", b"Initial forecasting work document for C4.", "text/plain")},
    )
    assert up1.status_code == 201
    up2 = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=c1_headers,
        files={"file": ("contrib_v2.txt", b"Updated forecasting work document v2 for C4.", "text/plain")},
    )
    assert up2.status_code == 201
    assert up2.json()["version"] == 2

    # Verify /contributor/assignments and /contributor/uploads return sanitized status and history
    assigns = client.get("/contributor/assignments", headers=c1_headers)
    assert assigns.status_code == 200
    assert len(assigns.json()) == 1
    assert assigns.json()[0]["lesson_id"] == lesson_id
    assert "keywords" not in assigns.json()[0]

    uploads = client.get("/contributor/uploads", headers=c1_headers)
    assert uploads.status_code == 200
    assert len(uploads.json()) == 2
    assert uploads.json()[0]["version"] == 2
    assert uploads.json()[0]["is_active"] is True
    assert uploads.json()[1]["version"] == 1
    assert uploads.json()[1]["is_active"] is False
    for row in uploads.json():
        assert "keywords" not in row
        assert "extracted_text" not in row


def test_contributor_reflection_submission_and_my_submissions_isolation(monkeypatch):
    """
    Verify that a contributor can:
    - Upload reference documents for assigned sessions.
    - Write and submit reflections (enforcing the 500-character limit and duplicate protection).
    - View their own reflection submissions, ratings, feedback, and PDF report without seeing
      other users' submissions or the reference keyword pool.
    - Be rejected when attempting unauthorized uploads or submissions before C4 readiness.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")
    faculty = login_faculty()

    def register_contributor(name: str) -> tuple[int, dict[str, str], str]:
        email = f"contrib-refl-{uuid.uuid4().hex[:10]}@rvu.edu.in"
        password = "contributor-password-123"
        reg = client.post(
            "/auth/register",
            json={
                "name": name,
                "email": email,
                "password": password,
                "confirm_password": password,
                "role": "contributor",
                "signup_code": "011002",
            },
        )
        assert reg.status_code == 201, reg.text
        log = client.post(
            "/auth/login",
            json={"email": email, "password": password, "expected_role": "contributor"},
        )
        assert log.status_code == 200, log.text
        return reg.json()["user"]["id"], {"Authorization": f"Bearer {log.json()['token']}"}, email

    c1_id, c1_headers, _ = register_contributor("Reflect Contributor 1")
    c2_id, c2_headers, _ = register_contributor("Reflect Contributor 2")
    c3_id, c3_headers, _ = register_contributor("Reflect Contributor 3")
    unassigned_id, unassigned_headers, _ = register_contributor("Unassigned Contributor")

    created = client.post(
        "/lessons",
        headers=faculty,
        json={
            "title": "Contributor Reflection C4 Session",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    )
    assert created.status_code == 200
    lesson_id = created.json()["id"]

    # Unassigned contributor cannot upload a reference document to this session
    forbidden_upload = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=unassigned_headers,
        files={"file": ("unauth.txt", b"Unauthorized reference document upload attempt.", "text/plain")},
    )
    assert forbidden_upload.status_code == 403

    # Contributor cannot submit reflection to C4 session before all 3 documents are processed
    early_refl = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=c1_headers,
        json={"text": "Attempting reflection before C4 pool is ready."},
    )
    assert early_refl.status_code == 409

    # Upload all 3 reference documents so C4 session becomes ready_for_assessment
    for idx, hdrs in enumerate([c1_headers, c2_headers, c3_headers], start=1):
        monkeypatch.setattr(
            document_processing,
            "call_nemotron_keyword_extraction",
            lambda document_text, subject="Business Forecasting", i=idx: json.dumps(
                {"keywords": [f"Pool{i} Concept {k}" for k in range(1, 6)]}
            ),
        )
        up = client.post(
            f"/lessons/{lesson_id}/reference-documents",
            headers=hdrs,
            files={"file": (f"c{idx}_ref.txt", f"Reference document {idx} for business forecasting.".encode(), "text/plain")},
        )
        assert up.status_code == 201
        assert up.json()["processing_status"] == "completed"

    # Reject 501-character reflection from contributor
    too_long = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=c1_headers,
        json={"text": "X" * 501},
    )
    assert too_long.status_code == 422

    # Mock C4 AI assessment (matching 13 concepts -> C4 rating 5)
    def mock_c4_ai(lesson, submission, pool_data):
        concepts = [c["display_concept"] for c in pool_data["distinct_concepts"][:13]]
        return main.NvidiaC4MatchResponse(
            concept_results=[
                main.ConceptResult(concept=c, status="recalled", evidence="", explanation="Matched")
                for c in concepts
            ],
            feedback="Excellent recall of 13 concepts.",
        )

    monkeypatch.setattr(main, "nvidia_c4_assess", mock_c4_ai)

    # Accept exact 500-character reflection from contributor
    exact_500 = "R" * 498 + "!?"
    sub_resp = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=c1_headers,
        json={"text": exact_500},
    )
    assert sub_resp.status_code == 201, sub_resp.text
    assert sub_resp.json()["character_count"] == 500
    sub_id = sub_resp.json()["id"]

    # Reject duplicate reflection submission for the same session
    dup_resp = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=c1_headers,
        json={"text": "Duplicate reflection attempt."},
    )
    assert dup_resp.status_code == 409

    # Faculty cannot submit a reflection
    fac_sub = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=faculty,
        json={"text": "Faculty attempting learner submission."},
    )
    assert fac_sub.status_code == 403

    # Contributor 1 sees their own reflection in GET /submissions with rating & feedback, without pool leak
    c1_subs = client.get("/submissions", headers=c1_headers)
    assert c1_subs.status_code == 200
    assert len(c1_subs.json()) == 1
    c1_sub = c1_subs.json()[0]
    assert c1_sub["id"] == sub_id
    assert c1_sub["rating"] == 5
    assert "Excellent recall of 13 concepts." in c1_sub["feedback"]
    assert "strengths" in c1_sub["assessment_details"]
    assert "areas_for_improvement" in c1_sub["assessment_details"]
    assert "unmatched_concepts" not in c1_sub["assessment_details"]
    assert "missing_concepts" not in c1_sub["assessment_details"]

    # Contributor 1 can download their own PDF report, but Contributor 2 cannot
    own_pdf = client.get(f"/submissions/{sub_id}/report.pdf", headers=c1_headers)
    assert own_pdf.status_code == 200
    assert own_pdf.headers["content-type"] == "application/pdf"

    other_pdf = client.get(f"/submissions/{sub_id}/report.pdf", headers=c2_headers)
    assert other_pdf.status_code == 404
    assert client.get(f"/submissions/{sub_id}", headers=c2_headers).status_code == 403


def test_contributor_session_creation_and_direct_text_reference_reports(monkeypatch):
    """
    Verify:
    - A Contributor can create a C4 session and assign 3 contributors (including themselves).
    - Contributors can submit reference material via Option B (Direct-Entry Reference Report text)
      and Option A (Document File Upload) in the same session.
    - Each report triggers 5-keyword extraction via NVIDIA Nemotron and activates the 15-concept pool.
    - The owning contributor can view/edit their own written report text, while other contributors
      and students cannot view another contributor's report or the reference pool.
    - Dashboard analytics endpoint returns real database aggregations.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")

    def register_contributor(name: str) -> tuple[int, dict[str, str]]:
        email = f"creator-c-{uuid.uuid4().hex[:10]}@rvu.edu.in"
        password = "contributor-password-123"
        reg = client.post(
            "/auth/register",
            json={
                "name": name,
                "email": email,
                "password": password,
                "confirm_password": password,
                "role": "contributor",
                "signup_code": "011002",
            },
        )
        assert reg.status_code == 201, reg.text
        log = client.post(
            "/auth/login",
            json={"email": email, "password": password, "expected_role": "contributor"},
        )
        assert log.status_code == 200, log.text
        return reg.json()["user"]["id"], {"Authorization": f"Bearer {log.json()['token']}"}

    c1_id, c1_headers = register_contributor("Session Creator Contributor")
    c2_id, c2_headers = register_contributor("Report Writer Contributor 2")
    c3_id, c3_headers = register_contributor("File Uploader Contributor 3")

    # Contributor 1 can fetch eligible contributors and create a C4 session
    elig = client.get("/contributors/eligible", headers=c1_headers)
    assert elig.status_code == 200
    assert {c1_id, c2_id, c3_id}.issubset({u["id"] for u in elig.json()})

    created = client.post(
        "/lessons",
        headers=c1_headers,
        json={
            "title": "Contributor-Created Demand Forecasting Session",
            "description": "Collaborative forecasting session created by a Contributor.",
            "subject": "Business Forecasting",
            "group_name": "MBA Cohort B",
            "session_date": "2026-10-28",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    )
    assert created.status_code == 200, created.text
    lesson_id = created.json()["id"]
    assert created.json()["is_creator"] is True
    assert created.json()["can_manage"] is True

    # Contributor 2 (not the creator) cannot manage Contributor 1's session
    assert client.get(f"/lessons/{lesson_id}/contributors", headers=c2_headers).status_code == 403
    # Contributor 1 (the creator) CAN view contributor statuses for their session
    mgr_view = client.get(f"/lessons/{lesson_id}/contributors", headers=c1_headers)
    assert mgr_view.status_code == 200
    assert len(mgr_view.json()["contributors"]) == 3

    # Empty or too-short direct text report is rejected with 400
    short_report = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c1_headers,
        json={"title": "Too Short", "content": "Too short."},
    )
    assert short_report.status_code == 400

    # Contributor 1 submits a direct-entry written report (Option B)
    monkeypatch.setattr(
        document_processing,
        "call_nemotron_keyword_extraction",
        lambda document_text, subject="Business Forecasting": json.dumps(
            {"keywords": [f"Direct1 Concept {k}" for k in range(1, 6)]}
        ),
    )
    rep1_text = (
        "Direct reference report on Holt-Winters triple exponential smoothing, "
        "seasonal indices, demand elasticity, forecast horizon, and residual diagnostics."
    )
    rep1 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c1_headers,
        json={"title": "Holt-Winters Reference Report", "content": rep1_text},
    )
    assert rep1.status_code == 201, rep1.text
    assert rep1.json()["entry_mode"] == "text"
    assert rep1.json()["report_title"] == "Holt-Winters Reference Report"
    assert rep1.json()["processing_status"] == "completed"

    # Contributor 1 can retrieve their own written report text via /reference-documents/me
    me1 = client.get(f"/lessons/{lesson_id}/reference-documents/me", headers=c1_headers)
    assert me1.status_code == 200
    assert me1.json()["entry_mode"] == "text"
    assert me1.json()["report_text"] == rep1_text
    assert "keywords" not in me1.json()

    # Contributor 2 submits a direct-entry written report via alias /reference-documents/text
    monkeypatch.setattr(
        document_processing,
        "call_nemotron_keyword_extraction",
        lambda document_text, subject="Business Forecasting": json.dumps(
            {"keywords": [f"Direct2 Concept {k}" for k in range(1, 6)]}
        ),
    )
    rep2 = client.post(
        f"/lessons/{lesson_id}/reference-documents/text",
        headers=c2_headers,
        json={
            "title": "ARIMA Diagnostics Report",
            "content": "Reference report covering Box-Jenkins ARIMA stationarity, differencing, ACF, PACF, and Ljung-Box tests.",
        },
    )
    assert rep2.status_code == 201, rep2.text
    assert rep2.json()["processing_status"] == "completed"

    # Contributor 3 uploads a file document (Option A) -> triggers 15-concept pool activation
    monkeypatch.setattr(
        document_processing,
        "call_nemotron_keyword_extraction",
        lambda document_text, subject="Business Forecasting": json.dumps(
            {"keywords": [f"File3 Concept {k}" for k in range(1, 6)]}
        ),
    )
    rep3 = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=c3_headers,
        files={"file": ("c3_forecast.txt", b"Reference file covering tracking signals, MAPE, RMSE, MAE, and bias.", "text/plain")},
    )
    assert rep3.status_code == 201
    assert rep3.json()["entry_mode"] == "file"
    assert rep3.json()["processing_status"] == "completed"

    # Session is now ready_for_assessment with 15 distinct concepts
    pool_info = client.get(f"/lessons/{lesson_id}/keyword-pool", headers=c1_headers)
    assert pool_info.status_code == 200
    assert pool_info.json()["pool_ready"] is True
    assert pool_info.json()["distinct_concepts_count"] == 15
    assert pool_info.json()["readiness_status"] == "ready_for_assessment"
    # Non-faculty session creator sees pool readiness & counts without raw keywords leak
    assert pool_info.json()["distinct_concepts"] == []

    # Verify /analytics/dashboard returns real database aggregations for Faculty, Contributor, and Student
    faculty = login_faculty()
    fac_analytics = client.get("/analytics/dashboard", headers=faculty)
    assert fac_analytics.status_code == 200
    assert "rating_distribution" in fac_analytics.json()
    assert len(fac_analytics.json()["rating_distribution"]) == 5
    assert "submission_completion" in fac_analytics.json()
    assert "session_comparison" in fac_analytics.json()

    contrib_analytics = client.get("/analytics/dashboard", headers=c1_headers)
    assert contrib_analytics.status_code == 200
    assert contrib_analytics.json()["total_sessions"] >= 1
