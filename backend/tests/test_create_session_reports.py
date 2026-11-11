import json
import uuid
import pytest
from fastapi.testclient import TestClient

from app import document_processing, main

client = TestClient(main.app)


def register_user(name: str, role: str, signup_code: str = "011002") -> tuple[int, dict[str, str]]:
    email = f"user-{uuid.uuid4().hex[:8]}@rvu.edu.in"
    password = "test-password-123"
    reg = client.post(
        "/auth/register",
        json={
            "name": name,
            "email": email,
            "password": password,
            "confirm_password": password,
            "role": role,
            "signup_code": signup_code if role in {"faculty", "contributor"} else None,
        },
    )
    assert reg.status_code == 201, reg.text
    login = client.post(
        "/auth/login",
        json={"email": email, "password": password, "expected_role": role},
    )
    assert login.status_code == 200, login.text
    return reg.json()["user"]["id"], {"Authorization": f"Bearer {login.json()['token']}"}


def test_session_creation_with_three_contributor_reports(monkeypatch):
    """
    Test:
    1. Faculty creates a C4 session with exactly 3 contributors.
    2. Faculty submits 3 separate 500-character reference reports for each contributor.
    3. Each report extracts 5 distinct concepts through Nemotron.
    4. Combined pool is automatically formed (15 concept entries total) and lesson is ready_for_assessment.
    5. Each report is associated with the correct contributor user ID.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")

    fac_id, fac_headers = register_user("Prof. Sharma", "faculty")
    c1_id, c1_headers = register_user("Aarav Sharma", "contributor")
    c2_id, c2_headers = register_user("Diya Verma", "contributor")
    c3_id, c3_headers = register_user("Kabir Mehta", "contributor")

    # 1. Create session
    create_res = client.post(
        "/lessons",
        headers=fac_headers,
        json={
            "title": "Holt-Winters & ARIMA Session",
            "session_date": "2026-11-01",
            "group_name": "Section B",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    )
    assert create_res.status_code == 200, create_res.text
    lesson_id = create_res.json()["id"]

    # 2. Mock Nemotron keyword extraction to return 5 distinct concepts per call
    extraction_calls = []

    def mock_extract(document_text: str, subject: str = "Business Forecasting"):
        extraction_calls.append(document_text)
        idx = len(extraction_calls)
        return json.dumps({
            "keywords": [f"Concept {idx}_{k}" for k in range(1, 6)]
        })

    monkeypatch.setattr(document_processing, "call_nemotron_keyword_extraction", mock_extract)

    # 3. Exactly 500 characters accepted
    report_500 = "A" * 50 + " " + "B" * 50 + " " + "C" * 50 + " " + "D" * 50 + " " + "E" * 50 + " " + "F" * 50 + " " + "G" * 50 + " " + "H" * 50 + " " + "I" * 50 + " " + "J" * 41
    assert len(report_500) == 500

    rep1 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=fac_headers,
        json={
            "contributor_id": c1_id,
            "title": "Aarav Report",
            "content": report_500,
        },
    )
    assert rep1.status_code == 201, rep1.text
    assert rep1.json()["contributor_id"] == c1_id
    assert rep1.json()["processing_status"] == "completed"

    rep2_text = "Diya Verma reference report on exponential smoothing, trend components, seasonality indices, damping factors, and mean absolute error evaluation."
    rep2 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=fac_headers,
        json={
            "contributor_id": c2_id,
            "title": "Diya Report",
            "content": rep2_text,
        },
    )
    assert rep2.status_code == 201, rep2.text
    assert rep2.json()["contributor_id"] == c2_id
    assert rep2.json()["processing_status"] == "completed"

    rep3_text = "Kabir Mehta reference report covering Box-Jenkins ARIMA stationarity, auto-correlation function ACF, PACF partial autocorrelation, differencing, and AIC."
    rep3 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=fac_headers,
        json={
            "contributor_id": c3_id,
            "title": "Kabir Report",
            "content": rep3_text,
        },
    )
    assert rep3.status_code == 201, rep3.text
    assert rep3.json()["contributor_id"] == c3_id
    assert rep3.json()["processing_status"] == "completed"

    # 4. Check keyword pool: all 3 reports processed, exactly 15 concepts in source entries
    pool_res = client.get(f"/lessons/{lesson_id}/keyword-pool", headers=fac_headers)
    assert pool_res.status_code == 200
    pool_data = pool_res.json()
    assert pool_data["pool_ready"] is True
    assert pool_data["total_raw_keywords"] == 15
    assert pool_data["distinct_concepts_count"] == 15
    assert pool_data["readiness_status"] == "ready_for_assessment"


def test_501_characters_rejected_and_empty_rejected(monkeypatch):
    """
    Test:
    - 501 characters is rejected by backend (422)
    - Empty report is rejected (422)
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")
    fac_id, fac_headers = register_user("Prof. Roy", "faculty")
    c1_id, _ = register_user("Student 1", "student")
    c2_id, _ = register_user("Student 2", "student")
    c3_id, _ = register_user("Student 3", "student")

    create_res = client.post(
        "/lessons",
        headers=fac_headers,
        json={
            "title": "Validation Test Session",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    )
    lesson_id = create_res.json()["id"]

    # 501 characters
    report_501 = "X" * 501
    res_501 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=fac_headers,
        json={
            "contributor_id": c1_id,
            "content": report_501,
        },
    )
    assert res_501.status_code == 422
    assert "500 characters or fewer" in res_501.text

    # Empty report
    res_empty = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=fac_headers,
        json={
            "contributor_id": c1_id,
            "content": "   ",
        },
    )
    assert res_empty.status_code == 422


def test_creator_contributor_and_self_selection(monkeypatch):
    """
    Test:
    - Contributor creates session and selects themselves as 1 of the 3 contributors.
    - Contributor submits reports for themselves and the other 2 contributors.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")
    c1_id, c1_headers = register_user("Contributor Creator", "contributor")
    c2_id, _ = register_user("Contributor Partner 2", "contributor")
    c3_id, _ = register_user("Contributor Partner 3", "contributor")

    create_res = client.post(
        "/lessons",
        headers=c1_headers,
        json={
            "title": "Contributor Managed Forecasting Session",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    )
    assert create_res.status_code == 200
    lesson_id = create_res.json()["id"]

    monkeypatch.setattr(
        document_processing,
        "call_nemotron_keyword_extraction",
        lambda document_text, subject="Business Forecasting": json.dumps({
            "keywords": ["Concept Alpha", "Concept Beta", "Concept Gamma", "Concept Delta", "Concept Epsilon"]
        }),
    )

    # Submitting for themselves (c1_id)
    r1 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c1_headers,
        json={
            "contributor_id": c1_id,
            "content": "Contributor creator report on exponential smoothing and tracking signals in demand forecasting.",
        },
    )
    assert r1.status_code == 201
    assert r1.json()["contributor_id"] == c1_id

    # Submitting for partner (c2_id)
    r2 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c1_headers,
        json={
            "contributor_id": c2_id,
            "content": "Contributor partner report covering ARIMA differencing, Box-Jenkins methodology, and residuals.",
        },
    )
    assert r2.status_code == 201
    assert r2.json()["contributor_id"] == c2_id


def test_partial_failure_reporting_and_idempotent_retry(monkeypatch):
    """
    Test:
    - If 1 report fails during Nemotron extraction, it is flagged as failed.
    - Retry succeeds without duplicate active document rows.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")
    fac_id, fac_headers = register_user("Prof. Partial", "faculty")
    c1_id, _ = register_user("Sub 1", "student")
    c2_id, _ = register_user("Sub 2", "student")
    c3_id, _ = register_user("Sub 3", "student")

    create_res = client.post(
        "/lessons",
        headers=fac_headers,
        json={
            "title": "Partial Failure & Retry Test",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    )
    lesson_id = create_res.json()["id"]

    call_count = 0

    def mock_extract(document_text: str, subject: str = "Business Forecasting"):
        nonlocal call_count
        call_count += 1
        if "FAIL_THIS" in document_text:
            raise ValueError("Simulated Nemotron concept extraction failure")
        return json.dumps({
            "keywords": [f"Concept {call_count}_{k}" for k in range(1, 6)]
        })

    monkeypatch.setattr(document_processing, "call_nemotron_keyword_extraction", mock_extract)

    # 1. Report 1 succeeds
    r1 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=fac_headers,
        json={"contributor_id": c1_id, "content": "Valid report 1 on Holt-Winters smoothing and seasonal adjustments."},
    )
    assert r1.status_code == 201
    assert r1.json()["processing_status"] == "completed"

    # 2. Report 2 fails
    r2 = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=fac_headers,
        json={"contributor_id": c2_id, "content": "FAIL_THIS report covering invalid data that triggers extraction failure."},
    )
    assert r2.status_code == 201
    assert r2.json()["processing_status"] == "failed"
    assert "Simulated Nemotron" in (r2.json()["error_message"] or "")

    # Lesson readiness should reflect extraction_failed
    less = client.get(f"/lessons/{lesson_id}/contributors", headers=fac_headers)
    assert less.json()["readiness_status"] == "extraction_failed"

    # 3. Retry Report 2 with valid text
    r2_retry = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=fac_headers,
        json={"contributor_id": c2_id, "content": "Repaired report for Contributor 2 covering Box-Jenkins ARIMA models."},
    )
    assert r2_retry.status_code == 201
    assert r2_retry.json()["processing_status"] == "completed"

    # Check that previous failed document is deactivated and only 1 active document exists for c2_id
    docs_res = client.get(f"/lessons/{lesson_id}/reference-documents", headers=fac_headers)
    assert docs_res.status_code == 200
    docs = docs_res.json()
    c2_docs = [d for d in docs if d["contributor_id"] == c2_id]
    assert len(c2_docs) == 1
    assert c2_docs[0]["processing_status"] == "completed"


def test_nemotron_does_not_fabricate_concepts_on_insufficient_output(monkeypatch):
    """
    Test:
    - If model returns fewer than 5 concepts (e.g. 3 concepts), extraction fails.
    - System refuses to fabricate or invent filler concepts.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")
    fac_id, fac_headers = register_user("Prof. Strict", "faculty")
    c1_id, _ = register_user("Contrib A", "student")
    c2_id, _ = register_user("Contrib B", "student")
    c3_id, _ = register_user("Contrib C", "student")

    create_res = client.post(
        "/lessons",
        headers=fac_headers,
        json={
            "title": "Strict Extraction Session",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    )
    lesson_id = create_res.json()["id"]

    # Model returns only 3 concepts (insufficient for 5 concepts)
    monkeypatch.setattr(
        document_processing,
        "call_nemotron_keyword_extraction",
        lambda document_text, subject="Business Forecasting": json.dumps({
            "keywords": ["Single Concept 1", "Single Concept 2", "Single Concept 3"]
        }),
    )

    rep = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=fac_headers,
        json={
            "contributor_id": c1_id,
            "content": "Short reference report with insufficient domain depth to yield 5 concepts.",
        },
    )
    assert rep.status_code == 201
    assert rep.json()["processing_status"] == "failed"
    assert "Expected 5 keywords from model, received 3" in (rep.json()["error_message"] or "")


def test_separate_limits_enforced_independently(monkeypatch):
    """
    Test Requirement 7:
    - 500-character limit applies to contributor reference reports in Create Session.
    - Uploaded reference documents accept larger text (e.g. > 1000 characters).
    - Student reflection independently enforces its 500-character limit.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")
    fac_id, fac_headers = register_user("Prof. Limits", "faculty")
    c1_id, c1_headers = register_user("Contrib L1", "contributor")
    c2_id, c2_headers = register_user("Contrib L2", "contributor")
    c3_id, c3_headers = register_user("Contrib L3", "contributor")
    stu_id, stu_headers = register_user("Student Lim", "student")

    create_res = client.post(
        "/lessons",
        headers=fac_headers,
        json={
            "title": "Limits Verification Session",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    )
    lesson_id = create_res.json()["id"]

    # 1. Contributor report in Create Session workflow rejects 501 characters
    rej = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=fac_headers,
        json={"contributor_id": c1_id, "content": "Z" * 501},
    )
    assert rej.status_code == 422

    # 2. Uploaded reference document allows > 1000 characters (e.g. 1500 chars)
    monkeypatch.setattr(
        document_processing,
        "call_nemotron_keyword_extraction",
        lambda document_text, subject="Business Forecasting": json.dumps({
            "keywords": [f"Concept {k}" for k in range(1, 6)]
        }),
    )
    large_doc_text = ("Forecasting reference notes on decomposition and seasonal smoothing. " * 30).encode("utf-8")
    assert len(large_doc_text) > 1000
    upload_res = client.post(
        f"/lessons/{lesson_id}/reference-documents",
        headers=c1_headers,
        files={"file": ("large_reference.txt", large_doc_text, "text/plain")},
    )
    assert upload_res.status_code == 201
    assert upload_res.json()["processing_status"] == "completed"

    # 3. Student reflection workflow independently rejects 501 characters but accepts 500
    stu_sub_501 = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=stu_headers,
        json={"text": "R" * 501},
    )
    assert stu_sub_501.status_code == 422
    assert "500 characters or fewer" in stu_sub_501.text

    # Student reflection accepts 500 characters
    stu_sub_500 = client.post(
        f"/lessons/{lesson_id}/submissions",
        headers=stu_headers,
        json={"text": "R" * 500},
    )
    # The session might be awaiting other uploads, so status code 409 (session not ready) or 201 (if ready)
    # But crucially, it did NOT fail with 422 validation error for text length!
    assert stu_sub_500.status_code in {201, 400, 409}
    if stu_sub_500.status_code in {400, 409}:
        assert "not ready for student submissions" in stu_sub_500.text


def test_simplified_create_session_and_independent_contributor_submission_workflow(monkeypatch):
    """
    Verification of Requirement 4:
    1. Create a session without entering any contributor report text.
    2. Confirm the session is saved and appears in the correct dashboards.
    3. Log in as each assigned contributor and open the session.
    4. Confirm each contributor can submit their own report through the existing page.
    5. Confirm the correct session and contributor IDs are stored.
    6. Confirm the session reaches the existing ready/assessment state after the required three reports are processed.
    7. Accidental duplicate submission is idempotent.
    """
    monkeypatch.setenv("FACULTY_SIGNUP_CODE", "011002")

    fac_id, fac_headers = register_user("Prof. Workflow", "faculty")
    c1_id, c1_headers = register_user("Independent Contrib 1", "contributor")
    c2_id, c2_headers = register_user("Independent Contrib 2", "contributor")
    c3_id, c3_headers = register_user("Independent Contrib 3", "contributor")
    outsider_id, outsider_headers = register_user("Outsider Student", "student")

    # 1. Create session without any contributor report text
    create_res = client.post(
        "/lessons",
        headers=fac_headers,
        json={
            "title": "Decoupled Workflow Session",
            "session_date": "2026-11-15",
            "group_name": "Section C",
            "subject": "Business Forecasting",
            "assessment_mode": "c4",
            "contributor_ids": [c1_id, c2_id, c3_id],
        },
    )
    assert create_res.status_code == 200, create_res.text
    session_data = create_res.json()
    lesson_id = session_data["id"]
    assert session_data["title"] == "Decoupled Workflow Session"
    assert session_data["readiness_status"] == "awaiting_uploads"

    # 2. Confirm session appears in creator's managed sessions
    fac_lessons = client.get("/lessons", headers=fac_headers).json()
    created_lesson = next((l for l in fac_lessons if l["id"] == lesson_id), None)
    assert created_lesson is not None
    assert created_lesson["title"] == "Decoupled Workflow Session"
    assert created_lesson["readiness_status"] == "awaiting_uploads"

    # 3. Each assigned contributor sees the session in their assignments dashboard
    for cid, hdrs in [(c1_id, c1_headers), (c2_id, c2_headers), (c3_id, c3_headers)]:
        assign_res = client.get("/contributor/assignments", headers=hdrs)
        assert assign_res.status_code == 200
        my_assignments = assign_res.json()
        item = next((a for a in my_assignments if a["lesson_id"] == lesson_id), None)
        assert item is not None, f"Lesson {lesson_id} not found in contributor {cid}'s assignments"
        assert item["document_uploaded"] is False
        assert item["processing_status"] == "awaiting_upload"
        assert item["session_readiness_status"] == "awaiting_uploads"

    # Unassigned contributor/student does NOT see it in their assignments
    outsider_assignments = client.get("/contributor/assignments", headers=outsider_headers).json()
    assert not any(a["lesson_id"] == lesson_id for a in outsider_assignments)

    # 4. Mock Nemotron concept extraction to return 5 distinct concepts per report
    report_count = 0

    def mock_extract(document_text: str, subject: str = "Business Forecasting"):
        nonlocal report_count
        report_count += 1
        return json.dumps({
            "keywords": [f"Decoupled Concept {report_count}_{k}" for k in range(1, 6)]
        })

    monkeypatch.setattr(document_processing, "call_nemotron_keyword_extraction", mock_extract)

    # 5. Contributor 1 opens the session and checks their report status (currently none)
    me_res1_empty = client.get(f"/lessons/{lesson_id}/reference-documents/me", headers=c1_headers)
    assert me_res1_empty.status_code == 404

    # Contributor 1 cannot submit on behalf of Contributor 2
    forbidden_sub = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c1_headers,
        json={
            "contributor_id": c2_id,
            "title": "Illegal Report",
            "content": "Contributor 1 attempting to forge report for Contributor 2 on ARIMA forecasting.",
        },
    )
    assert forbidden_sub.status_code == 403

    # Contributor 1 submits their own report
    rep1_text = "Contributor 1 detailed analysis of moving average processes and simple exponential smoothing."
    rep1_res = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c1_headers,
        json={
            "title": "C1 Written Report",
            "content": rep1_text,
            "lesson_id": lesson_id,
        },
    )
    assert rep1_res.status_code == 201, rep1_res.text
    rep1_data = rep1_res.json()
    assert rep1_data["contributor_id"] == c1_id
    assert rep1_data["lesson_id"] == lesson_id
    assert rep1_data["processing_status"] == "completed"

    # Confirm Contributor 1 can retrieve their own report
    me_c1 = client.get(f"/lessons/{lesson_id}/reference-documents/me", headers=c1_headers).json()
    assert me_c1["document_uploaded"] is True
    assert me_c1["report_title"] == "C1 Written Report"
    assert me_c1["report_text"] == rep1_text

    # Prevent accidental duplicate submission: resubmitting exact same text returns existing document
    dup_res = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c1_headers,
        json={
            "title": "C1 Written Report",
            "content": rep1_text,
            "lesson_id": lesson_id,
        },
    )
    assert dup_res.status_code == 201
    assert dup_res.json()["id"] == rep1_data["id"]

    # Session is still awaiting remaining contributor uploads
    c1_status_check = client.get(f"/lessons/{lesson_id}/contributors", headers=fac_headers).json()
    assert c1_status_check["readiness_status"] == "awaiting_uploads"

    # 6. Contributor 2 submits their own report
    rep2_text = "Contributor 2 reference document on Holt linear trend models, slope smoothing, and damped trends."
    rep2_res = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c2_headers,
        json={
            "title": "C2 Holt Analysis",
            "content": rep2_text,
            "lesson_id": lesson_id,
        },
    )
    assert rep2_res.status_code == 201, rep2_res.text
    assert rep2_res.json()["contributor_id"] == c2_id
    assert rep2_res.json()["lesson_id"] == lesson_id

    # 7. Contributor 3 submits their own report
    rep3_text = "Contributor 3 report on Box-Jenkins seasonal ARIMA modeling, ACF correlograms, and residual diagnostics."
    rep3_res = client.post(
        f"/lessons/{lesson_id}/reference-reports",
        headers=c3_headers,
        json={
            "title": "C3 ARIMA Models",
            "content": rep3_text,
            "lesson_id": lesson_id,
        },
    )
    assert rep3_res.status_code == 201, rep3_res.text
    assert rep3_res.json()["contributor_id"] == c3_id
    assert rep3_res.json()["lesson_id"] == lesson_id

    # 8. All three reports processed: session reaches ready_for_assessment state!
    updated_lesson = client.get(f"/lessons/{lesson_id}/contributors", headers=fac_headers).json()
    assert updated_lesson["readiness_status"] == "ready_for_assessment"

    # Confirm keyword pool is ready with 15 concepts (5 from each contributor)
    pool_res = client.get(f"/lessons/{lesson_id}/keyword-pool", headers=fac_headers).json()
    assert pool_res["pool_ready"] is True
    assert pool_res["readiness_status"] == "ready_for_assessment"
    assert pool_res["total_raw_keywords"] == 15
    assert pool_res["distinct_concepts_count"] == 15


