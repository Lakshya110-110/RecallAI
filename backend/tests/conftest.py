import os
from pathlib import Path
import tempfile

# Isolate pytest runs from recallai.db and demo_recallai.db before app.main is imported
_TEST_TMP_DIR = Path(tempfile.gettempdir()) / "recallai_pytest"
_TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_TMP_DIR / 'test_recallai.db'}"
os.environ["UPLOAD_DIR"] = str(_TEST_TMP_DIR / "uploads")

import pytest
from app import main, document_processing


@pytest.fixture(autouse=True)
def mock_nvidia_in_unit_tests(monkeypatch):
    """
    Prevent routine unit tests from making live network calls or consuming paid NVIDIA API credits.
    Individual tests can still override main.nvidia_assess, main.nvidia_c4_assess, or
    document_processing.call_nemotron_keyword_extraction via monkeypatch.
    """
    def default_mock_legacy_assess(lesson, submission):
        concepts = [c.strip() for c in (lesson.concepts or "").splitlines() if c.strip()]
        return main.NvidiaAssessment(
            recall_percentage=90.0,
            concept_results=[
                main.ConceptResult(
                    concept=c,
                    status="recalled",
                    evidence="",
                    explanation="Mocked unit test recall.",
                )
                for c in concepts[:1]
            ],
            missing_concepts=concepts[1:],
            feedback="Mocked assessment for unit test.",
        )

    def default_mock_c4_assess(lesson, submission, pool_data):
        return main.NvidiaC4MatchResponse(
            concept_results=[],
            feedback="Mocked C4 assessment feedback.",
        )

    def default_mock_keyword_extraction(document_text: str, subject: str = "Business Forecasting") -> str:
        return (
            '{"keywords": ["Historical demand patterns", "Causal drivers", '
            '"Forecast uncertainty", "Scenario outcomes", "Baseline estimation"]}'
        )

    monkeypatch.setattr(main, "nvidia_assess", default_mock_legacy_assess)
    monkeypatch.setattr(main, "nvidia_c4_assess", default_mock_c4_assess)
    monkeypatch.setattr(document_processing, "call_nemotron_keyword_extraction", default_mock_keyword_extraction)
