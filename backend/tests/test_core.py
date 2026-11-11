import pytest

from app.c4_scoring import (
    calculate_c4_coverage,
    calculate_c4_rating,
    rating_for_c4_coverage,
)
from app.scoring import calculate_c4_rating as scoring_c4_rating
from app.scoring import rating_for_c4_concepts, rating_for_recall
from app.similarity import compare_text


@pytest.mark.parametrize(
    "matched_concepts, expected_rating",
    [
        # 0–20% (0..3 of 15) -> 1
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        # Above 20–40% (4..6 of 15) -> 2
        (4, 2),
        (5, 2),
        (6, 2),
        # Above 40–60% (7..9 of 15) -> 3
        (7, 3),
        (8, 3),
        (9, 3),
        # Above 60–80% (10..12 of 15) -> 4
        (10, 4),
        (11, 4),
        (12, 4),
        # Above 80–100% (13..15 of 15) -> 5
        (13, 5),
        (14, 5),
        (15, 5),
    ],
)
def test_c4_scoring_boundaries_all_0_to_15(matched_concepts: int, expected_rating: int):
    assert calculate_c4_rating(matched_concepts) == expected_rating
    assert scoring_c4_rating(matched_concepts) == expected_rating
    assert rating_for_c4_concepts(matched_concepts) == expected_rating


@pytest.mark.parametrize(
    "coverage_pct, expected_rating",
    [
        (0.0, 1),
        (10.0, 1),
        (20.0, 1),
        (20.01, 2),
        (33.33, 2),
        (40.0, 2),
        (40.01, 3),
        (50.0, 3),
        (60.0, 3),
        (60.01, 4),
        (75.0, 4),
        (80.0, 4),
        (80.01, 5),
        (95.0, 5),
        (100.0, 5),
    ],
)
def test_c4_coverage_percentage_exact_band_boundaries(coverage_pct: float, expected_rating: int):
    assert rating_for_c4_coverage(coverage_pct) == expected_rating


def test_c4_scoring_handles_zero_concepts_safely_without_zero_division():
    assert calculate_c4_coverage(0, 0) == 0.0
    assert calculate_c4_rating(0, 0) == 1
    assert rating_for_c4_concepts(0, 0) == 1
    with pytest.raises(ValueError):
        calculate_c4_coverage(1, 0)
    with pytest.raises(ValueError):
        calculate_c4_rating(1, 0)


@pytest.mark.parametrize("invalid_value", [-1, -10, 16, 20, True, False, 4.5, "5", None])
def test_c4_scoring_rejects_outside_0_to_15_and_non_integers(invalid_value):
    with pytest.raises(ValueError):
        calculate_c4_rating(invalid_value)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        rating_for_c4_concepts(invalid_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value, expected",
    [(100, 5), (99, 4), (90, 4), (89.99, 3), (80, 3), (79.99, 0), (0, 0)],
)
def test_scoring_thresholds(value, expected):
    assert rating_for_recall(value) == expected


@pytest.mark.parametrize("invalid_pct", [-0.01, -5, 100.01, 120, True, False, "90", None])
def test_scoring_rejects_out_of_range(invalid_pct):
    with pytest.raises(ValueError):
        rating_for_recall(invalid_pct)  # type: ignore[arg-type]


def test_similarity_is_local_and_normalized():
    result = compare_text("Spaced practice supports memory", "SPACED practice supports memory!")
    assert result["percentage"] > 90
    assert "spaced" in result["matches"]


def test_different_reflections_produce_different_master_similarity():
    master = "Historical demand and uncertainty guide forecasting decisions."
    relevant = compare_text(master, "Historical demand guides the forecast.")
    unrelated = compare_text(master, "A completely unrelated sentence about cooking.")
    assert relevant["percentage"] != unrelated["percentage"]
