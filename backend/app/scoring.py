from decimal import Decimal

from .c4_scoring import (
    calculate_c4_coverage,
    calculate_c4_rating,
    rating_for_c4_coverage,
)


def rating_for_c4_concepts(matched_count: int, total_concepts: int = 15) -> int:
    """
    Deterministic C4 rating (1-5) based on coverage percentage:
      Coverage = (Matched canonical concepts / Total canonical concepts) * 100
      0–20%         -> 1
      Above 20–40%  -> 2
      Above 40–60%  -> 3
      Above 60–80%  -> 4
      Above 80–100% -> 5
    """
    return calculate_c4_rating(matched_count, total_concepts=total_concepts)


def rating_for_recall(percentage: float) -> int:
    """
    Legacy recall percentage rating (0-100%) for non-C4 sessions.
    """
    if isinstance(percentage, bool) or not isinstance(percentage, (int, float, Decimal)):
        raise ValueError("Recall percentage must be a number between 0 and 100")
    if not 0 <= float(percentage) <= 100:
        raise ValueError("Recall percentage must be between 0 and 100")
    value = Decimal(str(percentage))
    if value == 100:
        return 5
    if value >= 90:
        return 4
    if value >= 80:
        return 3
    return 0


__all__ = [
    "calculate_c4_coverage",
    "calculate_c4_rating",
    "rating_for_c4_concepts",
    "rating_for_c4_coverage",
    "rating_for_recall",
]
