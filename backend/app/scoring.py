from decimal import Decimal

from .c4_scoring import calculate_c4_rating


def rating_for_c4_concepts(matched_count: int) -> int:
    """
    Deterministic C4 rating (1-5) for 0-15 matched concepts:
      0-4   -> 1
      5-7   -> 2
      8-10  -> 3
      11-12 -> 4
      13-15 -> 5
    """
    return calculate_c4_rating(matched_count)


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
