import re
from typing import Any


PROMPT_VERSION = "c4-v2.0"


def calculate_c4_coverage(matched_count: int, total_concepts: int = 15) -> float:
    """
    Calculate C4 canonical concept coverage percentage:
      Coverage = (Matched canonical concepts / Total canonical concepts) * 100.
    Handles zero total_concepts safely and never divides by zero.
    """
    if (
        not isinstance(matched_count, int)
        or isinstance(matched_count, bool)
        or not isinstance(total_concepts, int)
        or isinstance(total_concepts, bool)
    ):
        raise ValueError("matched_count and total_concepts must be integers")
    if total_concepts < 0:
        raise ValueError("total_concepts must be non-negative")
    if total_concepts == 0:
        if matched_count < 0:
            raise ValueError("matched_count must be non-negative")
        if matched_count > 0:
            raise ValueError("matched_count cannot exceed total_concepts (0)")
        return 0.0
    if matched_count < 0 or matched_count > total_concepts:
        raise ValueError(f"matched_count must be between 0 and {total_concepts}")
    return round((matched_count / total_concepts) * 100.0, 2)


def rating_for_c4_coverage(coverage_percentage: float | int) -> int:
    """
    Map C4 coverage percentage (0–100%) to the 5-point rating band:
      0–20%         = 1
      Above 20–40%  = 2
      Above 40–60%  = 3
      Above 60–80%  = 4
      Above 80–100% = 5
    """
    if not isinstance(coverage_percentage, (int, float)) or isinstance(coverage_percentage, bool):
        raise ValueError("coverage_percentage must be a number between 0 and 100")
    pct = float(coverage_percentage)
    if pct < 0.0 or pct > 100.0:
        raise ValueError("coverage_percentage must be between 0 and 100")
    if pct <= 20.0:
        return 1
    if pct <= 40.0:
        return 2
    if pct <= 60.0:
        return 3
    if pct <= 80.0:
        return 4
    return 5


def calculate_c4_rating(matched_count: int, total_concepts: int = 15) -> int:
    """
    Deterministic C4 rating calculation based on coverage percentage:
      Coverage = (Matched canonical concepts / Total canonical concepts) * 100
    Rating bands:
      0–20%         -> 1
      Above 20–40%  -> 2
      Above 40–60%  -> 3
      Above 60–80%  -> 4
      Above 80–100% -> 5
    Handles zero concepts safely without dividing by zero (returns 1).
    """
    if (
        not isinstance(matched_count, int)
        or isinstance(matched_count, bool)
        or not isinstance(total_concepts, int)
        or isinstance(total_concepts, bool)
    ):
        raise ValueError("matched_count and total_concepts must be integers")
    if total_concepts < 0:
        raise ValueError("total_concepts must be non-negative")
    if total_concepts == 0:
        if matched_count < 0:
            raise ValueError("matched_count must be non-negative")
        if matched_count > 0:
            raise ValueError("matched_count cannot exceed total_concepts (0)")
        return 1
    if matched_count < 0 or matched_count > total_concepts:
        raise ValueError(f"matched_count must be between 0 and {total_concepts}")
    exact_coverage = (matched_count / total_concepts) * 100.0
    return rating_for_c4_coverage(exact_coverage)


def normalize_text_tokens(text: str) -> str:
    """Normalize casing, punctuation, and whitespace for consistent comparison."""
    if not text:
        return ""
    lowered = text.casefold()
    # Replace hyphens/slashes/punctuation with spaces while keeping alphanumeric tokens
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    cleaned = re.sub(r"_+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _stem_token(word: str) -> str:
    """Lightweight deterministic English inflection stemmer for reliable plural/tense matching."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if len(base) >= 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if len(base) >= 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if word.endswith("es") and len(word) > 4 and (word.endswith(("ses", "xes", "zes", "ches", "shes"))):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def normalize_concept(concept: str) -> str:
    """
    Normalize a concept phrase for deduplication and pool matching.
    Removes leading articles and normalizes tokens.
    """
    norm = normalize_text_tokens(concept)
    norm = re.sub(r"^(the|a|an)\s+", "", norm)
    tokens = [_stem_token(t) for t in norm.split() if t]
    return " ".join(tokens)


def normalize_concept_surface(concept: str) -> str:
    """Normalize surface form (preserving exact unstemmed words without punctuation/leading articles)."""
    norm = normalize_text_tokens(concept)
    return re.sub(r"^(the|a|an)\s+", "", norm)


# ---------------------------------------------------------------------------
# Domain-Aware Semantic Canonicalization for Business Forecasting
# ---------------------------------------------------------------------------

# Standalone acronym tokens expanded before concept comparison
_ACRONYM_TOKEN_EXPANSIONS: dict[str, str] = {
    "tes": "triple exponential smoothing",
    "ses": "simple exponential smoothing",
    "des": "double exponential smoothing",
    "mape": "mean absolute percentage error",
    "smape": "symmetric mean absolute percentage error",
    "rmse": "root mean squared error",
    "mse": "mean squared error",
    "mae": "mean absolute error",
    "mad": "mean absolute deviation",
    "mfe": "mean forecast error",
    "acf": "autocorrelation function",
    "pacf": "partial autocorrelation function",
    "arima": "autoregressive integrated moving average",
    "sarima": "seasonal autoregressive integrated moving average",
    "wma": "weighted moving average",
    "sma": "simple moving average",
    "aic": "akaike information criterion",
    "bic": "bayesian information criterion",
    "adf": "augmented dickey fuller",
    "vif": "variance inflation factor",
    "ols": "ordinary least squares",
}

# Non-distinguishing wrapper suffixes stripped when comparing canonical concept identity
_GENERIC_WRAPPER_SUFFIXES: set[str] = {
    "model",
    "models",
    "modeling",
    "modelling",
    "method",
    "methods",
    "methodology",
    "technique",
    "techniques",
    "approach",
    "approaches",
    "algorithm",
    "procedure",
    "process",
    "framework",
    "system",
    "metric",
    "metrics",
    "measure",
    "measures",
}

_STOPWORD_PREPOSITIONS: set[str] = {"of", "the", "a", "an", "in", "for", "to", "with", "on", "by"}

# Curated Business Forecasting canonical concept groups.
# Genuinely different concepts (e.g. Simple vs Double vs Triple Exponential Smoothing,
# ACF vs PACF, MAPE vs RMSE vs MAE vs MAD, Moving Average vs Weighted Moving Average)
# have separate canonical keys and are NEVER merged with each other.
_CANONICAL_CONCEPT_GROUPS: list[dict[str, Any]] = [
    {
        "canonical_key": "triple exponential smooth",
        "canonical_display": "Triple Exponential Smoothing",
        "prefer_canonical_display": True,
        "synonyms": [
            "tes",
            "tes model",
            "tes method",
            "tes technique",
            "tes forecasting",
            "tes forecast",
            "triple exponential smoothing",
            "triple exponential smoothing model",
            "triple exponential smoothing method",
            "holt winters",
            "holt winters model",
            "holt winters method",
            "holt winters smoothing",
            "holt winters exponential smoothing",
            "holt winters triple exponential smoothing",
            "winters exponential smoothing",
            "holt winters seasonal smoothing",
        ],
        "explanation_patterns": [
            r"\bsmooth(?:s|ing|ed)?\s+(?:the\s+)?level\s*,?\s*trend\s*,?\s*(?:and\s+)?seasonal",
            r"\blevel\s*,?\s*trend\s*,?\s*(?:and\s+)?seasonal(?:ity)?\s+smoothing\b",
            r"\bthree\s+smoothing\s+(?:parameters|equations|constants)\b",
        ],
    },
    {
        "canonical_key": "double exponential smooth",
        "canonical_display": "Double Exponential Smoothing",
        "prefer_canonical_display": True,
        "synonyms": [
            "des",
            "des model",
            "des method",
            "double exponential smoothing",
            "double exponential smoothing model",
            "double exponential smoothing method",
            "holt linear",
            "holt s linear",
            "holts linear",
            "holt linear trend",
            "holt s linear trend",
            "holt linear trend model",
            "holt linear trend method",
            "holt linear exponential smoothing",
            "holt two parameter exponential smoothing",
        ],
        "explanation_patterns": [
            r"\bsmooth(?:s|ing|ed)?\s+both\s+level\s+and\s+trend\b",
            r"\btwo\s+smoothing\s+(?:parameters|equations|constants)\s+for\s+level\s+and\s+trend\b",
        ],
    },
    {
        "canonical_key": "simple exponential smooth",
        "canonical_display": "Simple Exponential Smoothing",
        "prefer_canonical_display": True,
        "synonyms": [
            "ses",
            "ses model",
            "ses method",
            "simple exponential smoothing",
            "simple exponential smoothing model",
            "simple exponential smoothing method",
            "single exponential smoothing",
            "single exponential smoothing model",
            "brown simple exponential smoothing",
        ],
        "explanation_patterns": [
            r"\bsingle\s+smoothing\s+(?:parameter|constant)\s+alpha\b",
            r"\bexponential\s+smoothing\s+without\s+trend\s+or\s+seasonality\b",
        ],
    },
    {
        "canonical_key": "exponential smooth",
        "canonical_display": "Exponential Smoothing",
        "prefer_canonical_display": False,
        "synonyms": [
            "exponential smoothing",
            "exponential smoothing model",
            "exponential smoothing method",
            "exponential smoothing technique",
        ],
        "explanation_patterns": [
            r"\bexponentially\s+(?:decaying|decreasing)\s+weights\s+(?:to|on)\s+(?:past|older|historical)\s+observations\b",
        ],
    },
    {
        "canonical_key": "mean absolute percentage error",
        "canonical_display": "Mean Absolute Percentage Error",
        "prefer_canonical_display": True,
        "synonyms": [
            "mape",
            "mape metric",
            "mape error",
            "mape measure",
            "mean absolute percentage error",
            "mean absolute percent error",
        ],
        "explanation_patterns": [
            r"\baverage\s+(?:of\s+)?(?:the\s+)?absolute\s+percentage\s+(?:forecast\s+)?errors?\b",
            r"\bpercentage\s+forecast\s+error\s+relative\s+to\s+actual\b",
            r"\babsolute\s+error\s+divided\s+by\s+actual\s+(?:demand|values?)\s+in\s+percent",
        ],
    },
    {
        "canonical_key": "symmetric mean absolute percentage error",
        "canonical_display": "Symmetric Mean Absolute Percentage Error",
        "prefer_canonical_display": True,
        "synonyms": [
            "smape",
            "s mape",
            "symmetric mape",
            "symmetric mean absolute percentage error",
        ],
        "explanation_patterns": [],
    },
    {
        "canonical_key": "root mean squar error",
        "canonical_display": "Root Mean Squared Error",
        "prefer_canonical_display": True,
        "synonyms": [
            "rmse",
            "rmse metric",
            "rmse error",
            "root mean squared error",
            "root mean square error",
        ],
        "explanation_patterns": [
            r"\bsquare\s+root\s+of\s+(?:the\s+)?(?:mean|average)\s+(?:of\s+)?squared\s+(?:forecast\s+)?errors?\b",
        ],
    },
    {
        "canonical_key": "mean squar error",
        "canonical_display": "Mean Squared Error",
        "prefer_canonical_display": True,
        "synonyms": [
            "mse",
            "mse metric",
            "mse error",
            "mean squared error",
            "mean square error",
        ],
        "explanation_patterns": [
            r"\b(?:mean|average)\s+of\s+(?:the\s+)?squared\s+(?:forecast\s+)?errors?\b",
        ],
    },
    {
        "canonical_key": "mean absolute error",
        "canonical_display": "Mean Absolute Error",
        "prefer_canonical_display": True,
        "synonyms": [
            "mae",
            "mae metric",
            "mae error",
            "mean absolute error",
        ],
        "explanation_patterns": [
            r"\b(?:mean|average)\s+of\s+(?:the\s+)?absolute\s+(?:forecast\s+)?errors?\b",
        ],
    },
    {
        "canonical_key": "mean absolute deviation",
        "canonical_display": "Mean Absolute Deviation",
        "prefer_canonical_display": True,
        "synonyms": [
            "mad",
            "mad metric",
            "mad error",
            "mean absolute deviation",
        ],
        "explanation_patterns": [
            r"\baverage\s+absolute\s+deviation\s+of\s+forecast\s+errors?\b",
        ],
    },
    {
        "canonical_key": "autocorrelation function",
        "canonical_display": "Autocorrelation Function",
        "prefer_canonical_display": False,
        "synonyms": [
            "acf",
            "acf plot",
            "acf correlogram",
            "autocorrelation",
            "autocorrelation function",
            "autocorrelation analysis",
            "serial correlation",
        ],
        "explanation_patterns": [
            r"\bcorrelation\s+of\s+a\s+(?:time\s+)?series\s+with\s+its\s+(?:own\s+)?lagged\s+values\b",
            r"\bcorrelation\s+between\s+observations\s+at\s+different\s+time\s+lags\b",
        ],
    },
    {
        "canonical_key": "partial autocorrelation function",
        "canonical_display": "Partial Autocorrelation Function",
        "prefer_canonical_display": True,
        "synonyms": [
            "pacf",
            "pacf plot",
            "pacf correlogram",
            "partial autocorrelation",
            "partial autocorrelation function",
            "partial autocorrelation analysis",
        ],
        "explanation_patterns": [
            r"\bcorrelation\s+at\s+lag\s+\w+\s+after\s+(?:removing|controlling\s+for)\s+intermediate\s+lags\b",
        ],
    },
    {
        "canonical_key": "autoregressive integrat mov average",
        "canonical_display": "Autoregressive Integrated Moving Average",
        "prefer_canonical_display": True,
        "synonyms": [
            "arima",
            "arima model",
            "arima models",
            "arima modeling",
            "arima method",
            "arima process",
            "box jenkins",
            "box jenkins model",
            "box jenkins method",
            "box jenkins arima",
            "box jenkins arima model",
            "autoregressive integrated moving average",
            "autoregressive integrated moving average model",
        ],
        "explanation_patterns": [
            r"\bcombines\s+autoregressi(?:ve|on)\s*,?\s*differencing\s*,?\s*(?:and\s+)?moving\s+average\b",
        ],
    },
    {
        "canonical_key": "seasonal arima",
        "canonical_display": "Seasonal ARIMA",
        "prefer_canonical_display": True,
        "synonyms": [
            "sarima",
            "sarima model",
            "seasonal arima",
            "seasonal arima model",
            "seasonal autoregressive integrated moving average",
        ],
        "explanation_patterns": [],
    },
    {
        "canonical_key": "weight mov average",
        "canonical_display": "Weighted Moving Average",
        "prefer_canonical_display": True,
        "synonyms": [
            "wma",
            "wma model",
            "wma method",
            "weighted moving average",
            "weighted moving averages",
            "weighted moving average model",
            "weighted moving average method",
        ],
        "explanation_patterns": [
            r"\bassigns\s+(?:different|unequal|heavier)\s+weights\s+to\s+recent\s+observations\s+in\s+a\s+moving\s+window\b",
        ],
    },
    {
        "canonical_key": "mov average",
        "canonical_display": "Moving Average",
        "prefer_canonical_display": False,
        "synonyms": [
            "sma",
            "sma model",
            "moving average",
            "moving averages",
            "moving average model",
            "moving average method",
            "simple moving average",
            "simple moving average model",
        ],
        "explanation_patterns": [
            r"\baverages?\s+the\s+most\s+recent\s+\w+\s+observations\s+in\s+a\s+sliding\s+window\b",
        ],
    },
    {
        "canonical_key": "time series decomposition",
        "canonical_display": "Time Series Decomposition",
        "prefer_canonical_display": False,
        "synonyms": [
            "time series decomposition",
            "decomposition of time series",
            "classical time series decomposition",
        ],
        "explanation_patterns": [
            r"\bseparat(?:es|ing)\s+a\s+time\s+series\s+into\s+trend\s*,?\s*seasonal(?:ity)?\s*,?\s*(?:cyclical\s*,?\s*)?(?:and\s+)?(?:irregular|residual)\b",
        ],
    },
    {
        "canonical_key": "seasonal decomposition",
        "canonical_display": "Seasonal Decomposition",
        "prefer_canonical_display": False,
        "synonyms": [
            "seasonal decomposition",
            "decomposition of seasonality",
            "stl decomposition",
            "seasonal trend decomposition",
        ],
        "explanation_patterns": [],
    },
    {
        "canonical_key": "seasonality adjustment",
        "canonical_display": "Seasonality Adjustment",
        "prefer_canonical_display": False,
        "synonyms": [
            "seasonality adjustment",
            "seasonal adjustment",
            "deseasonalization",
            "deseasonalizing",
        ],
        "explanation_patterns": [
            r"\bremov(?:es|ing)\s+seasonal\s+(?:effects|fluctuations|variations)\s+from\s+(?:the\s+)?(?:time\s+)?series\b",
        ],
    },
    {
        "canonical_key": "seasonal index",
        "canonical_display": "Seasonal Index",
        "prefer_canonical_display": False,
        "synonyms": [
            "seasonal index",
            "seasonal indices",
            "seasonality index",
            "seasonality indices",
            "seasonal factor",
            "seasonal factors",
        ],
        "explanation_patterns": [
            r"\bratio\s+of\s+(?:a\s+)?period(?:'s)?\s+demand\s+to\s+the\s+overall\s+average\b",
        ],
    },
    {
        "canonical_key": "confidence interval",
        "canonical_display": "Confidence Intervals",
        "prefer_canonical_display": False,
        "synonyms": [
            "confidence interval",
            "confidence intervals",
            "confidence band",
            "confidence bands",
            "confidence bounds",
        ],
        "explanation_patterns": [
            r"\brange\s+of\s+values\s+quantifying\s+uncertainty\s+around\s+(?:the\s+)?(?:forecast|estimate)\b",
        ],
    },
    {
        "canonical_key": "prediction interval",
        "canonical_display": "Prediction Intervals",
        "prefer_canonical_display": False,
        "synonyms": [
            "prediction interval",
            "prediction intervals",
            "forecast interval",
            "forecast intervals",
            "prediction bounds",
        ],
        "explanation_patterns": [
            r"\binterval\s+within\s+which\s+future\s+observations\s+are\s+expected\s+to\s+fall\b",
        ],
    },
    {
        "canonical_key": "track signal",
        "canonical_display": "Tracking Signal",
        "prefer_canonical_display": False,
        "synonyms": [
            "tracking signal",
            "tracking signals",
            "tracking signal monitoring",
            "tracking signal bias",
        ],
        "explanation_patterns": [
            r"\bratio\s+of\s+(?:cumulative\s+forecast\s+error|rsfe)\s+to\s+(?:mean\s+absolute\s+deviation|mad)\b",
        ],
    },
    {
        "canonical_key": "akaike information criterion",
        "canonical_display": "Akaike Information Criterion",
        "prefer_canonical_display": True,
        "synonyms": [
            "aic",
            "aic metric",
            "akaike information criterion",
        ],
        "explanation_patterns": [],
    },
    {
        "canonical_key": "bayesian information criterion",
        "canonical_display": "Bayesian Information Criterion",
        "prefer_canonical_display": True,
        "synonyms": [
            "bic",
            "bic metric",
            "bayesian information criterion",
            "schwarz criterion",
        ],
        "explanation_patterns": [],
    },
    {
        "canonical_key": "ljung box test",
        "canonical_display": "Ljung-Box Test",
        "prefer_canonical_display": True,
        "synonyms": [
            "ljung box",
            "ljung box test",
            "ljung box q test",
            "ljung box statistic",
        ],
        "explanation_patterns": [],
    },
    {
        "canonical_key": "augment dickey fuller test",
        "canonical_display": "Augmented Dickey-Fuller Test",
        "prefer_canonical_display": True,
        "synonyms": [
            "adf",
            "adf test",
            "dickey fuller test",
            "augmented dickey fuller",
            "augmented dickey fuller test",
        ],
        "explanation_patterns": [],
    },
    {
        "canonical_key": "variance inflation factor",
        "canonical_display": "Variance Inflation Factor",
        "prefer_canonical_display": True,
        "synonyms": [
            "vif",
            "vif metric",
            "variance inflation factor",
        ],
        "explanation_patterns": [],
    },
]

# Precompute lookup maps from normalized/surface synonyms to canonical group
_SYNONYM_TO_GROUP: dict[str, dict[str, Any]] = {}
_GROUP_BY_KEY: dict[str, dict[str, Any]] = {}
for _grp in _CANONICAL_CONCEPT_GROUPS:
    _GROUP_BY_KEY[_grp["canonical_key"]] = _grp
    for _syn in _grp["synonyms"]:
        _surf = normalize_concept_surface(_syn)
        _stem = normalize_concept(_syn)
        if _surf:
            _SYNONYM_TO_GROUP[_surf] = _grp
        if _stem:
            _SYNONYM_TO_GROUP[_stem] = _grp


def _strip_generic_wrappers(tokens: list[str]) -> list[str]:
    """
    Remove non-distinguishing generic trailing wrapper words (like 'model', 'method', 'technique')
    and filler prepositions, provided at least one core token remains.
    """
    filtered = [t for t in tokens if t not in _STOPWORD_PREPOSITIONS]
    if not filtered:
        return tokens
    while len(filtered) > 1 and filtered[-1] in _GENERIC_WRAPPER_SUFFIXES:
        filtered = filtered[:-1]
    return filtered


def canonicalize_concept(raw_concept: str) -> dict[str, Any]:
    """
    Map a raw extracted keyword to its canonical concept representation.
    - Merges exact duplicates and semantic duplicates (e.g., 'TES model' and
      'Triple Exponential Smoothing' -> 'Triple Exponential Smoothing').
    - Does NOT merge genuinely different concepts.
    """
    cleaned_raw = str(raw_concept or "").strip()
    surf = normalize_concept_surface(cleaned_raw)
    stemmed = normalize_concept(cleaned_raw)
    if not surf or not stemmed:
        return {
            "canonical_key": "",
            "canonical_display": cleaned_raw,
            "prefer_canonical_display": False,
            "aliases": [],
            "explanation_patterns": [],
        }

    # 1. Direct lookup on surface or stemmed phrase
    matched_group = _SYNONYM_TO_GROUP.get(surf) or _SYNONYM_TO_GROUP.get(stemmed)

    # 2. Try expanding standalone acronym tokens and stripping generic wrapper suffixes ('model', 'method', etc.)
    if not matched_group:
        surf_tokens = surf.split()
        stripped_surf_tokens = _strip_generic_wrappers(surf_tokens)
        stripped_surf = " ".join(stripped_surf_tokens)
        stripped_stem = normalize_concept(stripped_surf)
        matched_group = _SYNONYM_TO_GROUP.get(stripped_surf) or _SYNONYM_TO_GROUP.get(stripped_stem)

        if not matched_group:
            expanded_tokens: list[str] = []
            for tok in stripped_surf_tokens:
                expanded_tokens.extend(_ACRONYM_TOKEN_EXPANSIONS.get(tok, tok).split())
            expanded_stripped = " ".join(_strip_generic_wrappers(expanded_tokens))
            expanded_stem = normalize_concept(expanded_stripped)
            matched_group = _SYNONYM_TO_GROUP.get(expanded_stripped) or _SYNONYM_TO_GROUP.get(expanded_stem)

    if matched_group:
        aliases = list(dict.fromkeys([surf, *matched_group["synonyms"]]))
        return {
            "canonical_key": matched_group["canonical_key"],
            "canonical_display": matched_group["canonical_display"],
            "prefer_canonical_display": bool(matched_group.get("prefer_canonical_display", False)),
            "aliases": aliases,
            "explanation_patterns": list(matched_group.get("explanation_patterns", [])),
        }

    # 3. Structural canonicalization for general phrases not in the curated dictionary:
    # Expand standalone acronyms, strip non-distinguishing wrapper words ('model', 'method', 'technique')
    # and prepositions ('of', 'in', 'the'), and stem tokens.
    surf_tokens = surf.split()
    expanded_tokens = []
    for tok in surf_tokens:
        expanded_tokens.extend(_ACRONYM_TOKEN_EXPANSIONS.get(tok, tok).split())
    core_tokens = _strip_generic_wrappers(expanded_tokens)
    core_stems = [_stem_token(t) for t in core_tokens if t]
    if not core_stems:
        core_stems = stemmed.split()

    # If the phrase used 'of' inversion (e.g., 'Decomposition of time series' vs 'Time series decomposition'),
    # sorting the core stems when 'of' is present or comparing core stem sets aligns equivalent reorderings.
    canonical_key = " ".join(sorted(core_stems)) if "of" in surf_tokens and len(core_stems) >= 2 else " ".join(core_stems)

    return {
        "canonical_key": canonical_key,
        "canonical_display": cleaned_raw,
        "prefer_canonical_display": False,
        "aliases": list(dict.fromkeys([surf, " ".join(core_tokens)])),
        "explanation_patterns": [],
    }


def _select_best_display_concept(
    existing_display: str,
    new_raw: str,
    canonical_info: dict[str, Any],
    occurrence_count: int,
) -> str:
    """
    Choose the clearest canonical concept label when entries are merged.
    When two or more entries merge (e.g., 'TES model' and 'Triple Exponential Smoothing'),
    prefer the canonical full name or the more descriptive surface phrase.
    """
    if canonical_info.get("prefer_canonical_display"):
        return canonical_info["canonical_display"]
    if occurrence_count > 1:
        if canonical_info.get("canonical_display") and len(canonical_info["canonical_display"]) > len(existing_display):
            return canonical_info["canonical_display"]
        clean_new = new_raw.strip(" .!?;:,")
        clean_existing = existing_display.strip(" .!?;:,")
        if len(clean_new) > len(clean_existing):
            return clean_new
    return existing_display


def build_concept_pool(source_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Combine extracted keywords from the 3 reference documents into a unified canonical concept pool.
    - Preserves all 15 original source entries with contributor attribution.
    - Identifies exact duplicates AND semantic duplicates (e.g., 'TES model' + 'Triple Exponential Smoothing')
      and merges them into canonical concepts.
    - Preserves original wording and records which entries were merged into each canonical concept.
    """
    distinct_map: dict[str, dict[str, Any]] = {}
    preserved_entries: list[dict[str, Any]] = []

    for entry in source_entries:
        raw_kw = str(entry.get("raw_keyword", "")).strip()
        if not raw_kw:
            continue
        norm_kw = normalize_concept(raw_kw)
        if not norm_kw:
            continue

        canonical_info = canonicalize_concept(raw_kw)
        canonical_key = canonical_info["canonical_key"] or norm_kw

        doc_id = entry.get("document_id")
        contrib_id = entry.get("contributor_id")
        contrib_name = str(entry.get("contributor_name") or "").strip()
        pos = entry.get("position", 0)

        surf = normalize_concept_surface(raw_kw)

        merged_entry_info = {
            "document_id": doc_id,
            "contributor_id": contrib_id,
            "contributor_name": contrib_name,
            "position": pos,
            "raw_keyword": raw_kw,
            "normalized_keyword": norm_kw,
        }

        if canonical_key not in distinct_map:
            initial_display = (
                canonical_info["canonical_display"]
                if canonical_info.get("prefer_canonical_display")
                and len(raw_kw.split()) <= 2
                and any(tok in _ACRONYM_TOKEN_EXPANSIONS for tok in surf.split())
                else raw_kw
            )
            aliases = list(dict.fromkeys([surf, *canonical_info.get("aliases", [])]))
            distinct_map[canonical_key] = {
                "normalized_concept": canonical_key,
                "canonical_concept": canonical_info["canonical_display"] or initial_display,
                "display_concept": initial_display,
                "surface_forms": [surf],
                "original_terms": [raw_kw],
                "aliases": aliases,
                "explanation_patterns": canonical_info.get("explanation_patterns", []),
                "merged_entries": [merged_entry_info],
                "is_merged": False,
                "source_document_ids": [doc_id] if doc_id is not None else [],
                "source_contributor_ids": [contrib_id] if contrib_id is not None else [],
                "source_contributor_names": [contrib_name] if contrib_name else [],
                "occurrence_count": 1,
            }
        else:
            existing = distinct_map[canonical_key]
            existing["occurrence_count"] += 1
            existing["is_merged"] = True
            existing["merged_entries"].append(merged_entry_info)
            if surf and surf not in existing["surface_forms"]:
                existing["surface_forms"].append(surf)
            if raw_kw not in existing["original_terms"]:
                existing["original_terms"].append(raw_kw)
            for al in canonical_info.get("aliases", []):
                if al and al not in existing["aliases"]:
                    existing["aliases"].append(al)
            for pat in canonical_info.get("explanation_patterns", []):
                if pat and pat not in existing["explanation_patterns"]:
                    existing["explanation_patterns"].append(pat)
            if doc_id is not None and doc_id not in existing["source_document_ids"]:
                existing["source_document_ids"].append(doc_id)
            if contrib_id is not None and contrib_id not in existing["source_contributor_ids"]:
                existing["source_contributor_ids"].append(contrib_id)
            if contrib_name and contrib_name not in existing["source_contributor_names"]:
                existing["source_contributor_names"].append(contrib_name)

            best_display = _select_best_display_concept(
                existing["display_concept"],
                raw_kw,
                canonical_info,
                existing["occurrence_count"],
            )
            existing["display_concept"] = best_display
            existing["canonical_concept"] = best_display

        item_record = {
            "document_id": doc_id,
            "contributor_id": contrib_id,
            "contributor_name": contrib_name,
            "position": pos,
            "raw_keyword": raw_kw,
            "normalized_keyword": norm_kw,
            "canonical_key": canonical_key,
            "canonical_concept": distinct_map[canonical_key]["display_concept"],
        }
        preserved_entries.append(item_record)

    # Refresh canonical_concept on preserved_entries after all merges are finalized
    for item_record in preserved_entries:
        ckey = item_record.get("canonical_key", "")
        if ckey in distinct_map:
            item_record["canonical_concept"] = distinct_map[ckey]["display_concept"]
            item_record["is_merged"] = distinct_map[ckey]["occurrence_count"] > 1

    distinct_concepts = list(distinct_map.values())
    total_raw = len(preserved_entries)
    distinct_count = len(distinct_concepts)
    duplicates = [c for c in distinct_concepts if c["occurrence_count"] > 1]
    has_duplicates = len(duplicates) > 0
    merged_entries_count = max(0, total_raw - distinct_count)

    warning = None
    if has_duplicates or distinct_count < 13:
        merged_summary = ", ".join(
            f"'{d['display_concept']}' ({d['occurrence_count']} entries: {', '.join(repr(t) for t in d.get('original_terms', []))})"
            for d in duplicates
        )
        if merged_summary:
            warning = (
                f"Normalized {total_raw} extracted entries into {distinct_count} distinct concepts "
                f"(canonical concepts merged: {merged_summary}). Scoring uses coverage out of {distinct_count} canonical concepts."
            )
        else:
            warning = (
                f"Combined pool has {distinct_count} distinct concepts (from {total_raw} extracted entries). "
                f"Scoring uses coverage out of {distinct_count} canonical concepts."
            )

    return {
        "total_raw_keywords": total_raw,
        "distinct_concepts_count": distinct_count,
        "canonical_concepts_count": distinct_count,
        "merged_entries_count": merged_entries_count,
        "has_duplicates": has_duplicates,
        "duplicate_concepts": duplicates,
        "distinct_concepts": distinct_concepts,
        "canonical_concepts": distinct_concepts,
        "source_entries": preserved_entries,
        "review_status": "pending_review",
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Student Reflection Understanding & Evidence Evaluation
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS = (
    r"\b(?:not|never|no|without|neither|nor|lack\s+of|lacks|lacking|ignores|ignoring|unrelated\s+to|instead\s+of)\s+(?:\w+\s+){0,2}"
)

# Explicit bare keyword-list prefixes (e.g., "Keywords: X, Y, Z" or "List of concepts: ...")
_BARE_LIST_HEADER_RE = re.compile(
    r"^\s*(?:keywords?|key\s+terms?|concepts?|terms?|topics?|list\s+of\s+(?:keywords?|concepts?|terms?))\s*[:\-=]",
    re.IGNORECASE,
)

# Meta-naming clauses that only name keywords without explaining them
_META_NAMING_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?(?:keywords?|concepts?|terms?|topics?)\s+(?:in\s+this\s+(?:session|lesson|class|report)\s+)?"
    r"(?:are|is|were|include|includes|included|consist\s+of)\b",
    re.IGNORECASE,
)

# Common English & analytical verbs/predicates indicating a real clause rather than a bare noun list
_EXPLANATORY_PREDICATE_WORDS: set[str] = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "can", "could", "will", "would", "should", "may", "might", "must",
    "do", "does", "did",
    "help", "helps", "helped", "helping",
    "use", "uses", "used", "using",
    "apply", "applies", "applied", "applying",
    "capture", "captures", "captured", "capturing",
    "measure", "measures", "measured", "measuring",
    "predict", "predicts", "predicted", "predicting",
    "forecast", "forecasts", "forecasted",
    "smooth", "smooths", "smoothed", "smoothing",
    "reduce", "reduces", "reduced", "reducing",
    "improve", "improves", "improved", "improving",
    "estimate", "estimates", "estimated", "estimating",
    "calculate", "calculates", "calculated", "calculating",
    "compute", "computes", "computed", "computing",
    "model", "models", "modeled", "modeling",
    "account", "accounts", "accounted", "accounting",
    "adjust", "adjusts", "adjusted", "adjusting",
    "remove", "removes", "removed", "removing",
    "separate", "separates", "separated", "separating",
    "decompose", "decomposes", "decomposed", "decomposing",
    "track", "tracks", "tracked",
    "detect", "detects", "detected", "detecting",
    "evaluate", "evaluates", "evaluated", "evaluating",
    "quantify", "quantifies", "quantified", "quantifying",
    "minimize", "minimizes", "minimized", "minimizing",
    "minimise", "minimises", "minimised",
    "penalize", "penalizes", "penalized", "penalizing",
    "weight", "weights", "weighted", "weighting",
    "assign", "assigns", "assigned", "assigning",
    "handle", "handles", "handled", "handling",
    "address", "addresses", "addressed", "addressing",
    "show", "shows", "showed", "showing",
    "indicate", "indicates", "indicated", "indicating",
    "reflect", "reflects", "reflected", "reflecting",
    "allow", "allows", "allowed", "allowing",
    "enable", "enables", "enabled", "enabling",
    "provide", "provides", "provided", "providing",
    "compare", "compares", "compared", "comparing",
    "require", "requires", "required", "requiring",
    "depend", "depends", "depended", "depending",
    "guide", "guides", "guided", "guiding",
    "support", "supports", "supported", "supporting",
    "explain", "explains", "explained", "explaining",
    "demonstrate", "demonstrates", "demonstrated", "demonstrating",
    "incorporate", "incorporates", "incorporated", "incorporating",
    "combine", "combines", "combined", "combining",
    "control", "controls", "controlled", "controlling",
    "test", "tests", "tested",
    "check", "checks", "checked", "checking",
    "validate", "validates", "validated", "validating",
    "divide", "divides", "divided", "dividing",
    "subtract", "subtracts", "subtracted", "subtracting",
    "multiply", "multiplies", "multiplied", "multiplying",
    "average", "averages", "averaged", "averaging",
    "correlate", "correlates", "correlated", "correlating",
    "differ", "differs", "differed", "differencing",
    "stabilize", "stabilizes", "stabilized", "stabilizing",
    "discuss", "discusses", "discussed", "discussing",
    "analyze", "analyzes", "analyzed", "analyzing",
    "assess", "assesses", "assessed", "assessing",
    "determine", "determines", "determined", "determining",
    "identify", "identifies", "identified", "identifying",
    "represent", "represents", "represented", "representing",
    "describe", "describes", "described", "describing",
    "yield", "yields", "yielded", "yielding",
    "produce", "produces", "produced", "producing",
    "generate", "generates", "generated", "generating",
    "prevent", "prevents", "prevented", "preventing",
    "avoid", "avoids", "avoided", "avoiding",
    "shape", "shapes", "shaped", "shaping",
    "extend", "extends", "extended", "extending",
    "adapt", "adapts", "adapted", "adapting",
    "filter", "filters", "filtered", "filtering",
    "project", "projects", "projected", "projecting",
    "extrapolate", "extrapolates", "extrapolated", "extrapolating",
    "monitor", "monitors", "monitored", "monitoring",
    "signal", "signals", "signaled",
    "because", "since", "where", "which", "when", "by", "through", "while", "whereas",
}

_TRIVIAL_META_WORDS: set[str] = {
    "i", "we", "you", "they", "he", "she", "it",
    "the", "a", "an", "and", "or", "but", "so", "also", "too",
    "in", "on", "at", "to", "for", "of", "with", "from", "about", "as", "into",
    "this", "that", "these", "those", "my", "our", "their", "its",
    "keyword", "keywords", "concept", "concepts", "term", "terms", "topic", "topics",
    "list", "listed", "listing", "mention", "mentioned", "mentions",
    "name", "named", "names", "include", "includes", "included",
    "learn", "learned", "learnt", "study", "studied", "cover", "covered", "covers",
    "know", "knew", "saw", "see", "read", "today", "session", "lesson", "class", "report", "course",
    "is", "are", "was", "were", "be", "been",
}


def _is_negated_in_text(normalized_reflection_raw: str, phrase_pattern: str) -> bool:
    """Check if every occurrence of the matched phrase in the reflection is preceded by negation."""
    matches = list(re.finditer(phrase_pattern, normalized_reflection_raw))
    if not matches:
        return False
    non_negated_found = False
    for m in matches:
        start = m.start()
        prefix = normalized_reflection_raw[max(0, start - 45):start]
        if not re.search(_NEGATION_PATTERNS + r"$", prefix):
            non_negated_found = True
            break
    return not non_negated_found


def _split_into_sentences(reflection_text: str) -> list[str]:
    """Split reflection text into individual sentences or line items."""
    if not reflection_text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", reflection_text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _sentence_demonstrates_understanding(
    sentence: str,
    matched_phrase_norm: str,
    all_pool_phrases_norm: list[str] | None = None,
) -> bool:
    """
    Determine whether a sentence containing a concept demonstrates conceptual understanding
    rather than merely listing keywords without context.
    Rejects:
      - Explicit keyword lists ("Keywords: TES model, MAPE, ARIMA")
      - Bare comma/bullet/numbered lists of terms with no predicate verb
      - Trivial meta-naming statements ("I learned about TES model, MAPE, and ARIMA today.")
        that contain no explanatory context after stripping listed concept names and meta words.
    """
    raw_sent = sentence.strip()
    if not raw_sent:
        return False

    # Reject explicit "Keywords: ..." or "The keywords are ..." headers
    if _BARE_LIST_HEADER_RE.search(raw_sent):
        return False

    sent_norm = normalize_text_tokens(raw_sent)
    if not sent_norm:
        return False

    words = sent_norm.split()
    # A bare 1–3 word fragment cannot explain a concept
    if len(words) < 4:
        return False

    # Check if there is at least one predicate/verb or relational marker in the sentence
    has_predicate = any(w in _EXPLANATORY_PREDICATE_WORDS for w in words)
    if not has_predicate:
        return False

    # Strip out the matched concept phrase (and any other pool concept phrases if this sentence
    # lists multiple concepts) to inspect the remaining explanatory words in the sentence.
    remainder = f" {sent_norm} "
    phrases_to_strip = [matched_phrase_norm]
    if all_pool_phrases_norm:
        phrases_to_strip.extend(all_pool_phrases_norm)
    # Sort longest first so multi-word concepts strip cleanly
    for ph in sorted(set(p for p in phrases_to_strip if p), key=len, reverse=True):
        remainder = re.sub(r"\b" + r"\s+".join(re.escape(t) for t in ph.split()) + r"\b", " ", remainder)

    rem_tokens = [t for t in remainder.split() if t and not t.isdigit()]
    if not rem_tokens:
        return False

    # If the sentence starts with "The concepts/keywords are ..." and has no explanatory clause
    if _META_NAMING_PREFIX_RE.search(raw_sent):
        meaningful_after_meta = [t for t in rem_tokens if t not in _TRIVIAL_META_WORDS]
        if len(meaningful_after_meta) < 2:
            return False

    # Filter out trivial meta-naming words ("I learned about ... in class today")
    meaningful_context = [t for t in rem_tokens if t not in _TRIVIAL_META_WORDS]
    if len(meaningful_context) < 1:
        return False

    # Check if the sentence contains a genuine analytical/action verb (beyond bare copulas like 'is'/'are')
    copula_aux = {
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "can", "could", "will", "would", "should", "may", "might", "must",
        "do", "does", "did",
    }
    has_action_verb = any(w in (_EXPLANATORY_PREDICATE_WORDS - copula_aux) for w in words)

    # If the sentence has 2+ commas/semicolons (a list structure) and NO analytical action verb,
    # require >= 2 substantive context words outside the listed keywords
    comma_count = raw_sent.count(",") + raw_sent.count(";")
    if comma_count >= 2 and not has_action_verb and len(meaningful_context) < 2:
        return False

    return True


def _candidate_phrases_for_concept(concept_entry: dict[str, Any]) -> list[str]:
    """Return all normalized surface phrases and aliases that represent a canonical concept."""
    display_concept = concept_entry.get("display_concept", "")
    canonical_concept = concept_entry.get("canonical_concept", "") or display_concept
    canonical_info = canonicalize_concept(canonical_concept)

    candidates: list[str] = []
    for raw_val in [
        display_concept,
        canonical_concept,
        *(concept_entry.get("surface_forms") or []),
        *(concept_entry.get("original_terms") or []),
        *(concept_entry.get("aliases") or []),
        *canonical_info.get("aliases", []),
    ]:
        surf = normalize_concept_surface(str(raw_val))
        if surf and surf not in candidates:
            candidates.append(surf)
    return candidates


def _linguistic_match_concept(
    reflection_text: str,
    concept_entry: dict[str, Any],
    all_pool_phrases_norm: list[str] | None = None,
) -> tuple[bool, str, bool]:
    """
    Deterministic word-boundary, alias, inflection, and explanation-aware matching.
    Returns (is_matched, supporting_evidence_phrase, appeared_only_as_bare_keyword).
    - Accepts canonical concept names, merged contributor aliases/acronyms, and equivalent explanations.
    - Rejects negated occurrences.
    - Rejects bare keyword listings that do not demonstrate understanding.
    - Extracts the exact supporting sentence/phrase from the student reflection as evidence.
    """
    if not reflection_text or not reflection_text.strip():
        return False, "", False

    sentences = _split_into_sentences(reflection_text)
    if not sentences:
        return False, "", False

    candidates = _candidate_phrases_for_concept(concept_entry)
    explanation_patterns = list(concept_entry.get("explanation_patterns") or [])
    if not explanation_patterns:
        canonical_info = canonicalize_concept(
            concept_entry.get("canonical_concept") or concept_entry.get("display_concept", "")
        )
        explanation_patterns = canonical_info.get("explanation_patterns", [])

    found_in_bare_list = False

    for sent in sentences:
        sent_surface = normalize_text_tokens(sent)
        if not sent_surface:
            continue
        sent_stems = [_stem_token(w) for w in sent_surface.split()]
        sent_stem_str = " ".join(sent_stems)

        # 1. Check exact surface candidates & aliases with word boundaries
        for cand in candidates:
            cand_tokens = cand.split()
            if not cand_tokens:
                continue
            pattern = r"\b" + r"\s+".join(re.escape(tok) for tok in cand_tokens) + r"\b"
            if re.search(pattern, sent_surface):
                if _is_negated_in_text(sent_surface, pattern):
                    continue
                if _sentence_demonstrates_understanding(sent, cand, all_pool_phrases_norm):
                    return True, sent.strip(), False
                found_in_bare_list = True

            # 2. Check stemmed word-boundary phrase match (plurals/inflections)
            cand_stems = [_stem_token(t) for t in cand_tokens]
            stem_pattern = r"\b" + r"\s+".join(re.escape(tok) for tok in cand_stems) + r"\b"
            if re.search(stem_pattern, sent_stem_str):
                if _is_negated_in_text(sent_stem_str, stem_pattern):
                    continue
                if _sentence_demonstrates_understanding(sent, " ".join(cand_tokens), all_pool_phrases_norm):
                    return True, sent.strip(), False
                found_in_bare_list = True

        # 3. Check conceptual explanation patterns (when student explains the concept in equivalent words)
        for exp_pat in explanation_patterns:
            if re.search(exp_pat, sent_surface, flags=re.IGNORECASE):
                if not _is_negated_in_text(sent_surface, exp_pat):
                    if _sentence_demonstrates_understanding(sent, "", all_pool_phrases_norm):
                        return True, sent.strip(), False

        # 4. Multi-word concept non-adjacent clause match (for 2-4 content words in same clause)
        primary_norm = concept_entry.get("normalized_concept", "") or normalize_concept(
            concept_entry.get("display_concept", "")
        )
        concept_stems = primary_norm.split()
        if len(concept_stems) >= 2:
            clauses = re.split(r"[;,:]+", sent)
            for clause in clauses:
                clause_surface = normalize_text_tokens(clause)
                if not clause_surface:
                    continue
                clause_stems = [_stem_token(w) for w in clause_surface.split()]
                clause_stem_set = set(clause_stems)
                if all(cs in clause_stem_set for cs in concept_stems):
                    negated = False
                    for cs in concept_stems:
                        p = r"\b" + re.escape(cs) + r"\b"
                        if _is_negated_in_text(" ".join(clause_stems), p):
                            negated = True
                            break
                    if not negated:
                        if _sentence_demonstrates_understanding(sent, " ".join(concept_stems), all_pool_phrases_norm):
                            return True, sent.strip(), False
                        found_in_bare_list = True

    return False, "", found_in_bare_list


def evaluate_c4_reflection(
    reflection_text: str,
    pool_data: dict[str, Any],
    ai_concept_matches: list[dict[str, Any]] | None = None,
    ai_feedback: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate a student reflection against the session's canonical reference concept pool.
    - Compares the student's reflection with every canonical concept in the pool.
    - Awards a maximum of 1 mark per canonical concept (never double-counting repeated mentions
      or synonym variations).
    - Accepts correct explanations and equivalent terminology.
    - Does NOT award marks for merely listing a keyword without demonstrating understanding.
    - Stores the supporting evidence phrase from the student reflection for every matched concept.
    - Displays matched and unmatched concepts separately.
    - Calculates Coverage = (Matched canonical concepts / Total canonical concepts) * 100
      and maps to the 5-point rating band safely (even when Total canonical concepts == 0).
    """
    raw_distinct_concepts: list[dict[str, Any]] = pool_data.get("distinct_concepts", [])

    # Ensure canonical pool entries are normalized and indexed without duplicates
    pool_by_norm: dict[str, dict[str, Any]] = {}
    alias_to_norm_key: dict[str, str] = {}

    for c in raw_distinct_concepts:
        display_name = c.get("display_concept") or c.get("canonical_concept", "")
        canonical_info = canonicalize_concept(display_name)
        norm_key = c.get("normalized_concept") or canonical_info["canonical_key"] or normalize_concept(display_name)
        if not norm_key:
            continue
        if norm_key not in pool_by_norm:
            pool_by_norm[norm_key] = c

        # Index all surface forms, original merged terms, and canonical aliases to norm_key
        for candidate in [
            norm_key,
            normalize_concept(display_name),
            normalize_concept(c.get("canonical_concept", "")),
            *(normalize_concept(sf) for sf in (c.get("surface_forms") or [])),
            *(normalize_concept(ot) for ot in (c.get("original_terms") or [])),
            *(normalize_concept(al) for al in (c.get("aliases") or [])),
            *(normalize_concept(al) for al in canonical_info.get("aliases", [])),
        ]:
            if candidate:
                alias_to_norm_key[candidate] = norm_key

    # Collect all normalized surface phrases in the pool for bare-list detection
    all_pool_phrases_norm: list[str] = []
    for c_entry in pool_by_norm.values():
        for cand in _candidate_phrases_for_concept(c_entry):
            if cand not in all_pool_phrases_norm:
                all_pool_phrases_norm.append(cand)

    # Index validated AI results if provided (mapping any alias/surface form to its canonical key)
    ai_map: dict[str, dict[str, Any]] = {}
    if ai_concept_matches:
        for item in ai_concept_matches:
            raw_c = str(item.get("concept", "")).strip()
            norm_c = normalize_concept(raw_c)
            canonical_c = canonicalize_concept(raw_c)["canonical_key"]
            resolved_key = (
                alias_to_norm_key.get(norm_c)
                or alias_to_norm_key.get(canonical_c)
                or (norm_c if norm_c in pool_by_norm else None)
            )
            if resolved_key and resolved_key in pool_by_norm:
                # Keep 'recalled' if multiple AI items mapped to the same canonical concept
                existing_ai = ai_map.get(resolved_key)
                if not existing_ai or str(item.get("status", "")).lower() in ("recalled", "matched"):
                    ai_map[resolved_key] = item

    matched_concepts: list[dict[str, Any]] = []
    unmatched_concepts: list[dict[str, Any]] = []
    concept_results_for_ui: list[dict[str, Any]] = []

    sentences = _split_into_sentences(reflection_text)
    fallback_sentence = sentences[0] if sentences else (reflection_text.strip()[:160] if reflection_text else "")

    for norm_key, c_entry in pool_by_norm.items():
        display_name = c_entry.get("canonical_concept") or c_entry["display_concept"]
        ling_matched, ling_evidence, bare_keyword_only = _linguistic_match_concept(
            reflection_text,
            c_entry,
            all_pool_phrases_norm=all_pool_phrases_norm,
        )
        ai_info = ai_map.get(norm_key)

        is_matched = False
        evidence = ""
        explanation = ""

        if ai_info:
            ai_status = str(ai_info.get("status", "")).lower()
            ai_ev = str(ai_info.get("evidence", "")).strip()
            ai_exp = str(ai_info.get("explanation", "")).strip()
            # Verify AI evidence exists in reflection if provided
            if ai_ev and normalize_text_tokens(ai_ev) not in normalize_text_tokens(reflection_text):
                ai_ev = ""

            if ai_status in ("misconception", "keyword_only", "listed_only"):
                is_matched = False
                explanation = ai_exp or (
                    f"Concept '{display_name}' was listed without demonstrating understanding."
                    if ai_status in ("keyword_only", "listed_only")
                    else f"Misconception or negated usage detected for '{display_name}'."
                )
            elif bare_keyword_only and not ling_matched:
                # Explicitly reject bare keyword listing even if AI marked it recalled
                is_matched = False
                explanation = (
                    f"Concept '{display_name}' was mentioned only in a bare keyword list without "
                    f"demonstrating conceptual understanding."
                )
            elif ai_status in ("recalled", "matched"):
                # Check none of the candidate phrases are negated in reflection
                candidates = _candidate_phrases_for_concept(c_entry)
                norm_refl = normalize_text_tokens(reflection_text)
                negated_any = False
                for cand in candidates:
                    surf_pat = r"\b" + r"\s+".join(re.escape(t) for t in cand.split()) + r"\b"
                    if re.search(surf_pat, norm_refl) and _is_negated_in_text(norm_refl, surf_pat):
                        negated_any = True
                        break
                if negated_any and not ling_matched:
                    is_matched = False
                    explanation = f"Negated usage of '{display_name}' does not count as a valid match."
                else:
                    is_matched = True
                    evidence = ai_ev or ling_evidence or fallback_sentence
                    explanation = ai_exp or f"Demonstrated understanding of '{display_name}' (1/1 mark)."
            elif ling_matched:
                is_matched = True
                evidence = ling_evidence or ai_ev or fallback_sentence
                explanation = ai_exp or f"Demonstrated understanding of '{display_name}' in reflection (1/1 mark)."
            else:
                is_matched = False
                explanation = ai_exp or f"Concept '{display_name}' was not demonstrated in the reflection."
        else:
            if ling_matched:
                is_matched = True
                evidence = ling_evidence or fallback_sentence
                explanation = f"Demonstrated understanding of '{display_name}' in reflection (1/1 mark)."
            elif bare_keyword_only:
                is_matched = False
                explanation = (
                    f"Concept '{display_name}' was only listed as a keyword without demonstrating understanding (0/1 mark)."
                )
            else:
                is_matched = False
                explanation = f"Concept '{display_name}' was not demonstrated in the reflection (0/1 mark)."

        record = {
            "concept": display_name,
            "canonical_concept": display_name,
            "normalized_concept": norm_key,
            "original_terms": c_entry.get("original_terms") or [c_entry.get("display_concept", display_name)],
            "merged_entries": c_entry.get("merged_entries", []),
            "source_document_ids": c_entry.get("source_document_ids", []),
            "source_contributor_ids": c_entry.get("source_contributor_ids", []),
            "source_contributor_names": c_entry.get("source_contributor_names", []),
            "marks_awarded": 1 if is_matched else 0,
            "max_marks": 1,
            "evidence": evidence if is_matched else "",
            "explanation": explanation,
        }

        if is_matched:
            matched_concepts.append(record)
            concept_results_for_ui.append({
                "concept": display_name,
                "status": "recalled",
                "evidence": evidence,
                "explanation": explanation,
            })
        else:
            unmatched_concepts.append(record)
            concept_results_for_ui.append({
                "concept": display_name,
                "status": "missing",
                "evidence": "",
                "explanation": explanation,
            })

    total_distinct = len(pool_by_norm)
    matched_count = len(matched_concepts)
    coverage_pct = calculate_c4_coverage(matched_count, total_distinct)
    c4_rating = calculate_c4_rating(matched_count, total_distinct)

    default_explanation = (
        f"Matched {matched_count} of {total_distinct} canonical concepts "
        f"(Coverage: {coverage_pct}%, C4 Rating: {c4_rating}/5)."
    )
    final_explanation = (
        f"{ai_feedback.strip()} ({default_explanation})"
        if (ai_feedback and ai_feedback.strip())
        else default_explanation
    )

    return {
        "total_raw_keywords": pool_data.get("total_raw_keywords", 15),
        "total_distinct_concepts": total_distinct,
        "total_canonical_concepts": total_distinct,
        "matched_count": matched_count,
        "c4_rating": c4_rating,
        "coverage_percentage": coverage_pct,
        "recall_percentage": coverage_pct,
        "matched_concepts": matched_concepts,
        "unmatched_concepts": unmatched_concepts,
        "concept_results": concept_results_for_ui,
        "missing_concepts": [u["concept"] for u in unmatched_concepts],
        "explanation": final_explanation,
    }
