import re
from typing import Any


PROMPT_VERSION = "c4-v1.0"


def calculate_c4_rating(matched_count: int) -> int:
    """
    Deterministic C4 rating calculation based on distinct matched reference concepts.
    Thresholds (inclusive):
      13-15 -> 5
      11-12 -> 4
      8-10  -> 3
      5-7   -> 2
      0-4   -> 1
    """
    if not isinstance(matched_count, int) or isinstance(matched_count, bool):
        raise ValueError("matched_count must be an integer")
    if matched_count < 0 or matched_count > 15:
        raise ValueError("matched_count must be between 0 and 15")
    if matched_count >= 13:
        return 5
    if matched_count >= 11:
        return 4
    if matched_count >= 8:
        return 3
    if matched_count >= 5:
        return 2
    return 1


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
    """Normalize surface form (preserving exact stemmed words vs unstemmed display)."""
    norm = normalize_text_tokens(concept)
    return re.sub(r"^(the|a|an)\s+", "", norm)


def build_concept_pool(source_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Combine extracted keywords from the 3 reference documents into a unified pool.
    Preserves all 15 source entries with provenance while deduplicating equivalent concepts
    for C4 scoring.
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
        doc_id = entry.get("document_id")
        contrib_id = entry.get("contributor_id")
        pos = entry.get("position", 0)

        item_record = {
            "document_id": doc_id,
            "contributor_id": contrib_id,
            "position": pos,
            "raw_keyword": raw_kw,
            "normalized_keyword": norm_kw,
        }
        preserved_entries.append(item_record)

        if norm_kw not in distinct_map:
            distinct_map[norm_kw] = {
                "normalized_concept": norm_kw,
                "display_concept": raw_kw,
                "surface_forms": [normalize_concept_surface(raw_kw)],
                "source_document_ids": [doc_id] if doc_id is not None else [],
                "source_contributor_ids": [contrib_id] if contrib_id is not None else [],
                "occurrence_count": 1,
            }
        else:
            existing = distinct_map[norm_kw]
            existing["occurrence_count"] += 1
            surf = normalize_concept_surface(raw_kw)
            if surf not in existing["surface_forms"]:
                existing["surface_forms"].append(surf)
            if doc_id is not None and doc_id not in existing["source_document_ids"]:
                existing["source_document_ids"].append(doc_id)
            if contrib_id is not None and contrib_id not in existing["source_contributor_ids"]:
                existing["source_contributor_ids"].append(contrib_id)

    distinct_concepts = list(distinct_map.values())
    total_raw = len(preserved_entries)
    distinct_count = len(distinct_concepts)
    duplicates = [c for c in distinct_concepts if c["occurrence_count"] > 1]
    has_duplicates = len(duplicates) > 0

    warning = None
    if distinct_count < 13:
        warning = (
            f"Combined pool has {distinct_count} distinct concepts (from {total_raw} extracted entries) "
            f"due to cross-document duplicates. Maximum achievable concept match count is {distinct_count}."
        )

    return {
        "total_raw_keywords": total_raw,
        "distinct_concepts_count": distinct_count,
        "has_duplicates": has_duplicates,
        "duplicate_concepts": duplicates,
        "distinct_concepts": distinct_concepts,
        "source_entries": preserved_entries,
        "warning": warning,
    }


_NEGATION_PATTERNS = (
    r"\b(?:not|never|no|without|neither|nor|lack\s+of|lacks|lacking|ignores|ignoring|unrelated\s+to|instead\s+of)\s+(?:\w+\s+){0,2}"
)


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


def _linguistic_match_concept(reflection_text: str, concept_entry: dict[str, Any]) -> tuple[bool, str]:
    """
    Deterministic word-boundary and inflection-aware phrase matching.
    Avoids matching a short keyword as an arbitrary substring inside an unrelated word.
    Avoids counting negated usage.
    """
    norm_reflection_surface = normalize_text_tokens(reflection_text)
    if not norm_reflection_surface:
        return False, ""

    reflection_words = norm_reflection_surface.split()
    reflection_stems = [_stem_token(w) for w in reflection_words]
    stemmed_reflection_str = " ".join(reflection_stems)

    display_concept = concept_entry.get("display_concept", "")
    norm_concept = concept_entry.get("normalized_concept", "") or normalize_concept(display_concept)
    if not norm_concept:
        return False, ""

    concept_stems = norm_concept.split()
    if not concept_stems:
        return False, ""

    # 1. Exact word-boundary phrase match on surface forms
    surface_forms = concept_entry.get("surface_forms") or [normalize_concept_surface(display_concept)]
    for surf in surface_forms:
        if not surf:
            continue
        pattern = r"\b" + r"\s+".join(re.escape(tok) for tok in surf.split()) + r"\b"
        if re.search(pattern, norm_reflection_surface):
            if _is_negated_in_text(norm_reflection_surface, pattern):
                return False, ""
            return True, display_concept

    # 2. Stemmed word-boundary phrase match (handles plurals/inflections reliably)
    stem_pattern = r"\b" + r"\s+".join(re.escape(tok) for tok in concept_stems) + r"\b"
    if re.search(stem_pattern, stemmed_reflection_str):
        if _is_negated_in_text(stemmed_reflection_str, stem_pattern):
            return False, ""
        return True, display_concept

    # 3. Multi-word concept non-adjacent clause match (for 2-4 content words in same sentence/clause)
    if len(concept_stems) >= 2:
        sentences = re.split(r"[.!?;,]+", reflection_text)
        for sent in sentences:
            sent_surface = normalize_text_tokens(sent)
            if not sent_surface:
                continue
            sent_stems = [_stem_token(w) for w in sent_surface.split()]
            sent_stem_set = set(sent_stems)
            if all(cs in sent_stem_set for cs in concept_stems):
                # Ensure none of the concept stems are preceded by negation in this clause
                negated = False
                for cs in concept_stems:
                    p = r"\b" + re.escape(cs) + r"\b"
                    if _is_negated_in_text(" ".join(sent_stems), p):
                        negated = True
                        break
                if not negated:
                    return True, sent.strip()[:120]

    return False, ""


def evaluate_c4_reflection(
    reflection_text: str,
    pool_data: dict[str, Any],
    ai_concept_matches: list[dict[str, Any]] | None = None,
    ai_feedback: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate a student reflection against the session's distinct reference concept pool.
    - Validates any AI-reported matches strictly against the stored distinct pool.
    - Combines validated AI semantic matches (when non-negated/valid) with deterministic
      linguistic phrase/inflection matches.
    - Counts each distinct reference concept at most once.
    - Calculates the final C4 rating deterministically via calculate_c4_rating().
    """
    distinct_concepts: list[dict[str, Any]] = pool_data.get("distinct_concepts", [])

    # Build lookup from normalized_concept and display_concept to canonical concept entry
    pool_by_norm: dict[str, dict[str, Any]] = {}
    for c in distinct_concepts:
        norm_key = c.get("normalized_concept") or normalize_concept(c.get("display_concept", ""))
        if norm_key:
            pool_by_norm[norm_key] = c

    # Index validated AI results if provided
    ai_map: dict[str, dict[str, Any]] = {}
    if ai_concept_matches:
        for item in ai_concept_matches:
            raw_c = str(item.get("concept", "")).strip()
            norm_c = normalize_concept(raw_c)
            if norm_c in pool_by_norm:
                ai_map[norm_c] = item

    matched_concepts: list[dict[str, Any]] = []
    unmatched_concepts: list[dict[str, Any]] = []
    concept_results_for_ui: list[dict[str, Any]] = []

    for norm_key, c_entry in pool_by_norm.items():
        display_name = c_entry["display_concept"]
        ling_matched, ling_evidence = _linguistic_match_concept(reflection_text, c_entry)
        ai_info = ai_map.get(norm_key)

        is_matched = False
        evidence = ""
        explanation = ""

        if ai_info:
            ai_status = str(ai_info.get("status", "")).lower()
            ai_ev = str(ai_info.get("evidence", "")).strip()
            ai_exp = str(ai_info.get("explanation", "")).strip()
            # Verify evidence actually exists in reflection if provided
            if ai_ev and normalize_text_tokens(ai_ev) not in normalize_text_tokens(reflection_text):
                ai_ev = ""

            if ai_status == "misconception":
                # Explicitly flagged as misconception / negated usage by AI
                is_matched = False
                explanation = ai_exp or f"Misconception or negated usage detected for '{display_name}'."
            elif ai_status in ("recalled", "matched"):
                # Verify it is not negated in reflection
                surf_pat = r"\b" + r"\s+".join(re.escape(t) for t in normalize_concept_surface(display_name).split()) + r"\b"
                if _is_negated_in_text(normalize_text_tokens(reflection_text), surf_pat):
                    is_matched = False
                    explanation = f"Negated usage of '{display_name}' does not count as a valid match."
                else:
                    is_matched = True
                    evidence = ai_ev or ling_evidence
                    explanation = ai_exp or f"Concept '{display_name}' matched in reflection."
            elif ling_matched:
                is_matched = True
                evidence = ling_evidence
                explanation = ai_exp or f"Concept '{display_name}' matched deterministically in reflection."
            else:
                is_matched = False
                explanation = ai_exp or f"Concept '{display_name}' was not present in the reflection."
        else:
            if ling_matched:
                is_matched = True
                evidence = ling_evidence
                explanation = f"Concept '{display_name}' matched in reflection."
            else:
                is_matched = False
                explanation = f"Concept '{display_name}' was not found in the reflection."

        record = {
            "concept": display_name,
            "normalized_concept": norm_key,
            "source_document_ids": c_entry.get("source_document_ids", []),
            "evidence": evidence,
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
    c4_rating = calculate_c4_rating(matched_count)
    recall_pct = round((matched_count / total_distinct) * 100.0, 2) if total_distinct > 0 else 0.0

    default_explanation = (
        f"Matched {matched_count} of {total_distinct} distinct reference concepts "
        f"(C4 Rating: {c4_rating}/5)."
    )
    final_explanation = f"{ai_feedback.strip()} ({default_explanation})" if (ai_feedback and ai_feedback.strip()) else default_explanation

    return {
        "total_distinct_concepts": total_distinct,
        "matched_count": matched_count,
        "c4_rating": c4_rating,
        "recall_percentage": recall_pct,
        "matched_concepts": matched_concepts,
        "unmatched_concepts": unmatched_concepts,
        "concept_results": concept_results_for_ui,
        "missing_concepts": [u["concept"] for u in unmatched_concepts],
        "explanation": final_explanation,
    }
