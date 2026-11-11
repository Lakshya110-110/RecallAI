import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def compare_text(reference: str, submission: str) -> dict:
    reference_normalized = normalize(reference)
    submission_normalized = normalize(submission)
    if not reference_normalized or not submission_normalized:
        return {"percentage": 0.0, "matches": []}
    matrix = TfidfVectorizer().fit_transform([reference_normalized, submission_normalized])
    score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100)
    reference_terms = set(re.findall(r"\b[\w']+\b", reference_normalized))
    submission_terms = set(re.findall(r"\b[\w']+\b", submission_normalized))
    return {"percentage": round(score, 2), "matches": sorted(reference_terms & submission_terms)[:20]}