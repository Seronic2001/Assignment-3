# ── Recommender Component ──────────────────────────────────
# Calls the FastAPI backend or runs inline if the API is down.

import requests
import streamlit as st
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_URL = "http://localhost:8000"
DB_PATH = Path(__file__).parent.parent.parent / "backend" / "recipe_db.json"
VEC_PATH = Path(__file__).parent.parent.parent / "backend" / "vectorizer.pkl"
MAT_PATH = Path(__file__).parent.parent.parent / "backend" / "tfidf_matrix.pkl"


# ── Inline fallback engine (no API needed) ─────────────────
@st.cache_resource(show_spinner="Loading recipe database…")
def _load_inline_db():
    if not DB_PATH.exists():
        return [], None, None
    with open(DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)
    if VEC_PATH.exists() and MAT_PATH.exists():
        with open(VEC_PATH, "rb") as f:
            vec = pickle.load(f)
        with open(MAT_PATH, "rb") as f:
            mat = pickle.load(f)
    else:
        vec = TfidfVectorizer()
        corpus = [" ".join(r.get("ingredients", [])) for r in db]
        mat = vec.fit_transform(corpus)
    return db, vec, mat


def _jaccard(user_set, recipe_set):
    if not recipe_set:
        return 0.0
    return len(user_set & recipe_set) / len(recipe_set)


def _cosine_score(user_set, vec, mat, idx):
    if vec is None:
        return 0.0
    user_text = " ".join(user_set)
    user_vec = vec.transform([user_text])
    return float(cosine_similarity(user_vec, mat[idx])[0][0])


def _inline_recommend(ingredients: list[str], pantry: list[str], diet: str | None, top_k: int):
    db, vec, mat = _load_inline_db()
    if not db:
        return []

    user_set = {i.lower().strip() for i in ingredients + pantry if i.strip()}
    results = []

    for idx, recipe in enumerate(db):
        if diet and diet.lower() not in recipe.get("diet", "").lower():
            continue
        recipe_set = {i.lower().strip() for i in recipe.get("ingredients", [])}
        score = 0.60 * _jaccard(user_set, recipe_set) + 0.40 * _cosine_score(user_set, vec, mat, idx)
        if score < 0.35:
            continue
        matched = user_set & recipe_set
        missing = list(recipe_set - user_set)
        results.append({
            "recipe": recipe,
            "score": round(score, 4),
            "matched_count": len(matched),
            "total_count": len(recipe_set),
            "missing_ingredients": missing[:6],
            "why": f"You have {len(matched)} of {len(recipe_set)} ingredients ({int(len(matched)/max(len(recipe_set),1)*100)}% coverage)",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ── Public interface ───────────────────────────────────────
def get_recommendations(
    scanned: list[str],
    pantry: list[str],
    diet: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Try FastAPI backend first; fall back to inline engine.
    Returns list of result dicts.
    """
    payload = {
        "scanned_ingredients": scanned,
        "pantry_ingredients": pantry,
        "diet_preference": diet,
        "top_k": top_k,
    }

    try:
        resp = requests.post(f"{API_URL}/api/recommend", json=payload, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            # Normalise API response to same shape as inline results
            return [
                {
                    "recipe": {
                        "name": r["name"],
                        "id": r["id"],
                        "diet": r["diet"],
                        "cuisine": r["cuisine"],
                        "time_mins": r.get("time_mins"),
                        "ingredients": r.get("ingredients", []),
                    },
                    "score": r["score"],
                    "matched_count": r["matched_count"],
                    "total_count": r["total_count"],
                    "missing_ingredients": r["missing_ingredients"],
                    "why": r["why"],
                }
                for r in data["results"]
            ]
    except Exception:
        pass  # Silently fall through to inline

    # Inline fallback
    return _inline_recommend(scanned, pantry, diet, top_k)
