"""
FastAPI Backend — Recipe Recommendation Engine
Hybrid Jaccard + TF-IDF cosine similarity scoring.
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import pickle
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import RecipeRequest, RecipeResult, RecommendResponse

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state (loaded once on startup) ─────────────────
RECIPE_DB: list[dict] = []
VECTORIZER: TfidfVectorizer | None = None
TFIDF_MATRIX = None
INGREDIENT_CLASSES: list[str] = []

DATA_DIR = Path(__file__).parent


def _load_assets():
    global RECIPE_DB, VECTORIZER, TFIDF_MATRIX, INGREDIENT_CLASSES

    # 1. Recipe database
    db_path = DATA_DIR / "recipe_db.json"
    if db_path.exists():
        with open(db_path, "r", encoding="utf-8") as f:
            RECIPE_DB = json.load(f)
        logger.info(f"Loaded {len(RECIPE_DB)} recipes.")
    else:
        logger.warning("recipe_db.json not found — using empty DB. Run the data prep notebook first.")
        RECIPE_DB = []

    # 2. TF-IDF vectorizer & matrix
    vec_path = DATA_DIR / "vectorizer.pkl"
    mat_path = DATA_DIR / "tfidf_matrix.pkl"
    if vec_path.exists() and mat_path.exists():
        with open(vec_path, "rb") as f:
            VECTORIZER = pickle.load(f)
        with open(mat_path, "rb") as f:
            TFIDF_MATRIX = pickle.load(f)
        logger.info("TF-IDF vectorizer + matrix loaded.")
    else:
        logger.warning("TF-IDF assets not found — building on-the-fly from recipe DB.")
        if RECIPE_DB:
            VECTORIZER = TfidfVectorizer()
            corpus = [" ".join(r.get("ingredients", [])) for r in RECIPE_DB]
            TFIDF_MATRIX = VECTORIZER.fit_transform(corpus)

    # 3. YOLO ingredient class labels (for frontend hints)
    cls_path = DATA_DIR / "ingredient_classes.json"
    if cls_path.exists():
        with open(cls_path, "r") as f:
            INGREDIENT_CLASSES = json.load(f)
    else:
        INGREDIENT_CLASSES = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_assets()
    yield


# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="Cook With What You Have — Recipe API",
    description="Ingredient-to-recipe recommendation using hybrid Jaccard + TF-IDF scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Scoring ────────────────────────────────────────────────
def _normalize(text: str) -> str:
    """Lowercase, strip whitespace."""
    return text.lower().strip()


def _jaccard(user_set: set, recipe_set: set) -> float:
    if not recipe_set:
        return 0.0
    return len(user_set & recipe_set) / len(recipe_set)


def _cosine(user_ingredients: set, recipe_idx: int) -> float:
    if VECTORIZER is None or TFIDF_MATRIX is None:
        return 0.0
    user_text = " ".join(user_ingredients)
    user_vec = VECTORIZER.transform([user_text])
    score = cosine_similarity(user_vec, TFIDF_MATRIX[recipe_idx])[0][0]
    return float(score)


def _hybrid_score(user_set: set, recipe_set: set, recipe_idx: int) -> float:
    """60% Jaccard coverage + 40% TF-IDF cosine semantic similarity."""
    return 0.60 * _jaccard(user_set, recipe_set) + 0.40 * _cosine(user_set, recipe_idx)


# ── Endpoints ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "ok",
        "recipes_loaded": len(RECIPE_DB),
        "tfidf_ready": TFIDF_MATRIX is not None,
    }


@app.get("/api/ingredient-classes")
async def ingredient_classes():
    """Return the YOLO model's known ingredient class labels."""
    return {"classes": INGREDIENT_CLASSES, "count": len(INGREDIENT_CLASSES)}


@app.get("/api/stats")
async def stats():
    return {
        "total_recipes": len(RECIPE_DB),
        "cuisines": list({r.get("cuisine", "Unknown") for r in RECIPE_DB}),
        "diets": list({r.get("diet", "Unknown") for r in RECIPE_DB}),
    }


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(req: RecipeRequest):
    if not RECIPE_DB:
        raise HTTPException(
            status_code=503,
            detail="Recipe database not loaded. Run the data prep notebook and restart.",
        )

    # Merge scanned + pantry ingredients, normalise
    all_user = {_normalize(i) for i in req.scanned_ingredients + req.pantry_ingredients if i.strip()}

    if not all_user:
        raise HTTPException(status_code=400, detail="No ingredients provided.")

    results: list[dict] = []

    for idx, recipe in enumerate(RECIPE_DB):
        # Hard filter: diet preference
        if req.diet_preference:
            recipe_diet = recipe.get("diet", "").lower()
            if req.diet_preference.lower() not in recipe_diet:
                continue

        recipe_set = {_normalize(i) for i in recipe.get("ingredients", [])}
        if not recipe_set:
            continue

        score = _hybrid_score(all_user, recipe_set, idx)

        # Only include if user has ≥ 35% of required ingredients
        if score < 0.35:
            continue

        matched = all_user & recipe_set
        missing = list(recipe_set - all_user)

        results.append({
            "recipe": recipe,
            "score": round(score, 4),
            "matched": matched,
            "missing": missing[:6],  # cap for UI
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[: req.top_k]

    recipe_results = []
    for r in top:
        recipe = r["recipe"]
        recipe_set = set(recipe.get("ingredients", []))
        recipe_results.append(
            RecipeResult(
                id=str(recipe.get("id", "")),
                name=recipe.get("name", "Unknown"),
                score=r["score"],
                matched_count=len(r["matched"]),
                total_count=len(recipe_set),
                missing_ingredients=r["missing"],
                why=(
                    f"You have {len(r['matched'])} of {len(recipe_set)} ingredients "
                    f"({int(len(r['matched'])/len(recipe_set)*100)}% coverage)"
                ),
                diet=recipe.get("diet", ""),
                cuisine=recipe.get("cuisine", ""),
                time_mins=recipe.get("time_mins"),
                ingredients=recipe.get("ingredients", []),
            )
        )

    return RecommendResponse(
        results=recipe_results,
        query_ingredients=list(all_user),
        total_scanned=len(all_user),
    )
