from pydantic import BaseModel
from typing import List, Optional


class RecipeRequest(BaseModel):
    scanned_ingredients: List[str]
    pantry_ingredients: List[str] = []
    diet_preference: Optional[str] = None  # e.g. "vegetarian", "non vegetarian"
    top_k: int = 10


class RecipeResult(BaseModel):
    id: str
    name: str
    score: float
    matched_count: int
    total_count: int
    missing_ingredients: List[str]
    why: str
    diet: str
    cuisine: str
    time_mins: Optional[float] = None
    ingredients: List[str]


class RecommendResponse(BaseModel):
    results: List[RecipeResult]
    query_ingredients: List[str]
    total_scanned: int
