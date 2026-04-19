# Indian Recipe Database Preparation — RecSys Engine
# Kaggle Notebook 2 | T14.1 SMAI Assignment 3
# Dataset: 6000-indian-recipes (kanishk307 / Kaggle)
# ─────────────────────────────────────────────────────────────

# ── Cell 1: Install dependencies ──────────────────────────────
# !pip install -q sentence-transformers scikit-learn nltk

# ── Cell 2: Imports ───────────────────────────────────────────
import json
import ast
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("omw-1.4", quiet=True)

WORK_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input/datasets/shubhadeepmandal/indianfooddataset")

# ── Cell 3: Load Dataset ──────────────────────────────────────
# Add the dataset via Kaggle UI: "6000 Indian Recipes"
# https://www.kaggle.com/datasets/nehaprabhavalkar/av-indian-food-dataset
# OR: kanishk307/IndianFoodDatasetGeneration

csv_candidates = list(INPUT_DIR.glob("*.csv"))
print(f"Found CSVs: {csv_candidates}")
df = pd.read_csv(csv_candidates[0])
print(f"Shape: {df.shape}")
print(df.columns.tolist())
df.head(3)

# ── Cell 4: Standardise Column Names ─────────────────────────
# Priority-ordered candidates for each logical field.
# We pick the FIRST matching column to avoid duplicate-column errors
# (which happen when the CSV has e.g. both "TranslatedIngredients"
# AND "Ingredients", causing df["ingredients_raw"] to be a DataFrame).

COL_CANDIDATES = {
    "name":            ["TranslatedRecipeName", "Recipe Name", "Translated-Recipe-Name", "name"],
    "ingredients_raw": ["TranslatedIngredients", "Ingredients", "Ingredient", "ingredients"],
    "diet":            ["Diet", "diet"],
    "cuisine":         ["Cuisine", "cuisine"],
    "time_mins":       ["TotalTimeInMins", "Total Time (in mins)", "PrepTimeInMins"],
    "id":              ["RecipeId", "Srno", "srno", "index"],
}

existing_cols = set(df.columns.tolist())
print("\nColumn resolution:")
for target, candidates in COL_CANDIDATES.items():
    matched = next((c for c in candidates if c in existing_cols), None)
    if matched and matched != target:
        df[target] = df[matched]   # copy as new column — never rename in-place
        print(f"  {target:20s} ← '{matched}'")
    elif matched:
        print(f"  {target:20s} ← '{matched}' (already correct name)")
    else:
        print(f"  {target:20s} ← NOT FOUND (will be NaN)")
        df[target] = None

# Verify the key columns are scalar Series (not DataFrames)
for col in ("name", "ingredients_raw"):
    if isinstance(df[col], pd.DataFrame):
        # Shouldn't happen after the fix, but guard anyway
        df[col] = df[col].iloc[:, 0]
        print(f"  WARNING: {col} was a DataFrame — kept first column only")

df.dropna(subset=["name", "ingredients_raw"], inplace=True)
df = df.reset_index(drop=True)
print(f"\nAfter dropping NA: {len(df)} recipes")
print(df[["name", "ingredients_raw"]].head(3))

# ── Cell 4b: Diagnose raw ingredient format ──────────────────
# Run this BEFORE touching Cell 5 to understand the actual format.
print("\n── Raw ingredient samples (first 5 rows) ──────────────────")
for i, val in enumerate(df["ingredients_raw"].head(5)):
    print(f"  [{i}] type={type(val).__name__}  repr={repr(str(val)[:120])}")


# ── Cell 5: Text Normalisation Pipeline ───────────────────────
lemmatizer = WordNetLemmatizer()
STOP = set(stopwords.words("english"))

# Only strip NUMERIC measurements — do NOT strip adjectives like
# 'fresh', 'dried' etc. at this stage so we keep ingredient words.
MEASUREMENT_RE = re.compile(
    r"\b(\d+\s*[-–]?\s*\d*\s*"
    r"(cups?|tbsps?|tsps?|grams?|g|kg|ml|liters?|litres?|oz|pounds?|lbs?|"
    r"inches?|cm|tablespoons?|teaspoons?|pinch|handful|bunch|cloves?|"
    r"pieces?|slices?|cans?|packets?)\b)",
    re.IGNORECASE,
)
# Phrases that are purely descriptive with no noun left
DESCRIPTOR_RE = re.compile(
    r"\b(chopped|diced|minced|grated|sliced|crushed|ground|whole|fresh|"
    r"dried|cooked|raw|peeled|optional|to taste|as needed|washed|soaked|"
    r"boiled|fried|roasted|powdered|blended|melted|softened|room temperature)\b",
    re.IGNORECASE,
)


def _split_raw(raw) -> list[str]:
    """
    Split the raw field into individual ingredient strings,
    handling every format we've seen in the wild:
      - Python list repr: "['tomato', 'onion']"
      - Comma-separated:  "tomato,onion,garlic"
      - Pipe-separated:   "tomato|onion|garlic"
      - Newline-separated
      - Already a Python list object
    """
    if isinstance(raw, list):
        return [str(x) for x in raw]

    s = str(raw).strip()

    # Try Python literal first (handles '["a","b"]' and "['a','b']")
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except Exception:
        pass

    # Pipe-separated
    if "|" in s:
        return [x.strip() for x in s.split("|")]
    # Newline-separated
    if "\n" in s:
        return [x.strip() for x in s.splitlines()]
    # Plain comma-separated (most common fallback)
    return [x.strip() for x in s.split(",")]


def normalize_ingredient(raw: str) -> str:
    """
    '2 cups of finely chopped red onions' → 'onion'
    'garam masala'                        → 'masala'  (last noun)
    'oil'                                 → 'oil'
    """
    text = raw.lower().strip()
    if not text:
        return ""

    text = MEASUREMENT_RE.sub(" ", text)       # strip numeric measures
    text = DESCRIPTOR_RE.sub(" ", text)        # strip pure descriptors
    text = re.sub(r"[^a-z\s]", " ", text)     # strip punctuation/digits
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    # Remove single-character tokens and very common filler words
    tokens = [t for t in tokens if len(t) > 1 and t not in STOP]

    if not tokens:
        # Fallback: if normalisation stripped everything, return the
        # longest word from the original (likely the ingredient noun)
        orig_tokens = re.sub(r"[^a-z\s]", "", raw.lower()).split()
        orig_tokens = [t for t in orig_tokens if len(t) > 1]
        if orig_tokens:
            return max(orig_tokens, key=len)  # longest = most specific noun
        return ""

    # Lemmatize and return last meaningful token (the head noun)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens[-1]


def parse_and_clean_ingredients(raw) -> list[str]:
    """Split, normalise, and deduplicate ingredients from any format."""
    items = _split_raw(raw)
    cleaned = []
    for item in items:
        norm = normalize_ingredient(item)
        if norm and len(norm) > 1:   # skip single chars
            cleaned.append(norm)
    return list(dict.fromkeys(cleaned))  # deduplicate, preserve order


df["ingredients"] = df["ingredients_raw"].apply(parse_and_clean_ingredients)

# ── Diagnostic: check normalisation worked ────────────────────
n_empty = (df["ingredients"].map(len) == 0).sum()
n_short = (df["ingredients"].map(len) < 3).sum()
print(f"\nIngredient parse results:")
print(f"  Rows with 0 ingredients : {n_empty}")
print(f"  Rows with <3 ingredients: {n_short}")
print(f"  Sample (row 0): {df['ingredients'].iloc[0]}")
print(f"  Sample (row 1): {df['ingredients'].iloc[1]}")

df = df[df["ingredients"].map(len) >= 3].reset_index(drop=True)
print(f"  After filtering: {len(df)} usable recipes")

# ── Cell 6: EDA ───────────────────────────────────────────────
all_ings = [ing for ings in df["ingredients"] for ing in ings]

if not all_ings:
    print("⚠️  all_ings is empty — ingredient parsing returned nothing.")
    print("    Check the 'Raw ingredient samples' output above and adjust")
    print("    _split_raw() if the format is unexpected.")
else:
    top_n = min(30, len(set(all_ings)))
    top_30 = Counter(all_ings).most_common(top_n)
    names_plot, counts_plot = zip(*top_30)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Indian Recipe Dataset — EDA", fontsize=14, fontweight="bold")

    # Horizontal bar — top ingredients
    axes[0].barh(list(names_plot[::-1]), list(counts_plot[::-1]), color="coral")
    axes[0].set_title(f"Top {top_n} Most Common Ingredients")
    axes[0].set_xlabel("Recipe Count")
    axes[0].tick_params(axis="y", labelsize=9)

    # Cuisine distribution
    if "cuisine" in df.columns:
        cuisine_counts = df["cuisine"].value_counts().head(15)
        axes[1].bar(range(len(cuisine_counts)), cuisine_counts.values, color="steelblue")
        axes[1].set_xticks(range(len(cuisine_counts)))
        axes[1].set_xticklabels(cuisine_counts.index, rotation=45, ha="right", fontsize=9)
        axes[1].set_title("Top 15 Cuisines")
        axes[1].set_ylabel("Recipe Count")
    else:
        # Ingredient count distribution instead
        lengths = df["ingredients"].map(len)
        axes[1].hist(lengths, bins=20, color="steelblue", edgecolor="white")
        axes[1].set_title("Ingredients per Recipe")
        axes[1].set_xlabel("Number of ingredients")
        axes[1].set_ylabel("Recipes")

    plt.tight_layout()
    plt.savefig(str(WORK_DIR / "eda_plots.png"), dpi=150)
    plt.show()
    print(f"\nTotal unique ingredients : {len(set(all_ings))}")
    print(f"Avg ingredients per recipe: {np.mean([len(i) for i in df['ingredients']]):.1f}")

# ── Cell 7: Build recipe_db.json ─────────────────────────────
recipe_db = []
for _, row in df.iterrows():
    recipe_db.append({
        "id": str(row.get("id", _)),
        "name": str(row["name"]),
        "ingredients": row["ingredients"],
        "diet": str(row.get("diet", "Unknown")),
        "cuisine": str(row.get("cuisine", "Unknown")),
        "time_mins": float(row["time_mins"]) if pd.notna(row.get("time_mins")) else None,
    })

out_db = WORK_DIR / "recipe_db.json"
with open(out_db, "w", encoding="utf-8") as f:
    json.dump(recipe_db, f, ensure_ascii=False, indent=2)
print(f"Saved {len(recipe_db)} recipes to {out_db}")

# ── Cell 8: Build TF-IDF Vectorizer ───────────────────────────
corpus = [" ".join(r["ingredients"]) for r in recipe_db]

vectorizer = TfidfVectorizer(
    min_df=2,           # ignore ingredients appearing in <2 recipes
    max_df=0.95,        # ignore very common stop-ingredients
    ngram_range=(1, 2), # unigrams + bigrams (e.g. "garam masala")
)
tfidf_matrix = vectorizer.fit_transform(corpus)
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

with open(WORK_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(WORK_DIR / "tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)
print("Saved vectorizer.pkl and tfidf_matrix.pkl")

# ── Cell 9: Sentence Embeddings (optional, slow) ─────────────
# Uncomment to add dense semantic retrieval alongside TF-IDF.
# model_st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# recipe_texts = [f"{r['name']}: " + ", ".join(r["ingredients"]) for r in recipe_db]
# embeddings = model_st.encode(recipe_texts, batch_size=64, show_progress_bar=True)
# np.save(WORK_DIR / "recipe_embeddings.npy", embeddings)
# print(f"Saved embeddings shape: {embeddings.shape}")

# ── Cell 10: Smoke Test — Query the Engine ───────────────────
print("\n── Smoke Test ──────────────────────────────────────────")
test_ingredients = {"tomato", "onion", "garlic", "ginger", "chicken"}
user_text = " ".join(test_ingredients)
user_vec = vectorizer.transform([user_text])
cosine_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

top_indices = cosine_scores.argsort()[::-1][:5]
print(f"Query: {test_ingredients}\n")
print("Top 5 TF-IDF matches:")
for rank, idx in enumerate(top_indices, 1):
    r = recipe_db[idx]
    jaccard = len(test_ingredients & set(r["ingredients"])) / len(set(r["ingredients"]))
    print(f"  {rank}. {r['name']:40s} cosine={cosine_scores[idx]:.3f}  jaccard={jaccard:.3f}  cuisine={r['cuisine']}")

# ── Cell 11: Export ingredient class list for YOLO labels ─────
# This list is used by the FastAPI /api/ingredient-classes endpoint.
# Adjust to the classes your YOLO model was trained on.
yolo_classes = sorted(set(all_ings))[:100]  # top 100 most common
with open(WORK_DIR / "ingredient_classes.json", "w") as f:
    json.dump(yolo_classes, f)
print(f"\nSaved {len(yolo_classes)} ingredient class labels")

# ── Cell 12: DOWNLOAD ALL OUTPUT FILES ───────────────────────
# Bundles every artifact into a single zip and triggers a
# browser download — no need to manually click each file in
# the Kaggle sidebar.
import zipfile
import shutil
from IPython.display import FileLink, display as ipy_display

OUTPUT_FILES = [
    WORK_DIR / "recipe_db.json",
    WORK_DIR / "vectorizer.pkl",
    WORK_DIR / "tfidf_matrix.pkl",
    WORK_DIR / "ingredient_classes.json",
    WORK_DIR / "eda_plots.png",
]

ZIP_PATH = WORK_DIR / "recsys_assets.zip"

print("\n── Packaging output files ───────────────────────────────")
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for fpath in OUTPUT_FILES:
        if fpath.exists():
            zf.write(fpath, arcname=fpath.name)
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✅  {fpath.name:35s} ({size_kb:.1f} KB)")
        else:
            print(f"  ⚠️   {fpath.name:35s} NOT FOUND — skipping")

zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
print(f"\n📦  recsys_assets.zip  →  {zip_size_mb:.2f} MB")

# ── Individual file links (clickable in Kaggle output panel) ──
print("\nIndividual download links:")
for fpath in OUTPUT_FILES:
    if fpath.exists():
        ipy_display(FileLink(str(fpath), result_html_prefix=f"⬇️  {fpath.name}: "))

# ── Zip download link ─────────────────────────────────────────
print("\nAll-in-one zip:")
ipy_display(FileLink(str(ZIP_PATH), result_html_prefix="📦  recsys_assets.zip: "))

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  All done!  Extract the zip and copy files as follows:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  recipe_db.json          →  Assignment 3/backend/
  vectorizer.pkl          →  Assignment 3/backend/
  tfidf_matrix.pkl        →  Assignment 3/backend/
  ingredient_classes.json →  Assignment 3/backend/
  eda_plots.png           →  (keep for your report)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Then run:
  cd backend && uvicorn main:app --reload --host 0.0.0.0
  cd streamlit_app && streamlit run app.py
""")
