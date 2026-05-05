import os
import re
import time
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Indian Recipe Recommender", page_icon="🍛", layout="wide")

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "IndianFoodDatasetCSV.csv")
MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["RecipeName", "TranslatedIngredients"]).reset_index(drop=True)
    df["TranslatedIngredients"] = df["TranslatedIngredients"].fillna("")
    df["Cuisine"] = df["Cuisine"].fillna("Unknown")
    df["Diet"] = df["Diet"].fillna("Unknown")
    df["Course"] = df["Course"].fillna("Unknown")
    df["combined_text"] = (
        df["TranslatedRecipeName"].fillna("") + " "
        + df["TranslatedIngredients"] + " "
        + df["Cuisine"] + " "
        + df["Diet"] + " "
        + df["Course"]
    )
    return df


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data
def compute_embeddings(texts: list[str]) -> np.ndarray:
    model = load_model()
    return model.encode(texts, show_progress_bar=False, batch_size=64)


FAISS_PATH = os.path.join(os.path.dirname(__file__), "recipe_index.faiss")


def recipe_key(row: pd.Series) -> str:
    url = str(row.get("URL", "")).strip()
    if url and url.lower().startswith("http"):
        return url.lower()
    name = str(row.get("TranslatedRecipeName", "")).strip()
    return name.lower()


@st.cache_resource
def build_faiss_index(embeddings: np.ndarray):
    if os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
        # notebook saves IVFFlat; app needs FlatIP — rebuild if wrong type
        if isinstance(index, faiss.IndexFlatIP):
            return index
    vecs = embeddings.astype("float32").copy()
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, FAISS_PATH)
    return index


def keyword_scores(query: str, df: pd.DataFrame) -> np.ndarray:
    keywords = [w.lower().strip() for w in query.replace(",", " ").split() if len(w) > 2]
    if not keywords:
        return np.zeros(len(df))
    return np.array([
        sum(k in row.lower() for k in keywords) / len(keywords)
        for row in df["TranslatedIngredients"]
    ])


def blend(cosine: np.ndarray, keyword: np.ndarray, kw_weight: float) -> np.ndarray:
    return (1 - kw_weight) * cosine + kw_weight * keyword


def recommend_brute(query, df, embeddings, top_k, kw_weight):
    model = load_model()
    t0 = time.perf_counter()
    query_vec = model.encode([query])
    cos = cosine_similarity(query_vec, embeddings)[0]
    kw  = keyword_scores(query, df) if kw_weight > 0 else np.zeros(len(df))
    scores = blend(cos, kw, kw_weight)
    top_idx = np.argsort(scores)[::-1][:top_k]
    elapsed = (time.perf_counter() - t0) * 1000
    results = df.iloc[top_idx].copy()
    results["similarity"] = scores[top_idx]
    return results, elapsed


def recommend_faiss(query, df, embeddings, top_k, kw_weight):
    model = load_model()
    index = build_faiss_index(embeddings)
    t0 = time.perf_counter()
    q = model.encode([query]).astype("float32")
    faiss.normalize_L2(q)
    fetch_k = min(len(df), top_k * 10)
    distances, indices = index.search(q, fetch_k)
    candidate_df = df.iloc[indices[0]].copy().reset_index(drop=True)
    kw_scores    = keyword_scores(query, candidate_df) if kw_weight > 0 else np.zeros(fetch_k)
    final_scores = blend(distances[0], kw_scores, kw_weight)
    top_idx      = np.argsort(final_scores)[::-1][:top_k]
    elapsed = (time.perf_counter() - t0) * 1000
    results = candidate_df.iloc[top_idx].copy()
    results["similarity"] = final_scores[top_idx]
    return results, elapsed


def make_why(row: pd.Series, query: str) -> str:
    query_words = set(w.lower() for w in query.replace(",", " ").split() if len(w) > 2)
    matched = [w for w in query_words if w in row["TranslatedIngredients"].lower()]
    parts = []
    if matched:
        parts.append(f"Contains: **{', '.join(matched)}**")
    if row["Diet"] not in ("Unknown", ""):
        parts.append(f"Diet: {row['Diet']}")
    if row["TotalTimeInMins"] > 0:
        parts.append(f"Ready in {int(row['TotalTimeInMins'])} min")
    return " · ".join(parts) if parts else "Semantically similar to your query"


def highlight_keywords(text: str, query: str) -> str:
    """Wrap every query keyword in the text with a yellow highlight span."""
    if not query:
        return text
    keywords = [w.strip() for w in query.replace(",", " ").split() if len(w) > 2]
    result = text
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        result = pattern.sub(
            lambda m: (
                f'<mark style="background-color:#f5a623;color:#000;'
                f'border-radius:3px;padding:0 3px;font-weight:600">'
                f'{m.group()}</mark>'
            ),
            result,
        )
    return result


def recipe_card(row, query, show_save=True):
    """Render a single recipe card. Used on both pages."""
    name = row["TranslatedRecipeName"]
    key = recipe_key(row)
    is_saved = key in st.session_state.favourites
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(name)
            if query:
                st.caption(make_why(row, query))
            else:
                st.caption(f"{row['Cuisine']} · {row['Course']}")
            with st.expander("Ingredients"):
                st.markdown(
                    highlight_keywords(row["TranslatedIngredients"], query),
                    unsafe_allow_html=True,
                )
            btn_col, link_col = st.columns([1, 3])
            with btn_col:
                if show_save:
                    if is_saved:
                        if st.button("❤️ Saved", key=f"fav_{key}", type="secondary"):
                            del st.session_state.favourites[key]
                            st.rerun()
                    else:
                        if st.button("🤍 Save", key=f"fav_{key}"):
                            st.session_state.favourites[key] = row.to_dict()
                            st.rerun()
                else:
                    if st.button("🗑 Remove", key=f"rm_{key}", type="secondary"):
                        del st.session_state.favourites[key]
                        st.rerun()
            with link_col:
                if pd.notna(row.get("URL")) and str(row["URL"]).startswith("http"):
                    st.markdown(f"[View full recipe ↗]({row['URL']})")
        with col2:
            if "similarity" in row and pd.notna(row.get("similarity")):
                st.metric("Match", f"{row['similarity']:.0%}")
            st.markdown(f"**Cuisine:** {row['Cuisine']}")
            st.markdown(f"**Diet:** {row['Diet']}")
            st.markdown(f"**Course:** {row['Course']}")
            if row["TotalTimeInMins"] > 0:
                st.markdown(f"**Time:** {int(row['TotalTimeInMins'])} min")
            if row["Servings"] > 0:
                st.markdown(f"**Serves:** {int(row['Servings'])}")


# ── Session state ───────────────────────────────────────────────────────────────
if "favourites" not in st.session_state:
    st.session_state.favourites = {}
else:
    migrated_favourites = {}
    for _, saved_row in st.session_state.favourites.items():
        row = pd.Series(saved_row)
        migrated_favourites[recipe_key(row)] = saved_row
    st.session_state.favourites = migrated_favourites

# ── Header row ─────────────────────────────────────────────────────────────────
df = load_data()
with st.spinner("Loading embeddings… (first run takes ~30s)"):
    embeddings = compute_embeddings(df["combined_text"].tolist())

title_col, dev_col = st.columns([8, 2])
title_col.title("🍛 Indian Recipe Recommender")
with dev_col:
    st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
    dev_mode = st.toggle("🛠 Dev tools", value=False)

# ── Navbar tabs ────────────────────────────────────────────────────────────────
fav_count = len(st.session_state.favourites)
tab_search, tab_favs = st.tabs([
    "🔍 Recommender",
    f"❤️ Favourites ({fav_count})",
])

# ── Sidebar (filters + dev, only relevant on search tab) ───────────────────────
with st.sidebar:
    st.header("Filters")
    cuisines = ["All"] + sorted(df["Cuisine"].unique().tolist())
    sel_cuisine = st.selectbox("Cuisine", cuisines)

    diets = ["All"] + sorted(df["Diet"].unique().tolist())
    sel_diet = st.selectbox("Diet", diets)

    courses = ["All"] + sorted(df["Course"].unique().tolist())
    sel_course = st.selectbox("Course", courses)

    max_time = st.slider("Max total time (mins)", 0, 300, 120, step=15)
    top_k = st.slider("Number of results", 5, 20, 10)
    sort_by = st.selectbox(
        "Sort results",
        ["Most relevant", "Shortest time", "Longest time", "Most servings"],
        help="Changes the order of the recipes after the search ranking is computed."
    )

    search_mode = "Brute-force (cosine)"
    use_keywords = True
    kw_weight = 0.35

    if dev_mode:
        st.divider()
        st.subheader("⚙️ Dev Features")
        search_mode = st.selectbox(
            "Search method",
            ["Brute-force (cosine)", "FAISS Index"],
            help="Brute-force checks every recipe; FAISS uses an optimized index."
        )
        st.markdown("**Scoring**")
        use_keywords = st.toggle(
            "Keyword boost", value=True,
            help="Boosts recipes that contain your exact query words in their ingredients."
        )
        kw_weight = 0.0
        if use_keywords:
            kw_weight = st.slider(
                "Keyword weight", 0.0, 1.0, 0.35, 0.05,
                help="0 = pure semantic, 1 = pure keyword. 0.3–0.4 works best."
            )
            st.caption(f"Final score = **{1-kw_weight:.0%}** semantic + **{kw_weight:.0%}** keyword")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Recommender
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    CHIPS = ["paneer, tomato, onion", "rice, dal", "chicken, spices", "potato, peas", "coconut, mustard"]
    st.markdown("**Quick picks:**")
    chip_cols = st.columns(len(CHIPS))
    chip_query = ""
    for i, chip in enumerate(CHIPS):
        if chip_cols[i].button(chip, key=f"chip_{i}"):
            chip_query = chip

    query = st.text_input(
        "Enter ingredients or dish name",
        value=chip_query,
        placeholder="e.g. paneer, tomato, cream"
    )

    if query:
        mask = pd.Series([True] * len(df), index=df.index)
        if sel_cuisine != "All":
            mask &= df["Cuisine"] == sel_cuisine
        if sel_diet != "All":
            mask &= df["Diet"] == sel_diet
        if sel_course != "All":
            mask &= df["Course"] == sel_course
        mask &= df["TotalTimeInMins"] <= max_time

        filtered_df   = df[mask].reset_index(drop=True)
        filtered_embs = embeddings[np.where(mask)[0]]

        if filtered_df.empty:
            st.warning("No recipes match the current filters. Try relaxing them.")
        else:
            if search_mode == "FAISS Index":
                results, elapsed = recommend_faiss(query, filtered_df, filtered_embs, top_k, kw_weight)
            else:
                results, elapsed = recommend_brute(query, filtered_df, filtered_embs, top_k, kw_weight)

            if sort_by == "Shortest time":
                results = results.sort_values(["TotalTimeInMins", "similarity"], ascending=[True, False])
            elif sort_by == "Longest time":
                results = results.sort_values(["TotalTimeInMins", "similarity"], ascending=[False, False])
            elif sort_by == "Most servings":
                results = results.sort_values(["Servings", "similarity"], ascending=[False, False])

            if dev_mode:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Search method", search_mode.split("(")[0].strip())
                m2.metric("Recipes searched", f"{len(filtered_df):,}")
                m3.metric("Query time", f"{elapsed:.1f} ms")
                m4.metric("Keyword weight", f"{kw_weight:.0%}" if use_keywords else "Off")

            active_filters = []
            if sel_cuisine != "All":
                active_filters.append(f"Cuisine: {sel_cuisine}")
            if sel_diet != "All":
                active_filters.append(f"Diet: {sel_diet}")
            if sel_course != "All":
                active_filters.append(f"Course: {sel_course}")
            active_filters.append(f"Max time: {max_time} min")
            st.caption("Filters: " + " · ".join(active_filters) + f"  |  Showing {len(results)} of {len(filtered_df):,} matching recipes")

            st.markdown(f"### Top {len(results)} Recipes for *\"{query}\"*")
            for _, row in results.iterrows():
                recipe_card(row, query, show_save=True)
    else:
        st.info("Enter ingredients above (e.g. *paneer, tomato, onion*) or click a quick pick to get started.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Favourites
# ══════════════════════════════════════════════════════════════════════════════
with tab_favs:
    if fav_count == 0:
        st.markdown("### No favourites yet")
        st.info("Head to the Recommender tab, search for something, and hit 🤍 Save on any recipe.")
    else:
        hdr_col, clr_col = st.columns([6, 1])
        hdr_col.markdown(f"### Your saved recipes ({fav_count})")
        if clr_col.button("Clear all", type="secondary"):
            st.session_state.favourites.clear()
            st.rerun()

        for _, saved_row in list(st.session_state.favourites.items()):
            recipe_card(pd.Series(saved_row), query="", show_save=False)
