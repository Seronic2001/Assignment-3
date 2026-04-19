"""
T14.1 — Cook With What You Have
Streamlit App — Main Entry Point

Run: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# ── Page config (must be first Streamlit call) ─────────────
st.set_page_config(
    page_title="Cook With What You Have 🍛",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "T14.1 — Personalised Indian Recipe Recommender | SMAI Assignment 3"
    },
)

# ── Import components ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from components.camera import camera_input_section, display_annotated_image
from components.detector import run_detection, unique_labels, Detection
from components.recommender import get_recommendations

# ── Custom CSS ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards */
    .recipe-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 14px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .recipe-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 167, 38, 0.5);
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
    }
    .score-high  { background: rgba(76,175,80,0.25); color: #81c784; border: 1px solid #4caf50; }
    .score-mid   { background: rgba(255,167,38,0.25); color: #ffca28; border: 1px solid #ffa726; }
    .score-low   { background: rgba(239,83,80,0.25);  color: #ef9a9a; border: 1px solid #ef5350; }

    /* Ingredient chips */
    .chip {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 13px;
        margin: 2px;
        color: #e0e0e0;
    }
    .chip-detected { background: rgba(255,167,38,0.2); border-color: #ffa726; color: #ffd180; }
    .chip-missing  { background: rgba(239,83,80,0.15); border-color: #ef5350; color: #ef9a9a; }

    /* Section headers */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffd180;
        margin-bottom: 0.2rem;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State Init ─────────────────────────────────────
if "pantry" not in st.session_state:
    st.session_state.pantry: list[str] = []
if "favourites" not in st.session_state:
    st.session_state.favourites: list[dict] = []
if "last_detections" not in st.session_state:
    st.session_state.last_detections: list[str] = []


# ── Sidebar — Pantry & Filters ─────────────────────────────
with st.sidebar:
    st.markdown("## 🧺 Virtual Pantry")
    st.caption("Ingredients you always have at home.")

    new_item = st.text_input("Add ingredient", placeholder="e.g. turmeric", key="pantry_input")
    if st.button("➕ Add", use_container_width=True) and new_item.strip():
        item = new_item.strip().lower()
        if item not in st.session_state.pantry:
            st.session_state.pantry.append(item)
        st.rerun()

    if st.session_state.pantry:
        st.markdown("**Pantry items:**")
        for i, item in enumerate(st.session_state.pantry):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"<span class='chip'>{item}</span>", unsafe_allow_html=True)
            if col2.button("✕", key=f"del_{i}", help="Remove"):
                st.session_state.pantry.pop(i)
                st.rerun()
    else:
        st.info("Your pantry is empty. Add staples like oil, salt, garlic…")

    st.divider()
    st.markdown("## 🎛️ Filters")
    diet = st.selectbox(
        "Diet preference",
        ["Any", "Vegetarian", "Non Vegetarian", "Vegan"],
        key="diet_filter",
    )
    top_k = st.slider("Max recipes to show", min_value=3, max_value=20, value=8, key="top_k")
    conf_thresh = st.slider("Detection confidence", min_value=0.3, max_value=0.9, value=0.45, step=0.05)

    st.divider()
    if st.session_state.favourites:
        st.markdown("## ❤️ Favourites")
        for fav in st.session_state.favourites:
            st.markdown(f"- {fav['recipe']['name']}")


# ── Main Layout ────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-family:"Playfair Display",serif; font-size:2.8rem;
                   background: linear-gradient(90deg,#ffd180,#ff8f00);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   margin-bottom:0;'>
            🍛 Cook With What You Have
        </h1>
        <p style='color:#90a4ae; font-size:1rem; margin-top:6px;'>
            Scan your ingredients → Get personalised Indian recipes
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_scan, tab_manual, tab_favs = st.tabs(["📸 Scan Ingredients", "✏️ Type Manually", "❤️ Favourites"])


# ─────────────────────────────────────────────────────────────
# SHARED RENDER FUNCTION
# ─────────────────────────────────────────────────────────────
def _render_recipe_cards(recs: list[dict], show_save: bool = True):
    for rec in recs:
        recipe = rec["recipe"]
        score = rec["score"]
        missing = rec.get("missing_ingredients", [])
        why = rec.get("why", "")

        score_pct = int(score * 100)
        if score_pct >= 70:
            badge_cls = "score-high"
        elif score_pct >= 50:
            badge_cls = "score-mid"
        else:
            badge_cls = "score-low"

        time_str = f"⏱️ {int(recipe.get('time_mins', 0))} min" if recipe.get('time_mins') else ""
        cuisine_str = recipe.get("cuisine", "")
        diet_str = recipe.get("diet", "")

        with st.container():
            st.markdown(
                f"""
                <div class='recipe-card'>
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                        <div>
                            <h3 style='margin:0;color:#ffe082;font-size:1.1rem;'>{recipe['name']}</h3>
                            <div style='margin-top:6px;'>
                                <span class='score-badge {badge_cls}'>{score_pct}% match</span>
                                <small style='color:#78909c;'>{cuisine_str} · {diet_str} {time_str}</small>
                            </div>
                        </div>
                    </div>
                    <p style='color:#90a4ae;font-size:0.85rem;margin:10px 0 6px;'>{why}</p>
                    {"<div><small style='color:#ef9a9a;'>Missing: " + "".join(f"<span class='chip chip-missing'>{m}</span>" for m in missing) + "</small></div>" if missing else "<p style='color:#81c784;font-size:0.85rem;'>✅ You have all ingredients!</p>"}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_exp, col_save = st.columns([3, 1])
            with col_exp:
                with st.expander("View all ingredients"):
                    all_ing = recipe.get("ingredients", [])
                    st.markdown(" ".join(f"<span class='chip'>{i}</span>" for i in all_ing), unsafe_allow_html=True)

            if show_save:
                with col_save:
                    fav_ids = [f["recipe"].get("id", f["recipe"]["name"]) for f in st.session_state.favourites]
                    recipe_id = recipe.get("id", recipe["name"])
                    if recipe_id not in fav_ids:
                        if st.button("❤️ Save", key=f"save_{recipe_id}"):
                            st.session_state.favourites.append(rec)
                            st.toast(f"Saved {recipe['name']}!", icon="❤️")
                            st.rerun()
                    else:
                        st.button("✅ Saved", key=f"saved_{recipe_id}", disabled=True)


# ─────────────────────────────────────────────────────────────
# TAB 1 — CAMERA SCAN
# ─────────────────────────────────────────────────────────────
with tab_scan:
    col_cam, col_results = st.columns([1, 1], gap="large")

    with col_cam:
        image = camera_input_section()

        if image:
            with st.spinner("🔍 Detecting ingredients…"):
                detections, annotated = run_detection(image, conf=conf_thresh)

            if annotated is not None:
                display_annotated_image(annotated, "Detection Results")
            else:
                st.image(image, caption="Captured Image", use_column_width=True)

            labels = unique_labels(detections)
            st.session_state.last_detections = labels

            if labels:
                st.markdown("**🥕 Detected ingredients:**")
                chips_html = "".join(
                    f"<span class='chip chip-detected'>{l} "
                    f"<small style='opacity:0.7'>{next(d.confidence for d in detections if d.label==l):.0%}</small>"
                    f"</span>"
                    for l in labels
                )
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.warning("No ingredients detected. Try better lighting or a closer shot.")

    with col_results:
        detected = st.session_state.last_detections
        if not detected and not st.session_state.pantry:
            st.markdown(
                "<div style='text-align:center;padding:4rem 1rem;color:#546e7a;'>"
                "<div style='font-size:4rem;'>📷</div>"
                "<p>Capture a photo on the left to detect ingredients</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            diet_val = None if diet == "Any" else diet
            all_ingredients = list(set(detected + st.session_state.pantry))

            st.markdown(
                f"<p class='section-title'>Recipes for you</p>"
                f"<p style='color:#78909c;font-size:0.85rem;'>Based on {len(all_ingredients)} ingredients</p>",
                unsafe_allow_html=True,
            )

            with st.spinner("🍳 Finding recipes…"):
                recs = get_recommendations(detected, st.session_state.pantry, diet_val, top_k)

            if not recs:
                st.info("No recipes found. Try adding more pantry staples (oil, salt, garlic).")
            else:
                _render_recipe_cards(recs)


# ─────────────────────────────────────────────────────────────
# TAB 2 — MANUAL INPUT
# ─────────────────────────────────────────────────────────────
with tab_manual:
    st.markdown("### ✏️ Type Your Ingredients")
    st.caption("Enter ingredients separated by commas.")

    manual_text = st.text_area(
        "Ingredients",
        placeholder="tomato, onion, chicken, ginger, garlic, chilli…",
        height=100,
        label_visibility="collapsed",
        key="manual_ingredients",
    )

    col_btn, col_clear = st.columns([2, 1])
    search_clicked = col_btn.button("🔍 Find Recipes", type="primary", use_container_width=True)
    if col_clear.button("Clear", use_container_width=True):
        st.session_state["manual_ingredients"] = ""
        st.rerun()

    if search_clicked and manual_text.strip():
        manual_list = [i.strip().lower() for i in manual_text.split(",") if i.strip()]
        diet_val = None if diet == "Any" else diet
        with st.spinner("🍳 Finding recipes…"):
            recs = get_recommendations(manual_list, st.session_state.pantry, diet_val, top_k)

        st.markdown(f"### Results ({len(recs)} recipes found)")
        if recs:
            _render_recipe_cards(recs)
        else:
            st.info("No recipes found. Try fewer or more common ingredients.")


# ─────────────────────────────────────────────────────────────
# TAB 3 — FAVOURITES
# ─────────────────────────────────────────────────────────────
with tab_favs:
    if not st.session_state.favourites:
        st.markdown(
            "<div style='text-align:center;padding:4rem;color:#546e7a;'>"
            "<div style='font-size:4rem;'>❤️</div>"
            "<p>No favourites yet. Save recipes from the Scan or Manual tabs!</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"### Your {len(st.session_state.favourites)} Saved Recipes")
        _render_recipe_cards(st.session_state.favourites, show_save=False)
        if st.button("🗑️ Clear All Favourites"):
            st.session_state.favourites = []
            st.rerun()



