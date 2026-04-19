# T14.1 — Cook With What You Have 🍛
### Camera-Based Personalised Indian Recipe Recommender | SMAI Assignment 3

---

## Project Overview
Point a camera at your fridge or kitchen counter. A **YOLOv8n** model detects the ingredients in real time. A **hybrid recommender engine** (Jaccard + TF-IDF cosine similarity) then returns ranked Indian recipes you can actually make — with explanations for each suggestion.

---

## Architecture

```
Camera (Webcam / Phone)
        │
        ▼
┌───────────────────┐       ┌────────────────────────┐
│  YOLOv8n (ONNX)  │──────▶│  Detected Ingredients  │
│  Streamlit / RN   │       │  [ tomato, onion, ... ] │
└───────────────────┘       └──────────┬─────────────┘
                                       │
                            ┌──────────▼─────────────┐
                            │  Virtual Pantry (state) │
                            └──────────┬──────────────┘
                                       │
                            ┌──────────▼─────────────┐
                            │   FastAPI Backend       │
                            │   Jaccard + TF-IDF      │
                            └──────────┬──────────────┘
                                       │
                    ┌──────────────────┴────────────────┐
                    │                                   │
         ┌──────────▼──────────┐             ┌──────────▼──────────┐
         │  Streamlit Web App  │             │  React Native (RN)  │
         │  (Primary Demo)     │             │  (Bonus / Mobile)   │
         └─────────────────────┘             └─────────────────────┘
```

---

## Project Structure

```
Assignment 3/
├── README.md
│
├── backend/                        ← FastAPI Recommendation API
│   ├── main.py                     ← API server (hybrid scoring)
│   ├── models.py                   ← Pydantic schemas
│   ├── requirements.txt
│   ├── recipe_db.json              ← [Download from Kaggle Notebook 2]
│   ├── vectorizer.pkl              ← [Download from Kaggle Notebook 2]
│   ├── tfidf_matrix.pkl            ← [Download from Kaggle Notebook 2]
│   └── ingredient_classes.json     ← [Download from Kaggle Notebook 2]
│
├── streamlit_app/                  ← Streamlit Web App (Primary Demo)
│   ├── app.py                      ← Main entry point
│   ├── requirements.txt
│   ├── assets/
│   │   ├── best.pt                 ← [Download from Kaggle Notebook 1]
│   │   └── best.onnx               ← [Download from Kaggle Notebook 1]
│   └── components/
│       ├── camera.py               ← Camera + upload widget
│       ├── detector.py             ← YOLOv8 inference (demo fallback)
│       └── recommender.py          ← API client + inline fallback engine
│
└── kaggle/
    ├── notebook1_cv_training.py    ← YOLOv8n training + export
    └── notebook2_data_prep.py      ← Recipe DB prep + TF-IDF build
```

---

## Execution Order

### Step 1 — Prepare the recipe database (Kaggle)
1. Open `kaggle/notebook2_data_prep.py` in a **new Kaggle Notebook**.
2. Add dataset: *6000 Indian Recipes* (search on Kaggle).
3. Run all cells. Download:
   - `recipe_db.json` → `backend/`
   - `vectorizer.pkl` → `backend/`
   - `tfidf_matrix.pkl` → `backend/`
   - `ingredient_classes.json` → `backend/`

### Step 2 — Run the FastAPI backend
```powershell
cd "Assignment 3\backend"
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Test: open http://localhost:8000/docs

### Step 3 — Run the Streamlit app (works in demo mode without the model)
```powershell
cd "Assignment 3\streamlit_app"
pip install -r requirements.txt
streamlit run app.py
```

### Step 4 — Train the CV model (Kaggle)
1. Open `kaggle/notebook1_cv_training.py` in a **new Kaggle Notebook** with **GPU T4 x2**.
2. Add one or more ingredient detection datasets (see Recommended Datasets below).
3. Run all cells. Download:
   - `best.pt` → `streamlit_app/assets/`
   - `best.onnx` → `streamlit_app/assets/`

### Step 5 — Full end-to-end run
Restart the Streamlit app. The camera tab will now use the real YOLO model instead of demo mode.

---

## Recommended Datasets

| Dataset | Platform | Notes |
|---|---|---|
| 6000 Indian Recipes | Kaggle | Recipe corpus (required) |
| Fruit and Vegetable Image Recognition | Kaggle | 36 produce classes for YOLO |
| Vegetable Image Dataset | Kaggle | 21K vegetable images |
| Open Images V7 (food subset) | Google | Diverse lighting/angles |
| Roboflow Universe — ingredient detection | Roboflow | Pre-annotated, YOLO-ready |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **YOLOv8n** | ~3MB export, real-time on CPU or any GPU |
| **ONNX export** | Python-native for Streamlit; no native compilation needed |
| **Hybrid Jaccard + TF-IDF** | Jaccard checks raw coverage; TF-IDF handles synonyms/related terms |
| **35% threshold** | Low enough to show results for partial pantry matches |
| **Inline fallback** | Streamlit works without the API running — safer for demos |
| **Demo mode** | App is fully demonstrable before model training completes |

---

## Skills Demonstrated
- Computer Vision: object detection, model quantisation, edge inference
- RecSys: cosine similarity, TF-IDF, Jaccard overlap, hybrid scoring
- NLP: ingredient normalisation, lemmatisation, stop-word filtering
- MLOps: Kaggle training pipeline, model export (ONNX/TFLite)
- Full-stack: FastAPI + Streamlit + React Native (bonus)
