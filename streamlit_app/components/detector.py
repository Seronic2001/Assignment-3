# ── Detector Component ─────────────────────────────────────────────────────
# Primary:   YOLOv8s  — object detection (bounding boxes, multi-object)
# Secondary: ResNet-50 — re-classifies each YOLO crop for higher label accuracy
#            (95% accuracy, 51 classes from Food_Ingredient_classification_51)

import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
from typing import NamedTuple

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH      = Path(__file__).parent.parent / "assets" / "best.pt"
ONNX_PATH       = Path(__file__).parent.parent / "assets" / "best.onnx"
CLASSIFIER_PATH = Path(__file__).parent.parent.parent / \
                  "Food_Ingredient_classification_51" / "fruits_vegetables_51.pth"

# Confidence thresholds
CONF_THRESH        = 0.25   # Lower threshold to catch more objects in cluttered scenes
                            # ResNet-50 re-classifier corrects any wrong labels
CLASSIFIER_THRESH  = 0.75   # ResNet must be ≥ this to override YOLO's label


# ── ResNet-50 class names (51 classes, lowercased for recipe matching) ────────
RESNET_CLASSES = [
    "amaranth", "apple", "banana", "beetroot", "bell pepper", "bitter gourd",
    "blueberry", "bottle gourd", "broccoli", "cabbage", "cantaloupe",
    "capsicum", "carrot", "cauliflower", "chilli", "coconut", "corn",
    "cucumber", "dragon fruit", "eggplant", "fig", "garlic", "ginger",
    "grape", "jalapeno", "kiwi", "lemon", "mango", "okra", "onion",
    "orange", "paprika", "pear", "pea", "pineapple", "pomegranate",
    "potato", "pumpkin", "radish", "raspberry", "ridge gourd", "soy bean",
    "spinach", "spiny gourd", "sponge gourd", "strawberry", "sweetcorn",
    "sweet potato", "tomato", "turnip", "watermelon",
]


class Detection(NamedTuple):
    label: str
    confidence: float
    box: tuple          # (x1, y1, x2, y2) normalised 0-1
    source: str = "yolo"  # "yolo" | "resnet"


# ── YOLO loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection model…")
def _load_yolo():
    """Returns (model, format_str) — no st.* calls (cache_resource restriction)."""
    try:
        from ultralytics import YOLO
        if MODEL_PATH.exists():
            return YOLO(str(MODEL_PATH)), "pt"
        elif ONNX_PATH.exists():
            return YOLO(str(ONNX_PATH), task="detect"), "onnx"
        return None, None
    except Exception as e:
        return None, str(e)


# ── ResNet-50 classifier loader ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading classifier model…")
def _load_classifier():
    """Returns (model, transform, device) or (None, None, None) if unavailable."""
    if not CLASSIFIER_PATH.exists():
        return None, None, None
    try:
        import torch
        from torchvision import models, transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        clf = models.resnet50(weights=None)
        clf.fc = torch.nn.Linear(clf.fc.in_features, len(RESNET_CLASSES))
        clf.load_state_dict(
            torch.load(str(CLASSIFIER_PATH), map_location=device)
        )
        clf.eval().to(device)

        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        return clf, tfm, device
    except Exception:
        return None, None, None


def _classify_crop(crop_pil: Image.Image) -> tuple[str, float] | None:
    """
    Run ResNet-50 on a cropped PIL image.
    Returns (class_name, confidence) or None if classifier unavailable.
    """
    clf, tfm, device = _load_classifier()
    if clf is None:
        return None
    import torch
    tensor = tfm(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = clf(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
    return RESNET_CLASSES[idx.item()], conf.item()


# ── Main detection function ───────────────────────────────────────────────────
def run_detection(image: Image.Image, conf: float = CONF_THRESH) -> tuple[list[Detection], np.ndarray | None]:
    """
    Run YOLOv8 detection + optional ResNet-50 re-classification per crop.

    Pipeline:
        1. YOLO detects bounding boxes in the full image
        2. Each crop is sent to ResNet-50
        3. If ResNet confidence >= CLASSIFIER_THRESH, its label overrides YOLO's

    Args:
        image : PIL Image (RGB)
        conf  : YOLO detection confidence threshold (default: CONF_THRESH)

    Returns:
        detections : list[Detection]
        annotated  : BGR numpy array with boxes drawn (or None)
    """
    yolo, status = _load_yolo()

    # ── One-time UI toast (must be outside cached fn) ────────────────────────
    if "model_toast_shown" not in st.session_state:
        st.session_state.model_toast_shown = True
        clf, _, _ = _load_classifier()
        if yolo is not None:
            extra = " + ResNet-50 classifier" if clf is not None else ""
            st.toast(f"✅ YOLOv8s{extra} loaded", icon="🤖")
        elif status:
            st.warning(f"YOLO load failed: {status}")

    # ── Demo fallback ─────────────────────────────────────────────────────────
    if yolo is None:
        st.info("⚠️ No trained model found in `assets/`. Running in **demo mode**.")
        return [
            Detection("tomato",  0.91, (0.1, 0.1, 0.3, 0.3)),
            Detection("onion",   0.87, (0.4, 0.2, 0.6, 0.5)),
            Detection("ginger",  0.72, (0.65, 0.6, 0.85, 0.9)),
        ], None

    # ── YOLO inference ────────────────────────────────────────────────────────
    img_rgb = np.array(image)
    img_bgr = img_rgb[:, :, ::-1]
    results = yolo(img_bgr, conf=conf, verbose=False)
    result  = results[0]
    h, w    = img_rgb.shape[:2]

    yolo_detections: list[Detection] = []
    for box in result.boxes:
        cls_id    = int(box.cls[0])
        yolo_conf = float(box.conf[0])
        yolo_lbl  = yolo.names[cls_id].lower()
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        final_label, final_conf, final_source = yolo_lbl, yolo_conf, "yolo"
        crop = image.crop((x1, y1, x2, y2))
        if crop.width > 10 and crop.height > 10:
            clf_result = _classify_crop(crop)
            if clf_result and clf_result[1] >= CLASSIFIER_THRESH:
                final_label, final_conf, final_source = *clf_result, "resnet"

        yolo_detections.append(Detection(
            label=final_label, confidence=final_conf,
            box=(x1/w, y1/h, x2/w, y2/h), source=final_source,
        ))

    # ── Filter bad YOLO: if >60% of detections share one label, discard all ──
    if yolo_detections:
        from collections import Counter
        label_counts = Counter(d.label for d in yolo_detections)
        dominant     = label_counts.most_common(1)[0]
        if dominant[1] / len(yolo_detections) > 0.6:
            yolo_detections = [d for d in yolo_detections
                               if d.label != dominant[0]]

    # ── ResNet-50 grid scan (always runs — primary ingredient finder) ─────────
    grid_detections = _grid_scan(image)

    # Merge: grid scan first, then well-labelled YOLO boxes
    seen_labels: set[str] = {d.label for d in grid_detections}
    for d in yolo_detections:
        if d.label not in seen_labels:
            grid_detections.append(d)
            seen_labels.add(d.label)

    annotated = result.plot()   # BGR numpy with YOLO boxes (for visual reference)
    return grid_detections, annotated


def _grid_scan(image: Image.Image,
               rows: int = 4, cols: int = 4,
               clf_thresh: float = 0.72) -> list[Detection]:
    """
    Divide the image into a rows×cols grid and classify each cell with ResNet-50.
    Cells whose prediction confidence < clf_thresh are skipped (background).
    Returns unique ingredient detections.
    """
    w, h = image.size
    cw, ch = w // cols, h // rows
    seen: dict[str, float] = {}
    detections: list[Detection] = []

    for row in range(rows):
        for col in range(cols):
            x1, y1 = col * cw, row * ch
            x2, y2 = x1 + cw, y1 + ch
            crop = image.crop((x1, y1, x2, y2))
            clf_result = _classify_crop(crop)
            if clf_result is None:
                continue
            lbl, c = clf_result
            if c >= clf_thresh and (lbl not in seen or c > seen[lbl]):
                seen[lbl] = c
                detections.append(Detection(
                    label=lbl, confidence=c,
                    box=(x1/w, y1/h, x2/w, y2/h),
                    source="resnet-grid",
                ))

    # deduplicate: keep highest-confidence entry per label
    best: dict[str, Detection] = {}
    for d in detections:
        if d.label not in best or d.confidence > best[d.label].confidence:
            best[d.label] = d
    return list(best.values())


def unique_labels(detections: list[Detection]) -> list[str]:
    """Return deduplicated ingredient labels (highest confidence wins)."""
    seen: dict[str, float] = {}
    for d in detections:
        if d.label not in seen or d.confidence > seen[d.label]:
            seen[d.label] = d.confidence
    return list(seen.keys())
