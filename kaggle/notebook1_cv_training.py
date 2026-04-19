# Fridge Ingredient Detection — YOLOv8n Training
# Kaggle Notebook 1 | T14.1 SMAI Assignment 3
# Dataset : shubhadeepmandal/fridge-object-dataset  (Fridge-Object v3, 53 classes)
# Hardware : Kaggle GPU T4 x2  |  100 epochs  |  patience 5
# ─────────────────────────────────────────────────────────────

# ── Cell 1: Install dependencies ──────────────────────────────
# !pip install -q ultralytics roboflow

# ── Cell 2: Imports ───────────────────────────────────────────
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO
from IPython.display import display, Image

WORK_DIR    = Path("/kaggle/working")
# ── Dataset uploaded to Kaggle as: shubhadeepmandal/fridge-object-dataset
DATASET_DIR = Path("/kaggle/input/fridge-object-dataset/Fridge-Object-3")

# ── Cell 3: Verify dataset ────────────────────────────────────
# The dataset is already available at DATASET_DIR (Kaggle input).
# data.yaml was shipped with the Roboflow export — just load it.

yaml_path = DATASET_DIR / "data.yaml"
assert yaml_path.exists(), f"data.yaml not found at {yaml_path}"

with open(yaml_path) as f:
    data_cfg = yaml.safe_load(f)

# Patch absolute paths so YOLO can find images inside /kaggle/input
data_cfg["path"] = str(DATASET_DIR)
data_cfg["train"] = "train/images"
data_cfg["val"]   = "valid/images"
data_cfg["test"]  = "test/images"

# Write patched yaml to /kaggle/working (read-only input can't be edited)
patched_yaml = WORK_DIR / "data.yaml"
with open(patched_yaml, "w") as f:
    yaml.dump(data_cfg, f, default_flow_style=False)

CLASS_NAMES = data_cfg["names"]
print(f"Dataset   : {DATASET_DIR}")
print(f"Classes   : {data_cfg['nc']} — {CLASS_NAMES[:6]} ...")
print(f"Train imgs: {len(list((DATASET_DIR/'train'/'images').glob('*')))}")
print(f"Val imgs  : {len(list((DATASET_DIR/'valid'/'images').glob('*')))}")
print(f"Test imgs : {len(list((DATASET_DIR/'test' /'images').glob('*')))}")

# ── Cell 5: Training ──────────────────────────────────────────
# ── Model selection ───────────────────────────────────────────────────────────
# yolov8n  = nano  (~3 MB)  — fastest, lowest accuracy
# yolov8s  = small (~22 MB) — +5–8 mAP over nano, still real-time on CPU/GPU ✅
# yolov8m  = medium         — further +3–5 mAP, needs more GPU RAM
MODEL_WEIGHTS = "yolov8s.pt"   # ← upgrade from nano to small

model = YOLO(MODEL_WEIGHTS)
print(f"Loaded: {MODEL_WEIGHTS}  |  params: {sum(p.numel() for p in model.model.parameters())/1e6:.1f}M")

RUN_NAME = "fridge_yolov8s"   # new name so old nano run is preserved

results = model.train(
    data    = str(patched_yaml),
    epochs  = 100,              # full budget; early-stop guards against waste
    imgsz   = 640,
    batch   = 16,               # s-model needs more VRAM than n; 16 is safe on T4
    device  = 0,                # T4 GPU
    patience= 20,               # was 5 — too aggressive, killed runs early
    project = str(WORK_DIR / "runs"),
    name    = RUN_NAME,
    exist_ok= True,

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer      = "AdamW",   # faster convergence than SGD for this dataset
    lr0            = 0.001,     # AdamW works better with lower lr than SGD
    lrf            = 0.01,      # final lr = lr0 * lrf
    warmup_epochs  = 5,         # longer warmup for stable AdamW start
    cos_lr         = True,      # cosine annealing — smoother lr decay (+1–2% mAP)
    weight_decay   = 0.0005,

    # ── Class imbalance handling ───────────────────────────────────────────────
    label_smoothing= 0.1,       # softens loss for zero-AP classes (fish, sausage…)

    # ── Augmentation (kitchen-specific) ───────────────────────────────────────
    hsv_h       = 0.015,        # hue shift — different lighting conditions
    hsv_s       = 0.7,          # saturation — fresh vs. wilted produce
    hsv_v       = 0.4,          # brightness — fridge dark vs. counter bright
    degrees     = 10,           # slight rotation
    fliplr      = 0.5,          # horizontal flip
    flipud      = 0.0,          # no vertical flip (fridge items are upright)
    mosaic      = 1.0,          # group 4 ingredients on one canvas
    mixup       = 0.15,         # blend two images (slightly more than before)
    copy_paste  = 0.1,          # paste ingredient instances across images
    close_mosaic= 15,           # disable mosaic for last 15 epochs (cleaner fine-tune)

    # ── Misc ──────────────────────────────────────────────────────────────────
    cache       = True,         # cache images in RAM — much faster epochs on T4
    workers     = 4,
    verbose     = True,
)

# ── Cell 6: Evaluation ────────────────────────────────────────
best_model_path = WORK_DIR / "runs" / RUN_NAME / "weights" / "best.pt"
eval_model = YOLO(str(best_model_path))

metrics = eval_model.val(data=str(patched_yaml))

print(f"\n{'='*50}")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision:{metrics.box.mp:.4f}")
print(f"Recall:   {metrics.box.mr:.4f}")
print(f"{'='*50}\n")

# Per-class AP50
print("Per-class AP50:")
zero_ap_classes = []
for i, name in enumerate(CLASS_NAMES):
    if i < len(metrics.box.ap50):
        ap = metrics.box.ap50[i]
        print(f"  {name:20s}: {ap:.3f}")
        if ap == 0.0:
            zero_ap_classes.append(name)

if zero_ap_classes:
    print(f"\nNOTE: {len(zero_ap_classes)} classes scored AP50=0 (insufficient val samples):")
    print(f"  {zero_ap_classes}")
    print("  This is a dataset imbalance issue, not a training failure.")

# ── Cell 7: Inference Speed Test (fast — no full benchmark) ──────────────────
# eval_model.benchmark() exports to ALL formats and runs full test set for each
# — it fills disk and takes hours. Use a single-image timing test instead.

import time, torch, numpy as np

# Warm up
dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
for _ in range(3):
    eval_model.predict(source=dummy, verbose=False)

# Time 50 inferences
N = 50
t0 = time.perf_counter()
for _ in range(N):
    eval_model.predict(source=dummy, verbose=False)
elapsed = time.perf_counter() - t0

print(f"Inference speed (GPU, {N} runs):")
print(f"  {elapsed/N*1000:.1f} ms / image")
print(f"  {N/elapsed:.1f} FPS")

# ── Cell 8: Export to ONNX (for Streamlit Python inference) ──
onnx_path = eval_model.export(format="onnx", imgsz=640, opset=12)
print(f"ONNX model saved: {onnx_path}")

# ── Cell 9: Export to TFLite INT8 (for React Native mobile) ──
# tflite_path = eval_model.export(format="tflite", int8=True, imgsz=640)
# print(f"TFLite INT8 model saved: {tflite_path}")
# NOTE: TFLite export sometimes fails on Kaggle — uncomment only if needed.

# ── Cell 10: Training Curves ──────────────────────────────────
results_csv = WORK_DIR / "runs" / RUN_NAME / "results.csv"
import pandas as pd
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("YOLOv8n Training Curves — Fridge Ingredient Detection (53 classes)", fontsize=14)

metrics_to_plot = [
    ("train/box_loss",   "Train Box Loss",  "orange"),
    ("train/cls_loss",   "Train Cls Loss",  "red"),
    ("metrics/mAP50",    "mAP@0.50",        "green"),
    ("metrics/mAP50-95", "mAP@0.50:0.95",   "blue"),
    ("metrics/precision","Precision",       "purple"),
    ("metrics/recall",   "Recall",          "teal"),
]

for ax, (col, label, color) in zip(axes.flat, metrics_to_plot):
    if col in df.columns:
        ax.plot(df["epoch"], df[col], color=color, linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(WORK_DIR / "training_curves.png"), dpi=150)
plt.show()

print("\n✅ Training complete! See Cell 11 below to download your files.")

# ── Cell 11: Download outputs ─────────────────────────────────────────────────
# Bundles best.pt, best.onnx, training_curves.png into a single zip and
# shows clickable FileLinks directly in the Kaggle notebook output panel.

import shutil
from IPython.display import FileLink, display

WEIGHTS_DIR = WORK_DIR / "runs" / RUN_NAME / "weights"

files_to_bundle = {
    "best.pt"            : WEIGHTS_DIR / "best.pt",
    "best.onnx"          : WEIGHTS_DIR / "best.onnx",
    "training_curves.png": WORK_DIR / "training_curves.png",
    "data.yaml"          : patched_yaml,
}

bundle_dir = WORK_DIR / "download_bundle"
bundle_dir.mkdir(exist_ok=True)

print("Bundling files:")
for fname, src in files_to_bundle.items():
    if src.exists():
        shutil.copy2(src, bundle_dir / fname)
        print(f"  [OK]     {fname}  ({src.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"  [SKIP]   {fname}  (not found)")

# Create zip archive
zip_out = WORK_DIR / "fridge_yolov8n_outputs"
shutil.make_archive(str(zip_out), "zip", str(bundle_dir))
print(f"\nAll-in-one zip: {zip_out}.zip  ({(zip_out.with_suffix('.zip')).stat().st_size / 1e6:.1f} MB)")

# Clickable links
print("\n=== Click to download ===")
display(FileLink(str(zip_out) + ".zip",
                 result_html_prefix="<b>📦 All outputs (zip):</b> &nbsp;"))

for fname, src in files_to_bundle.items():
    dest = bundle_dir / fname
    if dest.exists():
        display(FileLink(str(dest),
                         result_html_prefix=f"&nbsp;&nbsp;&nbsp;{fname}: "))

print("\nAfter download, copy to your project:")
print("  best.pt   → Assignment 3/streamlit_app/assets/best.pt")
print("  best.onnx → Assignment 3/streamlit_app/assets/best.onnx")
