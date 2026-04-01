# Data Creation & Model Training Guide

This guide covers everything needed to build a better dataset and train a higher-quality manga text detector.

---

## Part 1 — Why the Current Model Struggles

The shipped checkpoint (`models/checkpoints/custom_yolo_best.pt`) is a starting point, not a finished model. Common failure modes:

| Symptom | Root cause |
|---|---|
| Misses small text bubbles | Low-confidence detections filtered out; model undertrained on small instances |
| Overlapping boxes on same bubble | NMS IoU threshold too permissive; duplicate detections from different scales |
| Sound effects / signs missed | Class imbalance — Dialogue dominates the training set |
| Poor translation on badly cropped regions | OCR sees partial text because the bounding box is too tight or too loose |

The fix for all of these is **more diverse, well-balanced training data** and a **better training recipe**.

---

## Part 2 — Data Creation Pipeline

### 2.1 Where to Get Raw Images

| Source | Notes |
|---|---|
| `data/scrapers/` | Rawkuma scraper already in the repo — run it against chapters you have rights to use |
| Your own manga scans | JPG/PNG, any resolution — the pipeline handles resizing |
| Public-domain manga | Works published before copyright cutoff (e.g., old Tezuka works) |

**Minimum useful dataset size:** 500 pages per class for decent performance.
**Target:** 2,000+ pages with balanced class counts across all 5 classes.

### 2.2 Annotation Format

All labels must be in **YOLO `.txt` format**, one file per image, same filename stem:

```
# format: class_id cx cy w h   (all values 0-1, relative to image size)
0 0.512 0.234 0.310 0.187
1 0.780 0.410 0.120 0.065
```

Classes:
```
0  Dialogue       — speech bubbles with a tail pointing at a character
1  Sound_Effects  — onomatopoeia (sfx), usually drawn into the panel art
2  Signs          — background text: signs, labels, titles on objects
3  Text           — caption boxes, narration boxes, standalone text
4  removal        — regions to erase with no translation (stamps, watermarks)
```

### 2.3 Manual Annotation (Best Quality)

Use **LabelImg** (YOLO mode) or **Roboflow** (web-based):

```bash
pip install labelImg
labelImg data/manga_chapters/047 data/labels/047
```

Workflow:
1. Open a chapter folder in LabelImg
2. Set format to YOLO
3. Draw boxes around every text region, assign the correct class
4. Save — one `.txt` per image is created automatically

**Tips for better boxes:**
- Include the full bubble including its border (1-2px padding is fine)
- For overlapping bubbles, draw separate boxes — do not merge them
- Sound effects that span the whole panel: draw the tight bounding box around just the SFX text, not the whole panel
- Skip panels where text is 90%+ illegible (blurry, rotated > 45°)

### 2.4 Assisted / Semi-Automatic Annotation (Faster)

The repo already has a pseudo-label workflow. Use the current best model to generate candidates, then review:

```bash
# Step 1: generate candidates
python training/create_pseudo_labels.py \
  --model models/checkpoints/custom_yolo_best.pt \
  --input data/manga_chapters/new_chapter \
  --output data/pseudo_labels/new_chapter \
  --confidence 0.25

# Step 2: open the review manifest in the Admin → Data page
# Approve / reject / mark uncertain for each image
# Approved images go straight into training

# Step 3: apply decisions
python training/apply_pseudo_label_review.py \
  --manifest data/pseudo_labels/new_chapter/review_manifest.json
```

**Quality control rule:** only approve pages where ALL major text regions are correctly boxed. One missed bubble is acceptable; three missed bubbles means reject and annotate manually.

### 2.5 Dataset Structure

```
data/additional_data/
├── images/
│   ├── train/    ← 80% of pages
│   └── val/      ← 20% of pages (held out, never trained on)
└── labels/
    ├── train/    ← matching .txt files
    └── val/
```

The `custom_manga.yaml` config already points here. Add new pages by copying images and labels into the correct split folders.

**Recommended train/val split:** 80/20 random split *per chapter*, not per page.
This prevents the model from memorising chapter-specific art styles.

### 2.6 Class Balance Check

Run this before training to see class distribution:

```bash
python - <<'EOF'
from pathlib import Path
from collections import Counter

label_dir = Path("data/additional_data/labels/train")
counts = Counter()
for f in label_dir.glob("*.txt"):
    for line in f.read_text().splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1

names = {0:"Dialogue", 1:"Sound_Effects", 2:"Signs", 3:"Text", 4:"removal"}
total = sum(counts.values())
for cid, name in names.items():
    n = counts.get(cid, 0)
    print(f"  {name:16s}: {n:6d}  ({100*n/total:.1f}%)")
EOF
```

If any class is below 5% of total boxes, collect more examples of that class before training.

---

## Part 3 — Improved Training Methods

### 3.1 Recommended Model

Start from **YOLOv8m** (medium) or **YOLOv8l** (large). The current checkpoint uses a smaller variant.

```bash
# download base weights once
python - -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### 3.2 Two-Stage Training Recipe

#### Stage 1 — Frozen backbone, train head only (5 epochs)

Warms up the detection head before unfreezing. Prevents the pretrained backbone features from being destroyed in the first few gradient steps.

```bash
python training/advanced_train_yolo.py \
  --data training/datasets/custom_manga.yaml \
  --model yolov8m.pt \
  --epochs-stage1 5 \
  --epochs-stage2 60 \
  --imgsz 1280 \
  --batch 8 \
  --device 0
```

Key flags for manga:
- `--imgsz 1280` — manga pages are tall/high-res; 640 misses small SFX text
- `--batch 8` — reduce if you get OOM; increase if GPU has headroom
- `--device 0` — GPU; use `cpu` if no GPU available (much slower)

#### Stage 2 — Full fine-tune with augmentation

The `advanced_train_yolo.py` script handles unfreezing automatically after stage 1.

### 3.3 Hyperparameter Improvements

Create `training/manga_hyps.yaml`:

```yaml
# Manga-specific hyperparameters
lr0: 0.005          # Lower initial LR for fine-tuning (default 0.01)
lrf: 0.01           # Final LR as fraction of lr0
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8

# Augmentation — manga-specific
hsv_h: 0.0          # Manga is greyscale; hue shift wastes capacity
hsv_s: 0.2          # Minor saturation shift handles toned pages
hsv_v: 0.3          # Brightness shift handles scan quality variation
degrees: 5.0        # Small rotation — manga pages are mostly upright
translate: 0.1
scale: 0.5          # Scale jitter helps with different bubble sizes
flipud: 0.0         # Manga text is rarely upside-down
fliplr: 0.5         # Horizontal flip is safe for detection

# Class weights — boost underrepresented classes
cls: 0.5
```

Use it with:
```bash
python training/train_yolo.py \
  --data training/datasets/custom_manga.yaml \
  --hyp training/manga_hyps.yaml \
  --model yolov8m.pt \
  --epochs 80 \
  --imgsz 1280
```

### 3.4 Reducing Overlapping Boxes

Overlapping detections come from two sources:

**A. NMS threshold too high** — lower the IoU threshold so more overlaps are suppressed:
```python
# In config/settings.py
DEFAULT_IOU_THRESHOLD = 0.45   # was 0.55
```

**B. Multi-scale duplicates** — the model predicts the same bubble at two scales.
Fix during training by adding `overlap_mask: True` in the data yaml, or use the existing `src/models/advanced_nms.py` module (it is imported but not wired into the default path — wire it in):

```python
# In src/models/detector.py, after YOLO inference:
from src.models.advanced_nms import apply_advanced_nms
result = apply_advanced_nms(result, iou_threshold=0.45, class_agnostic=True)
```

**C. Model confidence calibration** — the overlap is worse at low confidence because many marginal detections overlap valid ones. The `MAX_REGIONS_PER_PAGE = 40` cap added to the pipeline prevents the worst cases.

### 3.5 Evaluate After Every Training Run

```bash
python evaluation/evaluate_model.py \
  --model yolo_train_run/YOUR_RUN/weights/best.pt \
  --data training/datasets/custom_manga.yaml
```

Look at the **per-class AP** in the HTML report. A model is ready when:
- Dialogue AP > 0.75
- Sound_Effects AP > 0.55
- Signs AP > 0.50
- Text AP > 0.55
- removal AP > 0.60
- Macro F1 > 0.60

If Sound_Effects or Signs lag behind, collect more pages that feature those classes and retrain.

### 3.6 Translation Quality Improvements

The detector quality only controls what gets sent to OCR and translation. For better translation output:

| Issue | Fix |
|---|---|
| OCR reads garbage from non-text regions | Raise confidence threshold (0.15–0.20 is a better default than 0.10 for clean results) |
| Gemma3 translates SFX literally | SFX class should not be translated — add a class-based skip in `pipeline.py` |
| Short text ("！") mistranslated | Pre/post-process: single punctuation → pass through unchanged |
| Long dialogue broken across lines | OCR already merges; if not, try `manga-ocr` with `force_cpu=False` |

**Skip SFX translation** (add to `pipeline.py` inside the region loop):

```python
# Sound Effects are onomatopoeia — translating them literally looks wrong.
# Keep the original Japanese SFX or skip rendering them.
if region_type == TextRegionType.SOUND_EFFECTS:
    # optionally: translated_text = original_text  # keep as-is
    continue  # or skip entirely — removes SFX bubble text
```

---

## Part 4 — Quick Reference Commands

```bash
# Activate environment
source .venv/bin/activate

# Check class balance before training
python tools/check_class_balance.py  # or the snippet in §2.6

# Two-stage training (recommended)
python training/advanced_train_yolo.py \
  --data training/datasets/custom_manga.yaml \
  --model yolov8m.pt \
  --imgsz 1280

# Evaluate
python evaluation/evaluate_model.py \
  --model yolo_train_run/advanced_train/weights/best.pt \
  --data training/datasets/custom_manga.yaml

# Generate pseudo-labels for a new chapter
python training/create_pseudo_labels.py \
  --model yolo_train_run/advanced_train/weights/best.pt \
  --input data/manga_chapters/NEW \
  --output data/pseudo_labels/NEW

# Threshold sweep to find best confidence cutoff
python evaluation/threshold_sweep.py \
  --model yolo_train_run/advanced_train/weights/best.pt \
  --data training/datasets/custom_manga.yaml
```

---

## Part 5 — Iteration Loop (The Fast Path to a Better Model)

```
1. Collect 20-30 new diverse manga pages
2. Run pseudo-label generation at confidence=0.25
3. Review in Admin → Data (approve/reject per image)
4. Copy approved labels to data/additional_data/labels/train
5. Re-run training (just stage 2, 20 more epochs, from last checkpoint)
6. Evaluate — check per-class AP
7. Repeat
```

Each iteration takes roughly:
- Manual review: 1–2 hours per 100 pages
- Training (GPU): 30–60 minutes for 20 epochs at imgsz=1280
- Evaluation: 5 minutes

After 3–4 iterations with fresh chapter data, Dialogue AP typically reaches 0.80+ and the other classes improve proportionally.
