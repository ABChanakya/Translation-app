# Quickstart

This is the shortest current path for running and demonstrating Auto Manga Translation.

## 1. Install

From [manga_translator_clean](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have the local project environment, just activate it.

## 2. Optional Admin Login

The public UI works without admin credentials. The admin area requires:

```bash
export FLASK_SECRET_KEY='replace-this'
export ADMIN_USERNAME='admin'
export ADMIN_PASSWORD_HASH='paste-a-werkzeug-password-hash-here'
```

Generate a password hash with:

```bash
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your-password'))"
```

## 3. Start The App

```bash
python web/app.py
```

Open:

- Public UI: `http://localhost:5000`
- Admin login: `http://localhost:5000/admin/login`

## 4. Demo Flow

Public demo:
1. Open `Translate`
2. Upload one manga page
3. Choose an available translation engine
4. Translate and show the rendered result
5. If useful, show `Batch` or `Colorize` as secondary features

Admin demo:
1. Log into `/admin`
2. Open `Evaluation` to show class-fair reports
3. Open `Data` to show pseudo-label generation and review
4. Open `Training` only if you want to show the training pipeline

## 5. Train The Detector

Basic training:

```bash
python training/train_yolo.py \
  --data training/datasets/custom_manga.yaml
```

Advanced training:

```bash
python training/advanced_train_yolo.py \
  --data training/datasets/custom_manga.yaml
```

Train and then evaluate:

```bash
python training/train_and_eval.py \
  --data training/datasets/custom_manga.yaml
```

## 6. Evaluate With Fairness Reporting

Run evaluation against a checkpoint:

```bash
python evaluation/evaluate_model.py \
  --model models/checkpoints/custom_yolo_best.pt \
  --data training/datasets/custom_manga.yaml
```

Run a threshold sweep:

```bash
python evaluation/threshold_sweep.py \
  --model models/checkpoints/custom_yolo_best.pt \
  --data training/datasets/custom_manga.yaml
```

The generated evaluation outputs include:

- aggregate metrics such as `mAP50` and `mAP50_95`
- macro and weighted precision/recall/F1
- per-class support and prediction counts
- per-class AP and matched-box IoU
- HTML reports for presentation

## 7. Assisted Data Generation

Generate pseudo-labels from an older detector:

```bash
python training/create_pseudo_labels.py \
  --model models/checkpoints/custom_yolo_best.pt \
  --input data/manga_chapters/047 \
  --output data/pseudo_labels/run_047
```

Apply review decisions after inspection:

```bash
python training/apply_pseudo_label_review.py \
  --manifest data/pseudo_labels/run_047/review_manifest.json
```

The intended workflow is still whole-image review:

- approve good images
- reject bad images
- keep uncertain items separate until reviewed

## 8. OCR Model Access

The runtime OCR path uses `manga-ocr`.

If OCR loading fails because the model cannot be downloaded:

```bash
huggingface-cli login
```

Then restart the app or rerun the script.

If you want low-network or repeatable usage later, cache the model locally before a demo and keep using the same environment.

Current OCR runtime code:
- [src/models/ocr.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/models/ocr.py)

## 9. Colorization

The public `Colorize` page is kept intentionally, but it depends on separate colorization weights.

See:
- [colorization/readme.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/colorization/readme.md)

If the weights are missing, the UI will show a setup-needed state instead of pretending the feature works.

## 10. Read Next

- Main overview: [README.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/README.md)
- Full engineering/context guide: [DEVELOPER_GUIDE.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/DEVELOPER_GUIDE.md)
- Data workflow: [data/README.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/README.md)
- Accessibility and scaling reservation: [FUTURE_ACCESSIBILITY_AND_SCALING.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/FUTURE_ACCESSIBILITY_AND_SCALING.md)
