# Auto Manga Translation

Auto Manga Translation is the canonical application in this repository.

It has two surfaces:

- `Public UI`: translate pages, batch-translate folders/uploads, and use manga colorization as a secondary feature
- `Admin UI`: launch training and evaluation runs, review reports, inspect engines/models, and generate pseudo-labels from older detector + OCR models

The project is designed to be easy to demonstrate in public while still exposing the full training workflow in a separate protected area.

## Main Workflow

Public flow:
- detect text regions
- OCR the detected text
- translate the extracted text
- remove the original text
- render translated text back onto the page
- optionally export batch ZIP/PDF outputs

Admin flow:
- train or fine-tune the 5-class detector
- evaluate with mAP plus macro/per-class fairness reporting
- inspect per-class precision, recall, F1, AP, and matched-box IoU
- generate assisted labels from older models
- review pseudo-label manifests as `approve`, `reject`, or `uncertain`

## Canonical Entrypoints

Web app:
- [web/app.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/web/app.py)

Training:
- [training/train_yolo.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/train_yolo.py)
- [training/advanced_train_yolo.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/advanced_train_yolo.py)
- [training/train_and_eval.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/train_and_eval.py)

Evaluation:
- [evaluation/evaluate_model.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/evaluation/evaluate_model.py)
- [evaluation/threshold_sweep.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/evaluation/threshold_sweep.py)

Pseudo-label workflow:
- [training/create_pseudo_labels.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/create_pseudo_labels.py)
- [training/apply_pseudo_label_review.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/apply_pseudo_label_review.py)

## Documentation Set

Start here:
- [DOCS_COMPACT.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/DOCS_COMPACT.md)

## Quick Run

From this directory:

```bash
python web/app.py
```

Open `http://localhost:5000`.

If you want the admin area enabled, set:

- `FLASK_SECRET_KEY`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD_HASH`

## Notes

- The canonical detector workflow is the current 5-class setup under [training/datasets/custom_manga.yaml](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/datasets/custom_manga.yaml).
- The current fairness-first evaluation adds macro/per-class reporting so weak classes do not disappear behind dominant ones.
- Colorization remains part of the product, but it depends on separate weights and may show a setup-needed state if those assets are missing.
- Older demos and historical material were intentionally reduced from the active narrative; use the root archive only when you need background context.

## What Changed

- Public UI is now framed as one product: `Auto Manga Translation`
- Training, evaluation, engines, and data tools now live under `/admin`
- Translation engine support is availability-gated instead of hardcoded
- Evaluation now reports:
  - `mAP50`
  - `mAP50-95`
  - macro precision / recall / F1
  - weighted precision / recall / F1
  - per-class support and prediction counts
  - per-class AP and matched box IoU
  - weakest-class highlighting
- Evaluation/training reports now save:
  - `metrics.json`
  - `fairness.json`
  - `summary.json`
  - `report.html`
  - chart artifacts
- Pseudo-label generation now writes a review manifest and can attach OCR summaries
- Batch translation now supports chunked processing for lower-RAM systems
- Colorization is preserved and reports an honest setup-needed state when weights are missing

## Defaults

- Canonical dataset YAML: `training/datasets/custom_manga.yaml`
- Canonical detection task: 5 classes
- Public inference NMS IoU default: `0.55`
- Evaluation validation NMS IoU default: `0.6`
- Matched box IoU analysis default: `0.6`

## Quick Start

Install dependencies:

```bash
cd manga_translator_clean
pip install -r requirements.txt
```

Optional admin credentials:

```bash
export ADMIN_USERNAME="admin"
export ADMIN_PASSWORD_HASH="..."
export FLASK_SECRET_KEY="change-me"
```

Generate a password hash with:

```bash
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your-password'))"
```

Run the app:

```bash
python web/app.py
```

Open `http://localhost:5000`.

Public routes:
- `/`
- `/translate`
- `/colorize`
- `/about`

Admin routes:
- `/admin/login`
- `/admin`
- `/admin/training`
- `/admin/evaluation`
- `/admin/data`
- `/admin/engines`
- `/admin/models`

## Training and Evaluation

Basic training:

```bash
python training/train_yolo.py --data training/datasets/custom_manga.yaml --device cpu
```

Two-stage training:

```bash
python training/advanced_train_yolo.py --data training/datasets/custom_manga.yaml --device cpu
```

Three-stage train + evaluate:

```bash
python training/train_and_eval.py --data training/datasets/custom_manga.yaml --device cpu
```

Standalone evaluation:

```bash
python evaluation/evaluate_model.py \
  --model yolo_train_run/<run>/weights/best.pt \
  --data training/datasets/custom_manga.yaml
```

Threshold sweep:

```bash
python evaluation/threshold_sweep.py \
  --model yolo_train_run/<run>/weights/best.pt \
  --data training/datasets/custom_manga.yaml
```

Reports are written under `evaluation/results/`.

## Assisted Dataset Generation

Generate pseudo-labels from an older detector:

```bash
python training/create_pseudo_labels.py \
  --input /path/to/raw/images \
  --output /path/to/pseudo_labels_run \
  --model yolo_train_run/<old_run>/weights/best.pt \
  --enable-ocr
```

This writes labels, optional overlay previews, and `review_manifest.json`.

Apply review decisions:

```bash
python training/apply_pseudo_label_review.py \
  --manifest /path/to/pseudo_labels_run/review_manifest.json \
  --reject-action remove
```

## Translation Engines

The engine registry keeps multiple engines structured in code without breaking the public UI.

Public UI shows only enabled engines.
Admin UI shows all tracked engines, including disabled ones and setup reasons.

Tracked engines:
- Google
- DeepL
- Gemma / Ollama
- Argos
- MarianMT
- NLLB
- Azure placeholder for future support

## Colorization

Colorization is preserved as a public secondary feature.

Current behavior:
- if the required weights are available, the backend can run
- if they are missing, the UI reports exactly what is missing and how to enable it

## Accessibility and Future Scaling

The project now leaves visible placeholders for future work on:

- keyboard-friendly public UI
- screen-reader-friendly upload/result sections
- OCR and translated text readout
- text-to-speech hooks
- queue-backed jobs
- resumable batches
- storage-backed manifests
- multi-machine execution
