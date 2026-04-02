# Compact Documentation

This file consolidates the previous focused guides into one place.

## 1) Quickstart

1. Install dependencies:
   - `cd manga_translator_clean`
   - `pip install -r requirements.txt`
2. Start app:
   - `python web/app.py`
3. Open:
   - `http://localhost:5000`

Optional admin setup:
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD_HASH`
- `FLASK_SECRET_KEY`

## 2) Product Surfaces

### Public UI
- Translate single manga pages
- Batch translation
- Optional colorization

### Admin UI
- Training runs
- Evaluation reports
- Model/engine status
- Assisted data generation and review

## 3) Translation Context Features

### Global story context
- User can provide chapter-level context for better term/name consistency.
- Context is injected into translator prompts when supported.

### Page-to-page context
- Batch flow can carry recent page context forward.
- Improves consistency in pronouns, speaker intent, and term usage.

## 4) Data Workflow

- Prepare raw manga pages
- Annotate text regions in YOLO format
- Train and evaluate detector
- Generate pseudo-labels from older checkpoints
- Review pseudo-label manifest (`approve` / `reject` / `uncertain`)
- Apply decisions and retrain

### Canonical training config
- `training/datasets/custom_manga.yaml`

## 5) Training and Evaluation

### Core scripts
- `training/train_yolo.py`
- `training/advanced_train_yolo.py`
- `training/train_and_eval.py`
- `evaluation/evaluate_model.py`
- `evaluation/threshold_sweep.py`

### Typical strategy
- Stage 1: freeze backbone, train head
- Stage 2: full fine-tune with manga-appropriate augmentation
- Track macro/per-class metrics, not only aggregate mAP

## 6) Verification Checklist

- UI sends context fields correctly
- Backend receives context and forwards to pipeline
- Translator receives context payload
- Python syntax checks pass
- Manual single-page and batch tests pass

## 7) Colorization Notes

- Colorization is secondary and optional
- Requires expected model assets/weights
- If assets are missing, app should report setup-needed state clearly

## 8) Scaling and Accessibility Direction

### Accessibility (reserved)
- Keyboard-first interactions
- Improved screen-reader semantics
- OCR/translation readout hooks

### Scaling (reserved)
- Queue-backed job orchestration
- Resumable batch processing
- Storage-backed manifests and artifacts
- Multi-worker/multi-machine execution path

## 9) Troubleshooting Quick Notes

- **Context seems ineffective**: verify selected translator supports context handling and verify payload propagation in logs.
- **High memory during long batches**: reduce context window and process smaller chunks.
- **Inconsistent outputs**: include clearer story context and key term constraints.

## 10) Canonical Files

- Web app: `web/app.py`
- Pipeline: `src/pipeline.py`
- Translator implementations: `src/translators/`
- Config defaults: `config/settings.py`
- Model layer: `src/models/`

---

This compact doc replaces the previous fragmented markdown set.
