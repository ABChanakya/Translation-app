# Data Workflow

This document covers the current data flow used by the detector training pipeline.

## Purpose

The data area supports:

- raw manga collection
- converted annotations
- training/evaluation dataset definitions
- pseudo-label generation from older models
- review before new data enters the canonical dataset

## Canonical Training Config

Use:

- [training/datasets/custom_manga.yaml](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/datasets/custom_manga.yaml)

That is the current default dataset config for detector training and evaluation.

## Current Detector Labels

The current workflow is based on the active 5-class detector setup used by the app and training scripts.

Keep your dataset YAML, label files, evaluation scripts, and demo language aligned to the same class set.

## Main Data Areas

Inside [data](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data) you may see a mix of real folders, symlinks, or local project-specific layouts depending on the machine.

Useful subpaths:

- [annotations](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/annotations)
- [scrapers](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/scrapers)
- [utils](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/utils)

The exact raw/training image storage may be local to your machine or symlinked outside the repo.

## Normal Data Flow

1. collect or scrape raw manga pages
2. prepare or import labels
3. train/evaluate the detector
4. use the older detector to generate pseudo-labels on new pages
5. review those generated labels
6. add only approved items into the canonical training set

## Assisted Generation And Review

Key scripts:

- [training/create_pseudo_labels.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/create_pseudo_labels.py)
- [training/apply_pseudo_label_review.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/apply_pseudo_label_review.py)

The intended workflow is:

```bash
python training/create_pseudo_labels.py \
  --model models/checkpoints/custom_yolo_best.pt \
  --input data/manga_chapters/047 \
  --output data/pseudo_labels/run_047
```

Then review the generated manifest and apply decisions:

```bash
python training/apply_pseudo_label_review.py \
  --manifest data/pseudo_labels/run_047/review_manifest.json
```

Review states:

- `approve`
- `reject`
- `uncertain`

The project still supports your whole-image reject workflow. The difference now is that the review state is recorded instead of being tracked only by manual file deletion.

## Useful Utilities

Scraping:
- [data/scrapers/rawkuma_scraper.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/scrapers/rawkuma_scraper.py)

Annotation conversion:
- [data/annotations/convert_to_yolo.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/annotations/convert_to_yolo.py)
- [data/annotations/convert_yolo_to_labelstudio.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/annotations/convert_yolo_to_labelstudio.py)

Utility helpers:
- [data/utils/dedupe.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/utils/dedupe.py)
- [data/utils/counts.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/utils/counts.py)
- [data/utils/deleter.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/utils/deleter.py)
- [data/utils/rename_seq.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/utils/rename_seq.py)
- [data/utils/page_upload.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/utils/page_upload.py)

## Training And Evaluation Commands

Train:

```bash
python training/train_yolo.py \
  --data training/datasets/custom_manga.yaml
```

Evaluate:

```bash
python evaluation/evaluate_model.py \
  --model models/checkpoints/custom_yolo_best.pt \
  --data training/datasets/custom_manga.yaml
```

## Notes

- Keep generated pseudo-label data separate from approved training data until review is complete.
- Keep the dataset story simple for demos: original data, model-assisted expansion, human review, retraining.
- If you change the class set, update the dataset YAML, label definitions, training scripts, and evaluation expectations together.

## Related Docs

- Main overview: [../README.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/README.md)
- Run/setup guide: [../QUICKSTART.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/QUICKSTART.md)
- Larger engineering guide: [../DEVELOPER_GUIDE.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/DEVELOPER_GUIDE.md)
