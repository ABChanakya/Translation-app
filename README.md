# Translation_tool-2 Repository Status

This repository now keeps only a small documentation set at the top level.

The active project is:

- [manga_translator_clean](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean)

If you want to run the app, train, evaluate, or prepare a demo, start there.

## Canonical Entrypoints

Public and admin web app:
- [manga_translator_clean/web/app.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/web/app.py)

Detection training:
- [manga_translator_clean/training/train_yolo.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/train_yolo.py)
- [manga_translator_clean/training/advanced_train_yolo.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/advanced_train_yolo.py)
- [manga_translator_clean/training/train_and_eval.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/train_and_eval.py)

Evaluation:
- [manga_translator_clean/evaluation/evaluate_model.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/evaluation/evaluate_model.py)
- [manga_translator_clean/evaluation/threshold_sweep.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/evaluation/threshold_sweep.py)

Assisted data generation:
- [manga_translator_clean/training/create_pseudo_labels.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/create_pseudo_labels.py)
- [manga_translator_clean/training/apply_pseudo_label_review.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/apply_pseudo_label_review.py)

## Docs To Read

Main product overview:
- [manga_translator_clean/README.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/README.md)

Run and setup guide:
- [manga_translator_clean/QUICKSTART.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/QUICKSTART.md)

Larger engineering/context guide:
- [manga_translator_clean/DEVELOPER_GUIDE.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/DEVELOPER_GUIDE.md)

Future accessibility and scaling notes:
- [manga_translator_clean/FUTURE_ACCESSIBILITY_AND_SCALING.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/FUTURE_ACCESSIBILITY_AND_SCALING.md)

## Root-Level Files Still Intentionally Kept

Legacy demos you explicitly wanted to keep:
- [examples/demo_2606.py](/home/chanakya/chanakya/Translation_tool-2/examples/demo_2606.py)
- [examples/demo_2606_refactored.py](/home/chanakya/chanakya/Translation_tool-2/examples/demo_2606_refactored.py)

Historical project brief:
- [Streamlit-based Manga Page Translator_ Techniques, Tools, and Design Patterns (1).pdf](/home/chanakya/chanakya/Translation_tool-2/Streamlit-based%20Manga%20Page%20Translator_%20Techniques,%20Tools,%20and%20Design%20Patterns%20(1).pdf)

## Archive And Cleanup Notes

Old root-level code, docs, data helpers, and generated outputs were archived into:

- [/.repo_archive/20260324_cleanup](/home/chanakya/chanakya/Translation_tool-2/.repo_archive/20260324_cleanup)

If a file exists both in the archive and inside `manga_translator_clean/`, trust the version inside `manga_translator_clean/`.

## Model Path Note

The active model-selection logic lives in:

- [manga_translator_clean/config/settings.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/config/settings.py)

At runtime, the app and scripts resolve the detector checkpoint from configuration and environment variables such as `YOLO_MODEL_PATH`.
