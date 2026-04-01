# Developer Guide

This is the single larger context document for the current codebase.

It replaces a long set of older migration, status, visual, and structure notes that had drifted out of sync with the actual project.

## 1. What This Project Is

Auto Manga Translation is a Flask-based system for:

- translating manga pages through a detection -> OCR -> translation -> text removal -> rendering pipeline
- training and evaluating the detector used by that pipeline
- generating new training data with older detector + OCR models
- reviewing pseudo-label outputs before they enter the canonical dataset
- optionally colorizing manga pages as a separate public feature

The current product is intentionally split into:

- `Public UI`: normal user flow
- `Admin UI`: training, evaluation, data generation, diagnostics

That separation keeps the demo simple while preserving the engineering workflow.

## 2. Project History And What Changed

Earlier versions of the repository accumulated many overlapping documents:

- migration reports
- completion reports
- file maps
- visual summaries
- setup variants
- one-off fix notes

Those files were useful while the project was moving quickly, but they became noisy and partly outdated after the app was re-centered around the current Flask workflow.

The current documentation policy is:

- keep a small set of current docs
- keep one larger context doc
- archive old material instead of pretending it is still canonical

Archived root-level material now lives under:

- [/.repo_archive/20260324_cleanup](/home/chanakya/chanakya/Translation_tool-2/.repo_archive/20260324_cleanup)

## 3. Canonical Structure

The only canonical implementation root is:

- [manga_translator_clean](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean)

Core entrypoints:

- web app: [web/app.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/web/app.py)
- translation pipeline: [src/pipeline.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/pipeline.py)
- batch processing: [src/batch_processor.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/batch_processor.py)
- translator registry: [src/translators/registry.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/translators/registry.py)
- detector training: [training/train_yolo.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/train_yolo.py)
- evaluation: [evaluation/evaluate_model.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/evaluation/evaluate_model.py)

The two root demo files intentionally kept for historical reference are:

- [examples/demo_2606.py](/home/chanakya/chanakya/Translation_tool-2/examples/demo_2606.py)
- [examples/demo_2606_refactored.py](/home/chanakya/chanakya/Translation_tool-2/examples/demo_2606_refactored.py)

They are not the main runtime path anymore.

## 4. Public UI

Public routes are meant to feel like one coherent product:

- `Translate`
- `Batch`
- `Colorize`
- `About`

The public side should remain easy to explain:

1. upload a page
2. detect text regions
3. OCR text from those regions
4. translate
5. remove the source text
6. render the translated result

Advanced model controls stay out of the public flow.

## 5. Admin UI

The admin area exists to keep training and developer tooling out of the public product surface.

Important routes:

- `/admin`
- `/admin/training`
- `/admin/evaluation`
- `/admin/data`
- `/admin/engines`
- `/admin/models`

Admin auth is session-based and controlled by:

- `ADMIN_USERNAME`
- `ADMIN_PASSWORD_HASH`
- `FLASK_SECRET_KEY`

## 6. Training And Evaluation

The canonical detection task is the current 5-class setup.

Current dataset default:

- [training/datasets/custom_manga.yaml](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/datasets/custom_manga.yaml)

Canonical classes are the current detector labels used by the project. Training, evaluation, and UI logic should stay aligned to that same setup instead of drifting into parallel 3-class and 5-class stories.

### Fairness-first reporting

The evaluation layer was intentionally changed so weak classes do not disappear behind aggregate scores.

The reporting now emphasizes:

- `mAP50`
- `mAP50_95`
- macro precision
- macro recall
- macro F1
- weighted precision/recall/F1
- per-class support counts
- per-class prediction counts
- per-class AP
- per-class matched-box IoU
- macro IoU

Important detail:

- the IoU reporting here is box-level matched IoU for detection analysis
- it is not segmentation IoU

This makes rare or underrepresented classes much easier to discuss in a demo and easier to debug during training.

### Reports

Training/evaluation runs generate machine-readable and demo-friendly outputs:

- `metrics.json`
- fairness/per-class JSON
- `summary.json`
- `report.html`

Those reports are surfaced in the admin UI.

## 7. Translation Engines

The project no longer assumes only one engine path.

The backend now uses an engine registry so the UI stays stable even when some engines are not configured yet.

Public UI behavior:

- only show engines that are currently available

Admin behavior:

- show all registered engines
- explain why unavailable engines are disabled
- make future enablement straightforward

This keeps the code extensible without making the normal demo unstable.

## 8. Assisted Data Generation

This is a first-class workflow, not a side script.

Purpose:

- reuse older trained detector + OCR models
- generate candidate boxes and text
- review the results before adding them to training data

Key files:

- [training/create_pseudo_labels.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/create_pseudo_labels.py)
- [training/apply_pseudo_label_review.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/apply_pseudo_label_review.py)

Expected review flow:

1. generate labels and overlays
2. inspect whole images
3. mark each item as `approve`, `reject`, or `uncertain`
4. export only approved items

This preserves your existing habit of rejecting bad images entirely, but makes the process trackable and safer.

## 9. OCR Runtime Note

The current OCR runtime path uses `manga-ocr`.

If model access fails because HuggingFace authentication is missing, the current guidance is:

```bash
huggingface-cli login
```

Relevant code:

- [src/models/ocr.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/models/ocr.py)

If you later want to add OCR model training as a formal packaged workflow, it should become a separate documented training path rather than being hidden inside the detector docs.

## 10. Colorization And LaMa

Colorization remains a kept product feature.

Relevant files:

- [src/colorization_service.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/colorization_service.py)
- [colorization/readme.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/colorization/readme.md)

Text removal in the translation pipeline is a separate concern from colorization and uses the inpainting path:

- [src/models/inpainter.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/models/inpainter.py)

The old LaMa explanation files were merged into the remaining colorization doc so the repo no longer needs multiple top-level notes for the same idea.

## 11. Low-RAM And Future Scaling

The current project should stay runnable on modest hardware where possible.

Already implemented:

- lazy heavy-model loading
- chunked batch processing
- disk-backed outputs
- background admin jobs

Reserved future work:

- queue-backed workers
- storage-backed manifests
- resumable jobs
- parallel workers
- multi-machine execution

See:

- [FUTURE_ACCESSIBILITY_AND_SCALING.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/FUTURE_ACCESSIBILITY_AND_SCALING.md)

## 12. Accessibility Reservation

Accessibility is not fully implemented yet, but it now has explicit reserved integration points in both docs and code.

Important current idea:

- keep OCR and translated text in structured outputs so later readout or TTS layers can be added without rewriting the pipeline

Again, see:

- [FUTURE_ACCESSIBILITY_AND_SCALING.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/FUTURE_ACCESSIBILITY_AND_SCALING.md)

## 13. Minimal Doc Set

The intended doc set after cleanup is:

- [README.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/README.md)
- [QUICKSTART.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/QUICKSTART.md)
- [DEVELOPER_GUIDE.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/DEVELOPER_GUIDE.md)
- [data/README.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/data/README.md)
- [colorization/readme.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/colorization/readme.md)
- [FUTURE_ACCESSIBILITY_AND_SCALING.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/FUTURE_ACCESSIBILITY_AND_SCALING.md)

That is the set you should actually read.
