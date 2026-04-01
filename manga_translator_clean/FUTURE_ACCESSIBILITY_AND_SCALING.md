# Future Accessibility And Scaling

This file reserves explicit future integration points for accessibility, larger-scale execution, and low-RAM-safe workflow growth.

## Accessibility Reservation

Current documentation entrypoints:

- [README.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/README.md)
- [web/templates/about.html](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/web/templates/about.html)
- [src/pipeline.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/pipeline.py)

### Reserved integration points

Structured OCR/translation output:
- [src/pipeline.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/pipeline.py)
  Why it matters:
  the `translations` list and per-region stats are kept structured so future text-to-speech, readout, or screen-reader summaries do not require rewriting the pipeline

Public UI result surfaces:
- [web/templates/translate.html](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/web/templates/translate.html)
- [web/templates/colorize.html](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/web/templates/colorize.html)
  Future work:
  add keyboard-first focus flow, ARIA labels, OCR readout controls, and result summaries for blind or visually impaired users

Admin dataset review:
- [web/templates/admin/data.html](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/web/templates/admin/data.html)
  Future work:
  add text-first review mode and auditory summary mode for region-level or image-level review

### Recommended future features

- screen-reader-friendly upload + result sections
- OCR text readout before translation
- translated text readout after rendering
- TTS controls for region-by-region playback
- optional page summary or alt-text generation
- keyboard-only navigation for the public UI

## Scaling And Parallelization Reservation

Current implementation entrypoints:

- [src/batch_processor.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/batch_processor.py)
- [web/app.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/web/app.py)
- [training/create_pseudo_labels.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/training/create_pseudo_labels.py)

### Already implemented now

- chunked public batch translation
- chunked pseudo-label generation
- disk-backed batch outputs
- background admin job launching

### Reserved future scaling features

- queue-backed job execution instead of local background subprocesses
- storage-backed run metadata instead of file-only manifests
- resumable public batch jobs
- resumable pseudo-label review jobs
- parallel worker pools for large image sets
- multi-machine execution for training/data generation
- object storage or shared-volume output handling

### Recommended integration path

1. Replace local background jobs in [web/app.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/web/app.py) with a queue worker.
2. Persist run manifests in a storage-backed database or durable JSON manifest store.
3. Split large batches into queue tasks instead of single-process loops.
4. Add retry/resume state to batch translation and pseudo-label review.
5. Add machine-local worker registration only after the single-node queue version is stable.

## Low-RAM Default Principle

The current project should keep prioritizing:

- lazy model loading
- chunked processing
- incremental disk output
- avoiding full-batch in-memory retention

That principle should stay in place even if future parallelization is added.
