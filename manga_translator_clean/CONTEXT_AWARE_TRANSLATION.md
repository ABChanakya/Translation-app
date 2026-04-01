# Context-Aware Translation for Batch Processing

## Overview

When translating manga batches (multiple pages in sequence), the translation system now maintains a **narrative context window** across pages. This allows Gemma3 and other translators to understand:

- Character relationships and tone shifts
- Recurring dialogue and character speech patterns
- Story context and plot progression
- Consistent term usage across chapters

## How It Works

### Single Page → Batch Translation Flow

**Single page translation:**
- Detects text regions
- Runs OCR
- Translates each bubble independently

**Batch translation (improved):**
1. Process page 1 → collect all dialogue translations
2. Pass page 1's translations as context to page 2
3. Page 2 translator receives: "Previous context: [dialogue from page 1]"
4. Process page 2 → collect translations
5. Pass page 1-2's translations (last 50 lines) as context to page 3
6. Continue through chapter

### Context Injection

The context is injected directly into Gemma3's prompt as a narrative hint:

```
[Previous page context for narrative continuity:
"Did you really intend to go there?"
"I have no choice anymore."]

=== JAPANESE TEXT ===
はい、もうここしかありません。
=== END TEXT ===
```

Gemma3 reads this context and can make translation decisions that feel coherent across pages.

## Configuration

### Context Window Size

In `src/batch_processor.py`:
```python
# Keep buffer size manageable (last 50 lines from last 2 pages)
if len(previous_page_translations) > 50:
    previous_page_translations = previous_page_translations[-50:]
```

**Default:** 50 translation lines (approximately 2 pages of dialogue)

To adjust, modify the threshold in `batch_processor.py` line ~130.

### Which Translators Support Context?

| Engine | Supports Context | Notes |
|--------|------------------|-------|
| **Gemma3** | ✅ Yes | Context injected into system prompt |
| **Google Translate** | ⚠️ Ignored | API doesn't support context hints |
| **DeepL** | ⚠️ Ignored | API doesn't support custom context |
| **Argos/MarianMT/NLLB** | ⚠️ Ignored | Offline engines ignore context |

**Note:** All translators accept the `context_prompt` parameter for API compatibility. Only Gemma3 actively uses it for improved translations.

## Usage

### Web UI Batch Upload

1. Upload 3+ manga pages in sequential order
2. Select "Gemma3" translator
3. Set confidence threshold
4. Click "Translate"
5. Pages will be processed with narrative context passed between them

### Programmatic Usage

```python
from src.pipeline import MangaTranslationPipeline

pipeline = MangaTranslationPipeline(translation_engine="Gemma3")

# Page 1
image1 = Image.open("page_1.png")
out1, logs1 = pipeline.process(image1)

# Page 2 — pass context from page 1
translations_from_page1 = [
    log["tgt_text"] for log in logs1 
    if log["class_id"] == TextRegionType.DIALOGUE
]
out2, logs2 = pipeline.process(image2, previous_page_context=translations_from_page1)

# Page 3 — pass combined context
translations_from_page2 = [
    log["tgt_text"] for log in logs2 
    if log["class_id"] == TextRegionType.DIALOGUE
]
combined_context = translations_from_page1[-25:] + translations_from_page2  # Last 25 from page 1 + all of page 2
out3, logs3 = pipeline.process(image3, previous_page_context=combined_context)
```

### Batch Processor (Automatic)

The batch processor automatically manages context windows:

```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor()
result = processor.process_batch(
    input_paths=["page_1.png", "page_2.png", "page_3.png"],
    process_func=pipeline.process_image,
    translation_engine="Gemma3"
)
```

The processor:
1. Processes pages in order
2. Extracts translations from each page's result
3. Automatically passes context to the next page
4. Keeps buffer size ≤ 50 lines

## Benefits

### Translation Quality

- **Character consistency:** Character names and titles remain consistent across pages
- **Tone continuity:** Dialogue feels natural and continuous across page breaks
- **Context understanding:** Gemma3 understands what just happened in the story
- **Term consistency:** Technical terms and proper nouns are translated consistently

### Example

**Without context:**
```
Page 1: "I'm going to find the legendary crystal."
Page 2: "The legendary stone holds the secret."  ← Different term!
Page 3: "That rock is useless."                   ← Yet another term!
```

**With context:**
```
Page 1: "I'm going to find the legendary crystal."
Page 2: "The legendary crystal holds the secret."  ← Consistent!
Page 3: "That crystal is useless."                 ← Consistent!
```

## Limitations & Future Work

### Current Limitations

1. **Context only flows forward** — Page 2 can't use context from page 3 or later
2. **Context limited to 50 lines** — Very long chapters may lose earlier plot points
3. **Only Gemma3 uses context actively** — Other translators ignore it (API limitations)
4. **No speaker identification** — Context includes all dialogue, not attributed to specific characters

### Potential Improvements

- **Dynamic context window:** Increase buffer for long chapters, decrease for short ones
- **Multi-engine context support:** Fine-tune or prompt-engineer Google/DeepL to accept context
- **Speaker-aware context:** Track who says what for better dialogue continuity
- **Bidirectional context:** Process pages in reverse to refine translations with forward/backward context
- **Glossary injection:** Auto-build glossary from first pages and inject into later translations

## Testing & Debugging

### Check Context Being Passed

Look for this in the console/logs when processing a batch:

```
📖 Step 2/4: OCR, Translation, and Text Removal...
📍 Region #1 (Dialogue)
   🌐 Translating with Gemma3...
   ✅ Translation: 'I understand completely.'
```

To debug context content:

```python
# In src/pipeline.py, around line 245
print(f"[DEBUG] Context passed: {context_prompt[:200]}")
```

### Verify Translations Array

Each page returns a `stats` dict with translations:

```python
print(result['stats']['translations'])
# Output:
# [
#   {'id': 1, 'type': 'Dialogue', 'original': '了解した', 'translated': 'I understand.', ...},
#   {'id': 2, 'type': 'Dialogue', 'original': 'ありがとう', 'translated': 'Thank you.', ...}
# ]
```

## Performance Impact

- **Processing time:** +0 seconds (context is just text in the prompt)
- **Memory:** +minimal (buffer is ~50 strings, ~5–10 KB)
- **API calls:** No change (same number of Gemma3 calls)

## Troubleshooting

### Context not affecting translation

**Problem:** Translations are inconsistent across pages despite using context.

**Solutions:**
1. Ensure Gemma3 is selected (other engines ignore context)
2. Check that pages are being uploaded in correct order
3. Verify Gemma3 model has sufficient context window (gemma3:latest supports 8K tokens)
4. Try increasing `DEFAULT_TEMPERATURE` in `config/settings.py` to make model more creative

### Memory issues in very long batches

**Problem:** Processing 20+ pages causes memory spike.

**Solution:** Reduce context buffer size in `src/batch_processor.py`:
```python
if len(previous_page_translations) > 20:  # was 50
    previous_page_translations = previous_page_translations[-20:]
```

### Context lost between pages

**Problem:** Page 3+ has no context history.

**Solution:** This is by design — only 1–2 pages of context are maintained to keep prompts concise. To extend, modify `src/batch_processor.py` line ~130.

## Example: Full Workflow

1. **Upload chapter files:** `ch01_p01.png`, `ch01_p02.png`, `ch01_p03.png`
2. **Translator:** Gemma3
3. **Confidence:** 0.15

**Processing:**
```
Batch started: 3 pages
─────────────────────
Processing page 1/3...
  Detected 12 bubbles
  Translated 10 dialogues
  Context collected: 10 lines
  ✅ Page 1 complete

Processing page 2/3...
  Context from page 1: 10 lines
  Detected 11 bubbles
  Translated 9 dialogues using context
  Context buffer now: 10 + 9 = 19 lines
  ✅ Page 2 complete

Processing page 3/3...
  Context from pages 1-2: 19 lines
  Detected 14 bubbles
  Translated 11 dialogues using context
  Context buffer: 19 + 11 = 30 lines (under 50 limit)
  ✅ Page 3 complete

ZIP ready for download!
```

## See Also

- [DATA_AND_TRAINING_GUIDE.md](./DATA_AND_TRAINING_GUIDE.md) — Training better models for improved base translations
- [src/translators/gemma.py](./src/translators/gemma.py) — Gemma3 translator implementation
- [src/batch_processor.py](./src/batch_processor.py) — Batch processing with context management
