"""
═══════════════════════════════════════════════════════════════════════════════
                    MANGA TRANSLATION PIPELINE - VISUAL FLOW
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                          📖 MANGA PAGE INPUT                                 │
│                                                                              │
│   ┌────────────────────────────────────────────────┐                        │
│   │  [Japanese Manga Page with Text]                │                        │
│   │  • Dialogue in speech bubbles                   │                        │
│   │  • Sound effects (SFX) over artwork            │                        │
│   │  • Signs in background                          │                        │
│   └────────────────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🔍 STEP 1: TEXT REGION DETECTION                          │
│                      (YOUR CUSTOM YOLO MODEL)                                │
│                                                                              │
│   Model: yolo_train_run/full_finetune_phase40/weights/best.pt              │
│                                                                              │
│   Detects 5 types of regions:                                               │
│   ┌──────────────┬────────────────────────────────────────┐                │
│   │ Class 0      │ DIALOGUE (speech bubble outlines)      │                │
│   │ Class 1      │ SOUND_EFFECTS (SFX text like "BOOM!")  │ ← Uses LaMa   │
│   │ Class 2      │ SIGNS (background text/labels)         │ ← Uses LaMa   │
│   │ Class 3      │ TEXT (general text overlays)           │ ← Uses LaMa   │
│   │ Class 4      │ REMOVAL (dialogue text to replace)     │                │
│   └──────────────┴────────────────────────────────────────┘                │
│                                                                              │
│   Output: Bounding boxes for each detected text region                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    📖 STEP 2: OCR (Text Extraction)                          │
│                         (Manga OCR Model)                                    │
│                                                                              │
│   For each detected region:                                                 │
│   1. Crop the image to the bounding box                                     │
│   2. Pass to Manga OCR model                                                │
│   3. Extract Japanese text                                                  │
│                                                                              │
│   Example:                                                                   │
│   Region [120, 45, 280, 95] → OCR → "ドカーン！"                            │
│   Region [50, 200, 150, 250] → OCR → "待って！"                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🌐 STEP 3: TRANSLATION                                    │
│                   (Multiple Engine Options)                                  │
│                                                                              │
│   Engines:                                                                   │
│   • Gemma3 (LLM via Ollama) ← Best quality                                 │
│   • Google Translate         ← Fast & free                                  │
│   • DeepL                    ← Professional                                 │
│   • Azure Translator         ← Enterprise                                   │
│   • Argos Translate          ← Offline                                      │
│   • MarianMT                 ← Open source                                  │
│   • NLLB                     ← Multilingual                                 │
│                                                                              │
│   Example:                                                                   │
│   "ドカーン！" → Translate → "BOOM!"                                         │
│   "待って！"   → Translate → "Wait!"                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              🎨 STEP 4: TEXT REMOVAL (THIS IS WHERE LAMA IS USED!)          │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  FOR SFX, SIGNS, TEXT (Classes 1, 2, 3):                            │  │
│   │  ════════════════════════════════════════                            │  │
│   │                                                                      │  │
│   │  if lama_model_available:                                           │  │
│   │      ┌────────────────────────────────────────┐                     │  │
│   │      │  🎨 LaMa INPAINTING                    │                     │  │
│   │      │  ─────────────────────                 │                     │  │
│   │      │  1. Create binary mask over text       │                     │  │
│   │      │     (white=remove, black=keep)         │                     │  │
│   │      │                                         │                     │  │
│   │      │  2. Pass to LaMa model:                │                     │  │
│   │      │     lama(image, mask)                  │                     │  │
│   │      │                                         │                     │  │
│   │      │  3. LaMa intelligently fills the       │                     │  │
│   │      │     masked area with background        │                     │  │
│   │      │     texture/pattern                    │                     │  │
│   │      │                                         │                     │  │
│   │      │  Result: Seamless background!          │                     │  │
│   │      └────────────────────────────────────────┘                     │  │
│   │                                                                      │  │
│   │  else:  # Fallback if LaMa not available                            │  │
│   │      Draw white/colored rectangle over text                         │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  FOR DIALOGUE REMOVAL (Class 4):                                    │  │
│   │  ═══════════════════════════                                        │  │
│   │                                                                      │  │
│   │  Simple rectangle fill (usually white)                              │  │
│   │  • Dialogue bubbles typically have plain backgrounds                │  │
│   │  • LaMa not needed here                                             │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ✏️ STEP 5: TEXT RENDERING                                 │
│                  (Draw Translated Text)                                      │
│                                                                              │
│   For each region with translation:                                         │
│   1. Intelligently fit text to bounding box                                 │
│      • Find largest font size that fits                                     │
│      • Wrap text across multiple lines if needed                            │
│                                                                              │
│   2. Create transparent overlay layer                                       │
│                                                                              │
│   3. Draw all translated text on overlay                                    │
│      • Centered in original text box                                        │
│      • User-chosen color                                                    │
│                                                                              │
│   4. Composite overlay onto cleaned image                                   │
│                                                                              │
│   Example:                                                                   │
│   "BOOM!" → Fit to [120, 45, 280, 95] → Draw centered                      │
│   "Wait!" → Fit to [50, 200, 150, 250] → Draw centered                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ✨ FINAL OUTPUT                                     │
│                                                                              │
│   ┌────────────────────────────────────────────────┐                        │
│   │  [Translated Manga Page]                        │                        │
│   │  • English text in place of Japanese            │                        │
│   │  • Clean backgrounds (thanks to LaMa!)          │                        │
│   │  • Professional-looking result                  │                        │
│   └────────────────────────────────────────────────┘                        │
│                                                                              │
│   Plus: Detailed logs of all translations                                   │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                        🎨 LAMA MODEL - DETAILED VIEW
═══════════════════════════════════════════════════════════════════════════════

Before LaMa:                       After LaMa:
┌─────────────────────┐            ┌─────────────────────┐
│ ╔═════════════╗     │            │                     │
│ ║  ドカーン！  ║     │            │   (clean artwork)   │
│ ║  (Japanese)  ║     │            │                     │
│ ╚═════════════╝     │            │                     │
│  [artwork below]    │    →→→     │  [artwork restored] │
│                     │            │                     │
└─────────────────────┘            └─────────────────────┘
                                             ↓
                                   ┌─────────────────────┐
                                   │   ┌──────────┐      │
                                   │   │  BOOM!   │      │
                                   │   └──────────┘      │
                                   │  [clean artwork]    │
                                   │                     │
                                   └─────────────────────┘
                                   (English translation added)


═══════════════════════════════════════════════════════════════════════════════
                        CODE STRUCTURE - WHERE IS LAMA?
═══════════════════════════════════════════════════════════════════════════════

File: DEMO_26.06_REFACTORED.py

Line ~70:    Import SimpleLama
             try:
                 from simple_lama_inpainting import SimpleLama

Line ~170:   Load LaMa model
             @st.cache_resource
             def load_lama_inpainter():
                 return SimpleLama()

Line ~230:   Inpainting function
             def inpaint_text_region(image, mask_box, lama_model):
                 # Create mask
                 # Call LaMa
                 # Return cleaned image

Line ~800:   Initialize in pipeline
             lama_model = load_lama_inpainter()

Line ~930:   USE LAMA HERE! ← MAIN USAGE POINT
             if region_type in (SFX, SIGNS, TEXT):
                 if lama_model:
                     output_image = inpaint_text_region(
                         output_image,
                         (x1, y1, x2, y2),
                         lama_model  # ← LaMa removes text here!
                     )


═══════════════════════════════════════════════════════════════════════════════
                              QUICK REFERENCE
═══════════════════════════════════════════════════════════════════════════════

Q: What does LaMa do?
A: Intelligently removes text by reconstructing the background

Q: When is it used?
A: For SFX (sound effects), SIGNS, and TEXT regions

Q: Why not use it for everything?
A: Dialogue bubbles usually have simple white backgrounds (don't need it)
   Also, LaMa is slower than simple fills

Q: What if I don't have LaMa installed?
A: Code automatically falls back to simple rectangle fills

Q: How do I install LaMa?
A: pip install simple-lama-inpainting
   OR use the lama_wrapper.py in manga-translator/ directory

Q: Is it required?
A: No! Optional but highly recommended for best results

═══════════════════════════════════════════════════════════════════════════════
"""
print(__doc__)
