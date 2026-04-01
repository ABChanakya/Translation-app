# 🔍 Quick Answer: Where is LaMa in the Code?

## TL;DR

**LaMa is at line ~930 in `DEMO_26.06_REFACTORED.py`** in the main processing loop.

It's used to **cleanly erase** the original Japanese text from SFX, SIGNS, and TEXT regions before drawing the English translation.

---

## The Key Line of Code

```python
# Line ~930 in DEMO_26.06_REFACTORED.py

if region_type in (TextRegionType.SOUND_EFFECTS, 
                   TextRegionType.SIGNS, 
                   TextRegionType.TEXT):
    if lama_model:
        print(f"   🎨 Using LaMa to cleanly remove {region_name}...")
        output_image = inpaint_text_region(  # ← LAMA IS CALLED HERE!
            output_image,
            (x1, y1, x2, y2),
            lama_model
        )
```

---

## What Happens in `inpaint_text_region()`?

```python
# Line ~230 in DEMO_26.06_REFACTORED.py

def inpaint_text_region(image, mask_box, lama_model):
    """Remove text using LaMa inpainting"""
    
    x1, y1, x2, y2 = mask_box
    image_np = np.array(image)
    
    # Create mask: 255 = remove this area
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    
    # Call LaMa model
    inpainted = lama_model(image_np, mask)  # ← MAGIC HAPPENS HERE!
    
    return Image.fromarray(inpainted)
```

---

## Complete Flow

```
1. YOUR YOLO MODEL detects text regions
   ↓
2. Manga OCR extracts Japanese text
   ↓
3. Translation engine converts to English
   ↓
4. 🎨 LAMA REMOVES ORIGINAL TEXT ← HERE!
   • Creates a mask over the text area
   • LaMa reconstructs the background
   • Returns clean image
   ↓
5. English text is drawn on the clean background
```

---

## Which Text Regions Use LaMa?

| Region Type      | Uses LaMa? | Why?                                    |
|------------------|------------|-----------------------------------------|
| SOUND_EFFECTS    | ✅ YES     | Over artwork, needs clean removal      |
| SIGNS            | ✅ YES     | Background elements, needs inpainting  |
| TEXT             | ✅ YES     | General overlays, benefits from LaMa   |
| DIALOGUE         | ❌ NO      | Just bubble outlines, not processed    |
| REMOVAL          | ❌ NO      | Simple white backgrounds, simple fill OK|

---

## How to Enable/Disable LaMa

### Enable (default):
```bash
pip install simple-lama-inpainting
```

Then run your script normally:
```bash
streamlit run DEMO_26.06_REFACTORED.py
```

Output:
```
✅ LaMa inpainting enabled - text will be removed cleanly
```

### Disable (fallback mode):
Don't install `simple-lama-inpainting`, code will automatically use simple rectangle fills:

Output:
```
⚠️ LaMa not available - using simple rectangle fill
```

---

## File Locations

1. **Import**: Line ~70
2. **Loader**: Line ~170-200 (`load_lama_inpainter()`)
3. **Inpainting Function**: Line ~230-280 (`inpaint_text_region()`)
4. **Main Usage**: Line ~930-945 (in processing loop)

---

## Visual Example

### Without LaMa:
```
[Original Image]
    "ドカーン!" over detailed artwork
         ↓
[White Rectangle]
    ▓▓▓▓▓▓▓▓▓ (blocky, obvious)
         ↓
[Add English]
    "BOOM!" on white rectangle
```

### With LaMa:
```
[Original Image]
    "ドカーン!" over detailed artwork
         ↓
[LaMa Inpainting]
    (reconstructed background with original textures)
         ↓
[Add English]
    "BOOM!" on seamless background ← Professional!
```

---

## Need More Info?

See these files:
- `LAMA_INPAINTING_EXPLAINED.md` - Detailed explanation
- `PIPELINE_VISUAL_FLOW.py` - Visual diagram of entire pipeline
- `manga-translator/website.py` - Original implementation reference
- `manga-translator/lama_wrapper.py` - Alternative LaMa wrapper

---

## Summary

**Location**: Line ~930 in `DEMO_26.06_REFACTORED.py`

**Purpose**: Remove Japanese text cleanly before drawing English

**Function**: `inpaint_text_region()` calls LaMa model

**Used For**: SFX, SIGNS, TEXT regions (not dialogue)

**Required**: No (has fallback to simple fills)

**Recommended**: Yes (much better results!)
