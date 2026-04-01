# 🎨 LaMa Inpainting in Manga Translator

## What is LaMa?

**LaMa** (Large Mask Inpainting) is a state-of-the-art deep learning model that can **intelligently remove objects** from images by filling in the masked areas with realistic background.

Think of it as "Photoshop's Content-Aware Fill" but powered by AI!

## Where is LaMa Used in Your Code?

LaMa is used in **Step 4** of the manga translation pipeline to **remove the original Japanese text cleanly** before drawing the English translation.

### Without LaMa:
```
Original manga → Simple white rectangle over text → Looks blocky and artificial
```

### With LaMa:
```
Original manga → LaMa removes text and reconstructs background → Seamless, professional look
```

## Code Location

### 1. **Import (Line ~70)**
```python
try:
    from simple_lama_inpainting import SimpleLama
    _HAS_SIMPLE_LAMA = True
except ImportError:
    _HAS_SIMPLE_LAMA = False
```

### 2. **Model Loader (Lines ~170-200)**
```python
@st.cache_resource(show_spinner=False, ttl=86400)
def load_lama_inpainter():
    """
    Load the LaMa (Large Mask Inpainting) model.
    
    LaMa is used to REMOVE the original text cleanly by intelligently
    filling in the masked areas with appropriate background patterns.
    """
    if _HAS_SIMPLE_LAMA:
        lama = SimpleLama()
        print("✅ SimpleLama model loaded")
        return lama
    return None
```

### 3. **Inpainting Function (Lines ~230-280)**
```python
def inpaint_text_region(
    image: Image.Image,
    mask_box: Tuple[int, int, int, int],
    lama_model=None
) -> Image.Image:
    """
    Remove text from an image region using LaMa inpainting.
    
    This is the KEY FUNCTION that uses the LaMa model to cleanly erase
    the original manga text while preserving the background.
    
    How it works:
        1. Create a binary mask (white=inpaint, black=keep)
        2. Pass image + mask to LaMa model
        3. LaMa fills in the masked area intelligently
    """
    x1, y1, x2, y2 = mask_box
    
    # Create binary mask
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255  # Mark this region for inpainting
    
    # Call LaMa model
    inpainted = lama_model(image_np, mask)
    
    return Image.fromarray(inpainted)
```

### 4. **Usage in Pipeline (Lines ~930-945)**
```python
# For SFX, SIGNS, and TEXT: use LaMa inpainting for clean removal
if region_type in (TextRegionType.SOUND_EFFECTS, 
                   TextRegionType.SIGNS, 
                   TextRegionType.TEXT):
    if lama_model:
        print(f"   🎨 Using LaMa to cleanly remove {region_name}...")
        output_image = inpaint_text_region(
            output_image,
            (x1, y1, x2, y2),
            lama_model
        )
    else:
        # Fallback: simple rectangle fill
        background_color = find_whitest_pixel(text_region_pixels)
        draw_context.rectangle([x1, y1, x2, y2], fill=background_color)
```

## Which Text Types Use LaMa?

LaMa is applied to these text region types detected by YOUR YOLO model:

✅ **SOUND_EFFECTS (SFX)** - "BOOM!", "WHOOSH!", etc.
✅ **SIGNS** - Background signs and labels  
✅ **TEXT** - General text overlays

❌ **REMOVAL** - Uses simple white fill (usually dialogue in speech bubbles)
❌ **DIALOGUE** - Only the bubble outline, not processed

## Why Not Use LaMa for Everything?

1. **Speed**: LaMa is slower than simple rectangle fill
2. **Dialogue bubbles**: Usually have simple white backgrounds anyway
3. **Computational cost**: SFX and signs benefit more from intelligent inpainting

## Installation

To enable LaMa inpainting, install:

```bash
pip install simple-lama-inpainting
```

Or use the included `lama_wrapper.py` in the `manga-translator/` directory.

## Visual Example

```
┌─────────────────────────────────────┐
│  [Original Manga Panel]             │
│                                     │
│     BOOM! ← SFX text over artwork  │
│                                     │
└─────────────────────────────────────┘
              ↓
        [YOLO detects SFX]
              ↓
┌─────────────────────────────────────┐
│  [LaMa Inpainting]                  │
│                                     │
│  Creates mask over "BOOM!"          │
│  Intelligently fills with background│
│                                     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  [Clean Background]                 │
│                                     │
│       (text removed seamlessly)     │
│                                     │
└─────────────────────────────────────┘
              ↓
    [Draw English translation]
              ↓
┌─────────────────────────────────────┐
│  [Final Result]                     │
│                                     │
│     BOOM! ← English text on clean bg│
│                                     │
└─────────────────────────────────────┘
```

## Performance Impact

- **Without LaMa**: ~0.5 seconds per page
- **With LaMa**: ~2-5 seconds per page (depends on GPU)
- **Recommendation**: Use GPU for best performance

## Fallback Behavior

If LaMa is not available:
```python
if lama_model:
    # Use intelligent inpainting
    output_image = inpaint_text_region(...)
else:
    # Fallback: simple rectangle fill
    draw_context.rectangle([x1, y1, x2, y2], fill=background_color)
```

The code gracefully falls back to simple rectangle fills, so it still works without LaMa!

## Summary

**LaMa's Role**: The "Eraser" that cleanly removes original manga text  
**Location**: Called in the main processing loop for SFX/SIGNS/TEXT  
**Benefit**: Professional-looking results with seamless backgrounds  
**Optional**: Code works without it (uses simple fills as fallback)
