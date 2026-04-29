# Frontend ↔ Backend Gap Analysis

**Date**: 2026-04-16  
**Scope**: All React components, API client functions, and FastAPI endpoints

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully wired — frontend calls backend, backend responds correctly |
| ⚠️ | Partially wired — endpoint exists but frontend doesn't use it, or UI exists but has no backend |
| ❌ | Gap — feature visible in UI but has no backend support, or backend exists with no frontend |
| 🔧 | Needs fix — both sides exist but the integration is broken |

---

## 1. Bubble Review Flow (BubbleEditor.tsx ↔ bubbles.py)

| Feature | Frontend | Backend | Status | Notes |
|---------|----------|---------|--------|-------|
| Accept bubble | `acceptBubble(id)` | `POST /bubbles/{id}/accept` | ✅ | Works. Updates DB status + feeds MangaProfile |
| Correct bubble | `correctBubble(id, text)` | `POST /bubbles/{id}/correct` | 🔧 | **DB updated but page image NOT re-rendered.** User sees old text on canvas. |
| Skip bubble | `skipBubble(id)` | `POST /bubbles/{id}/skip` | ✅ | Works |
| Delete bubble | `deleteBubble(id)` | `DELETE /bubbles/{id}` | ✅ | Works. Trash icon on hover in BubbleEditor |
| Relabel type | `updateBubbleType(id, type)` | `PATCH /bubbles/{id}/type` | ✅ | Works. Click type badge → dropdown |
| **Re-render after correction** | **MISSING** | `POST /bubbles/{id}/apply` | ❌ | **Critical gap.** BubbleEditor has no Apply button. Corrected text lives in DB only — never rendered to page image. |
| Font/color settings | Only in RegionProperties | `POST /bubbles/{id}/font` | ❌ | Pipeline-detected bubbles (shown in BubbleEditor) have no typography controls |
| Auto-advance to next | `goToNextPending()` | N/A (frontend-only) | ✅ | Works |

### Impact
When a user corrects a translation in BubbleEditor, the corrected text is saved to the database but the page image still shows the old pipeline-rendered text. The Export page will ship the old text. This is the #1 user-facing bug.

### Fix Required
Add an "Apply to Page" button in BubbleEditor that calls `POST /bubbles/{id}/apply` and bumps `pageImageVersion` to refresh the canvas.

---

## 2. Manual Annotation Flow (RegionProperties.tsx ↔ bubbles.py)

| Feature | Frontend | Backend | Status | Notes |
|---------|----------|---------|--------|-------|
| Mode selector (4 modes) | RegionProperties UI | `body.mode` in apply_bubble | ✅ | translate_and_inpaint, inpaint_only, manual_text, review_later |
| Run OCR | `rerunOcr(id)` | `POST /bubbles/{id}/ocr` | ✅ | Uses ocr_smart preprocessing |
| Translate | `translateBubble(id)` | `POST /bubbles/{id}/translate` | ✅ | Ollama/Gemma → Google fallback |
| Apply (inpaint + render) | `applyBubble(id, body)` | `POST /bubbles/{id}/apply` | ✅ | Split save: inpaint saved first, text render second |
| Font family | Bangers / DejaVu Sans selector | `_FONT_MAP` in bubbles.py | ✅ | Only 2 fonts with actual files |
| Font size | Number input (auto if blank) | Auto-size 28→8 in `_render_bubble_text` | ✅ | |
| Font color | Color picker | `_hex_to_rgb` | ✅ | |
| Stroke color + width | Color picker + dropdown | Pillow stroke_width/stroke_fill | ✅ | |
| Text alignment | Left/Center/Right buttons | `align` param in multiline_text | ✅ | |
| Delete region | Delete button | `DELETE /bubbles/{id}` | ✅ | |
| Create manual bubble | Canvas draw → `createManualBubble` | `POST /pages/{id}/bubbles/manual` | ✅ | |
| Update polygon | Canvas drag → `updatePolygon` | `POST /bubbles/{id}/polygon` | ✅ | |
| Canvas refresh after Apply | `bumpPageImage()` → `?v=N` cache bust | Images endpoint ignores query params | ✅ | |

### Status: Fully functional for manually-drawn bubbles.

---

## 3. Export Page (Export.tsx ↔ chapters.py)

| Feature | Frontend | Backend | Status | Notes |
|---------|----------|---------|--------|-------|
| Export as CBZ | `getExportUrl(id, "cbz")` download link | `GET /chapters/{id}/export?format=cbz` | ⚠️ | Works, but exports `inpainted_image_path` which may have OLD rendered text if user corrected translations without re-applying |
| Export as PDF | `getExportUrl(id, "pdf")` download link | `GET /chapters/{id}/export?format=pdf` | ⚠️ | Same issue as CBZ |
| Acceptance rate | Computed from `chapter.accepted_bubbles` | ChapterOut model | ✅ | |
| Correction count | `reviewed - accepted` | Derived client-side | ✅ | |
| Correction heatmap table | **Hardcoded placeholder data** | `GET /chapters/{id}/analytics` | ❌ | Backend has `top_discrepancies` data but Export.tsx uses fake rows |
| "Accuracy Improvement" chart | **Static SVG path** | No historical endpoint | ❌ | Decorative only — no real data |
| "Configure Metadata" button | Button exists | No endpoint | ❌ | Placeholder, no-op |
| "Validation Report" button | Button exists | No endpoint | ❌ | Placeholder, no-op |
| Quality score distribution | Not displayed | `analytics` endpoint has `quality_distribution` | ❌ | Backend ready, frontend ignores |

### Fix Required
1. Wire Export.tsx to `GET /chapters/{id}/analytics` for real heatmap data
2. Add "Re-render All" button (or auto-apply on export) so corrections are baked into images before export

---

## 4. Backend Endpoints with NO Frontend Consumer

| Endpoint | Purpose | Priority |
|----------|---------|----------|
| `PATCH /bubbles/{id}/notes` | Save editorial notes per bubble | Medium — useful for translator workflow |
| `POST /chapters/{id}/find-replace` | Batch find-replace across chapter translations | Medium — useful for terminology consistency |
| `GET /chapters/{id}/analytics` | Quality distribution, top discrepancies | Medium — Export page could use this |
| `GET /api/projects/{series}/stats` | `AccuracyStats` for a series | Low — ProjectView could show this |

---

## 5. Frontend Features with NO Backend Support

| Feature | Location | What's Missing |
|---------|----------|----------------|
| Filter button (funnel icon) | Review.tsx top bar | No filtering logic or endpoint — purely decorative |
| Search button (magnifying glass) | Review.tsx top bar | No search logic — decorative |
| "System Status: Ready" bar | Export.tsx bottom | Hardcoded, no health-check endpoint |
| "DB OK" / "LATENCY: OK" | Export.tsx bottom | Hardcoded, no backend monitoring |

---

## 6. Pipeline ↔ Apply Inconsistencies

| Issue | Location | Impact |
|-------|----------|--------|
| Pipeline renders text with hardcoded `#0000FF` (blue) | `pipeline.py` default `text_color` | Pipeline output has blue text instead of black. Config has `DEFAULT_TEXT_COLOR="#000000"` but pipeline doesn't use it |
| Pipeline output saved as `inpainted_image_path` | `chapters.py:448` | The "inpainted" image actually has text rendered on it — confusing naming |
| No `final_image_path` ever created | Never set by any code | Export falls back to `inpainted_image_path`. The concept of a separate "final" image isn't implemented |
| `apply_bubble` modifies `inpainted_image_path` in-place | `bubbles.py:366` | Applying one bubble's changes overwrites the image that other bubbles' text was already rendered on. Applying bubble B after bubble A erases A's text during inpainting step |

### Critical: Apply-one-at-a-time destroys previous applies
When `apply_bubble` runs, it:
1. Opens the current `inpainted_image_path`
2. Inpaints the bubble region (erasing ALL content in that bbox, including previously rendered text from other bubbles)
3. Renders this bubble's text
4. Saves back

If bubble A and bubble B overlap or if the user applies A then B, bubble A's rendered text may be partially or fully erased by step 2 of bubble B's apply. A "Re-render All Bubbles" endpoint would fix this by processing all bubbles in order on a clean inpainted base.

---

## 7. Priority Ranking

### P0 — Must Fix (Breaks core workflow)
1. **Add Apply button to BubbleEditor** so corrected pipeline-detected bubbles get re-rendered
2. **Add "Re-render Page" endpoint** that inpaints all regions then renders all text in one pass (prevents apply-one-destroys-previous issue)

### P1 — Should Fix (User asked for these)
3. Wire Export.tsx analytics table to real `GET /chapters/{id}/analytics` data
4. Fix pipeline text color: use `DEFAULT_TEXT_COLOR` from config instead of hardcoded blue
5. Add notes UI to BubbleEditor (backend endpoint exists)

### P2 — Nice to Have
6. Add Find-Replace UI (backend endpoint exists)
7. Add quality score display in bubble list
8. Wire Filter/Search buttons in Review page
9. Implement health-check endpoint for Export status bar

---

## 8. Backend Requirement List for Full Frontend Parity

To achieve 1:1 parity, the backend needs:

1. **`POST /pages/{page_id}/rerender`** — Re-render ALL bubbles on a page: start from original image, inpaint all non-review_later regions, then render all text. Save as `inpainted_image_path`. This is the safe version of apply that doesn't destroy previous work.

2. **`POST /chapters/{chapter_id}/rerender`** — Re-render all pages in a chapter (calls page rerender for each). Needed before export.

3. **`GET /health`** — Simple health check returning DB status, LaMa availability, Ollama status, disk space. Powers the Export status bar.

4. No additional endpoints needed for notes, find-replace, or analytics — **these already exist** and just need frontend wiring.
