import { useState, useEffect } from "react";
import { useStore } from "../store/useStore";
import { updateFont, rerunOcr, translateBubble, applyBubble, deleteBubble } from "../api/client";
import type { Bubble, BubbleMode } from "../api/types";

interface Props {
  bubble: Bubble;
}

const MODE_OPTIONS: { value: BubbleMode; label: string; color: string; hint: string }[] = [
  {
    value: "translate_and_inpaint",
    label: "Translate + Inpaint",
    color: "var(--color-mode-translate)",
    hint: "Remove Japanese text, render English translation",
  },
  {
    value: "inpaint_only",
    label: "Inpaint Only",
    color: "var(--color-mode-inpaint)",
    hint: "Remove Japanese text, leave bubble empty",
  },
  {
    value: "manual_text",
    label: "Manual Text",
    color: "var(--color-mode-manual)",
    hint: "Type your own text and render it on the page",
  },
  {
    value: "review_later",
    label: "Review Later",
    color: "var(--color-mode-review)",
    hint: "Skip for now, come back later",
  },
];

// Only fonts that have actual font files available
const FONT_OPTIONS = ["Bangers", "DejaVu Sans"];

export default function RegionProperties({ bubble }: Props) {
  const { updateBubble, removeBubble, bumpPageImage, addToast } = useStore();

  const [mode, setMode]               = useState<BubbleMode>(bubble.mode);
  const [manualText, setManualText]   = useState(bubble.human_translation || "");
  const [fontFamily, setFontFamily]   = useState(bubble.font_family || "Bangers");
  const [fontSize, setFontSize]       = useState<number | "">(bubble.font_size ?? "");
  const [fontColor, setFontColor]     = useState(bubble.font_color || "#000000");
  const [strokeColor, setStrokeColor] = useState(bubble.stroke_color || "#ffffff");
  const [strokeWidth, setStrokeWidth] = useState(bubble.stroke_width ?? 1);
  const [textAlign, setTextAlign]     = useState(bubble.text_align || "center");

  const [ocrLoading, setOcrLoading]         = useState(false);
  const [translateLoading, setTranslateLoading] = useState(false);
  const [applying, setApplying]             = useState(false);

  // Local OCR/translation state (updated after button clicks without waiting for store refresh)
  const [japaneseText, setJapaneseText]               = useState(bubble.japanese_text || "");
  const [suggestedTranslation, setSuggestedTranslation] = useState(bubble.suggested_translation || "");

  // Reset all local state when a different bubble is selected
  useEffect(() => {
    setMode(bubble.mode);
    setManualText(bubble.human_translation || "");
    setFontFamily(bubble.font_family || "Bangers");
    setFontSize(bubble.font_size ?? "");
    setFontColor(bubble.font_color || "#000000");
    setStrokeColor(bubble.stroke_color || "#ffffff");
    setStrokeWidth(bubble.stroke_width ?? 1);
    setTextAlign(bubble.text_align || "center");
    setJapaneseText(bubble.japanese_text || "");
    setSuggestedTranslation(bubble.suggested_translation || "");
  }, [bubble.id]);

  const handleRerunOcr = async () => {
    setOcrLoading(true);
    try {
      const res = await rerunOcr(bubble.id);
      setJapaneseText(res.japanese_text);
      updateBubble(bubble.id, {
        japanese_text: res.japanese_text,
        ocr_confidence: res.ocr_confidence,
      });
      addToast("success", "OCR complete", `Detected: "${res.japanese_text.slice(0, 40)}"`);
    } catch (e) {
      addToast("error", "OCR failed", String(e));
    }
    setOcrLoading(false);
  };

  const handleTranslate = async () => {
    if (!japaneseText) {
      addToast("error", "No Japanese text", "Run OCR first to detect text.");
      return;
    }
    setTranslateLoading(true);
    try {
      const res = await translateBubble(bubble.id);
      setSuggestedTranslation(res.suggested_translation);
      updateBubble(bubble.id, { suggested_translation: res.suggested_translation });
      addToast("success", "Translation ready", `"${res.suggested_translation.slice(0, 50)}"`);
    } catch (e) {
      addToast("error", "Translation failed", String(e));
    }
    setTranslateLoading(false);
  };

  const handleApply = async () => {
    setApplying(true);
    try {
      // Save font settings
      await updateFont(bubble.id, {
        font_family: fontFamily,
        font_size: fontSize || null,
        font_color: fontColor,
        stroke_color: strokeColor,
        stroke_width: strokeWidth,
        text_align: textAlign,
      });

      // Update Zustand store
      updateBubble(bubble.id, {
        mode,
        human_translation: mode === "manual_text" ? manualText : (bubble.human_translation ?? undefined) as any,
        font_family: fontFamily,
        font_size: fontSize || null,
        font_color: fontColor,
        stroke_color: strokeColor,
        stroke_width: strokeWidth,
        text_align: textAlign,
      });

      // Apply on backend (inpaint + render text)
      const result = await applyBubble(bubble.id, {
        mode,
        human_translation: mode === "manual_text" ? manualText
          : (suggestedTranslation || bubble.human_translation || undefined),
      });

      // Force canvas to reload the updated image
      bumpPageImage();

      if ((result as any).warning) {
        addToast("error", "Text render failed", (result as any).warning + " — inpainting was saved.");
      } else {
        addToast("success", "Applied!", "Page updated.");
      }
    } catch (e) {
      addToast("error", "Apply failed", String(e));
    }
    setApplying(false);
  };

  const handleDelete = async () => {
    if (!confirm("Delete this region?")) return;
    await deleteBubble(bubble.id);
    removeBubble(bubble.id);
    addToast("info", "Region deleted", "");
  };

  const showTextContent = mode === "translate_and_inpaint" || mode === "manual_text";

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-sm">Region Properties</h2>
        <span
          className={`text-xs px-2 py-0.5 rounded font-medium ${
            bubble.is_manual
              ? "bg-green-500/20 text-green-400"
              : "bg-blue-500/20 text-blue-400"
          }`}
        >
          {bubble.is_manual ? "MANUAL" : "AUTO"}
        </span>
      </div>

      {/* Coords */}
      <p className="text-xs text-[var(--color-text-muted)] font-mono">
        ({bubble.x1}, {bubble.y1}) &rarr; ({bubble.x2}, {bubble.y2})
      </p>

      {/* Mode selector */}
      <div>
        <label className="block text-xs text-[var(--color-text-muted)] mb-2 uppercase tracking-wide font-semibold">Mode</label>
        <div className="grid grid-cols-2 gap-1.5">
          {MODE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setMode(opt.value)}
              title={opt.hint}
              className={`text-xs px-3 py-2 rounded-lg border transition-all text-left ${
                mode === opt.value
                  ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 font-medium"
                  : "border-[var(--color-border)] hover:border-[var(--color-border)] opacity-70 hover:opacity-100"
              }`}
            >
              <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ background: opt.color }} />
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content area */}
      {showTextContent && (
        <div className="space-y-3">
          {mode === "translate_and_inpaint" && (
            <>
              {/* OCR */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-[var(--color-text-muted)]">Japanese (OCR)</label>
                  <button
                    onClick={handleRerunOcr}
                    disabled={ocrLoading}
                    className="text-xs px-2 py-0.5 rounded bg-[var(--color-surface-hover)] hover:bg-[var(--color-border)] disabled:opacity-50 transition-colors"
                  >
                    {ocrLoading ? "Running…" : "Run OCR"}
                  </button>
                </div>
                <div className="text-sm font-mono bg-[var(--color-bg)] rounded px-3 py-2 border border-[var(--color-border)] min-h-[2rem] text-[var(--color-text-muted)]">
                  {japaneseText || <span className="italic opacity-50">Click "Run OCR" to detect text</span>}
                </div>
              </div>

              {/* Translation */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-[var(--color-text-muted)]">English Translation</label>
                  <button
                    onClick={handleTranslate}
                    disabled={translateLoading || !japaneseText}
                    className="text-xs px-2 py-0.5 rounded bg-[var(--color-accent)]/20 hover:bg-[var(--color-accent)]/30 text-[var(--color-accent)] disabled:opacity-40 transition-colors"
                  >
                    {translateLoading ? "Translating…" : "Translate"}
                  </button>
                </div>
                <div className="text-sm bg-[var(--color-bg)] rounded px-3 py-2 border border-[var(--color-border)] min-h-[2rem] text-[var(--color-text-muted)]">
                  {suggestedTranslation || <span className="italic opacity-50">Click "Translate" after OCR</span>}
                </div>
              </div>
            </>
          )}

          {mode === "manual_text" && (
            <div>
              <label className="text-xs text-[var(--color-text-muted)] mb-1 block">Text to render</label>
              <textarea
                value={manualText}
                onChange={(e) => setManualText(e.target.value)}
                rows={3}
                className="w-full px-3 py-2 rounded bg-[var(--color-bg)] border border-[var(--color-border)] text-sm resize-none outline-none focus:border-[var(--color-accent)]"
                placeholder="Type your translation here…"
                autoFocus
              />
            </div>
          )}
        </div>
      )}

      {mode === "inpaint_only" && (
        <p className="text-xs text-[var(--color-text-muted)] bg-[var(--color-bg)] rounded px-3 py-2 border border-[var(--color-border)]">
          Japanese text will be removed from this region. No replacement text will be rendered.
        </p>
      )}

      {mode === "review_later" && (
        <p className="text-xs text-[var(--color-text-muted)] bg-[var(--color-bg)] rounded px-3 py-2 border border-[var(--color-border)]">
          This region is marked for later review. No changes will be applied.
        </p>
      )}

      {/* Font settings (only when rendering text) */}
      {showTextContent && (
        <div>
          <label className="block text-xs text-[var(--color-text-muted)] mb-2 uppercase tracking-wide font-semibold">Typography</label>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <select
              value={fontFamily}
              onChange={(e) => setFontFamily(e.target.value)}
              className="px-2 py-1.5 rounded bg-[var(--color-bg)] border border-[var(--color-border)] text-xs"
            >
              {FONT_OPTIONS.map((f) => <option key={f}>{f}</option>)}
            </select>
            <input
              type="number"
              placeholder="Auto"
              value={fontSize}
              min={6}
              max={72}
              onChange={(e) => setFontSize(e.target.value ? Number(e.target.value) : "")}
              className="px-2 py-1.5 rounded bg-[var(--color-bg)] border border-[var(--color-border)] text-xs text-center"
            />
          </div>

          <div className="grid grid-cols-3 gap-2 mb-2">
            <div>
              <p className="text-[10px] text-[var(--color-text-muted)] mb-0.5">Text</p>
              <input type="color" value={fontColor} onChange={(e) => setFontColor(e.target.value)} className="w-full h-7 rounded cursor-pointer border border-[var(--color-border)]" />
            </div>
            <div>
              <p className="text-[10px] text-[var(--color-text-muted)] mb-0.5">Stroke</p>
              <input type="color" value={strokeColor} onChange={(e) => setStrokeColor(e.target.value)} className="w-full h-7 rounded cursor-pointer border border-[var(--color-border)]" />
            </div>
            <div>
              <p className="text-[10px] text-[var(--color-text-muted)] mb-0.5">Stroke W</p>
              <select value={strokeWidth} onChange={(e) => setStrokeWidth(Number(e.target.value))} className="w-full px-1 py-1.5 rounded bg-[var(--color-bg)] border border-[var(--color-border)] text-xs">
                {[0, 1, 2, 3].map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          </div>

          <div className="flex gap-1">
            {(["left", "center", "right"] as const).map((a) => (
              <button
                key={a}
                onClick={() => setTextAlign(a)}
                className={`flex-1 text-xs py-1 rounded transition-colors ${
                  textAlign === a
                    ? "bg-[var(--color-accent)] text-white"
                    : "bg-[var(--color-bg)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)]"
                }`}
              >
                {a.charAt(0).toUpperCase() + a.slice(1)}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 pt-1">
        <button
          onClick={handleApply}
          disabled={applying}
          className="flex-1 px-4 py-2 rounded-lg bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white font-medium text-sm disabled:opacity-50 transition-colors"
        >
          {applying ? "Applying…" : "Apply"}
        </button>
        <button
          onClick={handleDelete}
          className="px-4 py-2 rounded-lg border border-[var(--color-danger)]/30 text-[var(--color-danger)] hover:bg-[var(--color-danger)]/10 text-sm transition-colors"
        >
          Delete
        </button>
      </div>
    </div>
  );
}
