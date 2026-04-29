import { useState, useEffect } from "react";
import { Trash2, RefreshCw, X as XIcon } from "lucide-react";
import { useStore } from "../store/useStore";
import { deleteBubble, updateBubbleType, applyBubble } from "../api/client";
import type { Bubble } from "../api/types";
import RegionProperties from "./RegionProperties";

interface Props {
  bubbles: Bubble[];
  activeBubbleId: number | null;
  onSelect: (id: number) => void;
  onAccept: () => void;
  onCorrect: (text: string) => void;
  onSkip: () => void;
}

const STATUS_COLORS: Record<string, string> = {
  pending:   "border-[var(--color-border)]",
  accepted:  "border-[var(--color-success)]",
  corrected: "border-[var(--color-warning)]",
  skipped:   "border-[var(--color-danger)]",
};

const BUBBLE_TYPES = [
  { value: "speech",    label: "SPEECH",    color: "bg-blue-500/20 text-blue-400" },
  { value: "sfx",       label: "SFX",       color: "bg-red-500/20 text-red-400" },
  { value: "narration", label: "NARRATION", color: "bg-purple-500/20 text-purple-400" },
  { value: "thought",   label: "THOUGHT",   color: "bg-cyan-500/20 text-cyan-400" },
  { value: "signs",     label: "SIGNS",     color: "bg-orange-500/20 text-orange-400" },
];

function typeBadge(type: string) {
  return BUBBLE_TYPES.find((t) => t.value === type) ?? BUBBLE_TYPES[0];
}

export default function BubbleEditor({
  bubbles, activeBubbleId, onSelect, onAccept, onSkip,
}: Props) {
  const { removeBubble, updateBubble, addToast, bumpPageImage } = useStore();
  const active = bubbles.find((b) => b.id === activeBubbleId) ?? null;

  const [editing, setEditing]             = useState(false);
  const [relabeling, setRelabeling]       = useState<number | null>(null); // bubble id being relabeled
  const [applying, setApplying]           = useState<number | null>(null); // bubble id being applied

  useEffect(() => {
    if (active) {
      setEditing(false);
      setRelabeling(null);
    }
  }, [active?.id]);

  const handleDelete = async (b: Bubble, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`Delete bubble #${b.id}?`)) return;
    try {
      await deleteBubble(b.id);
      removeBubble(b.id);
    } catch (err) {
      addToast("error", "Delete failed", String(err));
    }
  };

  const handleRelabel = async (b: Bubble, newType: string) => {
    try {
      const updated = await updateBubbleType(b.id, newType);
      updateBubble(b.id, { bubble_type: updated.bubble_type });
    } catch (err) {
      addToast("error", "Relabel failed", String(err));
    }
    setRelabeling(null);
  };

  const handleApply = async (b: Bubble) => {
    setApplying(b.id);
    try {
      const text = b.human_translation || b.suggested_translation || undefined;
      const result = await applyBubble(b.id, {
        mode: "translate_and_inpaint",
        human_translation: text,
      });
      bumpPageImage();
      if ((result as any).warning) {
        addToast("error", "Text render failed", (result as any).warning);
      } else {
        addToast("success", "Applied!", `Bubble #${b.id} rendered to page.`);
      }
    } catch (err) {
      addToast("error", "Apply failed", String(err));
    }
    setApplying(null);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-[var(--color-border)]">
        <h2 className="font-semibold">Translation Review</h2>
      </div>

      <div className="flex-1 overflow-y-auto">
        {bubbles.map((b, i) => {
          const isActive   = b.id === activeBubbleId;
          const badge      = typeBadge(b.bubble_type);
          const isRelabeling = relabeling === b.id;

          return (
            <div
              key={b.id}
              onClick={() => { onSelect(b.id); setRelabeling(null); }}
              className={`px-4 py-3 border-l-2 cursor-pointer transition-colors ${
                isActive
                  ? "bg-[var(--color-surface)] border-l-[var(--color-accent)]"
                  : STATUS_COLORS[b.status] + " hover:bg-[var(--color-surface-hover)]"
              }`}
            >
              {/* Row header: index · type badge · status badge · delete */}
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs font-mono text-[var(--color-text-muted)] w-5 shrink-0">
                  {i + 1}
                </span>

                {/* Type badge — click to relabel when active */}
                {isActive ? (
                  isRelabeling ? (
                    <div
                      className="flex gap-1 flex-wrap"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {BUBBLE_TYPES.map((t) => (
                        <button
                          key={t.value}
                          onClick={() => handleRelabel(b, t.value)}
                          className={`text-[10px] px-1.5 py-0.5 rounded font-medium border transition-colors ${
                            b.bubble_type === t.value
                              ? t.color + " border-current"
                              : "border-[var(--color-border)] hover:bg-[var(--color-surface-hover)]"
                          }`}
                        >
                          {t.label}
                        </button>
                      ))}
                      <button
                        onClick={(e) => { e.stopPropagation(); setRelabeling(null); }}
                        className="text-[10px] px-1.5 py-0.5 rounded border border-[var(--color-border)]"
                      >
                        ✕
                      </button>
                    </div>
                  ) : (
                    <button
                      title="Click to change type"
                      onClick={(e) => { e.stopPropagation(); setRelabeling(b.id); }}
                      className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${badge.color} hover:ring-1 hover:ring-current transition-all`}
                    >
                      {badge.label} ▾
                    </button>
                  )
                ) : (
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${badge.color}`}>
                    {badge.label}
                  </span>
                )}

                {b.status !== "pending" && (
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                    b.status === "accepted"  ? "bg-green-500/20 text-green-400" :
                    b.status === "corrected" ? "bg-yellow-500/20 text-yellow-400" :
                                               "bg-red-500/20 text-red-400"
                  }`}>
                    {b.status.toUpperCase()}
                  </span>
                )}

                {/* Delete button — always visible on hover, always accessible */}
                <button
                  title="Delete this bubble"
                  onClick={(e) => handleDelete(b, e)}
                  className={`ml-auto p-0.5 rounded transition-colors hover:text-[var(--color-danger)] ${
                    isActive ? "text-[var(--color-text-muted)]" : "text-transparent hover:text-[var(--color-danger)]"
                  }`}
                >
                  <Trash2 size={13} />
                </button>
              </div>

              {/* Japanese source */}
              <p className="text-xs text-[var(--color-text-muted)] mb-1 font-mono truncate">
                {b.japanese_text || <span className="italic opacity-50">(no OCR text)</span>}
              </p>

              {/* Translation */}
              {isActive && editing ? (
                <div
                  onClick={(e) => e.stopPropagation()}
                  className="mt-2 rounded-md border border-[var(--color-accent)]/40 bg-[var(--color-bg)]"
                >
                  <div className="flex items-center justify-between px-3 py-1.5 border-b border-[var(--color-border)]">
                    <span className="text-[10px] font-black uppercase tracking-widest text-[var(--color-text-muted)]">
                      Full Editor
                    </span>
                    <button
                      title="Close editor"
                      onClick={() => setEditing(false)}
                      className="p-0.5 rounded hover:bg-[var(--color-surface-hover)] text-[var(--color-text-muted)]"
                    >
                      <XIcon size={13} />
                    </button>
                  </div>
                  <RegionProperties bubble={b} />
                </div>
              ) : (
                <p
                  className={`text-sm ${isActive ? "text-[var(--color-text)]" : "text-[var(--color-text-muted)]"}`}
                  onDoubleClick={() => isActive && setEditing(true)}
                >
                  {b.human_translation || b.suggested_translation || (
                    <span className="italic opacity-40">(no translation)</span>
                  )}
                </p>
              )}

              {/* Action row for active pending bubble */}
              {isActive && !editing && !isRelabeling && b.status === "pending" && (
                <div className="flex gap-2 mt-2" onClick={(e) => e.stopPropagation()}>
                  <button
                    onClick={onAccept}
                    className="text-xs px-3 py-1 rounded bg-[var(--color-success)] text-white font-medium"
                  >
                    Accept
                  </button>
                  <button
                    onClick={() => setEditing(true)}
                    className="text-xs px-3 py-1 rounded border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)]"
                  >
                    Edit
                  </button>
                  <button
                    onClick={onSkip}
                    className="text-xs px-3 py-1 rounded border border-[var(--color-danger)]/30 text-[var(--color-danger)] hover:bg-[var(--color-danger)]/10"
                  >
                    Skip
                  </button>
                </div>
              )}

              {/* Apply to Page — re-render this bubble's translation onto the page image */}
              {isActive && !editing && !isRelabeling && (b.status === "corrected" || b.status === "accepted") && (
                <div className="flex gap-2 mt-2" onClick={(e) => e.stopPropagation()}>
                  <button
                    onClick={() => handleApply(b)}
                    disabled={applying === b.id}
                    className="text-xs px-3 py-1 rounded bg-[var(--color-accent)] text-white font-medium disabled:opacity-50 flex items-center gap-1.5"
                  >
                    <RefreshCw size={11} className={applying === b.id ? "animate-spin" : ""} />
                    {applying === b.id ? "Applying…" : "Apply to Page"}
                  </button>
                  <button
                    onClick={() => setEditing(true)}
                    className="text-xs px-3 py-1 rounded border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)]"
                  >
                    Edit
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
