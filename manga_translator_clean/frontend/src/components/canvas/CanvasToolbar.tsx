import { useStore, type CanvasTool } from "../../store/useStore";

const TOOLS: { id: CanvasTool; label: string; key: string }[] = [
  { id: "select", label: "Select", key: "V" },
  { id: "draw_rect", label: "Draw Rect", key: "R" },
  { id: "draw_polygon", label: "Draw Polygon", key: "P" },
  { id: "delete", label: "Delete", key: "D" },
];

const VIEW_OPTIONS = [
  { id: "original" as const, label: "Original" },
  { id: "inpainted" as const, label: "Inpainted" },
  { id: "final" as const, label: "Final" },
];

export default function CanvasToolbar() {
  const { canvasTool, setCanvasTool, imageView, setImageView } = useStore();

  return (
    <div className="flex items-center gap-4 px-4 py-1.5 border-b border-[var(--color-border)] bg-[var(--color-surface)] shrink-0">
      {/* Tools */}
      <div className="flex gap-1">
        {TOOLS.map((t) => (
          <button
            key={t.id}
            onClick={() => setCanvasTool(t.id)}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              canvasTool === t.id
                ? "bg-[var(--color-accent)] text-white"
                : "hover:bg-[var(--color-surface-hover)]"
            }`}
            title={`${t.label} (${t.key})`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="h-4 w-px bg-[var(--color-border)]" />

      {/* Image view toggle */}
      <div className="flex gap-1">
        {VIEW_OPTIONS.map((v) => (
          <button
            key={v.id}
            onClick={() => setImageView(v.id)}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              imageView === v.id
                ? "bg-[var(--color-surface-hover)] text-[var(--color-text)]"
                : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
            }`}
          >
            {v.label}
          </button>
        ))}
      </div>
    </div>
  );
}
