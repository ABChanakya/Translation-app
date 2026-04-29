import { useEffect, useCallback, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useStore } from "../store/useStore";
import {
  getChapter,
  getChapterPages,
  getPageBubbles,
  acceptBubble,
  correctBubble,
  skipBubble,
} from "../api/client";
import AnnotationCanvas from "../components/canvas/AnnotationCanvas";
import BubbleEditor from "../components/BubbleEditor";
import RegionProperties from "../components/RegionProperties";
import { cn } from "../lib/utils";
import { motion } from "motion/react";
import {
  MousePointer2,
  Square,
  Pentagon,
  Trash2,
  Image as ImageIcon,
  HeartPulse,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  ArrowLeft,
  MoreVertical,
  Filter,
  Search,
} from "lucide-react";

export default function Review() {
  const { chapterId } = useParams<{ chapterId: string }>();
  const navigate = useNavigate();
  const id = Number(chapterId);

  const {
    currentChapter, setCurrentChapter,
    pages, setPages,
    currentPageIndex, nextPage, prevPage,
    bubbles, setBubbles, updateBubble, activeBubbleId, setActiveBubbleId,
    canvasTool, setCanvasTool,
    imageView, setImageView,
    undo, pushUndo,
    addToast,
  } = useStore();

  const editorRef = useRef<HTMLDivElement>(null);
  const currentPage = pages[currentPageIndex] ?? null;

  // Load chapter + pages
  useEffect(() => {
    if (!id) return;
    getChapter(id).then(setCurrentChapter);
    getChapterPages(id).then(setPages);
  }, [id, setCurrentChapter, setPages]);

  // Load bubbles when page changes
  useEffect(() => {
    if (!currentPage) return;
    getPageBubbles(currentPage.id).then(setBubbles);
    setActiveBubbleId(null);
  }, [currentPage, setBubbles, setActiveBubbleId]);

  const reviewed = bubbles.filter((b) => b.status !== "pending").length;
  const total = bubbles.length;
  const progress = total ? (reviewed / total) * 100 : 0;

  const goToNextPending = useCallback(() => {
    const activeIdx = bubbles.findIndex((b) => b.id === activeBubbleId);
    for (let offset = 1; offset <= bubbles.length; offset++) {
      const b = bubbles[(activeIdx + offset) % bubbles.length];
      if (b.status === "pending") {
        setActiveBubbleId(b.id);
        return;
      }
    }
    if (currentPageIndex < pages.length - 1) {
      setTimeout(() => nextPage(), 800);
    }
  }, [bubbles, activeBubbleId, setActiveBubbleId, currentPageIndex, pages.length, nextPage]);

  const handleAccept = useCallback(async () => {
    if (!activeBubbleId) return;
    const updated = await acceptBubble(activeBubbleId);
    updateBubble(activeBubbleId, updated);
    pushUndo(() => updateBubble(activeBubbleId, { status: "pending", human_translation: null } as any));
    addToast("success", "Bubble accepted", `Bubble #${activeBubbleId} verified.`);
    goToNextPending();
  }, [activeBubbleId, updateBubble, pushUndo, goToNextPending, addToast]);

  const handleCorrect = useCallback(async (text: string) => {
    if (!activeBubbleId) return;
    const updated = await correctBubble(activeBubbleId, text);
    updateBubble(activeBubbleId, updated);
    goToNextPending();
  }, [activeBubbleId, updateBubble, goToNextPending]);

  const handleSkip = useCallback(async () => {
    if (!activeBubbleId) return;
    const updated = await skipBubble(activeBubbleId);
    updateBubble(activeBubbleId, updated);
    goToNextPending();
  }, [activeBubbleId, updateBubble, goToNextPending]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const inInput = target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable;

      if (!inInput) {
        if (e.key === "v" || e.key === "V") { useStore.getState().setCanvasTool("select"); return; }
        if (e.key === "r" || e.key === "R") { useStore.getState().setCanvasTool("draw_rect"); return; }
        if (e.key === "p" || e.key === "P") { useStore.getState().setCanvasTool("draw_polygon"); return; }
        if (e.key === "d" || e.key === "D") { useStore.getState().setCanvasTool("delete"); return; }
      }

      if (e.key === "Escape") {
        setActiveBubbleId(null);
        useStore.getState().setCanvasTool("select");
        return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        undo();
        return;
      }

      if (inInput) return;

      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        handleAccept();
        return;
      }
      if (e.key === "e" || e.key === "E") {
        const el = editorRef.current?.querySelector<HTMLTextAreaElement>("textarea");
        el?.focus();
        return;
      }
      if (e.key === "n" || e.key === "N") { handleSkip(); return; }
      if (e.key === "ArrowLeft") { prevPage(); return; }
      if (e.key === "ArrowRight") { nextPage(); return; }

      const num = parseInt(e.key);
      if (num >= 1 && num <= 9 && num <= bubbles.length) {
        setActiveBubbleId(bubbles[num - 1].id);
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleAccept, handleSkip, undo, prevPage, nextPage, bubbles, setActiveBubbleId]);

  // Auto-select first pending bubble
  useEffect(() => {
    if (bubbles.length && !activeBubbleId) {
      const first = bubbles.find((b) => b.status === "pending") ?? bubbles[0];
      setActiveBubbleId(first.id);
    }
  }, [bubbles, activeBubbleId, setActiveBubbleId]);

  const activeBubble = bubbles.find((b) => b.id === activeBubbleId) ?? null;

  const tools = [
    { key: "select" as const, icon: MousePointer2, title: "Select (V)", desc: "Select and move bubbles" },
    { key: "draw_rect" as const, icon: Square, title: "Rectangle (R)", desc: "Draw rectangular OCR regions" },
    { key: "draw_polygon" as const, icon: Pentagon, title: "Polygon (P)", desc: "Draw custom polygon regions" },
    { key: "delete" as const, icon: Trash2, title: "Delete (D)", desc: "Remove selected regions" },
  ];

  const viewModes = [
    { key: "original" as const, icon: ImageIcon, title: "Original", desc: "View raw scan" },
    { key: "inpainted" as const, icon: HeartPulse, title: "Inpainted", desc: "View cleaned scan (no text)" },
    { key: "final" as const, icon: Sparkles, title: "Final", desc: "View final typeset page" },
  ];

  return (
    <main className="flex-1 pt-14 pl-64 h-screen flex overflow-hidden">
      {/* Left toolbar */}
      <section className="w-16 bg-surface flex flex-col items-center py-4 gap-4 border-r border-outline-variant/10 z-20">
        <div className="flex flex-col gap-2">
          {tools.map((tool) => (
            <div key={tool.key} className="relative group/tool">
              <button
                onClick={() => setCanvasTool(tool.key)}
                aria-label={tool.title}
                className={cn(
                  "w-10 h-10 flex items-center justify-center transition-all active:scale-90",
                  canvasTool === tool.key
                    ? "bg-surface-container-highest text-primary active-glow"
                    : "text-on-surface-variant hover:bg-surface-container-high"
                )}
              >
                <tool.icon size={20} />
              </button>
              <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 bg-surface-container-highest text-[10px] font-black uppercase tracking-widest whitespace-nowrap opacity-0 group-hover/tool:opacity-100 pointer-events-none z-50 border border-outline-variant/20 shadow-xl">
                {tool.title}
                <div className="text-[8px] font-normal normal-case tracking-normal text-on-surface-variant/60">{tool.desc}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="h-[1px] w-8 bg-outline-variant/20 my-2" />
        <div className="flex flex-col gap-2">
          {viewModes.map((mode) => (
            <div key={mode.key} className="relative group/tool">
              <button
                onClick={() => setImageView(mode.key)}
                aria-label={mode.title}
                className={cn(
                  "w-10 h-10 flex items-center justify-center transition-all active:scale-90",
                  imageView === mode.key
                    ? "bg-surface-container-high text-primary"
                    : "text-on-surface-variant hover:bg-surface-container-high"
                )}
              >
                <mode.icon size={20} />
              </button>
              <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 bg-surface-container-highest text-[10px] font-black uppercase tracking-widest whitespace-nowrap opacity-0 group-hover/tool:opacity-100 pointer-events-none z-50 border border-outline-variant/20 shadow-xl">
                {mode.title}
                <div className="text-[8px] font-normal normal-case tracking-normal text-on-surface-variant/60">{mode.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Center canvas */}
      <section className="flex-1 bg-surface-container-lowest relative overflow-hidden flex flex-col">
        {/* Enhanced top info bar */}
        <div className="h-12 bg-surface/80 backdrop-blur-xl flex items-center justify-between px-6 text-[10px] uppercase tracking-widest font-black text-on-surface-variant border-b border-outline-variant/10 z-10">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate(-1)}
              className="flex items-center gap-2 hover:text-on-surface transition-colors group"
            >
              <ArrowLeft size={14} className="group-hover:-translate-x-0.5 transition-transform" />
              <span>Back to Project</span>
            </button>
            <div className="w-px h-4 bg-outline-variant/20" />
            <span className="text-on-surface">Chapter {currentChapter?.chapter_num ?? ""}</span>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={prevPage} disabled={currentPageIndex === 0} className="p-1 hover:text-primary disabled:opacity-30 transition-colors">
              <ChevronLeft size={16} />
            </button>
            <span className="text-on-surface font-mono bg-surface-container px-2 py-0.5 rounded-sm">
              Page {currentPage?.page_num ?? "?"} / {pages.length}
            </span>
            <button onClick={nextPage} disabled={currentPageIndex >= pages.length - 1} className="p-1 hover:text-primary disabled:opacity-30 transition-colors">
              <ChevronRight size={16} />
            </button>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-primary/60">Ch. {currentChapter?.chapter_num ?? "?"}</span>
            <button className="p-1 hover:text-on-surface transition-colors">
              <MoreVertical size={14} />
            </button>
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center p-8 overflow-auto">
          {currentPage ? (
            <div className="manga-shadow">
              <AnnotationCanvas pageId={currentPage.id} />
            </div>
          ) : (
            <p className="text-on-surface-variant/40">No page loaded</p>
          )}
        </div>
      </section>

      {/* Right panel: Bubble inspector */}
      <section
        ref={editorRef}
        className="w-[35%] max-w-[480px] bg-surface-container border-l border-outline-variant/10 flex flex-col shadow-2xl z-30"
      >
        <div className="flex flex-col border-b border-outline-variant/20 bg-surface">
          <div className="p-4 flex justify-between items-center">
            <h2 className="text-[10px] font-black uppercase tracking-widest text-on-surface-variant">Bubble Inspector</h2>
            <div className="flex items-center gap-2">
              <button aria-label="Filter" className="p-1.5 hover:bg-surface-container-high text-on-surface-variant transition-colors"><Filter size={14} /></button>
              <button aria-label="Search" className="p-1.5 hover:bg-surface-container-high text-on-surface-variant transition-colors"><Search size={14} /></button>
              <span className="text-[10px] text-on-surface-variant font-mono ml-1">{reviewed}/{total}</span>
            </div>
          </div>
          {/* Chapter Context (MangaProfile) */}
          <div className="px-4 py-2 bg-surface-container-low border-b border-outline-variant/10">
            <div className="flex items-center gap-2 mb-1">
              <Sparkles size={10} className="text-primary" />
              <span className="text-[8px] font-black uppercase tracking-[0.2em] text-on-surface-variant/60">Chapter Context (MangaProfile)</span>
            </div>
            <p className="text-[9px] text-on-surface-variant/40 italic leading-tight">
              {currentChapter
                ? `Chapter ${currentChapter.chapter_num} · ${currentChapter.total_bubbles} bubbles detected · ${Math.round((currentChapter.accepted_bubbles / Math.max(currentChapter.total_bubbles, 1)) * 100)}% accepted. Glossary and character names are loaded from the series profile and injected into each translation.`
                : "Load a chapter to see context."}
            </p>
          </div>
          {/* Animated progress bar */}
          <div className="h-1 w-full bg-surface-container-lowest" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              className="h-full bg-primary transition-all duration-500"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {activeBubble && activeBubble.is_manual ? (
            <RegionProperties bubble={activeBubble} />
          ) : (
            <BubbleEditor
              bubbles={bubbles}
              activeBubbleId={activeBubbleId}
              onSelect={setActiveBubbleId}
              onAccept={handleAccept}
              onCorrect={handleCorrect}
              onSkip={handleSkip}
            />
          )}
        </div>

        <div className="p-3 bg-surface text-[9px] uppercase tracking-widest font-black text-on-surface-variant/60 flex justify-between border-t border-outline-variant/10">
          <div className="flex gap-4">
            <span>Progress: {Math.round(progress)}%</span>
            <div className="w-px h-3 bg-outline-variant/20" />
            <span>Est. Time: {total ? `${Math.max(1, Math.round((total - reviewed) * 0.2))}m` : "---"}</span>
          </div>
          <div className="flex items-center gap-1.5 text-primary">
            <div className="w-1 h-1 rounded-full bg-primary animate-pulse" />
            <span>Auto-Save Active</span>
          </div>
        </div>
      </section>
    </main>
  );
}
