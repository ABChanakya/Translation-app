import { useEffect, useState, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useStore } from "../store/useStore";
import {
  getProject,
  getProjectStats,
  listProjectChapters,
  uploadChapter,
  getChapter,
} from "../api/client";
import type { AccuracyStats, Chapter } from "../api/types";
import { CloudUpload, Filter, Grid, ChevronRight, Plus, Sliders, Trash2, Terminal } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { cn } from "../lib/utils";
import React from "react";

export default function ProjectView() {
  const { series } = useParams<{ series: string }>();
  const navigate = useNavigate();
  const { currentProject, setCurrentProject, processingStatus, addProcessingStatus, clearProcessingStatus, showModal, addToast } = useStore();
  const [stats, setStats] = useState<AccuracyStats | null>(null);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [chapterNum, setChapterNum] = useState(1);
  const [confidence, setConfidence] = useState(0.10);
  const [iouThreshold, setIouThreshold] = useState(0.55);
  const [uploading, setUploading] = useState(false);
  const [processingChapterId, setProcessingChapterId] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!series) return;
    getProject(series).then(setCurrentProject).catch(console.error);
    getProjectStats(series).then(setStats).catch(console.error);
  }, [series, setCurrentProject]);

  useEffect(() => {
    if (!currentProject || !series) return;
    listProjectChapters(series).then(setChapters).catch(console.error);
  }, [currentProject, series]);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  // Live pipeline status stream: connect to the backend WebSocket while a
  // chapter is processing and push every event into the Console panel.
  useEffect(() => {
    if (!processingChapterId) return;

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    const url = `${proto}//${host}/api/chapters/${processingChapterId}/status`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        addProcessingStatus({
          page: msg.page ?? 0,
          stage: msg.stage ?? "info",
          message: msg.message ?? "",
          ts: msg.ts ?? new Date().toISOString(),
        });
      } catch {
        // non-JSON frame — ignore
      }
    };
    ws.onclose = () => {
      if (wsRef.current === ws) wsRef.current = null;
    };
    return () => {
      ws.close();
      if (wsRef.current === ws) wsRef.current = null;
    };
  }, [processingChapterId, addProcessingStatus]);

  // Auto-scroll console
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [processingStatus]);

  const handleUpload = useCallback(async () => {
    const files = fileRef.current?.files;
    if (!files?.length || !series) return;
    setUploading(true);
    clearProcessingStatus();

    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    const form = new FormData();
    for (const f of Array.from(files)) {
      form.append("files", f);
    }
    form.append("series_name", series);
    form.append("chapter_num", String(chapterNum));
    form.append("detection_confidence", String(confidence));
    form.append("nms_iou_threshold", String(iouThreshold));

    try {
      const res = await uploadChapter(form);
      setProcessingChapterId(res.chapter_id);
      addToast("info", "Processing started", `Chapter ${chapterNum} is now being ingested.`);

      addProcessingStatus({
        page: 0,
        stage: "processing",
        message: "[SYSTEM] Initializing OCR Pipeline...",
        ts: new Date().toISOString(),
      });

      pollRef.current = setInterval(async () => {
        try {
          const ch = await getChapter(res.chapter_id);
          if (ch.status === "ready" || ch.status === "complete") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setProcessingChapterId(null);
            setChapters((prev) => {
              const exists = prev.find((c) => c.id === ch.id);
              return exists
                ? prev.map((c) => (c.id === ch.id ? ch : c))
                : [...prev, ch];
            });
            addProcessingStatus({
              page: 0,
              stage: "done",
              message: `[DONE] ${ch.total_pages} pages, ${ch.total_bubbles} bubbles detected.`,
              ts: new Date().toISOString(),
            });
            addToast("success", "Processing complete", `Chapter ${ch.chapter_num}: ${ch.total_bubbles} bubbles detected.`);
          }
        } catch (e) {
          console.error("Poll error:", e);
        }
      }, 3000);
    } catch (e) {
      console.error("Upload failed:", e);
      addToast("error", "Upload failed", "Could not upload chapter. Check console for details.");
    } finally {
      setUploading(false);
    }
  }, [series, chapterNum, confidence, iouThreshold, clearProcessingStatus, addProcessingStatus, addToast]);

  const handleDeleteChapter = (id: number, num: number, e: React.MouseEvent) => {
    e.stopPropagation();
    showModal({
      title: "Delete Chapter",
      message: `Are you sure you want to delete Chapter ${num}? All OCR data and translations will be permanently removed.`,
      confirmLabel: "DELETE CHAPTER",
      variant: "danger",
      onConfirm: () => {
        setChapters((prev) => prev.filter((c) => c.id !== id));
        addToast("success", "Chapter deleted", `Chapter ${num} has been removed.`);
      },
    });
  };

  const summaryStats = [
    { label: "Bubbles Reviewed", value: stats ? String(stats.total_reviewed) : "0", color: "text-primary" },
    { label: "Acceptance Rate", value: stats ? `${Math.round(stats.acceptance_rate * 100)}%` : "0%", color: "text-tertiary" },
    { label: "Total Chapters", value: String(chapters.length), color: "text-on-surface" },
  ];

  return (
    <main className="mt-14 flex-1 flex flex-col overflow-hidden">
      {/* Project header */}
      <section className="p-8 bg-surface-dim border-b border-outline-variant/10">
        <div className="max-w-6xl mx-auto flex justify-between items-end">
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <h1 className="text-6xl font-black tracking-tighter text-on-surface mb-2">{series}</h1>
            <div className="flex gap-6">
              {summaryStats.map((stat, i) => (
                <React.Fragment key={i}>
                  <div className="flex flex-col">
                    <span className="text-[10px] uppercase tracking-widest text-on-surface-variant font-bold">{stat.label}</span>
                    <span className={cn("text-2xl font-mono", stat.color)}>{stat.value}</span>
                  </div>
                  {i < 2 && <div className="w-px h-10 bg-outline-variant/20 self-center" />}
                </React.Fragment>
              ))}
            </div>
          </motion.div>
          <div className="flex gap-3">
            <button
              onClick={() => navigate("/")}
              className="bg-surface-container-high hover:bg-surface-bright px-4 py-2 border border-outline-variant/20 text-[10px] font-black uppercase tracking-widest transition-all active:scale-95"
            >
              BACK TO HOME
            </button>
          </div>
        </div>
      </section>

      <div className="flex-1 overflow-y-auto bg-surface-container-lowest p-8">
        <div className="max-w-6xl mx-auto grid grid-cols-12 gap-8">
          {/* Left column: Upload + Console */}
          <div className="col-span-5 flex flex-col gap-6">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-surface-container p-6 border-l-4 border-primary"
            >
              <h2 className="text-[10px] font-black tracking-widest uppercase mb-4 text-on-surface">Ingest Chapter</h2>
              <div className="space-y-4">
                <div
                  className="border-2 border-dashed border-outline-variant/30 hover:border-primary/50 transition-colors p-8 flex flex-col items-center justify-center cursor-pointer group active:scale-[0.99]"
                  onClick={() => fileRef.current?.click()}
                >
                  <CloudUpload size={40} className="text-on-surface-variant group-hover:text-primary mb-2 transition-colors" />
                  <p className="text-[10px] uppercase tracking-widest font-bold text-on-surface-variant text-center">Drop chapter .zip or .jpg sequence here</p>
                </div>
                <input
                  ref={fileRef}
                  type="file"
                  multiple
                  accept="image/*,.zip,.cbz"
                  className="hidden"
                />
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col">
                    <label className="text-[10px] font-bold uppercase text-on-surface-variant mb-1">Chapter No.</label>
                    <input
                      className="bg-surface-container-lowest border-b border-outline-variant/30 focus:border-primary focus:ring-0 text-sm px-2 py-1 outline-none text-on-surface font-mono"
                      type="number"
                      min={1}
                      value={chapterNum}
                      onChange={(e) => setChapterNum(Number(e.target.value))}
                    />
                  </div>
                  <div className="flex items-end">
                    <button
                      onClick={handleUpload}
                      disabled={uploading || processingChapterId !== null}
                      className="gradient-cta w-full py-2.5 text-[10px] font-black uppercase tracking-widest text-on-primary disabled:opacity-50 active:scale-95 transition-all"
                    >
                      {processingChapterId ? "Processing..." : uploading ? "Uploading..." : "Start Processing"}
                    </button>
                  </div>
                </div>

                {/* Detection sliders */}
                <div className="pt-2 space-y-4 border-t border-outline-variant/10">
                  <div className="flex items-center gap-2 text-on-surface-variant/60 mb-2">
                    <Sliders size={12} />
                    <span className="text-[10px] font-black uppercase tracking-widest">Detection Parameters</span>
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-[10px] font-bold uppercase text-on-surface-variant">
                      <span>Confidence Threshold</span>
                      <span className="font-mono text-primary">{confidence.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min={0.01}
                      max={1.0}
                      step={0.01}
                      value={confidence}
                      onChange={(e) => setConfidence(Number(e.target.value))}
                      className="w-full h-1 bg-surface-container-lowest rounded-lg appearance-none cursor-pointer accent-primary"
                    />
                    <p className="text-[9px] text-on-surface-variant/60 leading-tight">
                      Lower values catch more text but increase false positives.
                    </p>
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-[10px] font-bold uppercase text-on-surface-variant">
                      <span>IoU Threshold (NMS)</span>
                      <span className="font-mono text-primary">{iouThreshold.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min={0.01}
                      max={1.0}
                      step={0.01}
                      value={iouThreshold}
                      onChange={(e) => setIouThreshold(Number(e.target.value))}
                      className="w-full h-1 bg-surface-container-lowest rounded-lg appearance-none cursor-pointer accent-primary"
                    />
                    <p className="text-[9px] text-on-surface-variant/60 leading-tight">
                      Controls overlap filtering. Lower values merge overlapping boxes more aggressively.
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Console output */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-surface-container-lowest border border-outline-variant/20 rounded-sm overflow-hidden flex flex-col shadow-2xl"
            >
              <div className="bg-surface-container-high px-3 py-1.5 flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <Terminal size={12} className="text-primary" />
                  <span className="font-mono text-[10px] uppercase font-bold text-on-surface-variant">Console_Output</span>
                </div>
                <div className="flex gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-error/40" />
                  <div className="w-2 h-2 rounded-full bg-tertiary/40" />
                  <div className="w-2 h-2 rounded-full bg-primary/40" />
                </div>
              </div>
              <div ref={consoleRef} className="p-4 font-mono text-[11px] leading-relaxed text-primary/80 h-64 overflow-y-auto">
                {processingStatus.length === 0 ? (
                  <div className="text-on-surface-variant/40">Waiting for input..._</div>
                ) : (
                  processingStatus.map((s, i) => (
                    <div
                      key={i}
                      className={
                        s.stage === "done"
                          ? "text-tertiary"
                          : s.stage === "error"
                          ? "text-error"
                          : ""
                      }
                    >
                      {s.message}
                    </div>
                  ))
                )}
              </div>
            </motion.div>
          </div>

          {/* Right column: Chapter library */}
          <div className="col-span-7 space-y-4">
            <div className="flex justify-between items-center mb-2">
              <h2 className="text-[10px] font-black tracking-widest uppercase text-on-surface">Chapter Library</h2>
              <div className="flex gap-2">
                <button aria-label="Filter Chapters" className="p-1.5 bg-surface-container hover:bg-surface-bright text-on-surface-variant transition-colors">
                  <Filter size={14} />
                </button>
                <button aria-label="Grid View" className="p-1.5 bg-surface-container hover:bg-surface-bright text-on-surface-variant transition-colors">
                  <Grid size={14} />
                </button>
              </div>
            </div>

            <AnimatePresence mode="popLayout">
              {chapters.length === 0 && !processingChapterId ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="border-2 border-dashed border-outline-variant/10 rounded-sm p-20 flex flex-col items-center justify-center text-center"
                >
                  <CloudUpload size={48} className="text-on-surface-variant/20 mb-4" />
                  <h3 className="text-lg font-black tracking-tighter text-on-surface-variant/60 uppercase">No chapters yet</h3>
                  <p className="text-xs text-on-surface-variant/40 mt-2 max-w-xs">Start by dropping a chapter file in the ingestion panel to begin processing.</p>
                </motion.div>
              ) : (
                chapters.map((chapter, index) => (
                  <motion.div
                    key={chapter.id}
                    layout
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ delay: index * 0.05 }}
                    onClick={() =>
                      chapter.status !== "processing" && navigate(`/review/${chapter.id}`)
                    }
                    className={cn(
                      "bg-surface-container p-4 flex items-center justify-between hover:bg-surface-container-high cursor-pointer transition-all border-l-4 border-transparent hover:border-primary group active:scale-[0.99]",
                      chapter.status === "processing" && "opacity-50 cursor-wait"
                    )}
                  >
                    <div className="flex items-center gap-6">
                      <div className="w-16 h-20 overflow-hidden relative border border-outline-variant/10 flex-shrink-0">
                        <div
                          className={cn(
                            "w-full h-full transition-all duration-500 group-hover:scale-110",
                            chapter.status === "processing" ? "opacity-30" : "brightness-50 group-hover:brightness-100 group-hover:grayscale-0 grayscale"
                          )}
                          style={{
                            background: `linear-gradient(${(chapter.chapter_num * 53) % 360}deg, hsl(${(chapter.chapter_num * 89) % 360},35%,10%) 0%, hsl(${(chapter.chapter_num * 137) % 360},45%,22%) 100%)`,
                          }}
                        />
                        <span className="absolute inset-0 flex items-center justify-center text-2xl font-black text-white/30 font-mono">
                          {chapter.chapter_num}
                        </span>
                      </div>
                      <div>
                        <h3 className="text-lg font-black tracking-tighter">Chapter {chapter.chapter_num}</h3>
                        <p className="text-[10px] uppercase tracking-widest font-bold text-on-surface-variant/60">
                          {chapter.total_pages} Pages &bull;{" "}
                          {chapter.status === "processing" ? "Processing OCR..." : `${chapter.total_bubbles} Bubbles`}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="flex flex-col items-end gap-2">
                        <span
                          className={cn(
                            "px-2 py-1 text-[10px] font-black uppercase tracking-widest border",
                            chapter.status === "processing"
                              ? "bg-tertiary/10 text-tertiary border-tertiary/20"
                              : chapter.status === "ready"
                              ? "bg-primary/10 text-primary border-primary/20"
                              : "bg-emerald-500/10 text-emerald-400 border-emerald-400/20"
                          )}
                        >
                          {chapter.status === "processing"
                            ? "Processing..."
                            : chapter.status === "ready"
                            ? `${chapter.reviewed_bubbles}/${chapter.total_bubbles} Reviewed`
                            : "Complete"}
                        </span>
                        {chapter.status === "processing" && (
                          <div className="w-32 h-1 bg-surface-container-lowest overflow-hidden" role="progressbar">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: "33%" }}
                              className="h-full bg-tertiary"
                            />
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={(e) => handleDeleteChapter(chapter.id, chapter.chapter_num, e)}
                          aria-label="Delete Chapter"
                          className="p-2 text-on-surface-variant/40 hover:text-error hover:bg-error/10 transition-all opacity-0 group-hover:opacity-100 rounded-sm"
                        >
                          <Trash2 size={16} />
                        </button>
                        {chapter.status !== "processing" && (
                          <ChevronRight size={18} className="text-on-surface-variant opacity-0 group-hover:opacity-100 transition-opacity" />
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      <button
        onClick={() => navigate("/")}
        className="fixed bottom-8 right-8 w-14 h-14 gradient-cta rounded-none shadow-2xl flex items-center justify-center group hover:scale-110 transition-transform active:scale-95"
      >
        <Plus size={30} className="text-on-primary" strokeWidth={3} />
        <span className="absolute right-full mr-4 bg-surface-container-highest border border-outline-variant/20 px-3 py-1.5 text-[10px] font-black uppercase tracking-widest whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-2xl">
          Back to Home
        </span>
      </button>
    </main>
  );
}
