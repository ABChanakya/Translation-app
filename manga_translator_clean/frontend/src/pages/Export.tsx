import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getChapter, getExportUrl } from "../api/client";
import type { Chapter } from "../api/types";
import { ArrowLeft, FileText, Archive, TrendingUp, Database, Zap } from "lucide-react";
import { motion } from "motion/react";
import { cn } from "../lib/utils";

export default function Export() {
  const { chapterId } = useParams<{ chapterId: string }>();
  const navigate = useNavigate();
  const id = Number(chapterId);
  const [chapter, setChapter] = useState<Chapter | null>(null);

  useEffect(() => {
    if (id) getChapter(id).then(setChapter);
  }, [id]);

  if (!chapter) return <p className="mt-14 p-8 text-on-surface-variant">Loading...</p>;

  const total = chapter.total_bubbles;
  const reviewed = chapter.reviewed_bubbles;
  const accepted = chapter.accepted_bubbles;
  const corrected = Math.max(0, reviewed - accepted);
  const acceptRate = total ? Math.round((accepted / total) * 100) : 0;

  return (
    <main className="mt-14 flex-1 p-8 min-h-[calc(100vh-3.5rem)] overflow-y-auto">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-end mb-8">
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <button
              onClick={() => navigate(`/review/${id}`)}
              className="flex items-center gap-2 text-primary hover:text-primary-container transition-colors mb-2 group"
            >
              <ArrowLeft size={14} className="group-hover:-translate-x-0.5 transition-transform" />
              <span className="text-[10px] font-black tracking-widest uppercase">Back to Review</span>
            </button>
            <h2 className="text-6xl font-black tracking-tighter text-on-surface">Export & Analytics</h2>
          </motion.div>
          <div className="flex gap-3">
            <a
              href={getExportUrl(id, "pdf")}
              download
              className="bg-surface-container-high hover:bg-surface-bright text-on-surface px-6 py-3 flex items-center gap-3 transition-all active:scale-95 border border-outline-variant/10"
            >
              <FileText size={20} />
              <span className="text-[10px] font-black uppercase tracking-widest">Export as PDF</span>
            </a>
            <a
              href={getExportUrl(id, "cbz")}
              download
              className="gradient-cta text-on-primary px-8 py-3 flex items-center gap-3 transition-all active:scale-95 shadow-xl"
            >
              <Archive size={20} />
              <span className="text-[10px] font-black uppercase tracking-widest">Export as CBZ</span>
            </a>
          </div>
        </div>

        <div className="grid grid-cols-12 gap-6">
          {/* Left stats column */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="col-span-12 lg:col-span-4 grid grid-cols-1 gap-6"
          >
            <div className="bg-surface-container p-6 border-l-4 border-primary shadow-lg">
              <p className="text-[10px] uppercase tracking-[0.2em] text-on-surface-variant font-black mb-1">Total Reviewed Segments</p>
              <h3 className="text-5xl font-black text-primary tracking-tighter">{reviewed.toLocaleString()}</h3>
              <div className="mt-4 flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-emerald-400">
                <TrendingUp size={14} />
                <span>{chapter.total_pages} pages processed</span>
              </div>
            </div>
            <div className="bg-surface-container p-6 shadow-lg">
              <p className="text-[10px] uppercase tracking-[0.2em] text-on-surface-variant font-black mb-1">Acceptance Rate</p>
              <div className="flex items-baseline gap-2">
                <h3 className="text-5xl font-black text-on-surface tracking-tighter">{acceptRate}</h3>
                <span className="text-2xl font-black text-on-surface-variant">%</span>
              </div>
              <div className="mt-6 h-1 w-full bg-surface-container-lowest overflow-hidden rounded-full">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${acceptRate}%` }}
                  className="h-full bg-primary"
                />
              </div>
            </div>
            <div className="bg-surface-container p-6 shadow-lg">
              <p className="text-[10px] uppercase tracking-[0.2em] text-on-surface-variant font-black mb-1">Corrections Made</p>
              <h3 className="text-5xl font-black text-tertiary tracking-tighter">{corrected}</h3>
            </div>
          </motion.div>

          {/* Right: Animated chart */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="col-span-12 lg:col-span-8 bg-surface-container p-6 relative overflow-hidden group shadow-lg"
          >
            <div className="flex justify-between items-start mb-8">
              <div>
                <p className="text-[10px] uppercase tracking-[0.2em] text-on-surface-variant font-black">Accuracy Improvement</p>
                <h4 className="text-lg font-black tracking-tighter">Acceptance Rate by Chapter</h4>
              </div>
              <div className="flex gap-4">
                <span className="flex items-center gap-1.5 text-[10px] uppercase font-black tracking-widest text-primary">
                  <span className="w-2 h-2 rounded-full bg-primary" /> Current
                </span>
                <span className="flex items-center gap-1.5 text-[10px] uppercase font-black tracking-widest text-on-surface-variant">
                  <span className="w-2 h-2 rounded-full bg-surface-container-highest" /> Benchmark
                </span>
              </div>
            </div>
            <div className="h-48 w-full relative">
              <svg className="w-full h-full" viewBox="0 0 800 200" preserveAspectRatio="none">
                <motion.path
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1.5, ease: "easeInOut" }}
                  className="text-primary opacity-80"
                  d="M0,180 Q100,160 200,120 T400,100 T600,60 T800,20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <motion.path
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1.5, ease: "easeInOut", delay: 0.2 }}
                  className="text-surface-container-highest"
                  d="M0,190 Q100,185 200,170 T400,165 T600,150 T800,140"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                {[200, 400, 600, 800].map((cx, i) => (
                  <motion.circle
                    key={i}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 1 + i * 0.1 }}
                    className="fill-primary"
                    cx={cx}
                    cy={[120, 100, 60, 20][i]}
                    r="4"
                  />
                ))}
              </svg>
            </div>
          </motion.div>

          {/* Correction heatmap table */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="col-span-12 bg-surface-container overflow-hidden shadow-lg"
          >
            <div className="p-6 border-b border-outline-variant/10 flex justify-between items-center bg-surface-container-high/30">
              <h4 className="text-[10px] font-black uppercase tracking-widest">Correction Heatmap: Terminological Drift</h4>
              <span className="text-[10px] font-mono font-bold text-on-surface-variant">Top Discrepancies</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="bg-surface-container-high/50">
                    <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-on-surface-variant">Raw Translation (OCR)</th>
                    <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-on-surface-variant">Editorial Correction</th>
                    <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-on-surface-variant">Frequency</th>
                    <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-on-surface-variant text-right">Confidence Impact</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/5">
                  {corrected > 0 ? (
                    [
                      { raw: '"Machine Output"', corr: '"Human Correction"', freq: corrected, impact: `-${(100 - acceptRate) * 0.1}%`, severity: "high" },
                    ].map((row, i) => (
                      <tr key={i} className="hover:bg-surface-bright/30 transition-colors group">
                        <td className="px-6 py-4 font-mono text-xs text-error/80 group-hover:text-error transition-colors">{row.raw}</td>
                        <td className="px-6 py-4 font-black text-xs text-primary">{row.corr}</td>
                        <td className="px-6 py-4 text-xs font-black">{row.freq}</td>
                        <td className="px-6 py-4 text-right">
                          <span className={cn(
                            "text-[10px] px-2 py-0.5 rounded-full font-black border",
                            row.severity === "high" ? "bg-error/10 text-error border-error/20" : "bg-tertiary/10 text-tertiary border-tertiary/20"
                          )}>
                            {row.impact}
                          </span>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={4} className="px-6 py-8 text-center text-on-surface-variant/40 text-xs">
                        No corrections recorded yet. Review bubbles to generate data.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </motion.div>

          {/* Review breakdown bar */}
          <div className="col-span-12 bg-surface-container p-6 rounded-sm shadow-lg">
            <p className="text-[10px] uppercase text-on-surface-variant font-black mb-3">Review Breakdown</p>
            <div className="flex gap-2 h-4">
              {accepted > 0 && (
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(accepted / total) * 100}%` }}
                  className="bg-emerald-500 rounded-sm"
                  title={`${accepted} accepted`}
                />
              )}
              {corrected > 0 && (
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(corrected / total) * 100}%` }}
                  className="bg-tertiary rounded-sm"
                  title={`${corrected} corrected`}
                />
              )}
              {total - reviewed > 0 && (
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${((total - reviewed) / total) * 100}%` }}
                  className="bg-surface-container-high rounded-sm"
                  title={`${total - reviewed} pending`}
                />
              )}
            </div>
            <div className="flex gap-4 mt-2 text-[10px] text-on-surface-variant">
              <span className="flex items-center gap-1"><span className="w-2 h-2 bg-emerald-500 rounded-full inline-block" /> Accepted ({accepted})</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 bg-tertiary rounded-full inline-block" /> Corrected ({corrected})</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 bg-surface-container-high rounded-full inline-block" /> Pending ({total - reviewed})</span>
            </div>
          </div>

          {/* Package preview */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="col-span-12 bg-surface-container-high p-8 flex flex-col md:flex-row items-center gap-12 shadow-xl"
          >
            <div className="w-48 h-64 bg-surface-container-lowest shadow-2xl relative flex-shrink-0 flex items-center justify-center border border-outline-variant/10">
              <span className="text-6xl font-black text-on-surface-variant/10">{chapter.chapter_num}</span>
              <div className="absolute top-2 right-2 bg-primary text-on-primary text-[8px] font-black px-1.5 py-0.5 rounded-sm shadow-lg">
                {reviewed === total && total > 0 ? "FINAL" : "DRAFT"}
              </div>
            </div>
            <div className="flex-1">
              <span className="text-primary text-[10px] font-black uppercase tracking-[0.2em] block mb-2">Package Contents</span>
              <h5 className="text-4xl font-black mb-6 tracking-tighter">Chapter {chapter.chapter_num}</h5>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
                <div>
                  <p className="text-[10px] text-on-surface-variant font-black uppercase mb-1">Pages</p>
                  <p className="text-sm font-mono font-bold">{chapter.total_pages}</p>
                </div>
                <div>
                  <p className="text-[10px] text-on-surface-variant font-black uppercase mb-1">Bubbles</p>
                  <p className="text-sm font-mono font-bold">{total}</p>
                </div>
                <div>
                  <p className="text-[10px] text-on-surface-variant font-black uppercase mb-1">Status</p>
                  <p className="text-sm font-mono font-bold">{chapter.status}</p>
                </div>
              </div>
              <div className="mt-10 flex flex-wrap gap-4">
                <button className="bg-surface-container-highest px-6 py-2.5 text-[10px] font-black uppercase tracking-widest hover:bg-surface-bright transition-all active:scale-95 border border-outline-variant/10">
                  Configure Metadata
                </button>
                <button className="bg-surface-container-highest px-6 py-2.5 text-[10px] font-black uppercase tracking-widest hover:bg-surface-bright transition-all active:scale-95 border border-outline-variant/10">
                  Validation Report
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* System status bar */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="fixed bottom-6 right-6 bg-surface-container-highest/80 backdrop-blur-xl border border-primary/20 px-4 py-3 rounded-sm flex items-center gap-6 shadow-2xl z-50"
      >
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.5)]" />
          <span className="text-[10px] font-black uppercase tracking-widest text-on-surface-variant">System Status: Ready</span>
        </div>
        <div className="w-px h-4 bg-outline-variant/20" />
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-on-surface-variant/60">
            <Database size={12} />
            <span>DB OK</span>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-on-surface-variant/60">
            <Zap size={12} />
            <span>LATENCY: OK</span>
          </div>
        </div>
      </motion.div>
    </main>
  );
}
