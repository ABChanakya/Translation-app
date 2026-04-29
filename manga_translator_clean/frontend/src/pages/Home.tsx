import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useStore } from "../store/useStore";
import { listProjects, createProject } from "../api/client";
import { ChevronRight, Plus, BookOpen, ArrowRight, PlusSquare, Trash2, Filter, SortAsc, Grid, List } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { cn } from "../lib/utils";

const SkeletonCard = () => (
  <div className="bg-surface-container border border-outline-variant/5 rounded-sm overflow-hidden animate-pulse">
    <div className="h-32 bg-surface-container-highest" />
    <div className="p-6 space-y-4">
      <div className="h-6 bg-surface-container-highest w-3/4" />
      <div className="h-4 bg-surface-container-highest w-1/2" />
      <div className="space-y-2 pt-4">
        <div className="h-1.5 bg-surface-container-highest w-full" />
        <div className="h-4 bg-surface-container-highest w-1/4" />
      </div>
    </div>
  </div>
);

export default function Home() {
  const { projects, setProjects, showModal, addToast } = useStore();
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    listProjects()
      .then(setProjects)
      .catch((e) => console.error("Failed to load projects:", e))
      .finally(() => setLoading(false));
  }, [setProjects]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    const project = await createProject(newName.trim());
    setProjects([project, ...projects]);
    setShowNew(false);
    setNewName("");
    addToast("success", "Project created", `"${project.series_name}" has been added to your repository.`);
    navigate(`/projects/${encodeURIComponent(project.series_name)}`);
  };

  const handleDelete = (id: number, name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    showModal({
      title: "Delete Project",
      message: `Are you sure you want to delete "${name}"? This action cannot be undone and all chapter data will be lost.`,
      confirmLabel: "DELETE PROJECT",
      variant: "danger",
      onConfirm: () => {
        setProjects(projects.filter((p) => p.id !== id));
        addToast("success", "Project deleted", `"${name}" has been removed from your repository.`);
      },
    });
  };

  const getStatusLabel = (rate: number, chapters: number) => {
    if (chapters === 0) return "New";
    if (rate >= 0.9) return "Ready";
    if (rate >= 0.5) return "In Progress";
    return "Needs Review";
  };

  const getPhaseLabel = (rate: number) => {
    if (rate >= 0.9) return "Final Proofing";
    if (rate >= 0.5) return "Critical Phase";
    return "Draft Review";
  };

  return (
    <main className="pt-20 px-10 pb-12 min-h-screen">
      <div className="flex justify-between items-end mb-10">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <nav className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-on-surface-variant/60 font-bold mb-2">
            <span>Main Canvas</span>
            <ChevronRight size={10} />
            <span className="text-primary/60">Library</span>
          </nav>
          <h2 className="text-4xl font-black tracking-tight text-on-surface leading-none">Manga Repository</h2>
        </motion.div>
        <button
          onClick={() => setShowNew(true)}
          className="flex items-center gap-2 bg-primary text-on-primary px-5 py-2.5 rounded-sm font-bold text-sm hover:brightness-110 active:scale-[0.98] transition-all shadow-lg shadow-primary/10"
        >
          <Plus size={18} strokeWidth={3} />
          <span>New Project</span>
        </button>
      </div>

      {/* New project modal */}
      {showNew && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-surface-container rounded-sm p-6 w-96 border border-outline-variant/20">
            <h2 className="text-xl font-black mb-4">New Project</h2>
            <input
              autoFocus
              type="text"
              placeholder="Series name (e.g. Berserk)"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              className="w-full px-4 py-2.5 bg-surface-container-lowest border border-outline-variant/30 text-on-surface placeholder:text-on-surface-variant/40 focus:border-primary outline-none rounded-sm"
            />
            <div className="flex gap-3 mt-4 justify-end">
              <button
                onClick={() => setShowNew(false)}
                className="px-4 py-2 bg-surface-container-high hover:bg-surface-bright text-on-surface text-sm font-bold transition-colors rounded-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                className="gradient-cta px-4 py-2 text-on-primary text-sm font-bold rounded-sm"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filter/sort toolbar */}
      <div className="flex items-center gap-4 mb-8 text-[10px] font-black uppercase tracking-widest border-b border-outline-variant/10 pb-4">
        <div className="flex items-center gap-2 text-on-surface-variant/60">
          <Filter size={12} />
          <span>Filter by:</span>
        </div>
        <button className="px-3 py-1 bg-surface-container text-on-surface rounded-sm border border-outline-variant/20">Status: All</button>
        <div className="flex items-center gap-2 text-on-surface-variant/60 ml-4">
          <SortAsc size={12} />
          <span>Sort:</span>
        </div>
        <button className="px-3 py-1 hover:bg-surface-container text-on-surface-variant transition-colors rounded-sm">Recent</button>
        <div className="ml-auto flex items-center gap-2 text-on-surface-variant/40">
          <button aria-label="Grid View" className="p-1.5 text-primary bg-surface-container rounded-sm"><Grid size={16} /></button>
          <button aria-label="List View" className="p-1.5 hover:text-on-surface transition-colors"><List size={16} /></button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {loading ? (
          <>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </>
        ) : (
          <AnimatePresence mode="popLayout">
            {projects.map((project, index) => {
              const rate = Math.round(project.acceptance_rate * 100);
              const status = getStatusLabel(project.acceptance_rate, project.chapters_count);
              return (
                <motion.div
                  key={project.id}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ delay: index * 0.05 }}
                  onClick={() => navigate(`/projects/${encodeURIComponent(project.series_name)}`)}
                  className="group bg-surface-container border border-outline-variant/5 rounded-sm overflow-hidden hover:bg-surface-container-high transition-all duration-300 cursor-pointer active:scale-[0.99]"
                >
                  <div className="h-48 overflow-hidden relative">
                    {/* Deterministic gradient cover based on project id */}
                    <div
                      className="w-full h-full transition-transform duration-700 group-hover:scale-105"
                      style={{
                        background: `linear-gradient(${(project.id * 47) % 360}deg, hsl(${(project.id * 73) % 360},30%,12%) 0%, hsl(${(project.id * 113) % 360},40%,20%) 100%)`,
                        filter: "grayscale(0.6)",
                      }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-surface-container via-transparent to-transparent opacity-90" />
                    <div className="absolute top-4 left-4 bg-surface-container-highest/80 backdrop-blur px-2 py-1 rounded-sm text-[10px] font-black uppercase tracking-tighter">
                      {status}
                    </div>
                    <button
                      onClick={(e) => handleDelete(project.id, project.series_name, e)}
                      aria-label="Delete Project"
                      className="absolute top-4 right-4 p-2 bg-surface-container-highest/80 backdrop-blur text-on-surface-variant/60 hover:text-error hover:bg-error/10 transition-all opacity-0 group-hover:opacity-100 rounded-sm"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                  <div className="p-6">
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-xl font-bold tracking-tight text-on-surface mb-1">{project.series_name}</h3>
                        <p className="text-xs text-on-surface-variant/70">
                          Created {new Date(project.created_at).toLocaleDateString()}
                        </p>
                      </div>
                      <span className="text-[10px] font-mono font-bold text-primary bg-primary/10 px-2 py-1 rounded-sm">
                        {rate > 0 ? `${rate}% ACC` : "---"}
                      </span>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-on-surface-variant/60">
                        <span>Acceptance Rate</span>
                        <span>{getPhaseLabel(project.acceptance_rate)}</span>
                      </div>
                      <div className="w-full bg-surface-container-lowest h-1.5 rounded-full overflow-hidden" role="progressbar" aria-valuenow={rate} aria-valuemin={0} aria-valuemax={100}>
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${rate}%` }}
                          transition={{ duration: 1, ease: "easeOut" }}
                          className={cn(
                            "h-full rounded-full",
                            status === "Needs Review"
                              ? "bg-tertiary-container"
                              : "bg-gradient-to-r from-primary to-primary-container"
                          )}
                        />
                      </div>
                      <div className="flex items-center justify-between pt-2">
                        <div className="flex items-center gap-2">
                          <BookOpen size={14} className="text-on-surface-variant/60" />
                          <span className="text-xs font-medium text-on-surface-variant/80">
                            {project.chapters_count} Chapter{project.chapters_count !== 1 ? "s" : ""}
                          </span>
                        </div>
                        <ArrowRight size={18} className="text-on-surface-variant/30 group-hover:text-primary transition-colors" />
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>
        )}

        {!loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: projects.length * 0.05 }}
            onClick={() => setShowNew(true)}
            className="group border-2 border-dashed border-outline-variant/20 rounded-sm flex flex-col items-center justify-center p-12 hover:border-primary/50 transition-colors cursor-pointer bg-surface-container-lowest active:scale-[0.98]"
          >
            <div className="w-12 h-12 rounded-full bg-surface-container-high flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <PlusSquare size={24} className="text-on-surface-variant group-hover:text-primary" />
            </div>
            <p className="text-sm font-bold text-on-surface-variant group-hover:text-on-surface">Start New Project</p>
            <p className="text-[10px] text-on-surface-variant/40 mt-1 uppercase tracking-widest font-black">Upload Raw Scans</p>
          </motion.div>
        )}
      </div>

      {/* Summary stats bar */}
      {!loading && (
        <div className="mt-20 grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { label: "Total Projects", value: String(projects.length) },
            { label: "Avg Accuracy", value: projects.length ? `${Math.round(projects.reduce((a, p) => a + p.acceptance_rate, 0) / projects.length * 100)}%` : "0%", highlight: true },
            { label: "Total Chapters", value: String(projects.reduce((a, p) => a + p.chapters_count, 0)) },
            { label: "Queue Status", value: "Optimal" },
          ].map((stat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + i * 0.1 }}
              className="bg-surface p-6 rounded-sm border border-outline-variant/5"
            >
              <p className="text-[10px] uppercase font-black tracking-widest text-on-surface-variant/60 mb-1">{stat.label}</p>
              <p className={cn(
                "text-2xl font-black tracking-tighter font-mono",
                stat.highlight ? "text-primary" : "text-on-surface"
              )}>
                {stat.value}
              </p>
            </motion.div>
          ))}
        </div>
      )}
    </main>
  );
}
