/**
 * Global Zustand store for the manga translation review tool.
 */

import { create } from "zustand";
import type { Project, Chapter, PageInfo, Bubble, ProcessingStatus } from "../api/types";

export type CanvasTool = "select" | "draw_rect" | "draw_polygon" | "delete";
export type ToastType = "success" | "error" | "info";

interface Toast {
  id: string;
  type: ToastType;
  message: string;
  description?: string;
}

interface ModalConfig {
  title: string;
  message: string;
  confirmLabel: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel?: () => void;
  variant?: "danger" | "primary";
}

interface AppState {
  // ── Project ─────────────────────────────────────
  projects: Project[];
  currentProject: Project | null;
  setProjects: (p: Project[]) => void;
  setCurrentProject: (p: Project | null) => void;

  // ── Chapter ─────────────────────────────────────
  currentChapter: Chapter | null;
  setCurrentChapter: (c: Chapter | null) => void;

  // ── Pages ───────────────────────────────────────
  pages: PageInfo[];
  currentPageIndex: number;
  setPages: (p: PageInfo[]) => void;
  setCurrentPageIndex: (i: number) => void;
  nextPage: () => void;
  prevPage: () => void;

  // ── Bubbles ─────────────────────────────────────
  bubbles: Bubble[];
  activeBubbleId: number | null;
  setBubbles: (b: Bubble[]) => void;
  updateBubble: (id: number, patch: Partial<Bubble>) => void;
  addBubble: (b: Bubble) => void;
  removeBubble: (id: number) => void;
  setActiveBubbleId: (id: number | null) => void;

  // ── Canvas ──────────────────────────────────────
  canvasTool: CanvasTool;
  setCanvasTool: (t: CanvasTool) => void;
  imageView: "original" | "inpainted" | "final";
  setImageView: (v: "original" | "inpainted" | "final") => void;
  /** Bumped after each apply so AnnotationCanvas busts the image cache */
  pageImageVersion: number;
  bumpPageImage: () => void;

  // ── Processing ──────────────────────────────────
  processingStatus: ProcessingStatus[];
  addProcessingStatus: (s: ProcessingStatus) => void;
  clearProcessingStatus: () => void;

  // ── Undo ────────────────────────────────────────
  undoStack: (() => void)[];
  pushUndo: (fn: () => void) => void;
  undo: () => void;

  // ── Toasts ─────────────────────────────────────
  toasts: Toast[];
  addToast: (type: ToastType, message: string, description?: string) => void;
  removeToast: (id: string) => void;

  // ── Modal ──────────────────────────────────────
  modal: ModalConfig | null;
  showModal: (config: ModalConfig) => void;
  hideModal: () => void;
}

export const useStore = create<AppState>((set, get) => ({
  // Project
  projects: [],
  currentProject: null,
  setProjects: (projects) => set({ projects }),
  setCurrentProject: (currentProject) => set({ currentProject }),

  // Chapter
  currentChapter: null,
  setCurrentChapter: (currentChapter) => set({ currentChapter }),

  // Pages
  pages: [],
  currentPageIndex: 0,
  setPages: (pages) => set({ pages, currentPageIndex: 0 }),
  setCurrentPageIndex: (i) => set({ currentPageIndex: i }),
  nextPage: () => {
    const { currentPageIndex, pages } = get();
    if (currentPageIndex < pages.length - 1) set({ currentPageIndex: currentPageIndex + 1 });
  },
  prevPage: () => {
    const { currentPageIndex } = get();
    if (currentPageIndex > 0) set({ currentPageIndex: currentPageIndex - 1 });
  },

  // Bubbles
  bubbles: [],
  activeBubbleId: null,
  setBubbles: (bubbles) => set({ bubbles }),
  updateBubble: (id, patch) =>
    set((s) => ({
      bubbles: s.bubbles.map((b) => (b.id === id ? { ...b, ...patch } : b)),
    })),
  addBubble: (b) => set((s) => ({ bubbles: [...s.bubbles, b] })),
  removeBubble: (id) =>
    set((s) => ({
      bubbles: s.bubbles.filter((b) => b.id !== id),
      activeBubbleId: s.activeBubbleId === id ? null : s.activeBubbleId,
    })),
  setActiveBubbleId: (id) => set({ activeBubbleId: id }),

  // Canvas
  canvasTool: "select",
  setCanvasTool: (canvasTool) => set({ canvasTool }),
  imageView: "inpainted",
  setImageView: (imageView) => set({ imageView }),
  pageImageVersion: 0,
  bumpPageImage: () => set((s) => ({ pageImageVersion: s.pageImageVersion + 1 })),

  // Processing
  processingStatus: [],
  addProcessingStatus: (s) =>
    set((state) => ({ processingStatus: [...state.processingStatus, s] })),
  clearProcessingStatus: () => set({ processingStatus: [] }),

  // Undo
  undoStack: [],
  pushUndo: (fn) => set((s) => ({ undoStack: [...s.undoStack.slice(-50), fn] })),
  undo: () => {
    const { undoStack } = get();
    if (undoStack.length === 0) return;
    const fn = undoStack[undoStack.length - 1];
    set({ undoStack: undoStack.slice(0, -1) });
    fn();
  },

  // Toasts
  toasts: [],
  addToast: (type, message, description) => {
    const id = Math.random().toString(36).substring(2, 9);
    set((s) => ({ toasts: [...s.toasts, { id, type, message, description }] }));
    setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    }, 3000);
  },
  removeToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  // Modal
  modal: null,
  showModal: (config) => set({ modal: config }),
  hideModal: () => set({ modal: null }),
}));
