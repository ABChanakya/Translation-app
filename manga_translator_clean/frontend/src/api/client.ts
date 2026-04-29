/**
 * Typed API client for the manga translation backend.
 * All endpoints go through Vite's dev proxy → FastAPI on :8000.
 */

import type {
  Project,
  Chapter,
  PageInfo,
  Bubble,
  AccuracyStats,
} from "./types";

const BASE = "";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

// ── Projects ──────────────────────────────────────────────────────

export const listProjects = () => json<Project[]>("/api/projects");

export const createProject = (series_name: string) =>
  json<Project>("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ series_name }),
  });

export const getProject = (series: string) =>
  json<Project>(`/api/projects/${encodeURIComponent(series)}`);

export const getProjectStats = (series: string) =>
  json<AccuracyStats>(`/api/projects/${encodeURIComponent(series)}/stats`);

export const listProjectChapters = (series: string) =>
  json<Chapter[]>(`/api/projects/${encodeURIComponent(series)}/chapters`);

// ── Chapters ──────────────────────────────────────────────────────

export const uploadChapter = (form: FormData) =>
  json<{ chapter_id: number; total_pages: number; status: string }>(
    "/api/chapters/upload",
    { method: "POST", body: form }
  );

export const getChapter = (id: number) =>
  json<Chapter>(`/api/chapters/${id}`);

export const getChapterPages = (id: number) =>
  json<PageInfo[]>(`/api/chapters/${id}/pages`);

// ── Bubbles ───────────────────────────────────────────────────────

export const getPageBubbles = (pageId: number) =>
  json<Bubble[]>(`/api/pages/${pageId}/bubbles`);

export const acceptBubble = (id: number) =>
  json<Bubble>(`/api/bubbles/${id}/accept`, { method: "POST" });

export const correctBubble = (id: number, human_translation: string) =>
  json<Bubble>(`/api/bubbles/${id}/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ human_translation }),
  });

export const skipBubble = (id: number) =>
  json<Bubble>(`/api/bubbles/${id}/skip`, { method: "POST" });

// ── Manual annotation ─────────────────────────────────────────────

export const createManualBubble = (pageId: number, data: Record<string, unknown>) =>
  json<Bubble>(`/api/pages/${pageId}/bubbles/manual`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const updatePolygon = (id: number, points: { x: number; y: number }[]) =>
  json<Bubble>(`/api/bubbles/${id}/polygon`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points }),
  });

export const updateFont = (id: number, data: Record<string, unknown>) =>
  json<Bubble>(`/api/bubbles/${id}/font`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const rerunOcr = (id: number) =>
  json<{ japanese_text: string; ocr_confidence: number }>(
    `/api/bubbles/${id}/ocr`,
    { method: "POST" }
  );

export const translateBubble = (id: number) =>
  json<{ suggested_translation: string }>(
    `/api/bubbles/${id}/translate`,
    { method: "POST" }
  );

export const applyBubble = (
  id: number,
  body?: { mode?: string; human_translation?: string }
) =>
  json<{ status: string; image_url: string; warning?: string }>(
    `/api/bubbles/${id}/apply`,
    {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    }
  );

export const deleteBubble = (id: number) =>
  json<{ status: string }>(`/api/bubbles/${id}`, { method: "DELETE" });

export const updateBubbleType = (id: number, bubble_type: string) =>
  json<Bubble>(`/api/bubbles/${id}/type`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bubble_type }),
  });

// ── Export ─────────────────────────────────────────────────────────

export const getExportUrl = (chapterId: number, format: "cbz" | "pdf") =>
  `/api/chapters/${chapterId}/export?format=${format}`;
