/** Shared API types — mirror the FastAPI Pydantic models */

export interface Project {
  id: number;
  series_name: string;
  created_at: string;
  profile_path: string | null;
  chapters_count: number;
  acceptance_rate: number;
}

export interface Chapter {
  id: number;
  project_id: number;
  chapter_num: number;
  status: "processing" | "ready" | "complete";
  total_pages: number;
  total_bubbles: number;
  reviewed_bubbles: number;
  accepted_bubbles: number;
  created_at: string | null;
}

export interface PageInfo {
  id: number;
  page_num: number;
  status: string;
  bubble_count: number;
  original_image_url: string | null;
  inpainted_image_url: string | null;
}

export type BubbleStatus = "pending" | "accepted" | "corrected" | "skipped";
export type BubbleMode =
  | "translate_and_inpaint"
  | "inpaint_only"
  | "manual_text"
  | "review_later";

export interface Bubble {
  id: number;
  bubble_index: number;
  bubble_type: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  mask_points: string | null;
  japanese_text: string | null;
  suggested_translation: string | null;
  human_translation: string | null;
  status: BubbleStatus;
  ocr_confidence: number | null;
  quality_score: number | null;
  edit_distance: number | null;
  notes: string | null;
  is_manual: boolean;
  mode: BubbleMode;
  mask_polygon: string | null;
  font_family: string | null;
  font_size: number | null;
  font_color: string | null;
  stroke_color: string | null;
  stroke_width: number | null;
  text_align: string | null;
}

export interface AccuracyStats {
  total_reviewed: number;
  acceptance_rate: number;
  most_corrected_terms: { japanese: string; count: number }[];
  improvement_by_chapter: Record<number, number>;
}

export interface ProcessingStatus {
  page: number;
  stage: string;
  message: string;
  ts: string;
}
