/**
 * TextPlacer — shape-aware text placement for manga bubbles.
 * Mirrors the Python text_placement.py logic on the frontend
 * so canvas overlays render text inside the actual bubble shape.
 */

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type BubbleShapeType = "ellipse" | "rect" | "polygon" | "cloud" | "thought";

export class TextPlacer {
  /**
   * Returns the optimal text render bounds for a given bubble shape.
   * For ellipses this is the largest inscribed rectangle (W/√2 × H/√2).
   * For polygons, a 10%-padded bounding box.
   * For rectangles, the bbox itself (with slight padding).
   */
  static getOptimalBounds(
    bbox: Rect,
    shape: BubbleShapeType = "rect",
    padding = 0.08
  ): Rect {
    if (shape === "ellipse" || shape === "thought") {
      const factor = 1 / Math.sqrt(2);
      const w = bbox.width * factor;
      const h = bbox.height * factor;
      return {
        x: bbox.x + (bbox.width - w) / 2,
        y: bbox.y + (bbox.height - h) / 2,
        width: w,
        height: h,
      };
    }

    if (shape === "polygon" || shape === "cloud") {
      return {
        x: bbox.x + bbox.width * padding,
        y: bbox.y + bbox.height * padding,
        width: bbox.width * (1 - 2 * padding),
        height: bbox.height * (1 - 2 * padding),
      };
    }

    // rect — small uniform padding
    return {
      x: bbox.x + bbox.width * padding,
      y: bbox.y + bbox.height * padding,
      width: bbox.width * (1 - 2 * padding),
      height: bbox.height * (1 - 2 * padding),
    };
  }

  /**
   * Word-wraps text to fit within maxWidth pixels given fontSize.
   * Uses a rough 0.6× character-width estimate (suitable for proportional fonts).
   */
  static wrapText(text: string, maxWidth: number, fontSize: number): string[] {
    const words = text.split(" ");
    const lines: string[] = [];
    let current = words[0] ?? "";

    for (let i = 1; i < words.length; i++) {
      const candidate = current + " " + words[i];
      const estimatedWidth = candidate.length * fontSize * 0.6;
      if (estimatedWidth <= maxWidth) {
        current = candidate;
      } else {
        lines.push(current);
        current = words[i];
      }
    }
    if (current) lines.push(current);
    return lines;
  }

  /**
   * Determines the largest font size that fits `lines` of text into `bounds`.
   */
  static fitFontSize(
    lines: string[],
    bounds: Rect,
    maxSize = 16,
    minSize = 8
  ): number {
    for (let size = maxSize; size >= minSize; size--) {
      const totalHeight = lines.length * size * 1.3;
      const maxLineWidth = Math.max(...lines.map((l) => l.length)) * size * 0.6;
      if (totalHeight <= bounds.height && maxLineWidth <= bounds.width) {
        return size;
      }
    }
    return minSize;
  }
}
