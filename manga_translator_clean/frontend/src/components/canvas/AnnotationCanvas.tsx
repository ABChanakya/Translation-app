/**
 * Annotation canvas — Konva-based interactive overlay on the manga page.
 *
 * Features:
 *   - Renders all detected bubbles as colored polygon/rect overlays
 *   - Select tool: click to select, drag to move, drag vertices to reshape
 *   - Draw Rect tool: click+drag to create a new rectangular region
 *   - Draw Polygon tool: click to place vertices, double-click to close
 *   - Delete tool: click a region to delete it
 *   - Edge midpoint handles to insert new vertices
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { Stage, Layer, Rect, Line, Circle, Image as KImage } from "react-konva";
import useImage from "use-image";
import { useStore } from "../../store/useStore";
import { createManualBubble, updatePolygon, deleteBubble } from "../../api/client";
import type { Bubble } from "../../api/types";

interface Props {
  pageId: number;
}

const MODE_COLORS: Record<string, string> = {
  translate_and_inpaint: "#3b82f6",
  inpaint_only: "#ef4444",
  manual_text: "#22c55e",
  review_later: "#f59e0b",
};

/** Convert a bubble bbox to flat polygon points [x1,y1, x2,y1, x2,y2, x1,y2] */
function bboxToPoints(b: Bubble): number[] {
  if (b.mask_polygon) {
    try {
      const pts: { x: number; y: number }[] = JSON.parse(b.mask_polygon);
      return pts.flatMap((p) => [p.x, p.y]);
    } catch { /* fall through */ }
  }
  return [b.x1, b.y1, b.x2, b.y1, b.x2, b.y2, b.x1, b.y2];
}

/** Flat points array → {x,y}[] */
function flatToXY(pts: number[]): { x: number; y: number }[] {
  const result: { x: number; y: number }[] = [];
  for (let i = 0; i < pts.length; i += 2) {
    result.push({ x: pts[i], y: pts[i + 1] });
  }
  return result;
}

export default function AnnotationCanvas({ pageId }: Props) {
  const {
    bubbles, activeBubbleId, setActiveBubbleId,
    canvasTool, addBubble, removeBubble, updateBubble,
    imageView, pushUndo, pageImageVersion,
  } = useStore();

  // Page image — ?v= busts browser cache after each apply
  const imgSrc =
    imageView === "original"
      ? `/api/images/${pageId}/original`
      : imageView === "final"
      ? `/api/images/${pageId}/final?v=${pageImageVersion}`
      : `/api/images/${pageId}/inpainted?v=${pageImageVersion}`;
  const [image] = useImage(imgSrc);
  const stageRef = useRef<any>(null);

  // Canvas dimensions — fit to the image
  const imgW = image?.naturalWidth ?? 800;
  const imgH = image?.naturalHeight ?? 1200;
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    if (!containerRef.current || !image) return;
    const cw = containerRef.current.clientWidth - 16;
    setScale(Math.min(1, cw / imgW));
  }, [image, imgW]);

  // ── Drawing state ──────────────────────────────────
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawEnd, setDrawEnd] = useState<{ x: number; y: number } | null>(null);
  const [polyPoints, setPolyPoints] = useState<number[]>([]);
  const [draggingVertex, setDraggingVertex] = useState<{
    bubbleId: number;
    vertexIndex: number;
  } | null>(null);

  // ── Get pointer position in image coords ───────────
  const getPointerPos = useCallback(() => {
    const stage = stageRef.current;
    if (!stage) return { x: 0, y: 0 };
    const pos = stage.getPointerPosition();
    return { x: Math.round(pos.x / scale), y: Math.round(pos.y / scale) };
  }, [scale]);

  // ── Stage mouse handlers ───────────────────────────
  const handleStageMouseDown = useCallback(() => {
    const pos = getPointerPos();

    if (canvasTool === "draw_rect") {
      setDrawStart(pos);
      setDrawEnd(pos);
      return;
    }

    if (canvasTool === "draw_polygon") {
      setPolyPoints((prev) => [...prev, pos.x, pos.y]);
      return;
    }
  }, [canvasTool, getPointerPos]);

  const handleStageMouseMove = useCallback(() => {
    if (canvasTool === "draw_rect" && drawStart) {
      setDrawEnd(getPointerPos());
    }
    if (draggingVertex) {
      const pos = getPointerPos();
      const b = bubbles.find((bb) => bb.id === draggingVertex.bubbleId);
      if (!b) return;
      const pts = bboxToPoints(b);
      const newPts = [...pts];
      newPts[draggingVertex.vertexIndex * 2] = pos.x;
      newPts[draggingVertex.vertexIndex * 2 + 1] = pos.y;
      updateBubble(b.id, { mask_polygon: JSON.stringify(flatToXY(newPts)) });
    }
  }, [canvasTool, drawStart, draggingVertex, bubbles, getPointerPos, updateBubble]);

  const handleStageMouseUp = useCallback(async () => {
    if (draggingVertex) {
      const b = bubbles.find((bb) => bb.id === draggingVertex.bubbleId);
      if (b && b.mask_polygon) {
        try {
          await updatePolygon(b.id, JSON.parse(b.mask_polygon));
        } catch { /* ignore */ }
      }
      setDraggingVertex(null);
      return;
    }

    if (canvasTool === "draw_rect" && drawStart && drawEnd) {
      const x1 = Math.min(drawStart.x, drawEnd.x);
      const y1 = Math.min(drawStart.y, drawEnd.y);
      const x2 = Math.max(drawStart.x, drawEnd.x);
      const y2 = Math.max(drawStart.y, drawEnd.y);

      if (x2 - x1 > 10 && y2 - y1 > 10) {
        const polygon = [
          { x: x1, y: y1 }, { x: x2, y: y1 },
          { x: x2, y: y2 }, { x: x1, y: y2 },
        ];
        try {
          const bubble = await createManualBubble(pageId, {
            x1, y1, x2, y2,
            polygon,
            mode: "manual_text",
          });
          addBubble(bubble);
          setActiveBubbleId(bubble.id);
          pushUndo(() => {
            deleteBubble(bubble.id);
            removeBubble(bubble.id);
          });
        } catch (e) {
          console.error("Failed to create manual bubble:", e);
        }
      }
      setDrawStart(null);
      setDrawEnd(null);
    }
  }, [canvasTool, drawStart, drawEnd, pageId, addBubble, setActiveBubbleId, pushUndo, removeBubble, draggingVertex, bubbles, updateBubble]);

  const handleStageDblClick = useCallback(async () => {
    if (canvasTool === "draw_polygon" && polyPoints.length >= 6) {
      const xyPts = flatToXY(polyPoints);
      const xs = xyPts.map((p) => p.x);
      const ys = xyPts.map((p) => p.y);
      const x1 = Math.min(...xs);
      const y1 = Math.min(...ys);
      const x2 = Math.max(...xs);
      const y2 = Math.max(...ys);

      try {
        const bubble = await createManualBubble(pageId, {
          x1, y1, x2, y2,
          polygon: xyPts,
          mode: "manual_text",
        });
        addBubble(bubble);
        setActiveBubbleId(bubble.id);
        pushUndo(() => {
          deleteBubble(bubble.id);
          removeBubble(bubble.id);
        });
      } catch (e) {
        console.error("Failed to create polygon bubble:", e);
      }
      setPolyPoints([]);
    }
  }, [canvasTool, polyPoints, pageId, addBubble, setActiveBubbleId, pushUndo, removeBubble]);

  // ── Click on a bubble overlay ──────────────────────
  const handleBubbleClick = useCallback(async (b: Bubble) => {
    if (canvasTool === "delete") {
      if (!confirm("Delete this region?")) return;
      await deleteBubble(b.id);
      removeBubble(b.id);
      return;
    }
    setActiveBubbleId(b.id);
  }, [canvasTool, setActiveBubbleId, removeBubble]);

  // ── Vertex drag start ──────────────────────────────
  const handleVertexMouseDown = useCallback((bubbleId: number, vertexIndex: number) => {
    if (canvasTool === "select") {
      setDraggingVertex({ bubbleId, vertexIndex });
    }
  }, [canvasTool]);

  // ── Insert vertex on edge double-click ─────────────
  const handleEdgeDoubleClick = useCallback(async (b: Bubble, edgeIndex: number) => {
    const pts = flatToXY(bboxToPoints(b));
    const p1 = pts[edgeIndex];
    const p2 = pts[(edgeIndex + 1) % pts.length];
    const mid = { x: Math.round((p1.x + p2.x) / 2), y: Math.round((p1.y + p2.y) / 2) };
    const newPts = [...pts];
    newPts.splice(edgeIndex + 1, 0, mid);
    updateBubble(b.id, { mask_polygon: JSON.stringify(newPts) });
    try {
      await updatePolygon(b.id, newPts);
    } catch { /* ignore */ }
  }, [updateBubble]);

  return (
    <div ref={containerRef} className="w-full">
      <Stage
        ref={stageRef}
        width={imgW * scale}
        height={imgH * scale}
        scaleX={scale}
        scaleY={scale}
        onMouseDown={handleStageMouseDown}
        onMouseMove={handleStageMouseMove}
        onMouseUp={handleStageMouseUp}
        onDblClick={handleStageDblClick}
        style={{ cursor: canvasTool === "draw_rect" || canvasTool === "draw_polygon" ? "crosshair" : canvasTool === "delete" ? "not-allowed" : "default" }}
      >
        {/* Base image layer */}
        <Layer>
          {image && <KImage image={image} x={0} y={0} width={imgW} height={imgH} />}
        </Layer>

        {/* Bubble overlays */}
        <Layer>
          {bubbles.map((b) => {
            const pts = bboxToPoints(b);
            const isActive = b.id === activeBubbleId;
            const color = MODE_COLORS[b.mode] ?? "#3b82f6";

            return (
              <Line
                key={`poly-${b.id}`}
                points={pts}
                closed
                fill={isActive ? `${color}22` : `${color}11`}
                stroke={color}
                strokeWidth={isActive ? 2.5 : 1.5}
                dash={b.is_manual ? [6, 3] : undefined}
                onClick={() => handleBubbleClick(b)}
                onTap={() => handleBubbleClick(b)}
              />
            );
          })}

          {/* Vertex handles for active bubble */}
          {activeBubbleId &&
            canvasTool === "select" &&
            (() => {
              const b = bubbles.find((bb) => bb.id === activeBubbleId);
              if (!b) return null;
              const pts = flatToXY(bboxToPoints(b));

              return (
                <>
                  {pts.map((p, i) => (
                    <Rect
                      key={`vertex-${b.id}-${i}`}
                      x={p.x - 4}
                      y={p.y - 4}
                      width={8}
                      height={8}
                      fill="white"
                      stroke="#3b82f6"
                      strokeWidth={1.5}
                      onMouseDown={() => handleVertexMouseDown(b.id, i)}
                      onDblClick={() => {
                        // Delete vertex (min 3)
                        if (pts.length > 3) {
                          const newPts = pts.filter((_, idx) => idx !== i);
                          updateBubble(b.id, { mask_polygon: JSON.stringify(newPts) });
                          updatePolygon(b.id, newPts);
                        }
                      }}
                      style={{ cursor: "grab" }}
                    />
                  ))}
                  {/* Edge midpoint handles */}
                  {pts.map((p, i) => {
                    const next = pts[(i + 1) % pts.length];
                    const mx = (p.x + next.x) / 2;
                    const my = (p.y + next.y) / 2;
                    return (
                      <Circle
                        key={`mid-${b.id}-${i}`}
                        x={mx}
                        y={my}
                        radius={3}
                        fill="rgba(59,130,246,0.4)"
                        stroke="#3b82f6"
                        strokeWidth={1}
                        onDblClick={() => handleEdgeDoubleClick(b, i)}
                        style={{ cursor: "copy" }}
                      />
                    );
                  })}
                </>
              );
            })()}

          {/* Draw rect preview */}
          {canvasTool === "draw_rect" && drawStart && drawEnd && (
            <Rect
              x={Math.min(drawStart.x, drawEnd.x)}
              y={Math.min(drawStart.y, drawEnd.y)}
              width={Math.abs(drawEnd.x - drawStart.x)}
              height={Math.abs(drawEnd.y - drawStart.y)}
              fill="rgba(59,130,246,0.15)"
              stroke="#3b82f6"
              strokeWidth={2}
              dash={[6, 3]}
            />
          )}

          {/* Draw polygon preview */}
          {canvasTool === "draw_polygon" && polyPoints.length >= 2 && (
            <Line
              points={polyPoints}
              stroke="#22c55e"
              strokeWidth={2}
              dash={[6, 3]}
            />
          )}
          {canvasTool === "draw_polygon" &&
            polyPoints.length >= 2 &&
            flatToXY(polyPoints).map((p, i) => (
              <Circle
                key={`ppv-${i}`}
                x={p.x}
                y={p.y}
                radius={4}
                fill="#22c55e"
              />
            ))}
        </Layer>
      </Stage>
    </div>
  );
}
