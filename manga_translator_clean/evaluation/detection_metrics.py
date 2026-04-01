from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from evaluation.reporting import write_report_artifacts
from training.common import (
    DEFAULT_RESULTS_DIR,
    format_metric,
    iter_image_files,
    load_yaml,
    normalize_class_names,
    prepare_ultralytics_dataset_yaml,
    resolve_artifact_reference,
    resolve_project_path,
    resolve_split_image_dir,
    resolve_split_label_dir,
)


DEFAULT_MATCH_IOU_THRESHOLD = 0.6


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """IoU for two axis-aligned ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions_to_ground_truth(
    predictions: list[tuple[tuple[float, float, float, float], float]],
    ground_truths: list[tuple[float, float, float, float]],
    match_iou_threshold: float,
) -> list[float]:
    """Greedily match predictions to ground-truth boxes in confidence order."""
    if not predictions or not ground_truths:
        return []

    used_ground_truth: set[int] = set()
    matched_ious: list[float] = []

    for predicted_box, _score in sorted(predictions, key=lambda item: item[1], reverse=True):
        best_index = None
        best_iou = 0.0
        for ground_truth_index, ground_truth_box in enumerate(ground_truths):
            if ground_truth_index in used_ground_truth:
                continue
            iou = box_iou(predicted_box, ground_truth_box)
            if iou > best_iou:
                best_iou = iou
                best_index = ground_truth_index

        if best_index is not None and best_iou >= match_iou_threshold:
            used_ground_truth.add(best_index)
            matched_ious.append(best_iou)

    return matched_ious


def load_ground_truth_boxes(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Load YOLO-normalized boxes and convert them to absolute ``xyxy`` coordinates."""
    boxes_by_class: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    if not label_path.exists():
        return boxes_by_class

    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, box_width, box_height = map(float, parts[1:5])
            x1 = (x_center - box_width / 2.0) * image_width
            y1 = (y_center - box_height / 2.0) * image_height
            x2 = (x_center + box_width / 2.0) * image_width
            y2 = (y_center + box_height / 2.0) * image_height
            boxes_by_class[class_id].append((x1, y1, x2, y2))

    return boxes_by_class


def extract_prediction_boxes(result: Any) -> dict[int, list[tuple[tuple[float, float, float, float], float]]]:
    boxes_by_class: dict[int, list[tuple[tuple[float, float, float, float], float]]] = defaultdict(list)
    if result.boxes is None or len(result.boxes) == 0:
        return boxes_by_class

    for box, score, class_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        boxes_by_class[int(class_id)].append((tuple(float(value) for value in box), float(score)))

    return boxes_by_class


def _flatten_class_values(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=int)
    if isinstance(values, list):
        arrays = [np.asarray(item).reshape(-1) for item in values if item is not None and np.asarray(item).size > 0]
        return np.concatenate(arrays).astype(int) if arrays else np.asarray([], dtype=int)
    return np.asarray(values).reshape(-1).astype(int)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _weighted_mean(rows: list[dict[str, Any]], metric_key: str) -> float:
    total_support = sum(row["support"] for row in rows)
    if total_support == 0:
        return 0.0
    return float(sum(row["support"] * row[metric_key] for row in rows) / total_support)


def compute_mean_iou(
    model: Any,
    data_yaml: str | Path,
    *,
    split: str = "val",
    conf: float = 0.001,
    nms_iou: float = 0.6,
    match_iou_threshold: float = DEFAULT_MATCH_IOU_THRESHOLD,
    device: str | None = None,
    max_det: int = 300,
    batch: int = 8,
    imgsz: int = 640,
) -> dict[str, Any]:
    """Compute a box-level IoU summary over a dataset split."""
    image_dir = resolve_split_image_dir(data_yaml, split)
    label_dir = resolve_split_label_dir(image_dir)
    data_config = load_yaml(data_yaml)
    class_names = normalize_class_names(data_config.get("names"))

    image_paths = iter_image_files(image_dir)
    if not image_paths:
        return {
            "macro_iou": 0.0,
            "matched_iou": 0.0,
            "matched_detections": 0,
            "ground_truth_boxes": 0,
            "per_class_iou": {},
            "per_class_matches": {},
            "per_class_ground_truth": {},
        }

    total_ground_truth_boxes = 0
    total_matched_boxes = 0
    all_matched_ious: list[float] = []
    per_class_ground_truth: dict[int, int] = defaultdict(int)
    per_class_match_count: dict[int, int] = defaultdict(int)
    per_class_iou_sum: dict[int, float] = defaultdict(float)

    batch_paths = [str(path) for path in image_paths]
    prediction_stream = model.predict(
        source=batch_paths,
        conf=conf,
        iou=nms_iou,
        device=device,
        max_det=max_det,
        imgsz=imgsz,
        batch=batch,
        verbose=False,
        stream=True,
    )

    for result in prediction_stream:
        image_path = Path(result.path)
        image_height, image_width = result.orig_shape
        label_path = label_dir / f"{image_path.stem}.txt"

        ground_truth_boxes = load_ground_truth_boxes(label_path, image_width, image_height)
        prediction_boxes = extract_prediction_boxes(result)

        class_ids = set(ground_truth_boxes) | set(prediction_boxes)
        for class_id in class_ids:
            class_ground_truths = ground_truth_boxes.get(class_id, [])
            class_predictions = prediction_boxes.get(class_id, [])

            per_class_ground_truth[class_id] += len(class_ground_truths)
            total_ground_truth_boxes += len(class_ground_truths)

            matched_ious = match_predictions_to_ground_truth(
                class_predictions,
                class_ground_truths,
                match_iou_threshold=match_iou_threshold,
            )
            per_class_iou_sum[class_id] += sum(matched_ious)
            per_class_match_count[class_id] += len(matched_ious)
            total_matched_boxes += len(matched_ious)
            all_matched_ious.extend(matched_ious)

    per_class_iou: dict[str, float] = {}
    per_class_matches: dict[str, int] = {}
    per_class_support: dict[str, int] = {}
    class_scores: list[float] = []

    for class_id, ground_truth_count in sorted(per_class_ground_truth.items()):
        if ground_truth_count == 0:
            continue
        class_name = class_names.get(class_id, f"class_{class_id}")
        class_score = per_class_iou_sum[class_id] / ground_truth_count
        per_class_iou[class_name] = float(class_score)
        per_class_matches[class_name] = int(per_class_match_count[class_id])
        per_class_support[class_name] = int(ground_truth_count)
        class_scores.append(class_score)

    return {
        "macro_iou": _mean(class_scores),
        "matched_iou": _mean(all_matched_ious),
        "matched_detections": total_matched_boxes,
        "ground_truth_boxes": total_ground_truth_boxes,
        "per_class_iou": per_class_iou,
        "per_class_matches": per_class_matches,
        "per_class_ground_truth": per_class_support,
    }


def _build_per_class_rows(
    validation_results: Any,
    class_names: Mapping[int, str],
    iou_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    box_metrics = validation_results.box
    ap_class_index = list(getattr(validation_results, "ap_class_index", []))
    precision_by_ap_index = np.asarray(getattr(box_metrics, "p", []), dtype=float)
    recall_by_ap_index = np.asarray(getattr(box_metrics, "r", []), dtype=float)
    f1_by_ap_index = np.asarray(getattr(box_metrics, "f1", []), dtype=float)
    raw_support_counts = getattr(validation_results, "nt_per_class", None)
    support_counts = np.asarray(raw_support_counts if raw_support_counts is not None else [], dtype=int)
    prediction_counts = np.bincount(
        _flatten_class_values(getattr(validation_results, "stats", {}).get("pred_cls")),
        minlength=max(len(class_names), len(support_counts) if support_counts.size else 0),
    )

    index_lookup = {class_id: index for index, class_id in enumerate(ap_class_index)}
    all_class_ids = sorted(set(class_names) | set(index_lookup) | set(range(len(support_counts))) | set(range(len(prediction_counts))))

    per_class_rows: list[dict[str, Any]] = []
    for class_id in all_class_ids:
        class_name = class_names.get(class_id, f"class_{class_id}")
        summary_index = index_lookup.get(class_id)
        support = int(support_counts[class_id]) if class_id < len(support_counts) else 0
        prediction_count = int(prediction_counts[class_id]) if class_id < len(prediction_counts) else 0

        if summary_index is not None:
            precision, recall, ap50, ap50_95 = box_metrics.class_result(summary_index)
            f1 = float(f1_by_ap_index[summary_index]) if summary_index < len(f1_by_ap_index) else 0.0
        else:
            precision = recall = f1 = ap50 = ap50_95 = 0.0

        per_class_rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "support": support,
                "prediction_count": prediction_count,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "ap50": float(ap50),
                "ap50_95": float(ap50_95),
                "matched_iou": float(iou_summary["per_class_iou"].get(class_name, 0.0)),
                "matched_boxes": int(iou_summary["per_class_matches"].get(class_name, 0)),
            }
        )

    return per_class_rows


def _detect_training_run_dir(model_reference: Any) -> str | None:
    candidate = Path(str(getattr(model_reference, "ckpt_path", model_reference)))
    if not candidate.is_absolute():
        candidate = Path(resolve_artifact_reference(candidate))
    if candidate.name in {"best.pt", "last.pt"} and candidate.parent.name == "weights":
        return str(candidate.parent.parent)
    return None


def evaluate_detection_model(
    model_or_path: Any,
    data_yaml: str | Path,
    *,
    split: str = "val",
    conf: float = 0.001,
    iou: float = 0.6,
    match_iou_threshold: float = DEFAULT_MATCH_IOU_THRESHOLD,
    device: str | None = None,
    max_det: int = 300,
    plots: bool = False,
    batch: int = 8,
    imgsz: int = 640,
    save_dir: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Run Ultralytics validation and augment it with class-fair detection reporting."""
    from ultralytics import YOLO

    resolved_data_yaml = resolve_project_path(data_yaml)
    prepared_data_yaml = prepare_ultralytics_dataset_yaml(resolved_data_yaml)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model_or_path if hasattr(model_or_path, "predict") else YOLO(resolve_artifact_reference(model_or_path))

    validation_results = model.val(
        data=str(prepared_data_yaml),
        split=split,
        device=resolved_device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        batch=batch,
        imgsz=imgsz,
        plots=plots,
        verbose=False,
    )

    iou_summary = compute_mean_iou(
        model,
        resolved_data_yaml,
        split=split,
        conf=conf,
        nms_iou=iou,
        match_iou_threshold=match_iou_threshold,
        device=resolved_device,
        max_det=max_det,
        batch=batch,
        imgsz=imgsz,
    )

    dataset_config = load_yaml(resolved_data_yaml)
    class_names = normalize_class_names(dataset_config.get("names"))
    per_class_rows = _build_per_class_rows(validation_results, class_names, iou_summary)
    classes_with_support = [row for row in per_class_rows if row["support"] > 0]

    precision = float(validation_results.box.mp)
    recall = float(validation_results.box.mr)
    aggregate_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall + 1e-9)),
        "mAP50": float(validation_results.box.map50),
        "mAP50_95": float(validation_results.box.map),
        "macro_precision": _mean(row["precision"] for row in classes_with_support),
        "macro_recall": _mean(row["recall"] for row in classes_with_support),
        "macro_f1": _mean(row["f1"] for row in classes_with_support),
        "weighted_precision": _weighted_mean(classes_with_support, "precision"),
        "weighted_recall": _weighted_mean(classes_with_support, "recall"),
        "weighted_f1": _weighted_mean(classes_with_support, "f1"),
        "macro_iou": float(iou_summary["macro_iou"]),
        "matched_iou": float(iou_summary["matched_iou"]),
        "matched_detections": int(iou_summary["matched_detections"]),
        "ground_truth_boxes": int(iou_summary["ground_truth_boxes"]),
        "support_total": int(sum(row["support"] for row in per_class_rows)),
        "prediction_total": int(sum(row["prediction_count"] for row in per_class_rows)),
        "classes_with_support": int(len(classes_with_support)),
    }

    weakest_classes = sorted(
        classes_with_support,
        key=lambda row: (row["recall"], row["f1"], row["matched_iou"], row["support"]),
    )[:3]
    strongest_classes = sorted(
        classes_with_support,
        key=lambda row: (row["f1"], row["matched_iou"], row["ap50_95"], row["support"]),
        reverse=True,
    )[:3]

    summary = {
        "model": str(getattr(model, "ckpt_path", model_or_path)),
        "data": str(resolved_data_yaml),
        "resolved_data_yaml": str(prepared_data_yaml),
        "split": split,
        "confidence": conf,
        "nms_iou": iou,
        "match_iou_threshold": match_iou_threshold,
        "aggregate_metrics": aggregate_metrics,
        "per_class": per_class_rows,
        "weakest_classes": weakest_classes,
        "strongest_classes": strongest_classes,
        "box_iou_label": "Box-level matched IoU for detection analysis",
        "timestamp": datetime.now().isoformat(),
        # Compatibility fields for existing scripts/UI.
        "precision": aggregate_metrics["precision"],
        "recall": aggregate_metrics["recall"],
        "f1": aggregate_metrics["f1"],
        "mAP50": aggregate_metrics["mAP50"],
        "mAP50-95": aggregate_metrics["mAP50_95"],
        "mIoU": aggregate_metrics["macro_iou"],
        "matched_iou": aggregate_metrics["matched_iou"],
        "matched_detections": aggregate_metrics["matched_detections"],
        "ground_truth_boxes": aggregate_metrics["ground_truth_boxes"],
        "macro_precision": aggregate_metrics["macro_precision"],
        "macro_recall": aggregate_metrics["macro_recall"],
        "macro_f1": aggregate_metrics["macro_f1"],
        "weighted_precision": aggregate_metrics["weighted_precision"],
        "weighted_recall": aggregate_metrics["weighted_recall"],
        "weighted_f1": aggregate_metrics["weighted_f1"],
        "macro_iou": aggregate_metrics["macro_iou"],
        "per_class_iou": {row["class_name"]: row["matched_iou"] for row in per_class_rows},
        "per_class_map50-95": {row["class_name"]: row["ap50_95"] for row in per_class_rows},
    }

    if save_dir:
        output_dir = resolve_project_path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = run_name or f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_artifacts = write_report_artifacts(
            summary,
            save_root=output_dir,
            run_name=stem,
            training_run_dir=_detect_training_run_dir(summary["model"]),
        )
        summary.update(report_artifacts)
        summary["saved_to"] = report_artifacts["summary_json"]

        # Keep the written summary in sync with report paths.
        Path(summary["saved_to"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


def print_evaluation_summary(summary: Mapping[str, Any]) -> None:
    """Consistent console summary for training and evaluation scripts."""
    aggregate = summary.get("aggregate_metrics", {})
    print("\n" + "=" * 72)
    print("Evaluation Summary")
    print("=" * 72)
    print(f"Model: {summary['model']}")
    print(f"Dataset: {summary['data']} [{summary['split']}]")
    print(f"Precision:        {format_metric(summary.get('precision'))}")
    print(f"Recall:           {format_metric(summary.get('recall'))}")
    print(f"Overall F1:       {format_metric(summary.get('f1'))}")
    print(f"mAP@0.5:          {format_metric(summary.get('mAP50'))}")
    print(f"mAP@0.5:0.95:     {format_metric(summary.get('mAP50-95'))}")
    print(f"Macro Precision:  {format_metric(summary.get('macro_precision'))}")
    print(f"Macro Recall:     {format_metric(summary.get('macro_recall'))}")
    print(f"Macro F1:         {format_metric(summary.get('macro_f1'))}")
    print(f"Weighted F1:      {format_metric(summary.get('weighted_f1'))}")
    print(f"Macro IoU (box):  {format_metric(summary.get('macro_iou'))}")
    print(f"Matched IoU:      {format_metric(summary.get('matched_iou'))}")
    print(
        f"Support / preds:  {aggregate.get('support_total', 0)} gt boxes, "
        f"{aggregate.get('prediction_total', 0)} predictions"
    )
    weakest_classes = summary.get("weakest_classes", [])
    if weakest_classes:
        print("\nWeakest classes:")
        for row in weakest_classes:
            print(
                f"  - {row['class_name']}: support={row['support']}, recall={row['recall']:.4f}, "
                f"F1={row['f1']:.4f}, IoU={row['matched_iou']:.4f}"
            )
    if summary.get("report_html"):
        print(f"\nReport HTML: {summary['report_html']}")
    if summary.get("metrics_json"):
        print(f"Metrics JSON: {summary['metrics_json']}")
    print("=" * 72)
