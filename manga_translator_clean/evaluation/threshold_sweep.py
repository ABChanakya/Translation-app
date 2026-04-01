"""Grid-search confidence/NMS thresholds with fairness-first reporting fields."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.detection_metrics import DEFAULT_MATCH_IOU_THRESHOLD, evaluate_detection_model  # noqa: E402
from training.common import ensure_directory  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep confidence and NMS thresholds for a YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument(
        "--data",
        type=str,
        default="training/datasets/custom_manga.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument("--conf-thrs", type=parse_float_list, default=[0.1, 0.2, 0.3, 0.4, 0.5], help="Comma-separated confidence thresholds")
    parser.add_argument("--iou-thrs", type=parse_float_list, default=[0.3, 0.5, 0.7], help="Comma-separated NMS IoU thresholds")
    parser.add_argument(
        "--match-iou",
        type=float,
        default=DEFAULT_MATCH_IOU_THRESHOLD,
        help="IoU threshold used for matched box IoU analysis",
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for prediction/validation")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Directory to store sweep CSV")
    args = parser.parse_args()

    rows = []
    for confidence in args.conf_thrs:
        for iou_threshold in args.iou_thrs:
            summary = evaluate_detection_model(
                args.model,
                args.data,
                split=args.split,
                conf=confidence,
                iou=iou_threshold,
                match_iou_threshold=args.match_iou,
                batch=args.batch,
                imgsz=args.imgsz,
                plots=False,
            )
            rows.append(summary)

    rows.sort(key=lambda item: item["macro_f1"], reverse=True)
    top_rows = rows[:5]

    print("\nTop 5 threshold settings by F1:")
    print("=" * 96)
    print(f"{'conf':<8} {'iou':<8} {'macro_p':<12} {'macro_r':<12} {'macro_f1':<12} {'mAP50':<12} {'macro_iou':<12}")
    print("-" * 96)
    for row in top_rows:
        print(
            f"{row['confidence']:<8.2f} "
            f"{row['nms_iou']:<8.2f} "
            f"{row['macro_precision']:<12.4f} "
            f"{row['macro_recall']:<12.4f} "
            f"{row['macro_f1']:<12.4f} "
            f"{row['mAP50']:<12.4f} "
            f"{row['macro_iou']:<12.4f}"
        )

    best = rows[0]
    print(
        "\nBest combo -> "
        f"conf={best['confidence']:.2f}, iou={best['nms_iou']:.2f}, "
        f"macro F1={best['macro_f1']:.3f}, mAP50={best['mAP50']:.3f}, mAP50-95={best['mAP50-95']:.3f}, "
        f"macro IoU={best['macro_iou']:.3f}"
    )

    output_dir = ensure_directory(args.output_dir)
    output_path = output_dir / f"threshold_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "confidence",
        "nms_iou",
        "precision",
        "recall",
        "f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "mAP50",
        "mAP50-95",
        "mIoU",
        "macro_iou",
        "matched_iou",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

    print(f"Saved sweep CSV: {output_path}")


if __name__ == "__main__":
    main()
