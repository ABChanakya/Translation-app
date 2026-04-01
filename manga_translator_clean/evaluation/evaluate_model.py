"""Evaluate one or more YOLO models with fairness-first detection reporting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.detection_metrics import DEFAULT_MATCH_IOU_THRESHOLD, evaluate_detection_model, print_evaluation_summary  # noqa: E402


def evaluate_single_model(
    model_path: str,
    data_yaml: str,
    *,
    save_dir: str,
    split: str,
    conf: float,
    iou: float,
    match_iou: float,
    batch: int,
    imgsz: int,
    plots: bool,
) -> dict:
    summary = evaluate_detection_model(
        model_path,
        data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        match_iou_threshold=match_iou,
        batch=batch,
        imgsz=imgsz,
        plots=plots,
        save_dir=save_dir,
        run_name=Path(model_path).stem,
    )
    print_evaluation_summary(summary)
    return summary


def compare_models(
    model_paths: list[str],
    data_yaml: str,
    *,
    split: str,
    conf: float,
    iou: float,
    match_iou: float,
    batch: int,
    imgsz: int,
) -> list[dict]:
    summaries = [
        evaluate_detection_model(
            model_path,
            data_yaml,
            split=split,
            conf=conf,
            iou=iou,
            match_iou_threshold=match_iou,
            batch=batch,
            imgsz=imgsz,
            plots=False,
        )
        for model_path in model_paths
    ]

    print("\n" + "=" * 104)
    print(
        f"{'Model':<35} {'mAP50':>8} {'mAP50-95':>10} {'MacroF1':>10} "
        f"{'MacroR':>8} {'MacroIoU':>10}"
    )
    print("-" * 104)
    for summary in sorted(summaries, key=lambda item: item["mAP50-95"], reverse=True):
        print(
            f"{Path(summary['model']).name:<35} "
            f"{summary['mAP50']:>8.4f} "
            f"{summary['mAP50-95']:>10.4f} "
            f"{summary['macro_f1']:>10.4f} "
            f"{summary['macro_recall']:>8.4f} "
            f"{summary['macro_iou']:>10.4f}"
        )
    print("=" * 104)
    return summaries


def threshold_analysis(
    model_path: str,
    data_yaml: str,
    *,
    split: str,
    thresholds: list[float],
    nms_iou: float,
    match_iou: float,
    batch: int,
    imgsz: int,
) -> list[dict]:
    rows = []
    for threshold in thresholds:
        summary = evaluate_detection_model(
            model_path,
            data_yaml,
            split=split,
            conf=threshold,
            iou=nms_iou,
            match_iou_threshold=match_iou,
            batch=batch,
            imgsz=imgsz,
            plots=False,
        )
        rows.append(summary)

    print("\n" + "=" * 88)
    print(f"{'Threshold':<12} {'Macro P':<12} {'Macro R':<12} {'Macro F1':<12} {'Macro IoU':<12}")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['confidence']:<12.2f} "
            f"{row['macro_precision']:<12.4f} "
            f"{row['macro_recall']:<12.4f} "
            f"{row['macro_f1']:<12.4f} "
            f"{row['macro_iou']:<12.4f}"
        )
    best = max(rows, key=lambda item: item["macro_f1"])
    print("-" * 88)
    print(
        f"Best threshold by macro F1: {best['confidence']:.2f} "
        f"(macro F1={best['macro_f1']:.4f}, macro IoU={best['macro_iou']:.4f})"
    )
    print("=" * 88)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO models for manga text detection")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument(
        "--data",
        type=str,
        default="training/datasets/custom_manga.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument("--save-dir", type=str, default="evaluation/results", help="Directory to save JSON summaries")
    parser.add_argument("--compare", nargs="+", help="Optional list of model paths to compare")
    parser.add_argument("--threshold-analysis", action="store_true", help="Evaluate a range of confidence thresholds")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for validation/prediction")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold for evaluation")
    parser.add_argument(
        "--match-iou",
        type=float,
        default=DEFAULT_MATCH_IOU_THRESHOLD,
        help="IoU threshold used for matched box IoU analysis",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size for prediction/validation")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics validation plots")
    args = parser.parse_args()

    if args.compare:
        compare_models(
            args.compare,
            args.data,
            split=args.split,
            conf=args.conf,
            iou=args.iou,
            match_iou=args.match_iou,
            batch=args.batch,
            imgsz=args.imgsz,
        )
        return

    if args.threshold_analysis:
        threshold_analysis(
            args.model,
            args.data,
            split=args.split,
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            nms_iou=args.iou,
            match_iou=args.match_iou,
            batch=args.batch,
            imgsz=args.imgsz,
        )
        return

    evaluate_single_model(
        args.model,
        args.data,
        save_dir=args.save_dir,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        match_iou=args.match_iou,
        batch=args.batch,
        imgsz=args.imgsz,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
