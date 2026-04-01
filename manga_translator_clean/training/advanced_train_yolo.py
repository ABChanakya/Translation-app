"""Simplified two-stage training entrypoint with post-training fairness reporting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.detection_metrics import DEFAULT_MATCH_IOU_THRESHOLD, evaluate_detection_model, print_evaluation_summary  # noqa: E402
from training.common import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASET_YAML,
    DEFAULT_RESULTS_DIR,
    TrainingPhase,
    run_training_plan,
)


def train_on_dataset(
    *,
    model_name: str = str(DEFAULT_BASE_MODEL),
    data_cfg: str = str(DEFAULT_DATASET_YAML),
    device: str = "cuda",
    head_epochs: int = 20,
    full_epochs: int = 80,
    batch_size: int = 16,
    imgsz: int = 640,
    lr_head: float = 1e-3,
    lr_full: float = 1e-4,
    patience_head: int = 12,
    patience_full: int = 18,
    project: str = "yolo_train_run",
    match_iou: float = DEFAULT_MATCH_IOU_THRESHOLD,
) -> tuple[object, list[dict[str, str]], dict]:
    phases = [
        TrainingPhase(
            name="head_warmup",
            epochs=head_epochs,
            lr0=lr_head,
            lrf=0.1,
            patience=patience_head,
            freeze=10,
            overrides={
                "augment": True,
                "mosaic": 0.5,
                "mixup": 0.2,
                "copy_paste": 0.0,
                "fliplr": 0.5,
                "flipud": 0.1,
                "degrees": 10.0,
                "translate": 0.1,
                "scale": 0.5,
            },
        ),
        TrainingPhase(
            name="full_finetune",
            epochs=full_epochs,
            lr0=lr_full,
            lrf=0.01,
            patience=patience_full,
            freeze=0,
            overrides={
                "augment": True,
                "mosaic": 0.4,
                "mixup": 0.1,
                "copy_paste": 0.0,
                "fliplr": 0.5,
                "flipud": 0.1,
                "degrees": 10.0,
                "translate": 0.1,
                "scale": 0.5,
            },
        ),
    ]

    model, phase_outputs = run_training_plan(
        model_name=model_name,
        data_yaml=data_cfg,
        phases=phases,
        device=device,
        batch_size=batch_size,
        imgsz=imgsz,
        project_dir=project,
        common_overrides={"optimizer": "AdamW"},
    )

    final_weights = phase_outputs[-1]["weights"]
    summary = evaluate_detection_model(
        final_weights,
        data_cfg,
        split="val",
        conf=0.001,
        iou=0.6,
        match_iou_threshold=match_iou,
        batch=batch_size,
        imgsz=imgsz,
        plots=False,
        save_dir=DEFAULT_RESULTS_DIR,
        run_name=Path(final_weights).parent.parent.name,
    )
    return model, phase_outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage YOLO training for manga text detection")
    parser.add_argument("--model", type=str, default=str(DEFAULT_BASE_MODEL), help="Base checkpoint or Ultralytics model name")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET_YAML), help="Path to dataset YAML")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--head-epochs", type=int, default=20, help="Warm-up epochs with frozen backbone")
    parser.add_argument("--full-epochs", type=int, default=80, help="Full fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate for head warm-up")
    parser.add_argument("--lr-full", type=float, default=1e-4, help="Learning rate for full fine-tuning")
    parser.add_argument("--patience-head", type=int, default=12, help="Early stopping patience for warm-up")
    parser.add_argument("--patience-full", type=int, default=18, help="Early stopping patience for full fine-tuning")
    parser.add_argument("--project", type=str, default="yolo_train_run", help="Training output directory")
    parser.add_argument(
        "--match-iou",
        type=float,
        default=DEFAULT_MATCH_IOU_THRESHOLD,
        help="IoU threshold used for matched box IoU analysis",
    )
    args = parser.parse_args()

    _model, phase_outputs, summary = train_on_dataset(
        model_name=args.model,
        data_cfg=args.data,
        device=args.device,
        head_epochs=args.head_epochs,
        full_epochs=args.full_epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        lr_head=args.lr_head,
        lr_full=args.lr_full,
        patience_head=args.patience_head,
        patience_full=args.patience_full,
        project=args.project,
        match_iou=args.match_iou,
    )

    print("\nGenerated checkpoints:")
    for output in phase_outputs:
        print(f"  - {output['phase']}: {output['weights']}")

    print_evaluation_summary(summary)


if __name__ == "__main__":
    main()
