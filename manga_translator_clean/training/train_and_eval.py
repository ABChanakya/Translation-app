#!/usr/bin/env python3
"""Three-phase YOLO training with shared class-fair evaluation reporting."""

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
    model_name: str,
    data_cfg: str,
    device: str,
    head_epochs: int,
    full_epochs: int,
    no_mosaic_epochs: int,
    freeze_layers: int,
    batch_size: int,
    lr_head: float,
    lr_full: float,
    patience_head: int,
    patience_full: int,
    img_size: int,
    project: str,
    match_iou: float,
) -> tuple[object, list[dict[str, str]], dict]:
    phase1_epochs = max(full_epochs - no_mosaic_epochs, 0)
    phases = [
        TrainingPhase(
            name="head_warmup",
            epochs=head_epochs,
            lr0=lr_head,
            lrf=0.1,
            patience=patience_head,
            freeze=freeze_layers,
            overrides={
                "augment": True,
                "mosaic": 0.5,
                "mixup": 0.2,
                "fliplr": 0.5,
                "flipud": 0.1,
                "degrees": 10.0,
                "translate": 0.1,
                "scale": 0.5,
            },
        ),
    ]

    if phase1_epochs > 0:
        phases.append(
            TrainingPhase(
                name="full_finetune_phase1",
                epochs=phase1_epochs,
                lr0=lr_full,
                lrf=0.01,
                patience=patience_full,
                freeze=0,
                overrides={
                    "augment": True,
                    "mosaic": 0.4,
                    "mixup": 0.15,
                    "fliplr": 0.5,
                    "flipud": 0.1,
                    "degrees": 10.0,
                    "translate": 0.1,
                    "scale": 0.5,
                },
            )
        )

    if no_mosaic_epochs > 0:
        phases.append(
            TrainingPhase(
                name="full_finetune_phase2",
                epochs=no_mosaic_epochs,
                lr0=lr_full * 0.1,
                lrf=0.01,
                patience=patience_full,
                freeze=0,
                overrides={
                    "augment": True,
                    "mosaic": 0.0,
                    "mixup": 0.0,
                    "fliplr": 0.5,
                    "flipud": 0.0,
                    "degrees": 5.0,
                    "translate": 0.05,
                    "scale": 0.3,
                },
            )
        )

    model, phase_outputs = run_training_plan(
        model_name=model_name,
        data_yaml=data_cfg,
        phases=phases,
        device=device,
        batch_size=batch_size,
        imgsz=img_size,
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
        imgsz=img_size,
        plots=False,
        save_dir=DEFAULT_RESULTS_DIR,
        run_name=Path(final_weights).parent.parent.name,
    )
    return model, phase_outputs, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-phase YOLO training with post-training evaluation")
    parser.add_argument("--model", type=str, default=str(DEFAULT_BASE_MODEL), help="Base checkpoint or Ultralytics model name")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET_YAML), help="Path to dataset YAML")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--head-epochs", type=int, default=15, help="Epochs to train the frozen head")
    parser.add_argument("--full-epochs", type=int, default=80, help="Total epochs for full-model training")
    parser.add_argument("--no-mosaic-epochs", type=int, default=15, help="Final refinement epochs without mosaic")
    parser.add_argument("--freeze-layers", type=int, default=10, help="How many layers to freeze during warm-up")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr-head", type=float, default=5e-4, help="Learning rate for head warm-up")
    parser.add_argument("--lr-full", type=float, default=1e-4, help="Learning rate for full fine-tuning")
    parser.add_argument("--patience-head", type=int, default=10, help="Early stopping patience for warm-up")
    parser.add_argument("--patience-full", type=int, default=15, help="Early stopping patience for full fine-tuning")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--project", type=str, default="yolo_train_run", help="Training output directory")
    parser.add_argument(
        "--match-iou",
        type=float,
        default=DEFAULT_MATCH_IOU_THRESHOLD,
        help="IoU threshold used for matched box IoU analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _model, phase_outputs, summary = train_on_dataset(
        model_name=args.model,
        data_cfg=args.data,
        device=args.device,
        head_epochs=args.head_epochs,
        full_epochs=args.full_epochs,
        no_mosaic_epochs=args.no_mosaic_epochs,
        freeze_layers=args.freeze_layers,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_full=args.lr_full,
        patience_head=args.patience_head,
        patience_full=args.patience_full,
        img_size=args.imgsz,
        project=args.project,
        match_iou=args.match_iou,
    )

    print("\nGenerated checkpoints:")
    for output in phase_outputs:
        print(f"  - {output['phase']}: {output['weights']}")

    print_evaluation_summary(summary)


if __name__ == "__main__":
    main()
