"""Basic YOLO training pipeline with shared dataset defaults and fairness reporting."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.detection_metrics import DEFAULT_MATCH_IOU_THRESHOLD, evaluate_detection_model, print_evaluation_summary  # noqa: E402
from training.common import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASET_YAML,
    DEFAULT_RESULTS_DIR,
    PROJECT_ROOT,
    YOLO_RUNS_DIR,
    prepare_ultralytics_dataset_yaml,
    resolve_artifact_reference,
    resolve_project_path,
)


class YOLOTrainer:
    """Minimal but consistent training wrapper for the demo project."""

    def __init__(
        self,
        data_yaml: str,
        model_size: str = str(DEFAULT_BASE_MODEL),
        project_name: str = "basic_train",
        experiment_name: str | None = None,
    ) -> None:
        self.data_yaml = str(resolve_project_path(data_yaml))
        self.ultralytics_data_yaml = str(prepare_ultralytics_dataset_yaml(self.data_yaml))
        self.model_size = model_size
        self.project_name = project_name
        self.experiment_name = experiment_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.project_dir = resolve_project_path(Path("yolo_train_run") / project_name)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        print("🚀 Initializing YOLO Trainer")
        print(f"   Project: {self.project_name}")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Model: {self.model_size}")
        print(f"   Data: {self.data_yaml}")
        print(f"   Resolved Data: {self.ultralytics_data_yaml}")

    def train(
        self,
        *,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 30,
        device: str = "0",
        workers: int = 8,
        **kwargs,
    ):
        print("\n" + "=" * 80)
        print("🎯 STARTING TRAINING")
        print("=" * 80)

        model = YOLO(resolve_artifact_reference(self.model_size, base=PROJECT_ROOT))
        train_params = {
            "data": self.ultralytics_data_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "patience": patience,
            "device": device,
            "workers": workers,
            "project": str(self.project_dir),
            "name": self.experiment_name,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.5,
            "mixup": 0.1,
            "copy_paste": 0.0,
            "save": True,
            "save_period": -1,
            "val": True,
            "plots": True,
            "verbose": True,
        }
        train_params.update(kwargs)
        results = model.train(**train_params)

        weights_path = self.project_dir / self.experiment_name / "weights" / "best.pt"
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best model saved to: {weights_path}")
        return results

    def resume_training(self, checkpoint_path: str):
        print(f"🔄 Resuming training from: {checkpoint_path}")
        model = YOLO(resolve_project_path(checkpoint_path))
        return model.train(resume=True)

    def validate(
        self,
        model_path: str,
        *,
        data_yaml: str | None = None,
        split: str = "val",
        conf: float = 0.001,
        iou: float = 0.6,
        match_iou: float = DEFAULT_MATCH_IOU_THRESHOLD,
    ) -> dict:
        summary = evaluate_detection_model(
            model_path,
            data_yaml or self.data_yaml,
            split=split,
            conf=conf,
            iou=iou,
            match_iou_threshold=match_iou,
            save_dir=DEFAULT_RESULTS_DIR,
            run_name=Path(model_path).stem,
        )
        print_evaluation_summary(summary)
        return summary

    def export(self, model_path: str, *, format: str = "onnx", imgsz: int = 640) -> str:
        print(f"📦 Exporting model to {format.upper()}")
        model = YOLO(resolve_project_path(model_path))
        export_path = model.export(format=format, imgsz=imgsz)
        print(f"✅ Exported to: {export_path}")
        return str(export_path)

    def predict_sample(self, model_path: str, source: str, *, conf: float = 0.25, save: bool = True):
        print(f"🔮 Running prediction on: {source}")
        model = YOLO(resolve_project_path(model_path))
        return model.predict(
            source=source,
            conf=conf,
            save=save,
            project=str(self.project_dir / "predictions"),
            name=self.experiment_name,
        )


def train_from_config(config_path: str):
    resolved_config_path = resolve_project_path(config_path)
    with open(resolved_config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    trainer = YOLOTrainer(
        data_yaml=config["data"],
        model_size=config.get("model", str(DEFAULT_BASE_MODEL)),
        project_name=config.get("project", "basic_train"),
        experiment_name=config.get("name"),
    )
    return trainer.train(**config.get("train", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO model for manga text detection")
    parser.add_argument("--config", type=str, help="Path to training config YAML")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET_YAML), help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default=str(DEFAULT_BASE_MODEL), help="Base checkpoint or model name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0, 1, cpu)")
    parser.add_argument("--project", type=str, default="basic_train", help="Project name")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--validate", type=str, help="Run evaluation on an existing checkpoint instead of training")
    parser.add_argument(
        "--match-iou",
        type=float,
        default=DEFAULT_MATCH_IOU_THRESHOLD,
        help="IoU threshold used for matched box IoU analysis",
    )
    args = parser.parse_args()

    if args.config:
        train_from_config(args.config)
        return

    trainer = YOLOTrainer(
        data_yaml=args.data,
        model_size=args.model,
        project_name=args.project,
        experiment_name=args.name,
    )

    if args.validate:
        trainer.validate(args.validate, match_iou=args.match_iou)
        return

    if args.resume:
        trainer.resume_training(args.resume)
        return

    trainer.train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
