#!/usr/bin/env python3
import argparse
import os
from ultralytics import YOLO
from PIL import ImageFile
import torch
from torchvision.ops import nms

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_grad_by_keyword(model, keyword: str, requires_grad: bool):
    """
    Toggle requires_grad on all parameters whose name includes keyword.
    """
    for name, param in model.model.named_parameters():
        if keyword in name:
            param.requires_grad = requires_grad


def train_on_dataset(
    model_name: str,
    data_cfg: str,
    device: str,
    head_epochs: int,
    full_epochs: int,
    batch_size: int,
    lr_head: float,
    lr_full: float,
    patience_head: int,
    patience_full: int,
    img_size: int = 640,
    augment_kwargs: dict = None
) -> YOLO:
    """
    Two-stage YOLO training:
      1) head-only warm-up with frozen backbone
      2) full-model fine-tune
    """
    model = YOLO(model_name)
    # Freeze backbone
    set_grad_by_keyword(model, keyword="model.", requires_grad=False)

    print(f"\n=== HEAD-ONLY WARM-UP: {head_epochs} epochs @ lr={lr_head} ===")
    model.train(
        data        = data_cfg,
        epochs      = head_epochs,
        batch       = batch_size,
        imgsz       = img_size,
        lr0         = lr_head,
        lrf         = 0.1,
        device      = device,
        patience    = patience_head,
        half        = True,
        project     = "yolo_train_run",
        name        = "head_warmup",
        exist_ok    = True,
        augment     = True,
        mosaic      = 0.5,
        mixup       = 0.3,
        copy_paste  = 0.0,
        fliplr      = 0.5,
        flipud      = 0.3,
        degrees     = 15.0,
        translate   = 0.15,
        scale       = 0.6,
        **(augment_kwargs or {})
    )

    # Unfreeze all layers for full fine-tune
    for _, param in model.model.named_parameters():
        param.requires_grad = True

    print(f"\n=== FULL-MODEL FINE-TUNE: {full_epochs} epochs @ lr={lr_full} ===")
    model.train(
        data        = data_cfg,
        epochs      = full_epochs,
        batch       = batch_size,
        imgsz       = img_size,
        lr0         = lr_full,
        lrf         = 0.01,
        device      = device,
        patience    = patience_full,
        half        = True,
        project     = "yolo_train_run",
        name        = "full_finetune",
        exist_ok    = True,
        augment     = True,
        mosaic      = 0.5,
        mixup       = 0.3,
        copy_paste  = 0.0,
        fliplr      = 0.5,
        flipud      = 0.3,
        degrees     = 15.0,
        translate   = 0.15,
        scale       = 0.6,
        **(augment_kwargs or {})
    )

    return model


def infer_with_tight_nms(
    model: YOLO,
    source,
    conf_thr: float = 0.1,
    iou_thr: float  = 0.3,
    max_det: int    = 50
):
    """
    Run inference and apply tighter NMS to reduce duplicate detections.
    """
    preds = model.predict(source=source, conf=conf_thr, iou=iou_thr, max_det=max_det)
    for det in preds:
        boxes  = det.boxes.xyxy  # (N,4)
        scores = det.boxes.conf  # (N,)
        keep   = nms(boxes, scores, iou_thr)
        det.boxes = det.boxes[keep]
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str, default="jameslahm/yolov10n")
    parser.add_argument("--data",         type=str, default="config.yaml")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--head_epochs",  type=int, default=30)
    parser.add_argument("--full_epochs",  type=int, default=100)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--lr_head",      type=float, default=1e-3)
    parser.add_argument("--lr_full",      type=float, default=1e-4)
    parser.add_argument("--patience_head",type=int, default=15)
    parser.add_argument("--patience_full",type=int, default=20)
    parser.add_argument("--imgsz",        type=int, default=640)
    args = parser.parse_args()

    model = train_on_dataset(
        model_name     = args.model,
        data_cfg       = args.data,
        device         = args.device,
        head_epochs    = args.head_epochs,
        full_epochs    = args.full_epochs,
        batch_size     = args.batch_size,
        lr_head        = args.lr_head,
        lr_full        = args.lr_full,
        patience_head  = args.patience_head,
        patience_full  = args.patience_full,
        img_size       = args.imgsz
    )

    # Example inference on validation images
    imgs = ["data/images/val/147.jpg", "data/images/val/152.jpg"]
    results = infer_with_tight_nms(model, source=imgs)
    print("\n=== Sample Inference Results ===")
    for det in results:
        for box, score, cls in zip(
            det.boxes.xyxy.cpu().tolist(),
            det.boxes.conf.cpu().tolist(),
            det.boxes.cls.cpu().tolist()
        ):
            print(f"Class {int(cls)} @ {box}  conf={score:.2f}")
