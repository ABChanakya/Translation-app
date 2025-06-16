#!/usr/bin/env python3
"""
train_and_eval.py

Two‑stage YOLOv8 training with:
 1) Head‑only warm‑up (freeze first N layers)
 2) Full fine‑tune in two phases:
    • Phase 1: heavy augmentation (mosaic, mixup)
    • Phase 2: no‐mosaic “fine” epochs
Finally runs inference with tighter NMS.

Usage:
    python train_and_eval.py \
      --model /home/chanakya/Downloads/comic-speech-bubble-detector.pt  \
      --data config.yaml \
      --device cuda \
      --head_epochs 20 \
      --full_epochs 80 \
      --no_mosaic_epochs 15 \
      --freeze_layers 10 \
      --batch_size 24 \
      --lr_head 5e-4 \
      --lr_full 1e-4 \
      --patience_head 10 \
      --patience_full 15 \
      --imgsz 640
"""

import argparse, sys
from ultralytics import YOLO
from PIL import ImageFile
import torch
from torchvision.ops import nms

# allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_grad_by_keyword(model, keyword: str, requires_grad: bool):
    """ Toggle requires_grad on all parameters whose name includes keyword """
    for name, param in model.model.named_parameters():
        if keyword in name:
            param.requires_grad = requires_grad

def train_on_dataset(
    model_name: str,
    data_cfg    : str,
    device      : str,
    head_epochs : int,
    full_epochs : int,
    no_mosaic_epochs: int,
    freeze_layers: int,
    batch_size  : int,
    lr_head     : float,
    lr_full     : float,
    patience_head: int,
    patience_full: int,
    img_size    : int
) -> YOLO:
    """ Two‑stage + two‑phase YOLOv8 training """
    # 1) Load pretrained model
    model = YOLO(model_name)

    # 2) HEAD‑ONLY warm‑up (freeze early layers)
    print(f"\n=== HEAD‑ONLY: {head_epochs} epochs @ lr={lr_head}, freeze={freeze_layers} ===")
    model.train(
        data       = data_cfg,
        epochs     = head_epochs,
        batch      = batch_size,
        imgsz      = img_size,
        lr0        = lr_head,
        lrf        = 0.1,
        device     = device,
        patience   = patience_head,
        freeze     = freeze_layers,
        augment    = True,
        mosaic     = 0.5,
        mixup      = 0.3,
        fliplr     = 0.5,
        flipud     = 0.3,
        degrees    = 15.0,
        translate  = 0.15,
        scale      = 0.6,
        project    = "yolo_train_run",
        name       = "head_warmup2",
        exist_ok   = True
    )

    # 3) FULL‑MODEL fine‑tune Phase 1 (heavy aug)
    phase1_epochs = full_epochs - no_mosaic_epochs
    print(f"\n=== FULL PHASE1: {phase1_epochs} epochs @ lr={lr_full}, heavy aug ===")
    model.train(
        data       = data_cfg,
        epochs     = phase1_epochs,
        batch      = batch_size,
        imgsz      = img_size,
        lr0        = lr_full,
        lrf        = 0.01,
        device     = device,
        patience   = patience_full,
        freeze     = 0,               # unfreeze all
        augment    = True,
        mosaic     = 0.5,
        mixup      = 0.3,
        fliplr     = 0.5,
        flipud     = 0.3,
        degrees    = 15.0,
        translate  = 0.15,
        scale      = 0.6,
        project    = "yolo_train_run",
        name       = "full_finetune_phase10",
        exist_ok   = True
    )

    # 4) FULL‑MODEL fine‑tune Phase 2 (no‐mosaic refine)
    print(f"\n=== FULL PHASE2: {no_mosaic_epochs} epochs @ lr={lr_full/10:.1e}, no mosaic ===")
    model.train(
        data       = data_cfg,
        epochs     = no_mosaic_epochs,
        batch      = batch_size,
        imgsz      = img_size,
        lr0        = lr_full * 0.1,   # finer LR
        lrf        = 0.01,
        device     = device,
        patience   = patience_full,
        freeze     = 0,
        augment    = True,
        mosaic     = 0.0,             # disable mosaic
        mixup      = 0.0,
        fliplr     = 0.5,
        flipud     = 0.3,
        degrees    = 10.0,
        translate  = 0.1,
        scale      = 0.5,
        project    = "yolo_train_run",
        name       = "full_finetune_phase20",
        exist_ok   = True
    )

    return model

def infer_with_tight_nms(model: YOLO, source, conf_thr=0.1, iou_thr=0.3, max_det=50):
    """ Predict + apply tighter NMS to remove near‑dupes """
    preds = model.predict(source=source, conf=conf_thr, iou=iou_thr, max_det=max_det)
    for det in preds:
        boxes  = det.boxes.xyxy
        scores = det.boxes.conf
        keep   = nms(boxes, scores, iou_thr)
        det.boxes = det.boxes[keep]
    return preds

def parse_args():
    p = argparse.ArgumentParser(description="Multi-phase YOLOv10 training script")
    p.add_argument("--model",            type=str,   default="yolo_train_run/full_finetune_phase2/weights/best.pt", help="Pretrained model or local checkpoint")
    p.add_argument("--data",             type=str,   default="config1.yaml",      help="Path to data config YAML")
    p.add_argument("--device",           type=str,   default="cuda",             help="Device for training (cpu, cuda)")
    p.add_argument("--head_epochs",      type=int,   default=10,                 help="Epochs to train head (frozen backbone)")
    p.add_argument("--full_epochs",      type=int,   default=80,                 help="Epochs to fine-tune full model")
    p.add_argument("--no_mosaic_epochs", type=int,   default=15,                 help="Epochs with mosaic off for final tuning")
    p.add_argument("--freeze_layers",    type=int,   default=10,                 help="Number of layers to freeze during head training")
    p.add_argument("--batch_size",       type=int,   default=16,                 help="Batch size")
    p.add_argument("--lr_head",          type=float, default=5e-4,               help="Learning rate for head training")
    p.add_argument("--lr_full",          type=float, default=1e-4,               help="Learning rate for full training")
    p.add_argument("--patience_head",    type=int,   default=10,                 help="Early stopping patience for head training")
    p.add_argument("--patience_full",    type=int,   default=15,                 help="Early stopping patience for full training")
    p.add_argument("--imgsz",            type=int,   default=512,                help="Image size")
    p.add_argument("--mosaic",           type=float, default=0.6,                help="mosaic augmentation prob")
    p.add_argument("--mixup",            type=float, default=0.3,                help="mixup augmentation prob")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = train_on_dataset(
        model_name       = args.model,
        data_cfg         = args.data,
        device           = args.device,
        head_epochs      = args.head_epochs,
        full_epochs      = args.full_epochs,
        no_mosaic_epochs = args.no_mosaic_epochs,
        freeze_layers    = args.freeze_layers,
        batch_size       = args.batch_size,
        lr_head          = args.lr_head,
        lr_full          = args.lr_full,
        patience_head    = args.patience_head,
        patience_full    = args.patience_full,
        img_size         = args.imgsz
    )

    # sample inference on your val images
    sample_imgs = ["data/images/val/0126.jpg", "data/images/val/0118.jpg"]
    results = infer_with_tight_nms(model, source=sample_imgs)
    print("\n=== Sample Inference Results ===")
    for det in results:
        for box, score, cls in zip(
            det.boxes.xyxy.cpu().tolist(),
            det.boxes.conf.cpu().tolist(),
            det.boxes.cls.cpu().tolist()
        ):
            print(f"Class {int(cls)} @ {box}  conf={score:.2f}")