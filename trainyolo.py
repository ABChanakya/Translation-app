from ultralytics import YOLO
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_grad_by_keyword(model, keyword, requires_grad: bool):
    """
    Toggle requires_grad on all parameters whose name includes keyword.
    """
    for name, param in model.model.named_parameters():
        if keyword in name:
            param.requires_grad = requires_grad

def train_on_dataset(
    model_name="jameslahm/yolov10n",
    data_cfg="config.yaml",
    device="cuda",
    head_epochs=30,
    full_epochs=100,
    batch_size=16,
    lr_head=1e-3,
    lr_full=1e-4,
    patience_head=15,
    patience_full=20,
    augment_kwargs=None
):
    """
    Two-stage YOLOv10 training for small datasets (e.g. 60 train / 20 val).
    """

    # 1) Load pretrained model
    model = YOLO(model_name)

    # 2) Freeze backbone (no grads on any 'model.*' layers except heads)
    set_grad_by_keyword(model, keyword="model.", requires_grad=False)

    # 3) HEAD-ONLY warm-up
    print(f"\n=== HEAD-ONLY WARM-UP: {head_epochs} epochs @ lr={lr_head} ===")
    model.train(
        data=data_cfg,
        epochs=head_epochs,
        batch=batch_size,
        lr0=lr_head,
        lrf=0.1,                     # final LR = lr0 * 0.1
        device=device,
        patience=patience_head,
        project="yolo_train_run",
        name="head_warmup_60_20",
        exist_ok=True,
        augment=True,
        # reduced augment to discourage duplicate learning
        mosaic=0.5,
        mixup=0.3,
        copy_paste=0.0,
        fliplr=0.5,
        flipud=0.3,
        degrees=15.0,
        translate=0.15,
        scale=0.6,
        **(augment_kwargs or {})
    )

    # 4) Unfreeze all parameters for full fine-tuning
    for _, param in model.model.named_parameters():
        param.requires_grad = True

    # 5) FULL-MODEL fine-tune
    print(f"\n=== FULL-MODEL FINE-TUNE: {full_epochs} epochs @ lr={lr_full} ===")
    model.train(
        data=data_cfg,
        epochs=full_epochs,
        batch=batch_size,
        lr0=lr_full,
        lrf=0.01,                    # even finer final LR
        device=device,
        patience=patience_full,
        project="yolo_train_run",
        name="full_finetune_60_20",
        exist_ok=True,
        augment=True,
        mosaic=0.5,
        mixup=0.3,
        copy_paste=0.0,
        fliplr=0.5,
        flipud=0.3,
        degrees=15.0,
        translate=0.15,
        scale=0.6,
        **(augment_kwargs or {})
    )

    return model

def infer_with_tight_nms(
    model,
    source,
    conf_thr=0.1,
    iou_thr=0.3,
    max_det=50
):
    """
    Simple inference with tighter NMS to suppress near‑duplicate boxes.
    """
    return model.predict(
        source=source,
        conf=conf_thr,
        iou=iou_thr,
        max_det=max_det
    )

if __name__ == "__main__":
    model = train_on_dataset()
    imgs = ["data/images/val/147.jpg", "data/images/val/152.jpg"]
    results = infer_with_tight_nms(model, source=imgs)

    for box, score, cls in zip(
        results[0].boxes.xyxy.cpu().tolist(),
        results[0].boxes.conf.cpu().tolist(),
        results[0].boxes.cls.cpu().tolist()
    ):
        print(f"Class {int(cls)} @ {box} conf={score:.2f}")
#  Best combo → conf=0.1, iou=0.3, F1=0.472, mAP50=0.481, mAP50-95=0.327
