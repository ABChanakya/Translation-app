import pandas as pd
from ultralytics import YOLO

# 1) Load your best model
model = YOLO("yolo_train_run/full_finetune_60_20/weights/best.pt")

# 2) Define your sweep grid
conf_thrs = [0.1, 0.2, 0.3, 0.4, 0.5]
iou_thrs  = [0.3, 0.5, 0.7]
data_cfg  = "config.yaml"

records = []
for conf in conf_thrs:
    for iou in iou_thrs:
        # 3) Run validation
        metrics = model.val(data=data_cfg, conf=conf, iou=iou, verbose=False)

        # 4) Pull mean precision & recall as attributes, not methods
        mp    = metrics.box.mp    # mean precision (numpy float)
        mr    = metrics.box.mr    # mean recall

        # 5) Compute global F1
        f1    = 2 * mp * mr / (mp + mr + 1e-9)

        # 6) And also mAPs as attributes
        map50 = metrics.box.map50    # mAP@0.5
        map95 = metrics.box.map      # mAP@0.5–0.95

        records.append({
            "conf_thr":  conf,
            "iou_thr":   iou,
            "precision": mp,
            "recall":    mr,
            "f1":        f1,
            "mAP50":     map50,
            "mAP50-95":  map95
        })

# 7) Summarize your top results
df   = pd.DataFrame(records)
best = df.sort_values("f1", ascending=False).iloc[0]
top5 = df.sort_values("f1", ascending=False).head(5)

print("\nTop 5 threshold settings by F1:")
print(top5.to_string(index=False))
print(
    f"\n👉 Best combo → conf={best.conf_thr}, iou={best.iou_thr}, "
    f"F1={best.f1:.3f}, mAP50={best.mAP50:.3f}, mAP50-95={best['mAP50-95']:.3f}"
)
