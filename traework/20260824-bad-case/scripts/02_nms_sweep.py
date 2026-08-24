# -*- coding: utf-8 -*-
"""
NMS IoU 阈值扫描 + 推理策略对照 (imgsz=960, TTA)
输出: data/nms_sweep.csv, runs/ 下各 val 结果
"""
import os
import pandas as pd
from ultralytics import YOLO

ROOT = r"D:\Data\New_Codes\Python_Codes\ultralytics"
WEIGHTS = os.path.join(ROOT, r"runs\ssdc_uav_train\size-s-base\yolo12s-baseline_ssdc_uav_exp01\weights\best.pt")
YAML = os.path.join(ROOT, r"datasets\SSDC-UAV_yolo\ssdc-uav.yaml")
OUT = os.path.join(ROOT, r"traework\20260824-bad-case")

model = YOLO(WEIGHTS)
rows = []


def run(tag, **kw):
    args = dict(data=YAML, split="test", conf=0.001, imgsz=640, batch=16, device="0",
                workers=0, project=os.path.join(OUT, "runs"), name=tag, exist_ok=True,
                plots=False, save_json=False, verbose=False)
    args.update(kw)
    m = model.val(**args)
    row = {
        "tag": tag,
        "iou_nms": kw.get("iou", 0.7),
        "imgsz": kw.get("imgsz", 640),
        "tta": kw.get("augment", False),
        "P": float(m.box.mp), "R": float(m.box.mr),
        "mAP50": float(m.box.map50), "mAP75": float(m.box.map75), "mAP50_95": float(m.box.map),
        "F1": float(m.box.f1.mean()),
    }
    rows.append(row)
    print(row, flush=True)


for iou in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    run(f"nms_iou{int(iou * 100):03d}", iou=iou)

run("imgsz960", imgsz=960)
run("tta640", augment=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "data", "nms_sweep.csv"), index=False, encoding="utf-8-sig")
print(df.to_string())
print("DONE")
