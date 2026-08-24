# -*- coding: utf-8 -*-
"""
Bad case 图库: 预测 vs 真值叠加可视化
输出: gallery/*.png, gallery/gallery_index.json, data/audit_high_bgfp.csv
"""
import json
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = r"D:\Data\New_Codes\Python_Codes\ultralytics"
OUT = os.path.join(ROOT, r"traework\20260824-bad-case")
IMG_DIR = os.path.join(ROOT, r"datasets\SSDC-UAV_yolo\images\test")
GAL = os.path.join(OUT, "gallery")
os.makedirs(GAL, exist_ok=True)

with open(os.path.join(OUT, "summary.json"), encoding="utf-8") as f:
    summary = json.load(f)
T_STAR = summary["operating_point"]["t_star"]

dfg = pd.read_csv(os.path.join(OUT, "data", "gt_cases.csv"))
dfp = pd.read_csv(os.path.join(OUT, "data", "pred_cases.csv"))

C_GT = (16, 185, 129)
C_TP = (59, 130, 246)
C_FP = (239, 68, 68)
C_LOW = (245, 158, 11)

try:
    FONT = ImageFont.truetype("msyh.ttc", 13)
    FONT_SM = ImageFont.truetype("msyh.ttc", 11)
except OSError:
    try:
        FONT = ImageFont.truetype("arial.ttf", 13)
        FONT_SM = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        FONT = ImageFont.load_default()
        FONT_SM = FONT


def dashed_rect(draw, box, color, width=2, dash=6):
    x1, y1, x2, y2 = box
    for (a, b, horiz) in [(x1, x2, True), (x1, x2, True)]:
        pass
    def dline(p1, p2):
        length = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if length == 0:
            return
        n = max(1, length // (dash * 2))
        for i in range(n + 1):
            s = i * 2 * dash
            e = min(s + dash, length)
            if s >= length:
                break
            t1 = s / length
            t2 = e / length
            q1 = (p1[0] + (p2[0] - p1[0]) * t1, p1[1] + (p2[1] - p1[1]) * t1)
            q2 = (p1[0] + (p2[0] - p1[0]) * t2, p1[1] + (p2[1] - p1[1]) * t2)
            draw.line([q1, q2], fill=color, width=width)
    dline((x1, y1), (x2, y1))
    dline((x2, y1), (x2, y2))
    dline((x2, y2), (x1, y2))
    dline((x1, y2), (x1, y1))


def draw_legend(draw, W):
    draw.rectangle([0, 0, W, 22], fill=(255, 255, 255))
    items = [("GT真值", C_GT), ("TP", C_TP), ("FP", C_FP), ("低分匹配/漏检GT", C_LOW)]
    x = 8
    for txt, c in items:
        draw.rectangle([x, 7, x + 10, 17], outline=c, width=2)
        draw.text((x + 14, 5), txt, fill=(30, 30, 30), font=FONT_SM)
        x += 14 + int(FONT_SM.getlength(txt)) + 16


def render(stem, out_path, highlight_gt=None, highlight_pred=None):
    """highlight_gt: set of gt_idx 用红色粗框标出; highlight_pred: pred_cases 行索引集合"""
    img = Image.open(os.path.join(IMG_DIR, stem + ".jpg")).convert("RGB")
    W, H = img.size
    canvas = Image.new("RGB", (W, H + 22), (255, 255, 255))
    canvas.paste(img, (0, 22))
    d = ImageDraw.Draw(canvas)
    draw_legend(d, W)

    gts = dfg[dfg["image"] == stem]
    prs = dfp[(dfp["image"] == stem) & (dfp["score"] >= 0.05)]

    for _, g in gts.iterrows():
        box = [g["x1"], g["y1"] + 22, g["x2"], g["y2"] + 22]
        if highlight_gt and int(g["gt_idx"]) in highlight_gt:
            d.rectangle(box, outline=C_FP if g["status"] != "FN_lowScore" else C_LOW, width=4)
        elif str(g["status"]).startswith("FN"):
            d.rectangle(box, outline=C_LOW, width=3)
        else:
            d.rectangle(box, outline=C_GT, width=2)

    for idx, p in prs.iterrows():
        box = [p["x1"], p["y1"] + 22, p["x2"], p["y2"] + 22]
        label = f'{p["score"]:.2f}'
        if highlight_pred and idx in highlight_pred:
            d.rectangle(box, outline=C_FP, width=4)
            d.text((box[0], max(24, box[1] - 14)), label, fill=C_FP, font=FONT)
        elif p["status"] == "TP":
            d.rectangle(box, outline=C_TP, width=2)
            d.text((box[0], max(24, box[1] - 14)), label, fill=C_TP, font=FONT_SM)
        elif p["status"] in ("dupFP", "locFP", "bgFP"):
            d.rectangle(box, outline=C_FP, width=2)
            d.text((box[0], max(24, box[1] - 14)), label, fill=C_FP, font=FONT_SM)
        elif p["status"] == "lowScoreTP":
            dashed_rect(d, box, C_LOW, width=2)
            d.text((box[0], max(24, box[1] - 14)), label, fill=C_LOW, font=FONT_SM)
    canvas.save(out_path)


index = []
used = set()

def add(cat, stem, note, hi_gt=None, hi_pred=None, fname=None):
    fname = fname or f"{cat}_{stem[-24:]}.png"
    render(stem, os.path.join(GAL, fname), hi_gt, hi_pred)
    index.append({"file": fname, "category": cat, "image": stem, "note": note})
    used.add(stem)

# 1. 密集低分漏检 top4
cnt = dfg[dfg["status"] == "FN_lowScore"].groupby("image").size().sort_values(ascending=False)
for stem in [s for s in cnt.index if s not in used][:4]:
    sub = dfg[(dfg["image"] == stem)]
    n_fn = int((sub["status"] == "FN_lowScore").sum())
    add("lowScoreFN", stem,
        f"GT {len(sub)} 个, 低分漏检 {n_fn} 个 (橙虚线为低于阈值 t*={T_STAR:.2f} 的匹配预测)",
        hi_gt=set(sub.loc[sub["status"] == "FN_lowScore", "gt_idx"].astype(int)))

# 2. 近失/未定位漏检
hard = dfg[dfg["status"].isin(["FN_nearMiss", "FN_neverLoc"])]
hcnt = hard.groupby("image").size().sort_values(ascending=False)
for stem in [s for s in hcnt.index if s not in used][:4]:
    sub = dfg[dfg["image"] == stem]
    hh = hard[hard["image"] == stem]
    add("hardMiss", stem,
        f"红粗框为近失/未定位漏检 GT (共 {len(hh)} 个, 全测试集仅 117 个)",
        hi_gt=set(hh["gt_idx"].astype(int)))

# 3. 高分背景FP (标注噪声嫌疑审计) top6
bg = dfp[dfp["status"] == "bgFP"].sort_values("score", ascending=False)
bg.to_csv(os.path.join(OUT, "data", "audit_high_bgfp.csv"), index=False, encoding="utf-8-sig")
picked = 0
for idx, row in bg.iterrows():
    if picked >= 6:
        break
    if row["image"] in used:
        continue
    add("auditBGFP", row["image"],
        f"红粗框: 高分背景FP score={row['score']:.3f}, 需人工核对是否漏标",
        hi_pred={idx})
    picked += 1

# 4. 重复FP
dup = dfp[dfp["status"] == "dupFP"].sort_values("score", ascending=False)
picked = 0
for idx, row in dup.iterrows():
    if picked >= 3:
        break
    if row["image"] in used:
        continue
    add("dupFP", row["image"], f"红粗框: 重复检测FP score={row['score']:.3f} (与TP重叠)", hi_pred={idx})
    picked += 1

# 5. 定位质量欠佳的 TP
locq = dfp[(dfp["status"] == "TP") & (dfp["iou50"] >= 0.5) & (dfp["iou50"] <= 0.65)].sort_values("iou50")
picked = 0
for idx, row in locq.iterrows():
    if picked >= 3:
        break
    if row["image"] in used:
        continue
    add("locQuality", row["image"],
        f"示例: TP 但 IoU={row['iou50']:.2f}<0.65, 定位偏差 (mAP75 差距来源)")
    picked += 1

# 6. 贴边漏检
edge_fn = dfg[(dfg["edge_dist"] <= 0) & (dfg["status"].str.startswith("FN"))]
picked = 0
for _, row in edge_fn.iterrows():
    if picked >= 2:
        break
    if row["image"] in used:
        continue
    add("edgeFN", row["image"], "橙粗框: 贴边 GT 漏检 (切片边缘截断苗)")
    picked += 1

with open(os.path.join(GAL, "gallery_index.json"), "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=2)
print(f"gallery: {len(index)} images")
for it in index:
    print(it["category"], it["file"])
print("DONE")
