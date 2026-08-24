# -*- coding: utf-8 -*-
"""
SSDC-UAV YOLO12s baseline bad-case 误差分解
输入: predictions.json (COCO xywh abs, conf=0.001) + YOLO GT labels
输出: data/*.csv, summary.json, charts/*.png
"""
import json
import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"D:\Data\New_Codes\Python_Codes\ultralytics"
PRED_JSON = os.path.join(ROOT, r"runs\ssdc_uav_test\size-s-base\yolo12s-baseline_ssdc_uav_test_exp01\predictions.json")
GT_DIR = os.path.join(ROOT, r"datasets\SSDC-UAV_yolo\labels\test")
IMG_DIR = os.path.join(ROOT, r"datasets\SSDC-UAV_yolo\images\test")
OUT = os.path.join(ROOT, r"traework\20260824-bad-case")
DATA_DIR = os.path.join(OUT, "data")
CHART_DIR = os.path.join(OUT, "charts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

EMERALD = "#10B981"
BLUE = "#3B82F6"
RED = "#EF4444"
AMBER = "#F59E0B"
GRAY = "#6B7280"


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def iou_matrix(a, b):
    # a: (N,4) xyxy, b: (M,4) xyxy
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def ap101(scores, tp, n_gt):
    """COCO 101-pt interpolated AP. scores/tp: aligned arrays for all preds."""
    order = np.argsort(-scores)
    tp = tp[order].astype(float)
    fp = 1 - tp
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.clip(ctp + cfp, 1e-9, None)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    thr = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, thr, side="left")
    idx = np.clip(idx, 0, len(mpre) - 1)
    return float(np.mean(mpre[idx])), mrec, mpre


def greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thr):
    """按分数降序贪心匹配。返回 pred_match: list of (gt_idx, iou) or None"""
    n = len(pred_boxes)
    pred_match = [None] * n
    if n == 0 or len(gt_boxes) == 0:
        return pred_match
    ious = iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores)
    matched = np.zeros(len(gt_boxes), dtype=bool)
    for pi in order:
        row = ious[pi].copy()
        row[matched] = -1
        gi = int(np.argmax(row))
        if row[gi] >= iou_thr:
            pred_match[pi] = (gi, float(row[gi]))
            matched[gi] = True
    return pred_match


# ---------------- 1. 读取数据 ----------------
with open(PRED_JSON, "r") as f:
    preds_raw = json.load(f)

preds_by_img = defaultdict(list)
for p in preds_raw:
    preds_by_img[p["image_id"]].append(p)

gt_files = [f for f in os.listdir(GT_DIR) if f.endswith(".txt")]
stems = sorted(f[:-4] for f in gt_files)
print(f"GT images: {len(stems)}, predictions: {len(preds_raw)}")

img_sizes = {}
gt_by_img = {}
for stem in stems:
    img_path = os.path.join(IMG_DIR, stem + ".jpg")
    with Image.open(img_path) as im:
        img_sizes[stem] = im.size  # (W,H)
    W, H = img_sizes[stem]
    boxes = []
    with open(os.path.join(GT_DIR, stem + ".txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    gt_by_img[stem] = np.array(boxes, dtype=float)

size_set = set(img_sizes.values())
print("image sizes:", size_set)

# ---------------- 2. 逐图匹配 (IoU 0.5 / 0.75) ----------------
all_pred_records = []   # 每条预测一行
all_gt_records = []     # 每个 GT 一行

for stem in stems:
    W, H = img_sizes[stem]
    gt = gt_by_img[stem]
    plist = preds_by_img.get(stem, [])
    if plist:
        pboxes = np.array([xywh_to_xyxy(p["bbox"]) for p in plist])
        pscores = np.array([p["score"] for p in plist])
    else:
        pboxes = np.zeros((0, 4))
        pscores = np.zeros(0)

    match50 = greedy_match(pboxes, pscores, gt, 0.5)
    match75 = greedy_match(pboxes, pscores, gt, 0.75)
    ious_all = iou_matrix(pboxes, gt)  # (n_pred, n_gt)

    # pred 记录
    for i, p in enumerate(plist):
        max_iou_any = float(ious_all[i].max()) if len(gt) else 0.0
        all_pred_records.append({
            "image": stem, "score": float(p["score"]),
            "x1": pboxes[i][0], "y1": pboxes[i][1], "x2": pboxes[i][2], "y2": pboxes[i][3],
            "tp50": match50[i] is not None, "iou50": match50[i][1] if match50[i] else 0.0,
            "gt50": match50[i][0] if match50[i] else -1,
            "tp75": match75[i] is not None,
            "max_iou_any_gt": max_iou_any,
        })

    # GT 特征
    n_gt = len(gt)
    if n_gt:
        gt_gt_iou = iou_matrix(gt, gt)
        np.fill_diagonal(gt_gt_iou, 0)
        max_gt_iou = gt_gt_iou.max(axis=1)
        ctr = np.stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2], axis=1)
        dist = np.sqrt(((ctr[:, None, :] - ctr[None, :, :]) ** 2).sum(axis=2))
        neighbors = ((dist < 100) & (dist > 0)).sum(axis=1)
        edge_dist = np.minimum(np.minimum(gt[:, 0], gt[:, 1]), np.minimum(W - gt[:, 2], H - gt[:, 3]))
    # GT 被匹配情况(取最高分匹配者: greedy 中每个 gt 只被一个 pred 匹配)
    gt_matched_by50 = {}
    gt_matched_by75 = {}
    for i, m in enumerate(match50):
        if m:
            gt_matched_by50[m[0]] = (i, float(p["score"] if False else pscores[i]), m[1])
    for i, m in enumerate(match75):
        if m:
            gt_matched_by75[m[0]] = (i, float(pscores[i]), m[1])
    max_pred_iou_gt = ious_all.max(axis=0) if len(plist) and n_gt else np.zeros(n_gt)

    for gi in range(n_gt):
        w = gt[gi][2] - gt[gi][0]
        h = gt[gi][3] - gt[gi][1]
        area = w * h
        all_gt_records.append({
            "image": stem, "gt_idx": gi,
            "x1": gt[gi][0], "y1": gt[gi][1], "x2": gt[gi][2], "y2": gt[gi][3],
            "w": w, "h": h, "area": area,
            "max_gt_iou": float(max_gt_iou[gi]),
            "neighbors_100px": int(neighbors[gi]),
            "edge_dist": float(edge_dist[gi]),
            "img_n_gt": n_gt,
            "matched50": gi in gt_matched_by50,
            "pred_score50": gt_matched_by50[gi][1] if gi in gt_matched_by50 else 0.0,
            "pred_iou50": gt_matched_by50[gi][2] if gi in gt_matched_by50 else 0.0,
            "matched75": gi in gt_matched_by75,
            "max_pred_iou": float(max_pred_iou_gt[gi]) if n_gt else 0.0,
        })

dfp = pd.DataFrame(all_pred_records)
dfg = pd.DataFrame(all_gt_records)
n_gt_total = len(dfg)
print(f"total GT: {n_gt_total}, total preds: {len(dfp)}")

# ---------------- 3. PR 曲线 / F1 最优阈值 (IoU=0.5) ----------------
scores_all = dfp["score"].values
tp50_all = dfp["tp50"].values.astype(bool)
ap50_recompute, mrec, mpre = ap101(scores_all, tp50_all, n_gt_total)
print(f"recomputed AP50 = {ap50_recompute:.5f} (official 0.88267)")

order = np.argsort(-scores_all)
tp_o = tp50_all[order].astype(float)
sc_o = scores_all[order]
ctp = np.cumsum(tp_o)
cfp = np.cumsum(1 - tp_o)
rec_curve = ctp / n_gt_total
prec_curve = ctp / np.clip(ctp + cfp, 1e-9, None)
f1_curve = 2 * prec_curve * rec_curve / np.clip(prec_curve + rec_curve, 1e-9, None)
best_i = int(np.argmax(f1_curve))
t_star = float(sc_o[best_i])
P_star = float(prec_curve[best_i])
R_star = float(rec_curve[best_i])
F1_star = float(f1_curve[best_i])
print(f"t*={t_star:.4f}  P={P_star:.4f}  R={R_star:.4f}  F1={F1_star:.4f} (official P=0.84911 R=0.81341)")

# Recall-priority operating points
def op_at_min_prec(min_p):
    ok = np.where(prec_curve >= min_p)[0]
    if len(ok) == 0:
        return None
    i = ok[np.argmax(rec_curve[ok])]
    return {"min_p": min_p, "conf": float(sc_o[i]), "P": float(prec_curve[i]), "R": float(rec_curve[i])}

ops = [op_at_min_prec(0.85), op_at_min_prec(0.80), op_at_min_prec(0.75)]

# ---------------- 4. 操作点 t* 下的误差分类 ----------------
dfp["status"] = "below_thr"
mask_hi = dfp["score"] >= t_star
dfp.loc[mask_hi & dfp["tp50"], "status"] = "TP"
dfp.loc[mask_hi & ~dfp["tp50"] & (dfp["max_iou_any_gt"] >= 0.5), "status"] = "dupFP"
dfp.loc[mask_hi & ~dfp["tp50"] & (dfp["max_iou_any_gt"] >= 0.1) & (dfp["max_iou_any_gt"] < 0.5), "status"] = "locFP"
dfp.loc[mask_hi & ~dfp["tp50"] & (dfp["max_iou_any_gt"] < 0.1), "status"] = "bgFP"
dfp.loc[~mask_hi & dfp["tp50"], "status"] = "lowScoreTP"  # 已定位但低于阈值

pred_counts = dfp["status"].value_counts().to_dict()
for k in ["TP", "dupFP", "locFP", "bgFP", "lowScoreTP", "below_thr"]:
    pred_counts.setdefault(k, 0)

# GT 侧 FN 分类
def fn_type(row):
    if row["matched50"] and row["pred_score50"] >= t_star:
        return "TP"
    if row["matched50"]:
        return "FN_lowScore"
    if row["max_pred_iou"] >= 0.1:
        return "FN_nearMiss"
    return "FN_neverLoc"

dfg["status"] = dfg.apply(fn_type, axis=1)
gt_counts = dfg["status"].value_counts().to_dict()
for k in ["TP", "FN_lowScore", "FN_nearMiss", "FN_neverLoc"]:
    gt_counts.setdefault(k, 0)
print("pred status:", pred_counts)
print("gt status:", gt_counts)

# ---------------- 5. Oracle 增益模拟 ----------------
def ap_from(dfp_sub):
    return ap101(dfp_sub["score"].values, dfp_sub["tp50"].values.astype(bool), n_gt_total)[0]

def pr_at(dfp_mod, thr):
    hi = dfp_mod["score"] >= thr
    tp = int((hi & dfp_mod["tp50"]).sum())
    fp = int((hi & ~dfp_mod["tp50"]).sum())
    p = tp / max(tp + fp, 1)
    # R 需按 GT 重新数: 简化用 pred 侧 TP / n_gt (每个 gt 最多一个 tp)
    r = tp / n_gt_total
    return p, r

base_P, base_R = P_star, R_star
base_AP = ap50_recompute

# O1: 打分修复 —— lowScoreTP 的分数抬到 t* 以上
dfp_o1 = dfp.copy()
dfp_o1.loc[dfp_o1["status"] == "lowScoreTP", "score"] = t_star + 1e-4
o1_AP = ap_from(dfp_o1)
o1_P, o1_R = pr_at(dfp_o1, t_star)

# O2: 去重 —— 删除 dupFP
dfp_o2 = dfp[dfp["status"] != "dupFP"]
o2_AP = ap_from(dfp_o2)
o2_P, o2_R = pr_at(dfp_o2, t_star)

# O3: 清除背景 FP —— 删除所有 max_iou<0.1 的预测
dfp_o3 = dfp[dfp["max_iou_any_gt"] >= 0.1]
o3_AP = ap_from(dfp_o3)
o3_P, o3_R = pr_at(dfp_o3, t_star)

# O4: 完美定位 —— IoU0.5 匹配者在 0.75 也算匹配 → AP75 上界
dfp_o4 = dfp.copy()
dfp_o4["tp75"] = dfp_o4["tp50"]
o4_AP75 = ap101(dfp_o4["score"].values, dfp_o4["tp75"].values.astype(bool), n_gt_total)[0]
base_AP75 = ap101(scores_all, dfp["tp75"].values.astype(bool), n_gt_total)[0]

# O5: 全部 FN 修复的召回上界 (=1) 按 FN 子类拆分贡献
fn_low_n = gt_counts["FN_lowScore"]
fn_near_n = gt_counts["FN_nearMiss"]
fn_never_n = gt_counts["FN_neverLoc"]

# O1+O2+O3 联合
dfp_o123 = dfp_o1[(dfp_o1["status"] != "dupFP") & (dfp_o1["max_iou_any_gt"] >= 0.1)]
o123_AP = ap_from(dfp_o123)
o123_P, o123_R = pr_at(dfp_o123, t_star)

oracle = {
    "base":  {"AP50": base_AP, "P": base_P, "R": base_R},
    "O1_scoring": {"AP50": o1_AP, "P": o1_P, "R": o1_R, "n_fixed": int((dfp["status"] == 'lowScoreTP').sum())},
    "O2_dedup": {"AP50": o2_AP, "P": o2_P, "R": o2_R, "n_fixed": pred_counts["dupFP"]},
    "O3_bgFP": {"AP50": o3_AP, "P": o3_P, "R": o3_R, "n_fixed": int((dfp["max_iou_any_gt"] < 0.1).sum())},
    "O4_localization": {"AP75_base": base_AP75, "AP75_oracle": o4_AP75,
                         "n_fixed": int((dfp["tp50"] & ~dfp["tp75"]).sum())},
    "O123_joint": {"AP50": o123_AP, "P": o123_P, "R": o123_R},
    "fn_recoverable": {"lowScore": fn_low_n, "nearMiss": fn_near_n, "neverLoc": fn_never_n,
                        "n_gt": n_gt_total},
}
print(json.dumps(oracle, indent=2, default=float))

# ---------------- 6. 切片分析 ----------------
def fn_rate_by(df, key, bins, labels):
    cats = pd.cut(df[key], bins=bins, labels=labels, include_lowest=True)
    g = df.groupby(cats, observed=False)["status"]
    total = g.count()
    fn = g.apply(lambda s: s.str.startswith("FN").sum())
    low = g.apply(lambda s: (s == "FN_lowScore").sum())
    return pd.DataFrame({"bin": labels, "n": total.values, "fn": fn.values,
                         "fn_lowScore": low.values,
                         "fn_rate": (fn / total.clip(lower=1)).values})

scale_bins = [0, 1024, 9216, 1e9]
scale_labels = ["small(<32²)", "medium(32²~96²)", "large(>96²)"]
sl_scale = fn_rate_by(dfg, "area", scale_bins, scale_labels)

ov_bins = [-0.001, 0.1, 0.3, 0.5, 0.7, 1.01]
ov_labels = ["[0,0.1)", "[0.1,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,1]"]
sl_overlap = fn_rate_by(dfg, "max_gt_iou", ov_bins, ov_labels)

edge_bins = [-1, 0, 16, 48, 1000]
edge_labels = ["贴边(=0px)", "0~16px", "16~48px", ">48px"]
sl_edge = fn_rate_by(dfg, "edge_dist", edge_bins, edge_labels)

dens_bins = [0, 3, 6, 9, 12, 1000]
dens_labels = ["1~3", "4~6", "7~9", "10~12", ">12"]
sl_dens = fn_rate_by(dfg, "img_n_gt", dens_bins, dens_labels)

# GT-GT 重叠结构性上限
struct_ceiling = {}
for t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    struct_ceiling[f"{t}"] = float((dfg["max_gt_iou"] >= t).mean())

# ---------------- 7. 保存 CSV / JSON ----------------
dfp_out = dfp[dfp["score"] >= 0.005].copy()
dfp_out.to_csv(os.path.join(DATA_DIR, "pred_cases.csv"), index=False, encoding="utf-8-sig")
dfg.to_csv(os.path.join(DATA_DIR, "gt_cases.csv"), index=False, encoding="utf-8-sig")
sl_scale.to_csv(os.path.join(DATA_DIR, "slice_scale.csv"), index=False, encoding="utf-8-sig")
sl_overlap.to_csv(os.path.join(DATA_DIR, "slice_overlap.csv"), index=False, encoding="utf-8-sig")
sl_edge.to_csv(os.path.join(DATA_DIR, "slice_edge.csv"), index=False, encoding="utf-8-sig")
sl_dens.to_csv(os.path.join(DATA_DIR, "slice_density.csv"), index=False, encoding="utf-8-sig")

thr_rows = []
idx_sample = np.linspace(0, len(sc_o) - 1, 400).astype(int)
for i in idx_sample:
    thr_rows.append({"conf": sc_o[i], "P": prec_curve[i], "R": rec_curve[i], "F1": f1_curve[i]})
pd.DataFrame(thr_rows).to_csv(os.path.join(DATA_DIR, "pr_curve.csv"), index=False, encoding="utf-8-sig")

summary = {
    "n_images": len(stems), "n_gt": n_gt_total, "n_preds": len(dfp),
    "img_sizes": [list(s) for s in size_set],
    "ap_check": {"recomputed_AP50": base_AP, "official_mAP50": 0.88267,
                 "recomputed_AP75": base_AP75, "official_mAP75": 0.60853},
    "operating_point": {"t_star": t_star, "P": P_star, "R": R_star, "F1": F1_star,
                         "official_P": 0.84911, "official_R": 0.81341},
    "recall_priority_ops": [o for o in ops if o],
    "pred_counts": pred_counts, "gt_counts": gt_counts,
    "oracle": oracle,
    "struct_ceiling_frac_gt_with_sibling_iou_ge": struct_ceiling,
    "gt_overlap_quantiles": {q: float(np.quantile(dfg["max_gt_iou"], float(q))) for q in ["0.5", "0.75", "0.9", "0.95", "0.99"]},
    "lowScoreTP_score_quantiles": {q: float(np.quantile(dfp.loc[dfp["status"] == "lowScoreTP", "score"], float(q)))
                                    if (dfp["status"] == "lowScoreTP").any() else None
                                    for q in ["0.25", "0.5", "0.75", "0.9"]},
    "bgFP_high_score_n": int((dfp["status"].eq("bgFP") & dfp["score"].ge(0.5)).sum()),
}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ---------------- 8. 图表 ----------------
def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_facecolor("white")

# 8.1 PR 曲线
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
ax.plot(mrec, mpre, color=EMERALD, lw=2, label=f"PR 曲线 (AP50={base_AP:.4f})")
ax.scatter([P_star and R_star], [P_star], color=RED, zorder=5, s=40)
ax.annotate(f"F1最优点 conf={t_star:.3f}\nP={P_star:.3f} R={R_star:.3f}",
            (R_star, P_star), textcoords="offset points", xytext=(-110, -30), fontsize=9,
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
for o in ops:
    if o:
        ax.scatter([o["R"]], [o["P"]], s=24, zorder=5, label=f"P≥{o['min_p']}: R={o['R']:.3f} (conf={o['conf']:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc="lower left"); style_ax(ax)
fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "pr_curve.png")); plt.close(fig)

# 8.2 F1/P/R vs conf
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
thr_df = pd.DataFrame(thr_rows)
thr_df = thr_df[thr_df["conf"] <= 0.95]
ax.plot(thr_df["conf"], thr_df["P"], color=BLUE, lw=1.6, label="Precision")
ax.plot(thr_df["conf"], thr_df["R"], color=AMBER, lw=1.6, label="Recall")
ax.plot(thr_df["conf"], thr_df["F1"], color=EMERALD, lw=2, label="F1")
ax.axvline(t_star, color=GRAY, ls="--", lw=1)
ax.annotate(f"t*={t_star:.3f}", (t_star, 0.05), rotation=90, fontsize=8, color=GRAY)
ax.set_xlabel("置信度阈值"); ax.set_ylabel("指标值"); ax.set_xlim(0, 0.95); ax.set_ylim(0, 1)
ax.legend(fontsize=9); style_ax(ax)
fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "f1_conf.png")); plt.close(fig)

# 8.3 误差构成
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
pc = [pred_counts["TP"], pred_counts["dupFP"], pred_counts["locFP"], pred_counts["bgFP"]]
plabels = [f"TP\n{pc[0]}", f"重复FP\n{pc[1]}", f"定位FP\n{pc[2]}", f"背景FP\n{pc[3]}"]
axes[0].bar(plabels, pc, color=[EMERALD, AMBER, "#F97316", RED])
axes[0].set_title(f"预测侧构成 (conf≥t*={t_star:.3f}, 共{sum(pc)}条)")
gc = [gt_counts["TP"], gt_counts["FN_lowScore"], gt_counts["FN_nearMiss"], gt_counts["FN_neverLoc"]]
glabels = [f"已检出TP\n{gc[0]}", f"低分漏检\n{gc[1]}", f"近失漏检\n{gc[2]}", f"未定位漏检\n{gc[3]}"]
axes[1].bar(glabels, gc, color=[EMERALD, AMBER, "#F97316", RED])
axes[1].set_title(f"真值侧构成 (共{n_gt_total}个GT)")
for ax in axes:
    style_ax(ax)
    ax.tick_params(axis="x", labelsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "error_composition.png")); plt.close(fig)

# 8.4 oracle 增益
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
names = ["打分修复\nO1", "去重复FP\nO2", "清背景FP\nO3", "O1+O2+O3"]
daps = [(o1_AP - base_AP) * 100, (o2_AP - base_AP) * 100, (o3_AP - base_AP) * 100, (o123_AP - base_AP) * 100]
bars = ax.bar(names, daps, color=[AMBER, BLUE, RED, EMERALD])
for b, v in zip(bars, daps):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"+{v:.2f}", ha="center", fontsize=10)
ax.set_ylabel("ΔAP50 (百分点)"); ax.set_title(f"Oracle 修复的 AP50 上界增益 (基线 AP50={base_AP:.4f})")
style_ax(ax)
fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "oracle_gains.png")); plt.close(fig)

# 8.5 FN率 vs GT-GT重叠
fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
ax.bar(sl_overlap["bin"], sl_overlap["fn_rate"] * 100, color=EMERALD, label="总FN率")
ax.bar(sl_overlap["bin"], sl_overlap["fn_lowScore"] / sl_overlap["n"].clip(lower=1) * 100,
       color=AMBER, label="其中: 低分漏检")
for i, (v, n) in enumerate(zip(sl_overlap["fn_rate"] * 100, sl_overlap["n"])):
    ax.text(i, v + 0.5, f"{v:.1f}%\n(n={n})", ha="center", fontsize=8)
ax.set_xlabel("该 GT 与另一 GT 的最大 IoU"); ax.set_ylabel("漏检率 (%)")
ax.set_title("GT 重叠程度 × 漏检率 (NMS 密集误删假说核心证据)")
ax.legend(fontsize=9); style_ax(ax)
fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "fn_by_overlap.png")); plt.close(fig)

# 8.6 FN率 vs 尺度
fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=150)
ax.bar(sl_scale["bin"], sl_scale["fn_rate"] * 100, color=EMERALD)
for i, (v, n) in enumerate(zip(sl_scale["fn_rate"] * 100, sl_scale["n"])):
    ax.text(i, v + 0.3, f"{v:.1f}%\n(n={n})", ha="center", fontsize=9)
ax.set_ylabel("漏检率 (%)"); ax.set_title("目标尺度 × 漏检率")
style_ax(ax)
fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "fn_by_scale.png")); plt.close(fig)

# 8.7 FN率 vs 边缘距离
fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=150)
ax.bar(sl_edge["bin"], sl_edge["fn_rate"] * 100, color=BLUE)
for i, (v, n) in enumerate(zip(sl_edge["fn_rate"] * 100, sl_edge["n"])):
    ax.text(i, v + 0.3, f"{v:.1f}%\n(n={n})", ha="center", fontsize=9)
ax.set_xlabel("GT 框到切片边缘的最小距离"); ax.set_ylabel("漏检率 (%)")
ax.set_title("切片边缘效应 (步长608, 重叠32px)")
style_ax(ax)
fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "fn_by_edge.png")); plt.close(fig)

# 8.8 分数分布
fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
bins = np.linspace(0, 1, 51)
ax.hist(dfp.loc[dfp["tp50"], "score"], bins=bins, alpha=0.75, color=EMERALD, label="已匹配预测(TP候选)", density=False)
fp_mask = ~dfp["tp50"] & (dfp["score"] >= 0.001)
ax.hist(dfp.loc[fp_mask, "score"], bins=bins, alpha=0.65, color=RED, label="未匹配预测(FP候选)", density=False)
ax.axvline(t_star, color=GRAY, ls="--", lw=1)
ax.set_yscale("log")
ax.set_xlabel("置信度"); ax.set_ylabel("数量 (log)")
ax.set_title("预测置信度分布: 匹配 vs 未匹配")
ax.legend(fontsize=9); style_ax(ax)
fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "score_hist.png")); plt.close(fig)

# 8.9 GT-GT 重叠 CDF + 结构上限
fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
sorted_iou = np.sort(dfg["max_gt_iou"].values)
cdf = 1 - np.arange(1, len(sorted_iou) + 1) / len(sorted_iou)
ax.plot(sorted_iou, cdf * 100, color=EMERALD, lw=2)
for t in [0.5, 0.7, 0.9]:
    frac = (dfg["max_gt_iou"] >= t).mean() * 100
    ax.scatter([t], [frac], color=RED, zorder=5, s=30)
    ax.annotate(f"IoU≥{t}: {frac:.2f}%GT", (t, frac), textcoords="offset points",
                xytext=(10, 6), fontsize=8.5)
ax.set_xlabel("GT 与其他 GT 的最大 IoU"); ax.set_ylabel("占比 (%)")
ax.set_title("GT 间最大重叠的反向累积分布 (结构性召回上限证据)")
style_ax(ax)
fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "gt_overlap_cdf.png")); plt.close(fig)

print("\nDONE. outputs in", OUT)
