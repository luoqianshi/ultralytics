# SSDC-UAV TCSAE 实验方案 · 证据数据准备脚本（只读源数据，输出到 traework/20260901/data）
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"d:\Data\New_Codes\Python_Codes\ultralytics")
OUT = ROOT / "traework" / "20260901" / "data"
OUT.mkdir(parents=True, exist_ok=True)

REN = {
    "Parameters (M)": "Params_M",
    "GFLOPS (imgsz=640)": "GFLOPs",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1-Score": "F1",
    "mAP50 (IoU=0.50)": "mAP50",
    "mAP75 (IoU=0.75)": "mAP75",
    "mAP50-95 (IoU=0.50:0.95)": "mAP50_95",
}


def load(p):
    return pd.read_csv(p, encoding="utf-8-sig").rename(columns=REN)


s = load(ROOT / "runs/test_result/SSDC-UAV_Test_Result.csv")
n = load(ROOT / "runs/test_result/SSDC-UAV_Test_Result_n.csv")
nms = load(ROOT / "runs/test_result/SSDC-UAV_Test_Result_NMS微调实验.csv")
re0 = load(ROOT / "runs/test_result_re0/SSDC-UAV_Test_Result_re0.csv")

# ---------- 1. improvement_history.csv ----------
hist = pd.concat([s.assign(Size="s"), n.assign(Size="n")], ignore_index=True)
base_map = {"s": 88.267, "n": 87.880}  # YOLOv12s-baseline / YOLOv12n(250e) 测试集 mAP50
hist["Delta_mAP50_vs_base"] = hist.apply(
    lambda r: round(r["mAP50"] - base_map[r["Size"]], 3), axis=1
)
hist = hist.sort_values("mAP50", ascending=False)
hist.to_csv(OUT / "improvement_history.csv", index=False, encoding="utf-8-sig")

# ---------- 2. baseline_summary.csv ----------
rows = []


def pick(df, model, proto, size, note):
    r = df[df["Model"] == model]
    if r.empty:
        return
    r = r.iloc[0]
    rows.append(
        {
            "Model": model,
            "Size": size,
            "Protocol": proto,
            "Epochs": r["Epoch"],
            "Params_M": r["Params_M"],
            "GFLOPs": r["GFLOPs"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "mAP50": r["mAP50"],
            "mAP75": r["mAP75"],
            "mAP50_95": r["mAP50_95"],
            "Note": note,
        }
    )


pick(s, "YOLOv12s-baseline", "coco_pretrain", "s", "主基线（预训练协议）")
pick(re0, "YOLOv12s", "from_scratch", "s", "from-scratch 150ep")
rows.append(
    {
        "Model": "YOLOv12s(300ep)", "Size": "s", "Protocol": "from_scratch", "Epochs": 300,
        "Params_M": 9.25, "GFLOPs": 21.52, "Precision": 84.546, "Recall": 80.618,
        "mAP50": 88.121, "mAP75": 60.030, "mAP50_95": 55.128,
        "Note": "from-scratch 300ep（论文主协议基线）",
    }
)
pick(n, "YOLOv12n-300e/250e/150e", "coco_pretrain", "n", "n 尺寸基线（尺寸消融对照）")
pick(s, "YOLOv12s-Facaler-CIoU", "coco_pretrain", "s", "历史最佳改进（预训练协议）")
pick(re0, "YOLOv12s-Focaler-CIoU", "from_scratch", "s", "同一改进在 from-scratch 下增益归零（协议交互证据）")
pick(nms, "YOLOv12s-baseline_nms60", "coco_pretrain", "s", "NMS iou 0.60 免费收益")
pick(nms, "YOLOv12s-baseline_nms50", "coco_pretrain", "s", "NMS iou 0.50（Recall 最优）")
for m in ["YOLOv11s", "YOLOv8s", "YOLOv26s", "YOLOv5su"]:
    pick(s, m, "coco_pretrain", "s", "对比模型（预训练）")
for m in ["YOLOv11s", "YOLOv8s", "YOLOv26s"]:
    pick(re0, m, "from_scratch", "s", "对比模型（from-scratch）")
for m in ["YOLOv8n", "YOLOv11n", "YOLOv5nu", "YOLOv26n"]:
    pick(n, m, "coco_pretrain", "n", "n 尺寸对比模型")
pd.DataFrame(rows).to_csv(OUT / "baseline_summary.csv", index=False, encoding="utf-8-sig")

# ---------- 3. error_decomposition.csv ----------
sj = json.loads((ROOT / "traework/20260824-bad-case/summary.json").read_text(encoding="utf-8"))
pc, gc, orc = sj["pred_counts"], sj["gt_counts"], sj["oracle"]
n_gt, n_preds = sj["n_gt"], sj["n_preds"]
ed = []
for k, v, note in [
    ("TP", pc["TP"], "正确检测"),
    ("dupFP", pc["dupFP"], "重复框（同一株框两次）"),
    ("locFP", pc["locFP"], "定位偏差/框到邻苗"),
    ("bgFP", pc["bgFP"], "背景误检（含332个高分疑似漏标）"),
    ("below_thr", pc["below_thr"], "低于操作阈值的背景预测"),
]:
    ed.append({"Section": "预测侧构成", "Category": k, "Count": v,
               "Share_pct": round(100 * v / n_preds, 2), "Note": note})
for k, v, note in [
    ("TP", gc["TP"], "已检出"),
    ("FN_lowScore", gc["FN_lowScore"], "低分漏检（框已出但分数低）"),
    ("FN_nearMiss", gc["FN_nearMiss"], "近失漏检（最佳IoU 0.1~0.5）"),
    ("FN_neverLoc", gc["FN_neverLoc"], "完全未定位漏检"),
]:
    ed.append({"Section": "GT侧构成", "Category": k, "Count": v,
               "Share_pct": round(100 * v / n_gt, 2), "Note": note})
base_ap = orc["base"]["AP50"]
for k, v, note in [
    ("base", base_ap, "当前基线（复算AP50）"),
    ("O1_修复低分漏检", orc["O1_scoring"]["AP50"], f"修复{orc['O1_scoring']['n_fixed']}个低分漏检"),
    ("O2_去重", orc["O2_dedup"]["AP50"], f"修复{orc['O2_dedup']['n_fixed']}个重复FP"),
    ("O3_背景抑制", orc["O3_bgFP"]["AP50"], f"修复{orc['O3_bgFP']['n_fixed']}条背景预测"),
    ("O1+O2+O3_联合上界", orc["O123_joint"]["AP50"], "理论回收上限"),
]:
    ed.append({"Section": "Oracle模拟", "Category": k, "Count": "",
               "Share_pct": round(100 * (v - base_ap), 2), "Note": f"AP50={v:.4f}（Δ={100*(v-base_ap):+.2f}pp）· {note}"})
ed.append({"Section": "Oracle模拟", "Category": "O4_定位质量", "Count": "",
           "Share_pct": "", "Note": f"AP75 {orc['O4_localization']['AP75_base']:.4f} → {orc['O4_localization']['AP75_oracle']:.4f}（修复{orc['O4_localization']['n_fixed']}个TP定位）"})
pd.DataFrame(ed).to_csv(OUT / "error_decomposition.csv", index=False, encoding="utf-8-sig")

# ---------- 4. counting_quick_eval.csv ----------
bc = ROOT / "traework/20260824-bad-case/data"
gt = pd.read_csv(bc / "gt_cases.csv")
pred = pd.read_csv(bc / "pred_cases.csv")
gt_cnt = gt.groupby("image")["img_n_gt"].first()
all_imgs = set(gt_cnt.index) | set(pred["image"].unique())
gt_arr = np.array([int(gt_cnt.get(im, 0)) for im in sorted(all_imgs)])
img_order = sorted(all_imgs)
pred_img = pred["image"].values
pred_score = pred["score"].values
sort_idx = np.argsort(pred_img)
pred_img_s, pred_score_s = pred_img[sort_idx], pred_score[sort_idx]

res = []
for name, t in [("F1最优点_conf0.378", 0.37762), ("Recall优先_P≥0.80_conf0.257", 0.25734),
                ("Precision优先_P≥0.85_conf0.355", 0.3545), ("高置信_conf0.5", 0.5)]:
    kept = pred_score_s >= t
    cnt = pd.Series(kept, index=pred_img_s).groupby(level=0).sum().reindex(img_order, fill_value=0).values.astype(float)
    ss_res = float(((gt_arr - cnt) ** 2).sum())
    ss_tot = float(((gt_arr - gt_arr.mean()) ** 2).sum())
    r2_direct = 1 - ss_res / ss_tot
    slope, intercept = np.polyfit(gt_arr, cnt, 1)
    fit = slope * gt_arr + intercept
    r2_fit = 1 - float(((cnt - fit) ** 2).sum()) / ss_tot
    mae = float(np.abs(cnt - gt_arr).mean())
    rmse = float(np.sqrt(((cnt - gt_arr) ** 2).mean()))
    res.append({
        "OperatingPoint": name, "ConfThr": t, "PredTotal": int(cnt.sum()), "GTTotal": int(gt_arr.sum()),
        "R2_fit": round(r2_fit, 4), "R2_direct": round(r2_direct, 4),
        "MAE": round(mae, 4), "RMSE": round(rmse, 4), "Slope": round(slope, 4),
    })
pd.DataFrame(res).to_csv(OUT / "counting_quick_eval.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(res).to_string(index=False))

# ---------- 5. experiment_protocol.csv ----------
proto = pd.DataFrame([
    ["P0 协议", "E0.1", "预训练 vs from-scratch 协议对照表", "论文试验节协议正当性说明", "YOLOv12s 两协议（已有数据 88.267 vs 88.121）", "—", "—", "—", "已完成"],
    ["P1 数据侧", "E1.1", "标注审计", "复核332个高分疑似漏标 + 抽检低分漏检边界苗", "audit_high_bgfp.csv 全量人工复核", "—", "—", "—", "待执行（人力）"],
    ["P1 数据侧", "E1.2", "干净测试子集重建与复评", "量化标注噪声对指标的影响", "修订后重建 test-clean，复评基线", "—", "—", "—", "待执行"],
    ["P2 基线", "E2.1", "主基线多种子", "显著性基准", "YOLOv12s from-scratch 300ep", "from_scratch/640/SGD", 300, 3, "待执行（已有1种子）"],
    ["P2 基线", "E2.2", "对比模型补齐", "对比试验表", "RT-DETR-r（from-scratch）；可选 Faster R-CNN", "统一协议", 300, 1, "待执行"],
    ["P3 消融", "E3.0", "消融基线", "消融起点", "YOLOv12s", "from_scratch/640/SGD", 300, 1, "复用E2.1"],
    ["P3 消融", "E3.1", "+NWD辅助损失", "密集/小苗漏检（漏检率16.4%→50%）", "box_loss=CIoU+λ·NWD", "同上", 300, 1, "待执行（脚本已有）"],
    ["P3 消融", "E3.2", "+内容感知上采样", "小苗漏检率46%（大苗4.8倍）", "CARAFE（需实现）或 DySample 替换最近邻上采样", "同上", 300, 1, "待执行"],
    ["P3 消融", "E3.3", "+检测头改进", "94.7%漏检为低分漏检（打分校准）", "DyHead 或 共享卷积轻量化头", "同上", 300, 1, "待执行"],
    ["P3 消融", "E3.4", "+Soft-NMS后处理", "密集打分抑制/重复框影响计数", "Soft-NMS 替换硬NMS（先重测现有best.pt）", "推理阶段", "—", "—", "待执行（零训练成本）"],
    ["P3 消融", "E3.5", "关键两两组合×2~3", "模块互补/抑制分析", "损失×上采样、损失×头 等", "同上", 300, 1, "待执行"],
    ["P3 消融", "E3.6", "全组合×3种子", "最终模型与显著性", "最优组合", "同上", 300, 3, "待执行"],
    ["P4 计数", "E4.1", "tile级计数评价", "计数指标主表", "每切片检测框数 vs GT框数，报R²/MAE/RMSE", "—", "—", "—", "基线快评已完成（见counting_quick_eval.csv）"],
    ["P4 计数", "E4.2", "原始大图级计数评价", "对标人工真值", "30张测试大图（8540株人工真值）聚合计数回归", "—", "—", "—", "待执行"],
    ["P5 可视化", "E5.1", "Grad-CAM热力图+检测对比图", "可视化分析节", "基线 vs 最终模型", "—", "—", "—", "待执行"],
    ["P5 部署", "E5.2", "边缘设备实测（可选加分项）", "轻量化叙事", "n尺寸模型 Jetson FPS", "—", "—", "—", "视设备而定"],
], columns=["Phase", "ExpID", "Name", "Purpose", "Config", "Protocol", "Epochs", "Seeds", "Status"])
proto.to_csv(OUT / "experiment_protocol.csv", index=False, encoding="utf-8-sig")

print("\nOK: all CSVs written to", OUT)
