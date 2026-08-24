# -*- coding: utf-8 -*-
"""NMS 扫描结果图"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"D:\Data\New_Codes\Python_Codes\ultralytics"
OUT = os.path.join(ROOT, r"traework\20260824-bad-case")
df = pd.read_csv(os.path.join(OUT, "data", "nms_sweep.csv"))
sweep = df[df["tag"].str.startswith("nms_iou")].sort_values("iou_nms")

EMERALD, BLUE, AMBER, GRAY = "#10B981", "#3B82F6", "#F59E0B", "#6B7280"

fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=150)
ax.plot(sweep["iou_nms"], sweep["mAP50"] * 100, "-o", color=EMERALD, lw=2, ms=5, label="mAP50")
ax.plot(sweep["iou_nms"], sweep["R"] * 100, "-s", color=AMBER, lw=2, ms=5, label="Recall@F1最优")
ax.plot(sweep["iou_nms"], sweep["P"] * 100, "-^", color=BLUE, lw=1.4, ms=4, alpha=0.7, label="Precision@F1最优")
ax.plot(sweep["iou_nms"], sweep["mAP75"] * 100, "--d", color=GRAY, lw=1.4, ms=4, alpha=0.8, label="mAP75")
ax.axvline(0.7, color=GRAY, ls=":", lw=1)
ax.annotate("baseline 默认 iou=0.7", (0.7, 70), rotation=90, fontsize=8, color=GRAY)
best = sweep.loc[sweep["mAP50"].idxmax()]
ax.scatter([best["iou_nms"]], [best["mAP50"] * 100], s=90, facecolor="none", edgecolor="red", zorder=5)
ax.annotate(f"最优 iou={best['iou_nms']:.2f}: mAP50={best['mAP50']*100:.2f}, R={best['R']*100:.2f}",
            (best["iou_nms"], best["mAP50"] * 100), textcoords="offset points", xytext=(-10, -26), fontsize=9,
            ha="right",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
ax.set_xlabel("NMS IoU 阈值 (越小抑制越强)")
ax.set_ylabel("指标 (%)")
ax.set_title("NMS IoU 扫描: 放松 NMS 全面变差, 「NMS 误删是 FN 主源」假说被证伪", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25, linewidth=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "charts", "nms_sweep.png"))
print("saved nms_sweep.png")
