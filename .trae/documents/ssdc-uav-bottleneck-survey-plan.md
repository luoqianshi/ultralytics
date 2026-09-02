# SSDC-UAV 检测瓶颈归结与 YOLO 改进方案文献调研计划

## Summary

基于 2026-08-24 bad-case 难例挖掘的量化结果，把 SSDC-UAV 上 YOLO12s 的检测性能瓶颈归结为 **3 个问题**（附证据排除清单），然后针对每个瓶颈类别，从 **CCF 顶刊顶会** 与 **计算机×农业工程交叉 SCI 一二区** 文献中系统检索可用的 YOLO 类改进方案，产出一份带引用、带推荐矩阵的中文 HTML 调研报告。本任务只做调研与报告，不跑训练实验，但报告末尾给出"下一批实验候选清单"。

## Current State Analysis（数据与事实依据）

### 来自 20260824-bad-case 的硬数据（traework/20260824-bad-case/summary.json、report.html）
- 测试集：1703 切片 / 11263 GT；baseline：P=0.849, R=0.813, mAP50=0.883, mAP75=0.609, mAP50-95=0.556；F1 最优 t*=0.378。
- FN 解剖：2190 个漏检中 **2073（94.7%）为低分漏检**（已框出 IoU≥0.5，分数中位数仅 0.123）；近失 111；完全未定位 6。
- 维度切片：小目标 FN 率 46.0%（大目标 9.5%）；密度 >12 株/图 FN 率 27.1%（稀疏 8.7%）；GT-GT 重叠 [0.5,0.7) FN 率 41.8%；边缘效应阴性。
- Oracle 上界：O1 打分修复 +4.58 AP50（R→0.990）；O3 清背景 FP +2.22；O4 完美定位 AP75 +27.2；O2 去重 ≈0。
- NMS 假说证伪：GT-GT IoU≥0.7 仅 0.036%；放松 NMS 全面变差；收紧 iou=0.5~0.6 免费 +0.4~0.5 mAP50。
- 标注噪声嫌疑：332 个高分背景 FP（score≥0.5）。

### 项目已试过清单（来自 runs/、AddModules/、记忆 8/5–8/30）
- 勉强有效：Focaler-CIoU、Powerful-CIoUv2(PIoU2)、DySample(dyscope)、SD-Loss；组合 DySample+PIoU2+DyHead 已配置待训。
- 无效/剔除：P2 检测头（极小目标仅 0.24%）、A2C2f 内注意力（EMA/SCSA/MoCA/MCA/Mona）、BiFPN/FreqFusion 颈部或 head 重构（预训练迁移率 342/729，权重断裂）、SPDConv（构建事故）、Soft-NMS/密度感知 NMS（8/24 证伪）、推理 imgsz960/TTA（负收益）。
- 方法论约束：噪声地板 σ≈0.10 mAP50；结构大改破坏 COCO 预训练迁移；损失函数是迄今最大增益来源；用户有效性标准 = mAP50≥+1.0、P/R 不同时降、Recall 优先。

## 瓶颈归结（报告第 1 部分的结论框架）

- **B1 难样本判别力/置信度抑制（Recall 主瓶颈，上界 +4.6 mAP50）**：小目标+密集+重叠场景下 cls/conf 分支系统性低估；2073 低分 FN 为证。
- **B2 边界框回归精度（mAP75/mAP50-95 主瓶颈，上界 +27 AP75）**：3347 个 TP 的 IoU 卡在 [0.5,0.75)。
- **B3 背景判别与标注噪声（Precision 侧，上界 +2.2 mAP50）**：332 高分疑似漏标 + 全分数段 24541 条背景 FP；同时污染评估可信度。
- **排除清单**（带证据）：NMS 策略、切片边缘效应、重复检测、P2 极小目标覆盖。

## Proposed Changes / 执行步骤

### 步骤 1：基线复核（只读）
- 重读 `traework/20260824-bad-case/summary.json` 与 report.html 第 4–6 节，把上述数字固化进报告；不重算。

### 步骤 2：并行文献调研（3 个 Explore 子代理，恰好 3 个，一瓶颈一代理）
每个代理返回结构化条目表，字段：方法名 / 出处(venue+年份) / CCF 等级或 JCR 分区 / 核心机制(1-2 句) / 对应瓶颈 / 在 UAV·农业·小密目标场景的实证 / YOLO 集成形态(损失|模块|增强|分配|训练策略) / 预训练兼容性 / 预期收益指标 / 实现成本 / 参考链接。检索范围 2019–2026，优先 2023–2026。

**代理 1（B1 难样本判别力/小密目标）** 检索主题：
Varifocal Loss(CVPR21)、Quality/Generalized Focal Loss(ECCV20)、TOOD(ICCV21)、IoU-aware/PAA(ECCV20)、NWD 小目标损失、OHEM 及难例加权、置信度校准与证据深度学习(EDL)检测、copy-paste 增强(CVPR21)、高分辨率训练；交叉场景：CEA/Remote Sensing 上 UAV 小麦·玉米·甘蔗苗·杂草的 YOLO 小密目标论文。

**代理 2（B2 定位质量）** 检索主题：
IoU 损失族（SIoU、EIoU、WIoU v3、MPDIoU、Inner-IoU、Shape-IoU、Focaler-IoU、PIoU v2）、Distribution Focal Loss/GFLv2、Gaussian/KL 回归、DySample(ICCV23)、CARAFE(ICCV19)、DyHead(CVPR21)、任务对齐分配对定位的影响；农业遥感 SCI 中 YOLO 定位损失对比类论文。

**代理 3（B3 噪声与背景）** 检索主题：
检测任务的 noisy-label learning（鲁棒损失、标签修正、Co-teaching 系）、伪标签精炼（Soft Teacher 等半监督）、data-centric 标注质量/主动学习审计、农田背景抑制注意力、负样本挖掘；农业数据集中标注噪声处理的 SCI 论文。

### 步骤 3：主会话补充检索（WebSearch，跨瓶颈综述与场景论文）
- small object detection survey TPAMI 2024/2025；UAV detection survey Remote Sensing 2025；YOLO 农业应用综述 CEA 2025/2026；sugarcane/seedling counting UAV 论文。用于报告"场景证据"与引用兜底。

### 步骤 4：去重、映射、排序
- 与"已试过清单"逐条核对，标注 本项目状态（已试有效/已试无效/未试）。
- 按用户标准排序：①对 B1 的 Recall 空间是否 ≥+1.0 mAP50 量级；②预训练兼容（插件式损失/模块 > head 重构）；③实现成本。
- 产出推荐矩阵：每瓶颈 Top 3–5 方案，含"为什么这次不一样"（对照历史失败原因，如权重断裂、口径不对症）。

### 步骤 5：生成 HTML 报告
- 路径：`traework/20260901-bottleneck-survey/report.html`（单文件、浅色极简、Emerald #10B981、SVG 图标、无 emoji，遵循用户视觉偏好）。
- 结构：
  1. TL;DR（3 瓶颈 + 排除清单 + Top 推荐 3 条）
  2. 瓶颈归结（B1/B2/B3 数字证据表 + 排除清单）
  3. 分瓶颈方案全景（每瓶颈一张方法表 + 重点方法卡片：机制/证据/集成方式/成本/兼容性）
  4. 推荐矩阵与下一批实验候选清单（5 组以内，标明预期收益与验证方式，含种子复测提醒）
  5. 已试过方向复盘映射（瓶颈 × 已试方案 矩阵，解释为何有效/无效）
  6. 参考文献（全部为实际检索到的来源，markdown 链接）

## Assumptions & Decisions
- 交付物仅调研报告 + 实验候选清单，不实施训练/代码改动。
- 输出目录新建 `traework/20260901-bottleneck-survey/`，与 20260824-bad-case 平级。
- 报告语言中文，文献标题保留英文原文。
- 检索来源限定：CCF A/B 会议期刊（CVPR/ICCV/ECCV/AAAI/IJCAI/NeurIPS/TPAMI/TIP/IJCV 等）与 JCR Q1/Q2 交叉刊（Computers and Electronics in Agriculture、Remote Sensing、IEEE JSTARS、Precision Agriculture、Plant Phenomics、Biosystems Engineering、Smart Agricultural Technology、Agronomy）。
- 不推荐已被本项目证伪/剔除的方向（NMS 系、P2、head 重构类），除非新文献给出与本项目失败原因不冲突的新证据。

## Verification
- 报告每个方法条目含可点击引用链接；无检索来源的条目不入库。
- 核对"本项目状态"列与 AddModules/runs/记忆一致。
- 数字与 summary.json 逐项一致（B1/B2/B3 证据表）。
- 文件产出后以 Read 抽查 HTML 关键节与表格渲染内容完整。
