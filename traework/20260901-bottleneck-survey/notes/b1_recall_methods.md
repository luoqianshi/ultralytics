# B1「难样本判别力/置信度抑制」改进方案文献调研

- 调研日期：2026-09-01
- 项目：SSDC-UAV 甘蔗苗无人机检测（Ultralytics YOLO12s，COCO 预训练微调 150 epoch，imgsz=640，切片 640×640）
- 基线：P=0.849, R=0.813, mAP50=0.883, mAP75=0.609, mAP50-95=0.556
- B1 定义（Recall 主瓶颈）：小目标+密集+重叠场景下分类/置信度分支系统性低估。证据：2190 个 FN 中 2073 个（94.7%）为"低分漏检"（框已出、IoU≥0.5，但分数中位数仅 0.123）；小目标 FN 率 46%（大目标 9.5%）；密度>12 株/图 FN 率 27.1%（稀疏 8.7%）；GT-GT 重叠 [0.5,0.7) 区域 FN 率 41.8%。Oracle 上界：仅修复打分排序即 +4.58 AP50，Recall→0.990。
- 项目约束：结构大改破坏 COCO 预训练迁移（BiFPN/FreqFusion/A2C2f 重构已试无效）；优先损失级/增强级/插件式小模块级/标签分配与训练策略级方案。有效性标准：mAP50 ≥ +1.0 且 P/R 不同时降，Recall 优先。
- 已试无效（B1 相关）：P2 检测头（-0.26）、SD-Loss（+0.19 噪声内）、推理 imgsz960/TTA（负收益）、Soft-NMS（证伪）。

## 检索与核实说明

- 时间范围 2019–2026 优先；两条奠基性工作（OHEM 2016、SNIP/SNIPER 2018）因任务主题显式要求而收录，并明确标注"窗口外/奠基"。
- 所有条目的 venue/年份/链接均经 WebSearch/WebFetch 实际核实（arXiv abs 页、openaccess.thecvf.com、proceedings.neurips.cc、ecva.net、DOI 页、官方 GitHub）。
- 任务书勘误：
  1) 任务书称 GFL 为 ECCV 2020，经核实 Generalized Focal Loss 发表于 **NeurIPS 2020**（proceedings.neurips.cc 可查）。
  2) 任务书称 "IoU-aware Classification Score (ICCV 2021 oral)"；IACS 概念实际出自 **VarifocalNet（CVPR 2021 Oral）**，ICCV 2021 Oral 对应的是 **TOOD**，两篇均已收录。
- 检索后丢弃的条目（无可靠符合等级的出处）：
  - E-DETR（Evidential Deep Learning for End-to-End Uncertainty Estimation in Object Detection）：OpenReview 显示为 **ICLR 2025 Withdrawn Submission**，未正式发表，丢弃（在 §10 中说明 EDL 现状）。
  - CertainNet（RA-L 2021）：机器人信刊、不在任务限定的农业/遥感交叉刊清单内，仅在 §10 附注。
  - SAHI 切片推理（IEEE ICIP 2022，CCF C）：等级不达标且项目已用切片，丢弃。
  - 若干 arXiv-only 预印本（YOLOX/SimOTA 例外：以 OTA CVPR 2021 为正式出处，SimOTA 作为其简化工程变体附注）。

## 快速索引表

| # | 方法 | 出处 | 等级 | 集成形态 | 预训练兼容性 | 预期收益（原文量级） | 成本 |
|---|------|------|------|----------|--------------|----------------------|------|
| 1 | Varifocal Loss / VFNet | CVPR 2021 Oral | CCF A | 损失 | 高 | ~+2.0 AP（COCO，vs 基线） | 低 |
| 2 | Generalized Focal Loss (QFL+DFL) | NeurIPS 2020 | CCF A | 损失 | 高 | 最佳单模 48.2 AP（COCO test-dev） | 低 |
| 3 | GFL V2 (DGQP) | NeurIPS 2021 | CCF A | 损失+小模块 | 中 | ~+1 AP（vs GFLV1，近零开销） | 中 |
| 4 | TOOD (T-Head+TAL) | ICCV 2021 Oral | CCF A | 头+标签分配 | 低(头)/内置(TAL) | 51.1 AP 单模单尺度 | 高/零 |
| 5 | PAA | ECCV 2020 | CCF B | 标签分配+IoU分支 | 中 | 最佳单模 49.0 AP；密集簇 recall 最高 | 中 |
| 6 | NWD / NWD-RKA | arXiv 2021 + ISPRS J P&RS 2022 | JCR Q1 | 损失/度量/分配 | 高 | AI-TOD-v2 +4.3 AP（vs SOTA） | 低 |
| 7 | RFLA | ECCV 2022 | CCF B | 标签分配 | 中 | AI-TOD +4.0 AP（vs SOTA） | 中 |
| 8 | DCFL | CVPR 2023 | CCF A | 标签分配+课程 | 中低 | AI-TOD-R +2.9~6.7 AP | 高 |
| 9 | OHEM | CVPR 2016（窗口外/奠基） | CCF A | 训练策略 | 高 | VOC 一致提升（~2 点 mAP 量级） | 低 |
| 10 | Cal-DETR（置信度校准） | NeurIPS 2023 | CCF A | 训练策略/校准 | 低（DETR 专用，概念迁移） | ECE 改善 5.4%/7.6%，AP 不降 | 高 |
| 11 | Simple Copy-Paste | CVPR 2021 | CCF A | 增强 | 高 | +1.5 box AP；LVIS 稀有类 +3.6 | 中 |
| 12 | CrowdAug（拥挤 Copy-Paste） | AAAI 2023 | CCF A | 增强 | 高 | +2.2/+3.3 AP（拥挤基准） | 中 |
| 13 | OTA / SimOTA | CVPR 2021 | CCF A | 标签分配 | 高 | FCOS-R50 1x 40.7 AP；拥挤场景优势 | 低中 |
| 14 | DSLA | Pattern Recognition 2022 | CCF B / JCR Q1 | 标签分配（软标签） | 中 | COCO 一致提升；NanoDet-Plus 采用 | 中 |
| 15 | SNIP / SNIPER 多尺度训练 | CVPR 2018 / NeurIPS 2018（窗口外/奠基） | CCF A | 训练策略 | 高 | SNIPER COCO 47.6 mAP | 高 |
| 16 | Sugarcane-YOLO（农业实证） | Agronomy 2024 | JCR Q1 | 配方实证 | 高 | 蔗芽识别准确率 97.42% | — |
| 17 | PSDS-YOLOv8 麦穗（农业实证） | Frontiers in Plant Science 2025 | JCR Q1 | 配方实证 | 高 | 密集小麦穗漏检/误检改善 | — |
| 18 | ADL-YOLOv8 杂草（农业实证） | Agronomy 2024 | JCR Q1 | 配方实证 | 高 | R +2.45%，mAP50 +3.07% | — |

---

## 1. Varifocal Loss (VFL) / VarifocalNet（IACS 打分）

- 方法名：Varifocal Loss（VarifocalNet: An IoU-aware Dense Object Detector）
- 出处：CVPR 2021（**Oral**）；论文标题 "VarifocalNet: An IoU-aware Dense Object Detector"（Haoyang Zhang, Ying Wang, Feras Dayoub, Niko Sünderhauf）
- 等级：CCF A
- 核心机制：训练分类分支直接预测 IACS（IoU-Aware Classification Score）= 目标存在性与定位质量的联合表示；正样本标签用预测框与 GT 的 IoU（软标签），负样本用 focal 形式非对称降权（α=0.75, γ=2.0）。
- 对 B1 的作用机理：B1 的本质是"框定位尚可（IoU≥0.5）但分类分数被系统性压低（中位数 0.123）"。VFL 把分类目标从 0/1 改为真实 IoU，使"定位中等但确实命中"的框获得 0.5–0.8 量级的监督目标而非被当作负样本压低——直接对症低分漏检；同时密集重叠场景下高质量框与低质量框的分数排序更准，减少 NMS 前排序错误。Oracle 实验（仅修打分 +4.58 AP50）表明打分修复空间巨大，VFL 正是打分修复的首选插件。
- 场景实证：IEEE TGRS 低空航拍"非刚性小行人密集检测"论文（xplorestaging.ieee.org/document/9779100）明确以 VFL 替换 focal loss 用于低空航拍密集小目标，报告性能提升；mmdetection 生态广泛复现。
- YOLO 集成形态：损失（替换检测头 cls 分支的 BCE；需由预测框在线计算 IoU 软标签）
- 预训练兼容性：高（插件式损失，头部结构不变，COCO 权重可直接迁移微调）
- 预期收益指标与量级：原文 COCO 上 VFNet 较 FCOS+ATSS 基线一致提升 ~2.0 AP（不同 backbone）；最佳单模单尺度 55.1 AP（test-dev）。动机实验中用 gt_IoU 作分类目标可把排序性能从 56.1 提到 74.7 AP，说明"打分修复"上限高。
- 实现成本：低。Ultralytics 生态已有多个 VFL 移植（YOLOv5/8 社区版），核心是 ~50 行损失函数 + IoU 目标计算；需注意与 DFL/TAL 的配合。
- 参考链接：
  - arXiv: https://arxiv.org/abs/2008.13367
  - CVPR 2021 open access: https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.pdf
  - 代码: https://github.com/hyz-xmaster/VarifocalNet

## 2. Generalized Focal Loss（Quality Focal Loss + Distribution Focal Loss）

- 方法名：Generalized Focal Loss（QFL + DFL）
- 出处：**NeurIPS 2020**（任务书原写 ECCV 2020，经 proceedings.neurips.cc 核实为 NeurIPS 2020）；论文标题 "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection"（Xiang Li 等）
- 等级：CCF A
- 核心机制：QFL 把分类分支训练为预测"分类×定位质量"的连续值（软标签=预测框 IoU），统一训练与推理时的打分口径；DFL 把框回归建模为离散分布学习，缓解模糊边界的不确定性。
- 对 B1 的作用机理：与 VFL 同族但更温和：QFL 让所有候选框（含密集重叠产生的中等质量框）都获得与其真实 IoU 成正比的连续监督，避免硬 0/1 标签把"命中但不够完美"的框压成低分——正对 94.7% 低分漏检。DFL 对小目标模糊边界的回归更稳，间接提高预测框 IoU，从而抬高打分目标。注意：Ultralytics YOLOv8+ 已内置 DFL，缺的是 QFL 这一半。
- 场景实证：CVPR 2023 Workshop 的高粱蚜虫密集簇检测数据集研究（arXiv 2307.05929）在密集簇场景对 VFNet/GFLV2/PAA/ATSS 做了对比（见 §5）；GFL 系列是遥感小目标 YOLO 改进论文中最常被引用的打分方案之一。
- YOLO 集成形态：损失（cls 分支目标改为 IoU 软标签；DFL 已内置）
- 预训练兼容性：高（纯损失级插件）
- 预期收益指标与量级：原文最佳单模 48.2 AP（COCO test-dev，大 backbone；该数字引自 TOOD 论文对比表）；GFL 是后续 GFLV2/TOOD 的基线。
- 实现成本：低。在 Ultralytics 中把 cls 目标从 one-hot 换成 bbox IoU 软标签并改写 BCE 为 QFL 形式；与内置 DFL 天然兼容。
- 参考链接：
  - arXiv: https://arxiv.org/abs/2006.04388
  - NeurIPS 2020 论文: https://proceedings.neurips.cc/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf

## 3. Generalized Focal Loss V2（DGQP 定位质量估计）

- 方法名：GFL V2（Distribution-Guided Quality Predictor）
- 出处：NeurIPS 2021；论文标题 "Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection"（Xiang Li, Wenhai Wang, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang）
- 等级：CCF A
- 核心机制：发现回归分支学到的框分布统计量（DFL 输出的尖锐/平坦程度）与真实定位质量高度相关；用极轻量的 DGQP 从分布统计量预测 LQE（定位质量），与分类分数相乘得到最终打分。
- 对 B1 的作用机理：B1 中"框已出（IoU≥0.5）但分数低"的样本，其回归分布通常是尖锐的（定位确定），DGQP 会给它们较高的质量分，从而把最终打分拉回合理区间——这是在不改分类分支的前提下修复打分的第二条路径，与 QFL/VFL 互补（可叠加）。对密集重叠场景，LQE 还能帮助 NMS 保留定位更好的框。
- 场景实证：蚜虫密集簇检测研究（arXiv 2307.05929，CVPR 2023 Workshop）报告 GFLV2 在密集簇上 recall 79.2（原始设置），低于 PAA 的 84.1，但仍显著高于普通分类器；官方仓库确认在 COCO 上 ~+1 AP。
- YOLO 集成形态：损失 + 插件式小模块（DGQP：1–2 层卷积/MLP，输入为 DFL 回归分布统计量）
- 预训练兼容性：中（新增小模块随机初始化，但主干/颈部/头主体权重保留；YOLO 已输出 DFL 分布，取统计量方便）
- 预期收益指标与量级：官方仓库与论文报告较 GFLV1 提升 ~1 AP、几乎无额外计算开销（COCO，多 backbone 一致）。
- 实现成本：中。需从 reg 分支抽取分布统计量并接一个小 DGQP 头，推理时乘到置信度上；改动集中在检测头 forward 与 loss。
- 参考链接：
  - arXiv: https://arxiv.org/abs/2011.12885
  - NeurIPS 2021 openreview: https://openreview.net/forum?id=YHs84VHQHj
  - 代码: https://github.com/YunhaoLee/GFocalV2

## 4. TOOD: Task-aligned One-stage Object Detection（T-Head + TAL）

- 方法名：TOOD（Task-aligned Head + Task Alignment Learning）
- 出处：ICCV 2021（**Oral**）；论文标题 "TOOD: Task-aligned One-stage Object Detection"（Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R. Scott, Weilin Huang）
- 等级：CCF A
- 核心机制：T-Head 用任务交互层让分类/回归共享对齐表示；TAL 用 task-aligned 度量（cls_score^α × IoU^β）动态选正样本并施加对齐损失，消除分类与定位两任务最优锚点的空间错位。
- 对 B1 的作用机理：低分漏检的一个隐含成因是"分类分支看到的特征位置与回归分支最优位置错位"，密集小目标尤甚。TAL 按分类×定位联合质量选正样本，保证被训练为正的样本本身就是"能定位好"的样本，从源头减少"框好但分低"的矛盾样本。注意：**Ultralytics YOLOv8/YOLO11/YOLO12 的 TaskAlignedAssigner 即源自 TOOD 的 TAL**——本项目已在使用其分配思想，因此增量空间主要在 T-Head（需换头，兼容性差）或调整 TAL 的 α/β 与 topk（零成本可调）。
- 场景实证：TOOD 思想已被 YOLO 系列大规模工程验证（Ultralytics 默认分配器）；另有跨领域应用（如 arXiv 2503.16483 太阳射电暴检测采用 TOOD 结构）。
- YOLO 集成形态：标签分配（TAL，已内置）| 模块（T-Head，需换头）
- 预训练兼容性：T-Head 低（头部重构，破坏迁移）；TAL 调参零成本（已内置）
- 预期收益指标与量级：原文单模单尺度 51.1 AP（COCO），大幅超过 ATSS 47.7 / GFL 48.2 / PAA 49.0（同为大 backbone 设定）。
- 实现成本：T-Head 高（不建议，违反项目约束）；TAL 超参（topk、α/β、阈值）扫描低成本，可作为零结构改动的调优项。
- 参考链接：
  - arXiv: https://arxiv.org/abs/2108.07755
  - ICCV 2021 open access: https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html
  - 代码: https://github.com/fcjian/TOOD

## 5. PAA: Probabilistic Anchor Assignment with IoU Prediction

- 方法名：PAA（Probabilistic Anchor Assignment + IoU 预测分支）
- 出处：ECCV 2020；论文标题 "Probabilistic Anchor Assignment with IoU Prediction for Object Detection"（Kang Kim, Hee Seok Lee）
- 等级：CCF B
- 核心机制：按模型当前学习状态对锚点分数拟合概率分布（GMM），以概率方式自适应划分正负样本；额外预测检测框 IoU 作为定位质量，与分类分相乘用于 NMS 选框，弥合训练/测试目标差异。
- 对 B1 的作用机理：两条通路都对症：(a) 概率化分配让"当前模型还没学好"的难样本（低分但实为 GT 的小密目标）更可能留在正样本集合继续被训练，而不是被永久判负→缓解系统性低估；(b) IoU 预测分支提供独立于分类分支的定位质量分，推理时组合打分可把"框好分低"的候选拉回。蚜虫密集簇实证中 PAA 的 recall 显著高于同族方法（见下），支持其在密集簇场景的召回优势。
- 场景实证：高粱蚜虫密集簇检测数据集研究（arXiv 2307.05929，部分内容发表于 CVPR 2023 Workshops）：密集簇原始设置下 **PAA recall 84.1 vs ATSS 80.0、VFNet 80.4、GFLV2 79.2**；在簇合并后 PAA recall 达 87.6→98.4，均为四者最高——密集簇召回最强的打分/分配方案之一。
- YOLO 集成形态：标签分配（概率化正负划分）+ 模块（新增一个 IoU 预测卷积层，原文称仅加单层卷积）
- 预训练兼容性：中（分配策略为训练期插件；IoU 分支为新增小卷积，主干/颈部权重保留）
- 预期收益指标与量级：原文在 COCO test-dev 上以多种 backbone 刷新单阶段检测器记录；TOOD 论文对比表中 PAA 最佳单模 49.0 AP。
- 实现成本：中。需实现 GMM 分配 + IoU 预测头 + 组合打分；mmdetection 有完整参考实现。
- 参考链接：
  - arXiv: https://arxiv.org/abs/2007.08103
  - ECCV 2020 论文 PDF: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700358.pdf
  - 代码: https://github.com/kkhoot/PAA

## 6. NWD: Normalized Gaussian Wasserstein Distance（含 NWD-RKA，ISPRS 2022）

- 方法名：NWD（Normalized Gaussian Wasserstein Distance）；扩展版 NWD-RKA（RanKing-based Assigning）
- 出处：
  - 原始版：arXiv 2021，"A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"（Jinwang Wang, Chang Xu, Wen Yang, Huai Yu, Lei Yu, Gui-Song Xia），arXiv:2110.13389
  - 扩展版（正式发表）：**ISPRS Journal of Photogrammetry and Remote Sensing, 2022, 190: 79–93**，"Detecting tiny objects in aerial images: A normalized Wasserstein distance and a new benchmark"，DOI: 10.1016/j.isprsjprs.2022.06.002（arXiv:2206.13996）
- 等级：ISPRS J P&RS 为遥感顶刊，JCR Q1（IF≈10）
- 核心机制：把边界框建模为二维高斯分布，用归一化 Wasserstein 距离度量框相似度，替代对微小偏移极度敏感的 IoU；可嵌入损失、标签分配与 NMS。扩展版配套 RKA 按 NWD 排序选正样本。
- 对 B1 的作用机理：甘蔗苗在 640 切片下属小目标，IoU 对 1–2 像素偏移剧变→训练时小目标正样本被 IoU 阈值"误杀"为负样本，分类分支学到"这类框不该给高分"，推理即低分漏检。NWD 对小偏移平滑，保证小目标的正样本监督不流失；用于损失时回归梯度也更平滑。这是从"度量层面"修复小目标监督不足，与 VFL/QFL 的"打分层面"修复正交可叠加。
- 场景实证：ISPRS 2022 版在 AI-TOD/AI-TOD-v2 等 4 个航拍微小目标数据集上"一致且大幅提升"；DetectoRS+NWD-RKA 在 AI-TOD-v2 上较 SOTA **+4.3 AP**；后续被大量遥感/工业小目标工作采用（如 X 射线铸造缺陷检测中以 NWD 替换 CIoU 提升小缺陷精度，PMC11845526）。
- YOLO 集成形态：损失（NWD 替换/混合 CIoU 用于小框）| 标签分配（NWD 阈值替换 IoU 阈值）| 度量（NMS 可选）
- 预训练兼容性：高（纯度量/损失替换，无结构改动）
- 预期收益指标与量级：原文（ISPRS 2022）报告在四个航拍数据集上一致提升，DetectoRS 设定 +4.3 AP（AI-TOD-v2 vs SOTA）；社区在 VisDrone/DOTA 类任务普遍报告 +1~3 AP（小目标子集更明显）。
- 实现成本：低。NWD 公式为闭式高斯 Wasserstein 距离（~30 行），可只对小于阈值的框启用（混合 IoU+NWD），风险小。
- 参考链接：
  - arXiv（原始）: https://arxiv.org/abs/2110.13389
  - arXiv（扩展）: https://arxiv.org/abs/2206.13996
  - DOI: https://doi.org/10.1016/j.isprsjprs.2022.06.002
  - 代码: https://github.com/jwwangchn/NWD ；https://github.com/Chasel-Tsui/mmdet-aitod

## 7. RFLA: Gaussian Receptive Field based Label Assignment

- 方法名：RFLA（Receptive Field Distance + Hierarchical Label Assignment）
- 出处：ECCV 2022（LNCS pp. 526–543）；论文标题 "RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection"（Chang Xu, Jinwang Wang, Wen Yang, Huai Yu, Lei Yu, Gui-Song Xia）
- 等级：CCF B
- 核心机制：利用特征感受野近似高斯分布的先验，提出 RFD（Receptive Field Distance）度量高斯感受野与 GT 的相似度替代 IoU/中心采样；HLA 分层分配，纠正 IoU 阈值与中心采样对大目标的偏置，让小目标获得均衡正样本。
- 对 B1 的作用机理：B1 的小目标 FN 率 46%（大目标 9.5%）说明小目标在训练期被系统性欠监督。RFLA 直接解决"现有分配策略产生大量 outlier 小目标、检测器对小目标关注不足"的问题——给小目标足够且合理的正样本，分类分支才有机会学到"该给高分"。与 NWD 同属小目标监督修复，但作用在分配环节。
- 场景实证：原文在 AI-TOD 等 4 个数据集验证，**AI-TOD 上较 SOTA +4.0 AP**；后续被 SAR 舰船检测（HCA-RFLA, Electronics 2024）、SR-TOD（arXiv 2405.11276，AP 39.0）等遥感微小任务采用。
- YOLO 集成形态：标签分配（自定义 assigner，替换 TaskAlignedAssigner 或与 SimOTA 结合）
- 预训练兼容性：中（训练期插件，无结构改动；但需重写分配器并调参）
- 预期收益指标与量级：原文 AI-TOD +4.0 AP（vs SOTA）；消融显示对 tiny 子集增益最大。
- 实现成本：中。需实现 RFD 计算 + 分层分配逻辑；mmdetection 有官方实现（Chasel-Tsui/mmdet-rfla）可移植。
- 参考链接：
  - arXiv: https://arxiv.org/abs/2208.08738
  - 代码: https://github.com/Chasel-Tsui/mmdet-rfla （另见 Small-Object-Detection/mmdet-RFLA）

## 8. DCFL: Dynamic Coarse-to-Fine Learning（面向微小目标的难样本课程学习）

- 方法名：DCFL（Dynamic Prior + Coarse-to-Fine Sample Learning）
- 出处：CVPR 2023；论文标题 "Dynamic Coarse-to-Fine Learning for Oriented Tiny Object Detection"（Chang Xu, Jian Ding, Jinwang Wang, Wen Yang, Huai Yu, Lei Yu, Gui-Song Xia）
- 等级：CCF A
- 核心机制：针对微小目标训练管线的 mismatch/imbalance：动态更新先验位置；粗阶段用 GJSD 保证每个 GT 获得充足多样的正样本，细阶段用动态高斯混合模型约束剔除低质量样本，实现由粗到细的难样本课程学习。
- 对 B1 的作用机理：与 B1 证据链高度吻合——"置信的目标越来越置信，脆弱的小目标被进一步边缘化"（其后继 TPAMI 工作明确命名为 learning bias）。DCFL 的粗阶段强制给脆弱小目标正样本配额，正是对"小目标系统性欠监督→低分漏检"的训练期修复。
- 场景实证：航拍定向微小目标数据集 AI-TOD-R 官方基准：S²A-Net+DCFL 较基线 **+2.9 AP（1x）/ +6.7 AP（40e）**，AP50 +6.3/+16.2（项目页 chasel-tsui.github.io/AI-TOD-R 表格）。
- YOLO 集成形态：标签分配 + 训练策略（课程式两阶段分配）
- 预训练兼容性：中低（为定向框设计，迁移到水平框需简化改造；训练流程改动较大）
- 预期收益指标与量级：如上，AI-TOD-R 上 +2.9~6.7 AP，AP50 增益尤为显著（与本项目以 AP50 为主要指标一致）。
- 实现成本：高。动态先验+两阶段分配+高斯混合约束，工程量大；建议只借鉴其"粗阶段保底正样本配额"思想做简化版。
- 参考链接：
  - arXiv: https://arxiv.org/abs/2304.08876
  - CVPR 2023 open access: https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Dynamic_Coarse-To-Fine_Learning_for_Oriented_Tiny_Object_Detection_CVPR_2023_paper.pdf
  - 基准页: https://chasel-tsui.github.io/AI-TOD-R/

## 9. OHEM: Online Hard Example Mining（窗口外/奠基，任务书指定主题）

- 方法名：OHEM（Online Hard Example Mining）
- 出处：CVPR 2016，"Training Region-based Object Detectors with Online Hard Example Mining"（Abhinav Shrivastava, Abhinav Gupta, Ross Girshick），pp. 761–769，DOI: 10.1109/CVPR.2016.87
- 等级：CCF A（注：2016 年，超出 2019–2026 窗口，作为难例加权训练策略的奠基工作收录）
- 核心机制：每个 batch 用当前网络前向计算所有候选的损失，按损失排序选取难例（高损失样本）反传，自动聚焦难样本、免手工启发式。
- 对 B1 的作用机理：B1 的难样本画像明确（低分漏检框、小目标、高密度、重叠区），这些样本在普通训练中被海量简单负样本/简单正样本稀释。OHEM 强制把梯度预算花在高损失样本上——其中就包括"被错误压低的真目标"（分类损失大的正样本），有助于纠正系统性低估。注意与 focal loss 的区别：focal 是静态降权易例，OHEM 是在线动态选难例，二者可叠加。
- 场景实证：密集微小害虫检测 Pest-YOLO（PMC9783619）等农业密集场景工作均以 OHEM 为难例处理基线并讨论其不足（忽略易样本、对噪声标签敏感）；大量遥感小目标 YOLO 改进论文将 OHEM 作为训练策略组件。
- YOLO 集成形态：训练策略（batch 内按损失选难例；对 anchor-free 密集头可按锚点/样本级损失实现）
- 预训练兼容性：高（纯训练过程策略，零结构改动）
- 预期收益指标与量级：原文在 PASCAL VOC 与 COCO 上对 Fast R-CNN/Faster R-CNN 一致提升（约 2 个 mAP 点量级，多 backbone 一致）；增益依赖数据中难例比例——本项目难例占比高（94.7% FN 为低分漏检），预期收益条件好。
- 实现成本：低。Ultralytics 训练循环中加一次前向+损失排序选样即可；注意显存与噪声标签风险（可配合置信度阈值过滤）。
- 参考链接：
  - arXiv: https://arxiv.org/abs/1604.03540
  - CVPR 2016: https://doi.org/10.1109/CVPR.2016.87

## 10. Cal-DETR: Calibrated Detection Transformer（置信度校准；含 EDL 现状说明）

- 方法名：Cal-DETR（不确定性引导的 logit 调制 + logit mixing）
- 出处：NeurIPS 2023；论文标题 "Cal-DETR: Calibrated Detection Transformer"（Muhammad Akhtar Munir, Muhammad Haris Khan, Mohsen Ali 等）
- 等级：CCF A
- 核心机制：用不同 decoder 层输出的方差量化每个 logit 的不确定性，对高不确定性 logit 降尺度调制；辅以检测专用 mixup 正则（logit mixing），训练期校准，使预测概率与真实正确概率对齐。
- 对 B1 的作用机理：B1 的"分数中位数 0.123 但框 IoU≥0.5"是典型的置信度欠校准（系统性低估）。校准类方法不改变检出框集合、只修分数——与 Oracle 实验（仅修打分 +4.58 AP50）思路一致。Cal-DETR 证明"训练期校准可以在不降 AP 的前提下显著改善置信度质量"，其"不确定性高→压低分数、不确定性低→放行"的调制思想可概念迁移到 YOLO（例如用分类/回归分支的一致性、或多次增广预测的方差作为不确定性代理）。
- 场景实证：无农业实证；原文在 COCO 域内与 4 个域外场景验证，**校准（ECE）改善 5.4%（域内）/7.6%（域外），检测 AP 保持或提升**。
- YOLO 集成形态：训练策略/损失（校准正则 + logit 调制；需为 YOLO 设计不确定性代理）
- 预训练兼容性：低-中（Cal-DETR 本体依赖 DETR 多层 decoder 方差，不能直接用于 CNN 检测头；仅概念迁移）
- 预期收益指标与量级：原文以 ECE 相对改善 5.4–7.6% 为主指标，AP 不降；对 B1 的价值在于"修分数不改框"，潜在收益参照 Oracle +4.58 AP50 的上界。
- 实现成本：高（需为 YOLO 设计并验证不确定性代理与调制机制，无现成移植）。
- EDL 现状说明：任务主题含"证据深度学习（EDL）用于检测置信度"。经核实：E-DETR（EDL+DETR）为 **ICLR 2025 撤稿投稿**，无正式出处，丢弃；CertainNet（RA-L 2021）为采样-free 检测不确定性，但发表于机器人信刊、不在任务限定刊单，仅列此备查；EDL 综述见 TPAMI 2026（DOI 见 computer.org/csdl/journal/tp/2026/03/11217233）。在限定来源内，检测置信度校准的可靠代表即 Cal-DETR。
- 参考链接：
  - arXiv: https://arxiv.org/abs/2311.03570
  - NeurIPS 2023 论文: https://proceedings.nips.cc/paper_files/paper/2023/file/e271e30de7a2e462ca1f85cefa816380-Paper-Conference.pdf
  - 代码: https://github.com/akhtarvision/cal-detr

## 11. Simple Copy-Paste（实例级复制粘贴增强）

- 方法名：Simple Copy-Paste
- 出处：CVPR 2021；论文标题 "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"（Golnaz Ghiasi, Yin Cui, Aravind Srinivas, 等）
- 等级：CCF A
- 核心机制：把其他图像中的目标实例（含 mask）随机粘贴到当前图像，自动生成新训练样本；粘贴比例与规模随机化。
- 对 B1 的作用机理：B1 的密度效应（>12 株/图 FN 率 27.1% vs 稀疏 8.7%）说明模型在高密度构型上训练不足。Copy-Paste 可人为提高每张图的实例数与密集度，直接增加"密集小目标"正样本供给；同时粘贴产生的新上下文能降低分类分支对特定背景的依赖，缓解系统性低估。对重叠场景，可通过控制粘贴重叠率制造 [0.5,0.7) IoU 的重叠对（恰是 FN 率 41.8% 的区间）。
- 场景实证：原文在 LVIS 稀有类别上 **+3.6 mask AP**（稀有≈样本不足，与甘蔗苗密集难例的稀缺性类似）；农业应用实例：SAM+Copy-Paste 用于小数据集同色背景下青果检测（PMC11011402），验证其在农业实例检测中的可迁移性。
- YOLO 集成形态：增强（训练数据管线）
- 预训练兼容性：高（零结构改动）
- 预期收益指标与量级：原文 COCO 达 57.3 box AP / 49.1 mask AP，较前 SOTA **+1.5 box AP / +0.6 mask AP**；LVIS 稀有类 +3.6 mask AP；增益与粘贴实例数正相关。
- 实现成本：中。需要实例 mask（或用框内裁剪近似）；Ultralytics 已支持 copy-paste（参数化），主要成本在调粘贴策略（数量/尺度/重叠率）以匹配甘蔗苗密集分布。
- 参考链接：
  - arXiv: https://arxiv.org/abs/2012.07177
  - CVPR 2021 open access: https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html
  - DOI: https://doi.org/10.1109/CVPR46437.2021.00294

## 12. CrowdAug: Improving Crowded Object Detection via Copy-Paste

- 方法名：CrowdAug（面向拥挤场景的 Copy-Paste）
- 出处：AAAI 2023；论文标题 "Improving crowded object detection via copy-paste"（AAAI v37i1, 2023）
- 等级：CCF A
- 核心机制：指出拥挤场景两大病因——ICD（IoU-Confidence correlation Disturbance，IoU 与置信度的相关性被拥挤扰乱）与 CDD（Confused De-Duplication，NMS 去重混淆）；设计专门制造拥挤场景的 copy-paste 方案，让检测器在训练期见到并学会拥挤构型。
- 对 B1 的作用机理：这是与 B1 机理匹配度最高的增强工作：B1 的 GT-GT 重叠 [0.5,0.7) 区 FN 率 41.8% 正是"拥挤扰乱 IoU-置信度相关性"的表现——重叠目标被分类分支系统性压低。CrowdAug 通过可控粘贴制造重叠拥挤样本，使分类分支学到"重叠≠背景/低分"，直接修复 ICD；同时缓解 CDD 带来的训练噪声。
- 场景实证：原文在拥挤行人基准（CityPersons/CrowdHuman 类）上较无增强强基线 **Faster R-CNN +2.2 AP、RetinaNet +3.3 AP**，并显著提升 CrowdDet/ProgS-RCNN 等 SOTA 拥挤检测器；无直接农业实证，但甘蔗苗行内拥挤与其场景同构。
- YOLO 集成形态：增强（拥挤导向的粘贴策略）
- 预训练兼容性：高（零结构改动）
- 预期收益指标与量级：+2.2~3.3 AP（拥挤基准，原文报告）；对拥挤子集的增益大于全集——与本项目"密度>12 株/图"子集预期一致。
- 实现成本：中。在 Copy-Paste 基础上增加拥挤构型采样（按现有目标位置粘贴、控制重叠 IoU 分布）；arXiv 版有实现细节（2211.12110）。
- 参考链接：
  - DOI: https://doi.org/10.1609/aaai.v37i1.25124
  - arXiv: https://arxiv.org/abs/2211.12110

## 13. OTA / SimOTA: Optimal Transport Assignment

- 方法名：OTA（Optimal Transport Assignment）；工程简化版 SimOTA（YOLOX）
- 出处：CVPR 2021；论文标题 "OTA: Optimal Transport Assignment for Object Detection"（Zheng Ge, Songtao Liu, Zeming Li, Osamu Yoshie, Jian Sun）
- 等级：CCF A
- 核心机制：把标签分配建模为最优传输问题：全局视角下以"分类+回归损失加权和"为传输代价，求解锚点（需求方）到 GT（供给方）的全局最优分配，避免逐 GT 独立分配产生的歧义与冲突。
- 对 B1 的作用机理：密集重叠场景下逐 GT 独立分配常把同一锚点重复分配或把难锚点漏分，导致监督冲突→分类分支学到矛盾信号→低分。OTA 的全局最优分配消除歧义锚点，"提高训练数据利用率"，对拥挤场景尤其有效（原文在 CrowdHuman 验证）。SimOTA 是其轻量近似（动态 topk + 代价矩阵），已成为 YOLO 系事实标准之一。
- 场景实证：原文在 COCO 与 **CrowdHuman 拥挤场景**验证，"尤其在拥挤场景展现优势"；YOLOX（arXiv 2107.08430 技术报告）以 SimOTA 为默认分配并达成 51.2 AP（YOLOX-X）。
- YOLO 集成形态：标签分配（训练期插件）
- 预训练兼容性：高（无结构改动；Ultralytics 已内置相近的 TaskAlignedAssigner，可对比/替换为 SimOTA 变体）
- 预期收益指标与量级：原文 FCOS-ResNet-50 1x 达 **40.7 AP**，超过当时所有分配方法；消融显示对拥挤/歧义场景增益最大。
- 实现成本：低-中。SimOTA 实现成熟（YOLOX/ByteTrack 生态、mmdetection 均有），主要工作在超参（topk、代价权重）。
- 参考链接：
  - arXiv: https://arxiv.org/abs/2103.14259
  - CVPR 2021: https://www.computer.org/csdl/proceedings-article/cvpr/2021/450900a303/1yeM4Che0cU
  - 代码: https://github.com/Megvii-BaseDetection/OTA ；SimOTA 见 YOLOX: https://arxiv.org/abs/2107.08430

## 14. DSLA: Dynamic Smooth Label Assignment

- 方法名：DSLA（Dynamic Smooth Label Assignment，动态平滑标签分配）
- 出处：Pattern Recognition, 2022, 129: 108868；论文标题 "DSLA: Dynamic smooth label assignment for efficient anchor-free object detection"（Hu Su, Yonghao He, Rui Jiang, Jiabin Zhang, Wei Zou, Bin Fan）
- 等级：CCF B / JCR Q1（Pattern Recognition, IF≈7.5）
- 核心机制：基于 FCOS centerness 思想，把正负样本标签从硬 0/1 平滑为 [0,1] 连续值，并按模型状态动态调整；用 IoU 感知权重让正负样本间过渡平稳，缓解 anchor-free 检测中分配不一致导致的 NMS 误抑制。
- 对 B1 的作用机理：硬标签把"接近正样本边界"的候选（密集重叠下大量存在）一刀切为负样本，是低分漏检的训练期根源之一。平滑标签给这些边界样本与质量成正比的软监督，分类分支输出的分数谱更连续，减少"好框被压到低分段"的现象；动态性保证随训练推进标准自适应收紧。
- 场景实证：原文在 COCO 上对 FCOS 等 anchor-free 检测器一致提升；被轻量检测器 NanoDet-Plus 采用（与 AGM 辅助模块配合）作为默认分配策略。
- YOLO 集成形态：标签分配（软标签 + 动态权重；需修改分类目标生成）
- 预训练兼容性：中（训练期插件，无结构改动，但需改目标生成与损失）
- 预期收益指标与量级：原文报告在 MS COCO 上对 anchor-free 基线一致提升 AP（多模型验证）；NanoDet-Plus 凭该策略在轻量设定下达到 COCO 30.4 mAP。
- 实现成本：中。核心是软标签生成逻辑 + 与现有损失的对接；arXiv 版（2208.00817）与社区实现（Liyi4578/PDSLA）可参考。
- 参考链接：
  - DOI: https://doi.org/10.1016/j.patcog.2022.108868
  - arXiv: https://arxiv.org/abs/2208.00817

## 15. SNIP / SNIPER: 多尺度/高分辨率训练策略（窗口外/奠基，任务书指定主题）

- 方法名：SNIP（Scale Normalization for Image Pyramids）与 SNIPER（高效多尺度训练）
- 出处：
  - SNIP：CVPR 2018，"An Analysis of Scale Invariance in Object Detection"（Bharat Singh, Larry S. Davis）
  - SNIPER：NeurIPS 2018，"SNIPER: Efficient Multi-Scale Training"（Bharat Singh, Mahyar Najibi, Larry S. Davis）
- 等级：CCF A（注：2018 年，超出 2019–2026 窗口，作为高分辨率/多尺度训练主题的奠基工作收录）
- 核心机制：SNIP 在图像金字塔上训练、按目标尺度选择性反传（不同尺度只学对应尺寸的目标），消除尺度间样本不平衡；SNIPER 进一步只在目标周围按合适尺度采样 context chip，兼顾高分辨率训练与效率。
- 对 B1 的作用机理：本项目已试"推理期 imgsz960"为负收益，说明瓶颈不在推理分辨率而在训练期尺度分布：训练时 640 切片内小目标像素占比低、特征弱，分类分支学不到判别性→低分。多尺度/高分辨率训练（在更大尺度上训练小目标、或对切片做 2× 上采样训练）直接增加小目标的有效像素与正样本质量。SNIP/SNIPER 的"尺度归一化"思想可简化为：训练时随机多尺度（如 640/800/960 切片）+ 按尺度加权采样密集图。
- 场景实证：原文以 COCO 为主（SNIPER 达 47.6 mAP，Faster R-CNN-ResNet101，单 GPU 5 img/s）；UAV 小目标社区普遍沿用多尺度训练作为标配（如 Drones 2025 高分辨率预测层工作报告 mAP50 +2.31、Recall +1.23，mdpi.com/2504-446X/10/8/624，属辅助佐证）。
- YOLO 集成形态：训练策略（多尺度训练、尺度加权采样、切片尺度混合）
- 预训练兼容性：高（纯训练管线改动）
- 预期收益指标与量级：SNIPER 在 COCO 达 47.6 mAP 且训练效率大幅提升；原文结论"高分辨率/多尺度训练对小目标 AP 提升显著"。对本项目预期主要体现在小目标子集 Recall。
- 实现成本：高（完整 SNIPER chip 采样管线重）；简化版（多尺度训练 + 大尺度切片混合）成本低-中，建议先做简化版。
- 参考链接：
  - SNIPER arXiv: https://arxiv.org/abs/1805.09300
  - SNIPER NeurIPS 2018: https://proceedings.neurips.cc/paper/2018 （sniper_efficient_multiscale_training）

## 16. 农业实证：Sugarcane-YOLO（甘蔗种芽，Agronomy 2024）

- 方法名：Sugarcane-YOLO（YOLOv8s + SimAM + SPD-Conv + E-IoU + P2 小目标层）
- 出处：Agronomy, 2024, 14(10): 2412；论文标题 "Sugarcane-YOLO: An Improved YOLOv8 Model for Accurate Identification of Sugarcane Seed Sprouts"
- 等级：JCR Q1（MDPI Agronomy）
- 核心机制：颈部加 SimAM 注意力、尾部卷积换 SPD-Conv（保留小目标空间细节）、回归损失换 E-IoU、加 P2 小目标检测层。
- 对 B1 的作用机理（实证参考价值）：与本项目同域（甘蔗、苗/芽级小目标）。其配方中 SPD-Conv 与 E-IoU 属插件级、预训练友好；但其 P2 层与本项目已证伪的 P2 方案冲突，提示同域论文中"结构加法"未必可迁移，应优先借鉴其损失/卷积级组件。
- 场景实证：自建甘蔗种芽数据集上优于主流模型、速度-精度平衡最佳（论文报告识别准确率约 97.4%）。
- YOLO 集成形态：配方实证（组件可拆借：损失 / 模块）
- 预训练兼容性：高（组件均为插件式）
- 预期收益指标与量级：原文报告较 YOLOv8s 基线显著提升（以准确率/速度平衡为主指标）。
- 实现成本：低-中（组件均有现成实现）。
- 参考链接：
  - DOI: https://doi.org/10.3390/agronomy14102412

## 17. 农业实证：PSDS-YOLOv8（UAV 麦穗密集检测，Frontiers in Plant Science 2025）

- 方法名：PSDS-YOLOv8（P2 + SPD-Conv + DySample + SCAM）
- 出处：Frontiers in Plant Science, 2025, 16: 1536017；论文标题 "A lightweight wheat ear counting model in UAV images based on improved YOLOv8"
- 等级：JCR Q1（Frontiers in Plant Science）
- 核心机制：针对 UAV 麦穗"密集分布、小尺寸、高重叠"导致的漏检/误检，组合 P2 层、SPD-Conv、DySample 上采样与 SCAM 注意力。
- 对 B1 的作用机理（实证参考价值）：场景与 B1 几乎同构（UAV、密集、小、重叠、以漏检为主诉）。该论文证实此类组合在"密集小重叠目标"上能同时压低漏检与误检；其中 DySample 属本项目 AddModules 已有组件，可单独评估其对打分/召回的影响。
- 场景实证：UAV 麦穗数据集（GWHD 类）消融显示各组件对漏检改善有效（原文报告）。
- YOLO 集成形态：配方实证（组件可拆借：模块/上采样器）
- 预训练兼容性：高（插件式组件）
- 预期收益指标与量级：原文报告较 YOLOv8 基线在密集麦穗上显著降低漏检率（具体数值见原文表格）。
- 实现成本：低（DySample/SPD-Conv 均有现成实现）。
- 参考链接：
  - DOI: https://doi.org/10.3389/fpls.2025.1536017

## 18. 农业实证：ADL-YOLOv8（田间杂草，Agronomy 2024）

- 方法名：ADL-YOLOv8（田间作物杂草检测改进 YOLOv8）
- 出处：Agronomy, 2024, 14(10): 2355；论文标题 "ADL-YOLOv8: A Field Crop Weed Detection Model Based on Improved YOLOv8"
- 等级：JCR Q1（MDPI Agronomy）
- 核心机制：针对田间杂草漏检/误检的改进组合（注意力与特征融合类插件）。
- 对 B1 的作用机理（实证参考价值）：报告了明确的 Recall 提升（+2.45%）且 Precision 同升（+2.2%），满足本项目"mAP50 升且 P/R 不同时降"的有效性判据形态，可作为农业密集场景"召回与精度兼得"的参照案例。
- 场景实证：自建田间杂草数据集：**Precision +2.2%、Recall +2.45%、mAP@0.5 +3.07%、mAP@0.95 +1.9%**，同时模型体积 -15.77%、计算量 -10.98%（原文报告）。
- YOLO 集成形态：配方实证
- 预训练兼容性：高（插件式）
- 预期收益指标与量级：见上（mAP50 +3.07 超过本项目 +1.0 阈值，且 R 升幅大于 P）。
- 实现成本：低-中。
- 参考链接：
  - 论文页: https://www.mdpi.com/2073-4395/14/10/2355 （DOI: 10.3390/agronomy14102355）

---

## 附：其他核实过的辅助证据（不单独成条）

- YOLOv5-T（UAV 玉米雄穗检测）：Computers and Electronics in Agriculture, 2024, 219: 108991（JCR Q1），UAV 低空遥感密集雄穗 AP 达 98.70%，证明"密集小目标+UAV"任务上 YOLO 系可达高召回。DOI: https://doi.org/10.1016/j.compag.2024.108991
- 高粱蚜虫密集簇检测数据集研究（CVPR 2023 Workshops + 扩展版）：arXiv 2307.05929。密集簇基准上 PAA recall 84.1 > ATSS 80.0 > VFNet 80.4 > GFLV2 79.2（原始设置），为"分配/打分策略决定密集簇召回"提供直接对照证据。
- Mamba-WheatNet（Scientific Reports 2026, s41598-026-45083-2）：GWHD-2021 UAV 麦穗上报告 accuracy/recall/mAP50 超 YOLOv13 与 RT-DETR，佐证麦穗类密集任务的 recall 竞争态势（辅助参考）。

## B1 优先级建议（基于证据链匹配度）

1. **打分类修复（直接对症 94.7% 低分漏检，Oracle 上界 +4.58 AP50）**：VFL（§1）≈ QFL（§2）> GFLV2-DGQP（§3）。三者均为插件式、预训练兼容高，建议先试 VFL 或 QFL（二选一，避免同时改坏归因），再考虑 DGQP 叠加。
2. **数据侧（密度/重叠构型供给）**：CrowdAug（§12）> Simple Copy-Paste（§11）。CrowdAug 的 ICD 机理与 B1 重叠区证据（[0.5,0.7) FN 41.8%）直接对应。
3. **小目标监督修复（小目标 FN 46%）**：NWD（§6，成本最低）→ RFLA（§7）→ SimOTA 变体（§13）。
4. **训练策略**：OHEM（§9，低成本）与多尺度/大切片训练简化版（§15）。
5. **谨慎项**：TOOD T-Head（§4，换头，违反约束；TAL 调参可做）、Cal-DETR（§10，概念迁移成本高）、DCFL（§8，工程量大，借鉴思想）。
