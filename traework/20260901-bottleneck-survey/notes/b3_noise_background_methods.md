# B3「背景判别与标注噪声」改进方案文献调研

- 调研日期：2026-09-01
- 项目：SSDC-UAV 甘蔗苗无人机检测（Ultralytics YOLO12s，COCO 预训练微调 150 epoch，imgsz=640）
- 基线：P=0.849, R=0.813, mAP50=0.883
- B3 定义（Precision 侧 + 评估可信度）：572 个背景误检中 332 个为高分（score≥0.5）疑似漏标/标注噪声；全分数段共 24541 条背景预测；Oracle 清除全部背景 FP 仅 +2.22 AP50 但 P→0.911；另有 763 个 locFP（框到邻苗）。怀疑标注本身有噪声（漏标、框不准），污染训练与评估。
- 项目约束：结构大改破坏 COCO 预训练迁移；优先数据级/损失级/训练策略级方案；有效性标准 mAP50≥+1.0 且 P/R 不同时降。
- 检索范围：2019–2026（优先 2023–2026）；来源限定 CCF A/B 会议期刊或 JCR Q1/Q2 交叉刊；每条均经 WebSearch 核实真实出处（venue/年份/链接），查无可靠出处的条目已丢弃（见文末附录）。

---

## ELDET（Early-Learning Distillation with Noisy Labels for Object Detection）

- 出处：NeurIPS 2025（Main Conference Track），"ELDET: Early-Learning Distillation with Noisy Labels for Object Detection"，Dongmin Choi, Sangbin Lee, EungGu Yun, Jonghyuk Baek, Frank C. Park，DOI: 10.52202/085713-2331
- 等级：CCF A
- 核心机制：利用噪声标签训练的"early-learning"两阶段现象（网络先拟合干净样本、后记忆错误标签，且定位噪声的记忆晚于分类噪声），将 early-learning 阶段的模型冻结为教师，蒸馏出对分类噪声+定位噪声同时鲁棒的检测器；以即插即用模块形式兼容通用检测架构。
- 对 B3 的作用机理：SSDC-UAV 的高分背景误检很大程度源于模型对"漏标区域被当背景"和"不准框"的记忆性过拟合；ELDET 在记忆化发生前截断训练并以早期教师蒸馏，直接抑制对漏标/错框的记忆，从训练源头减少高分背景 FP；不改网络结构，COCO 迁移完整保留。
- 场景实证：PASCAL VOC、MS COCO、VinDr-CXR（医学）；无直接农业/遥感实证，但密集小目标+噪声标注设定与 SSDC-UAV 同构。
- YOLO 集成形态：训练策略（蒸馏插件模块）
- 预训练兼容性：高
- 预期收益指标与量级：原文（OpenReview 作者回复）：PASCAL VOC 上跨检测器与噪声设置平均 +2.29 AP，87.5%（35/40）设置为正提升；MS COCO 上平均 +1.27 AP。
- 实现成本：中（需确定 early-learning 端点并做两阶段教师-学生训练；无结构改动，代码量中等）
- 参考链接：
  - https://proceedings.neurips.cc/paper_files/paper/2025/hash/6460e378f24da3a79f20ac2640732a00-Abstract-Conference.html
  - https://openreview.net/forum?id=IWEc6kpy8O

## DN-TOD（DeNoising Tiny Object Detector：CLC + TLS）

- 出处：Pattern Recognition（2026），"DN-TOD: Robust tiny object detection amidst label noise"，Haoran Zhu, Chang Xu, Wen Yang, Ruixiang Zhang, Yan Zhang, Gui-Song Xia，DOI: 10.1016/j.patcog.2026.113448；预印本 arXiv:2401.08056 "Robust Tiny Object Detection in Aerial Images amidst Label Noise"
- 等级：CCF B / JCR Q1（AI 类顶刊）
- 核心机制：CLC（Class-aware Label Correction）识别并过滤类别漂移的正样本；TLS（Trend-guided Learning Strategy）通过样本重加权与边界框再生成处理框噪声；可无缝嵌入 one-stage 与 two-stage 检测管线。
- 对 B3 的作用机理：甘蔗苗属小目标，漏标与不准框正是 B3 两大噪声源。CLC 把"高分预测但与 GT 冲突"的样本（即疑似漏标）识别出来并过滤，避免其被当作背景负样本训练（直接对应 332 个高分背景误检的成因）；TLS 的框再生成缓解框不准导致的监督污染（对应 763 个 locFP）。
- 场景实证：合成噪声（noisy AI-TOD-v2.0、DOTA-v2.0）与真实噪声（AI-TOD）航空小目标数据集，与无人机农田小目标场景高度同构。
- YOLO 集成形态：数据流程 + 损失（样本重加权）+ 训练策略
- 预训练兼容性：高（不改网络结构）
- 预期收益指标与量级：原文摘要：在强基线 RFLA 上，40% 混合噪声下性能提升 4.9 个点（"a noteworthy performance improvement of 4.9 points under 40% mixed noise"）。
- 实现成本：中（需实现类别漂移正样本过滤、趋势引导重加权与框再生成；逻辑与检测器解耦）
- 参考链接：
  - https://arxiv.org/abs/2401.08056
  - https://doi.org/10.1016/j.patcog.2026.113448

## Soft Teacher（End-to-End Semi-Supervised Object Detection）

- 出处：ICCV 2021，"End-to-End Semi-Supervised Object Detection with Soft Teacher"，Mengde Xu, Zheng Zhang, Han Hu, Jianfeng Wang, Lijuan Wang, Fangyun Wei, Xiang Bai, Zicheng Liu，arXiv:2106.09018
- 等级：CCF A
- 核心机制：端到端师生课程训练，伪标签质量随训练渐进提升；soft teacher 机制用教师分类分数对每个框的分类损失加权（软权重替代硬阈值）；box jittering 筛选可靠伪框用于回归学习。
- 对 B3 的作用机理：软加权思想可迁移到噪声 GT 场景——对疑似漏标/不准框的样本按教师分数降权而非硬 0/1 监督，避免模型把漏标区域学成"背景"（高分背景 FP 的主要训练来源）；box jittering 式可靠样本选择防止不准框污染回归分支。同时其伪标签管线可直接用于补标（识别漏标）。
- 场景实证：MS-COCO（1%/5%/10% 及全量标签设定）。
- YOLO 集成形态：训练策略（师生课程训练）
- 预训练兼容性：高（结构不变，教师与学生均可用 YOLO12s COCO 权重初始化）
- 预期收益指标与量级：原文摘要：全量 COCO 训练的 40.9 mAP 基线，借助 123K 无标注图像 +3.6 mAP 达 44.5 mAP；在 58.9 mAP 的 Swin 基线上仍 +1.5 mAP 达 60.4 mAP。
- 实现成本：高（需双模型端到端训练循环与课程调度；官方实现基于 mmdetection，移植到 ultralytics 工程量较大）
- 参考链接：
  - https://arxiv.org/abs/2106.09018
  - https://www.microsoft.com/en-us/research/publication/end-to-end-semi-supervised-object-detection-with-soft-teacher/

## DSL（Dense Learning based Semi-Supervised Object Detection）

- 出处：CVPR 2022，"Dense Learning based Semi-Supervised Object Detection"，Binghui Chen, Pengyu Li, Xiang Chen, Biao Wang, Lei Zhang, Xian-Sheng Hua，arXiv:2204.07300
- 等级：CCF A
- 核心机制：面向 anchor-free 检测器的 SSOD：Adaptive Filtering 分配多层、精确的稠密像素级伪标签；Aggregated Teacher 产生稳定伪标签；跨尺度与 patch-shuffle 的不确定性一致性正则提升泛化。
- 对 B3 的作用机理：YOLO12 为 anchor-free，DSL 的稠密伪标签自适应过滤可直接用于训练集"审计-补标"：用聚合教师对全量训练图重推理，补回漏标框、过滤低质标注；一致性正则降低对噪声标注的过拟合，间接压制高分背景 FP。
- 场景实证：MS-COCO、PASCAL-VOC。
- YOLO 集成形态：训练策略 + 数据流程（伪标签审计/补标）
- 预训练兼容性：高
- 预期收益指标与量级：原文摘要：在 MS-COCO 与 PASCAL-VOC 上取得当时 SSOD SOTA，"surpassing existing methods by a large margin"；论文表格显示 10% COCO 标签下约 36.2 mAP（消融行 "+L_scale: 36.2"）。
- 实现成本：高（官方实现基于 mmdetection/FCOS 系 anchor-free 头；思想移植到 YOLO 头工作量较大）
- 参考链接：
  - https://arxiv.org/abs/2204.07300
  - https://github.com/chenbinghui1/DSL

## Confident Learning（CL / cleanlab）

- 出处：Journal of Artificial Intelligence Research（JAIR）Vol. 70, 2021，"Confident Learning: Estimating Uncertainty in Dataset Labels"，Curtis Northcutt, Lu Jiang, Isaac Chuang，DOI: 10.1613/jair.1.12125
- 等级：CCF B
- 核心机制：基于类条件噪声过程假设，用（交叉验证的）预测概率与给定标签直接估计噪声标签与真实标签的联合分布，实现噪声样本剪枝、噪声率估计与按置信度排序训练；与模型和数据模态解耦。
- 对 B3 的作用机理：直接用于 SSDC-UAV 标注审计：以交叉验证预测概率定位"高分预测但无 GT"（疑似漏标）与"有 GT 但持续低分"（错标/不准框）样本，输出复核优先级清单；剪枝或修正后重训，同时提升训练纯净度与评估可信度（B3 的评估污染问题）。
- 场景实证：CIFAR 噪声基准、ImageNet（原文估计 645 张 "missile" 被错标为父类 "projectile"）、MNIST、Amazon Reviews 文本；通用框架，模态无关，无农业直接实证。
- YOLO 集成形态：数据流程（标注审计/清洗，不触碰训练代码）
- 预训练兼容性：高（纯数据侧）
- 预期收益指标与量级：原文摘要：在 CIFAR 噪声学习上优于 7 种近年竞争方法；发现 10 个常用 ML 基准测试集中的标签错误；ImageNet 上清洗数据后训练 "moderately increase model accuracy (e.g., for ResNet)"。对 B3 的直接收益以审计清单质量为主，建议配合人工复核小样本验证。
- 实现成本：低（开源 cleanlab；仅需对训练集跑交叉验证推理并套用 CL 框架）
- 参考链接：
  - https://jair.org/index.php/jair/article/view/12125
  - https://github.com/cleanlab/cleanlab

## SSOD-AT（Active Teaching 遥感半监督+主动学习选样）

- 出处：IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1-5, 2024，"Boosting Semi-Supervised Object Detection in Remote Sensing Images With Active Teaching"，Boxuan Zhang, Zengmao Wang, Bo Du，DOI: 10.1109/LGRS.2024.3357098
- 等级：JCR Q1（Remote Sensing）
- 核心机制：教师-学生网络 + RoI Comparison 模块（RoICM）：生成高置信伪标签的同时识别 top-K 最不确定图像；用基于类别原型的多样性准则去冗余后送人工标注。
- 对 B3 的作用机理：为 SSDC-UAV 提供"不确定性驱动的标注复核选样"现成范式：优先选出高分预测与 GT 冲突（漏标集中区）的图像复核，原型多样性准则避免重复选择相似农田切片；复核修正后重训，从数据源头提升 P 与评估可信度。
- 场景实证：遥感 DOTA、DIOR 数据集。
- YOLO 集成形态：数据流程 + 训练策略（主动学习循环）
- 预训练兼容性：高
- 预期收益指标与量级：原文摘要：在 DOTA 与 DIOR 上 "achieves 1 percent improvement in most cases in the whole AL"（相对 SOTA 方法的完整主动学习循环）。
- 实现成本：中（需实现 RoICM 对比与原型多样性选择；训练流程为标准师生结构）
- 参考链接：
  - https://arxiv.org/abs/2402.18958
  - https://doi.org/10.1109/LGRS.2024.3357098

## IBS-Net（Irrelevant Background Suppression Network）

- 出处：IEEE Transactions on Geoscience and Remote Sensing, vol. 63, 2024（在线发表 2024-12-19），"Background Suppression Network With Attention Collapse Inhibited Transformer for Optical Remote Sensing Object Detection"，Jiaojiao Li, Hailei Li, Haitao Xu, Rui Song, Yunsong Li, Qian Du，DOI: 10.1109/TGRS.2024.3520299
- 等级：JCR Q1（遥感顶刊）
- 核心机制：backbone 后接 Background Detach Module（BDM）抑制无关背景、增强前景；composite-sampler 采样前景与上下文向量以扩大判别感受野；ACI-former 用部分残差连接抑制 transformer 注意力坍缩、保留小目标特征。
- 对 B3 的作用机理：原文明确针对遥感图像"无关背景导致判别特征提取困难，产生相似目标的 FP/FN"——与甘蔗田土垄/杂草/阴影引发的背景误检同构；BDM 从特征层面削弱类苗背景区域的响应，直接压低高分背景 FP，与数据/损失侧方案互补。
- 场景实证：两个光学遥感检测基准（论文摘要称相对主流检测方法取得突出结果，具体增量见原文表格）。
- YOLO 集成形态：模块（backbone 后的背景抑制模块）
- 预训练兼容性：中（新增模块需初始化训练，但 backbone 权重保留；需注意插入位置对 P3–P5 特征对齐的影响）
- 预期收益指标与量级：原文摘要："conducted related experiments on two benchmarks, which demonstrate that our method has achieved prominent results compared with other mainstream detection methods"（摘要未给统一增量数字，具体见论文表格）。
- 实现成本：中（需实现 BDM / composite-sampler 并适配 YOLO neck）
- 参考链接：
  - https://doi.org/10.1109/TGRS.2024.3520299
  - https://xplorestaging.ieee.org/document/10807360/

## AODet（Aerial Object Detection Using Transformers for Foreground Regions）

- 出处：IEEE Transactions on Geoscience and Remote Sensing, vol. 62, article no. 4106711, 2024，"AODet: Aerial Object Detection Using Transformers for Foreground Regions"（IEEE Xplore document 10546305）
- 等级：JCR Q1（遥感顶刊）
- 核心机制：先识别背景区域，仅在最可能包含前景目标的区域上执行检测，利用 transformer 建模前景区域间上下文，显著减少背景冗余计算与误检。
- 对 B3 的作用机理：为无人机农田检测提供"背景先行"策略参考：若能可靠判定土垄/行间为纯背景区域，则这些区域的高分误检可被先验直接压制；原框架为 DETR 系，直接集成困难，可退化为区域先验掩膜 + 后处理过滤的轻量形式。
- 场景实证：VisDrone、DOTA 航空数据集。
- YOLO 集成形态：训练策略/推理流程（区域先验）
- 预训练兼容性：低（原框架与 YOLO 单阶段 anchor-free 流程差异大；仅思想可借鉴）
- 预期收益指标与量级：原文摘要：VisDrone 上 40.9 AP、DOTA 上 79.6 mAP（绝对值，非增量）。
- 实现成本：高（完整复现不现实；若简化为区域先验掩膜+后处理过滤则为中）
- 参考链接：
  - https://xplorestaging.ieee.org/document/10546305/

## BPC（Bridging Precision and Confidence 训练期校准损失）

- 出处：CVPR 2023，"Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection"，Muhammad Akhtar Munir, Muhammad Haris Khan, Salman Khan, Fahad Shahbaz Khan，arXiv:2303.14404，DOI: 10.1109/CVPR52729.2023.01104
- 等级：CCF A
- 核心机制：训练期辅助损失，显式将边界框类别置信度与预测准确度（precision）对齐；原式依赖 mini-batch 内 TP/FP 计数，论文给出可微代理损失，可按置信度-精度四象限分组惩罚失准预测。
- 对 B3 的作用机理：B3 的核心症状是"高分但错"（332/572 背景误检 score≥0.5）；BPC 直接惩罚高置信度假阳性，迫使背景误检的置信度回落，等效于在损失层面提升 P 的置信度可靠性；作为附加损失项不改结构，COCO 迁移保留。
- 场景实证：MS-COCO、Cityscapes、Sim10k、BDD100k 等 6 个数据集，覆盖 in-domain 与 out-domain（域漂移）场景。
- YOLO 集成形态：损失（辅助损失项）
- 预训练兼容性：高
- 预期收益指标与量级：原文摘要：在 in-domain 与 out-domain 场景中 "surpasses strong calibration baselines in reducing calibration error"（具体 ECE 降幅见论文表格，摘要未给统一数字）。
- 实现成本：低-中（实现 TP/FP 可微代理损失并调一个权重超参；可作为附加项随时启停）
- 参考链接：
  - https://arxiv.org/abs/2303.14404
  - https://github.com/akhtarvision/bpc_calibration

## On Calibration of Object Detectors（检测器校准评测框架 + PS/IR 后校准基线）

- 出处：ECCV 2024，"On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines"，Seyit Kuzucu, Kemal Oksuz 等，arXiv:2405.20459
- 等级：CCF B
- 核心机制：系统揭示现有检测器校准评测的陷阱与隐含假设，提出 D-ECE + AP 联合评测框架，并引入为目标检测定制的 Platt Scaling（PS）与 Isotonic Regression（IR）后校准基线。
- 对 B3 的作用机理：零训练成本方案：在留出校准集上学习置信度映射，把失准的高分背景误检置信度压低（便于以阈值压制），同时纠正低分但正确的预测；其联合评测框架还能让 B3 的"校准+精度"结论更可信，避免温度缩放等方法的误评。
- 场景实证：论文构建三个不同特性数据集并系统评测多种检测器。
- YOLO 集成形态：数据流程/后处理（post-hoc 校准，不改权重）
- 预训练兼容性：高（零训练）
- 预期收益指标与量级：论文结论：在正确评测下 PS 与 IR 是高效的后校准器、优于此前常用的温度缩放做法（具体 ECE/AP 数字见原文表格，摘要未给统一数字）。
- 实现成本：低（官方开源工具 fiveai/detection_calibration；仅需在验证集上推理并拟合校准函数）
- 参考链接：
  - https://arxiv.org/abs/2405.20459
  - https://github.com/fiveai/detection_calibration

## ReFocal（学习失衡校正的再平衡 Focal 损失）

- 出处：IEEE Geoscience and Remote Sensing Letters, vol. 22, pp. 1-5, 2025，"ReFocal: Addressing Learning Imbalances for Accurate Tiny Object Detection in Aerial Imagery"，Zijuan Chen, Chang Xu, Haoran Zhu, Yuxin Li, Wen Yang，DOI: 10.1109/LGRS.2024.3507209
- 等级：JCR Q1（Remote Sensing）
- 核心机制：ReFocal Loss 用 magnitude factor 调节不同样本数量目标的学习强度，并用 focal rate adjuster 在样本级区分样本质量，使检测器优先学习高质量样本；ReFocal FPN 以零额外算力动态增强高层特征图的细节信息。
- 对 B3 的作用机理：SSDC-UAV 训练存在前景-背景、易-难样本严重失衡；样本级质量加权可下调疑似噪声样本与"易背景样本"的学习权重，特征再聚焦增强苗株判别性，间接抑制背景误检；属损失级微调，对结构侵入低，与 BPC 可叠加。
- 场景实证：AI-TOD-v2、TinyPerson 航空/无人机小目标数据集。
- YOLO 集成形态：损失（+可选 FPN 模块）
- 预训练兼容性：高（损失替换低侵入；FPN 部分为中）
- 预期收益指标与量级：原文（OpenReview 摘要）："Extensive experiments on AI-TOD-v2 and TinyPerson datasets demonstrate the superiority of our method"（具体增量见论文表格）。
- 实现成本：低（损失部分）/ 中（FPN 部分）
- 参考链接：
  - https://doi.org/10.1109/LGRS.2024.3507209
  - https://openreview.net/forum?id=43xy15Up9R

## NWD-RKA（Normalized Wasserstein Distance + RanKing-based Assigning）

- 出处：ISPRS Journal of Photogrammetry and Remote Sensing, Vol. 190, pp. 79-93, 2022，"Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance and A New Benchmark"，Chang Xu, Jinwang Wang, Wen Yang, Huai Yu, Lei Yu, Gui-Song Xia
- 等级：JCR Q1（遥感顶刊，IF≈11）
- 核心机制：将边界框建模为 2D 高斯分布，用归一化 Wasserstein 距离（NWD）度量相似性，替代对小目标偏移极度敏感的 IoU 阈值；RKA 以排序方式分配正样本，缓解正负样本失衡、尺度样本失衡与样本补偿失效。
- 对 B3 的作用机理：甘蔗苗为小目标，IoU 阈值分配对框偏移极敏感，导致邻苗漏分配/误分配为背景（763 个 locFP 与框粘连重复检测的来源之一）；NWD 用于分配/损失/NMS 可让密集邻苗的分配更稳定，减少框粘连与误判背景。该文同时通过专家复核将 AI-TOD 重标为 AI-TOD-v2（补漏标、纠位置错误），是"标注审计 + 度量改进"双管齐下的模板案例，直接对应 B3 的评估可信度问题。
- 场景实证：AI-TOD / AI-TOD-v2、VisDrone2019、DOTA-v2.0 四个航空小目标数据集。
- YOLO 集成形态：损失 + 分配策略（+NMS 度量）
- 预训练兼容性：高（仅改度量与分配逻辑，不改结构）
- 预期收益指标与量级：原文/项目页：NWD-RKA 嵌入 DetectoRS 后在 AI-TOD-v2 上较 SOTA 提升 4.3 AP，达 24.7 AP / 57.2 AP0.5；在 AI-TOD、VisDrone2019、DOTA-v2 上"consistently improve ... by a large margin"。
- 实现成本：低-中（NWD 有成熟开源实现；RKA 需适配 YOLO 的 anchor-free TAL 分配）
- 参考链接：
  - https://www.sciencedirect.com/science/article/abs/pii/S0924271622001599
  - https://chasel-tsui.github.io/AI-TOD-v2/
  - https://github.com/jwwangchn/NWD

## DCFL（Dynamic Coarse-to-Fine Learning，AI-TOD-R）

- 出处：IEEE Transactions on Pattern Analysis and Machine Intelligence（2025 年在线，2026 年 3 月卷期），"Oriented Tiny Object Detection: A Dataset, Benchmark, and Dynamic Unbiased Learning"，Chang Xu, Ruixiang Zhang, Wen Yang, Haoran Zhu, Fang Xu, Jian Ding, Gui-Song Xia，DOI: 10.1109/TPAMI.2025.3634161
- 等级：CCF A
- 核心机制：揭示检测训练中的学习偏差（置信样本越来越置信、脆弱小目标被边缘化）；DCFL 动态更新先验位置以对齐小目标有限区域，并在样本数量与质量间做平衡分配，实现尺度无偏学习。
- 对 B3 的作用机理：UAV 密集甘蔗苗在分配中易被边缘化（漏分配→被当背景→高分 FP 或漏检）；DCFL 的动态先验对齐与量质平衡分配可减少分配偏差累积，与 NWD-RKA 互补（前者治"分配偏差累积"，后者治"度量对偏移敏感"）。
- 场景实证：AI-TOD-R（28,036 图、752,460 旋转小目标框、平均尺寸 10.6² 像素）等 10 个检测数据集。
- YOLO 集成形态：训练策略/分配策略
- 预训练兼容性：中（分配逻辑改动较深，但不改网络结构）
- 预期收益指标与量级：原文摘要：跨 10 个数据集 "DCFL achieves state-of-the-art accuracy, high efficiency, and remarkable versatility"（具体增量见论文表格）。
- 实现成本：中-高（需实现动态先验更新与平衡分配，比 NWD 复杂）
- 参考链接：
  - https://doi.org/10.1109/TPAMI.2025.3634161
  - https://chasel-tsui.github.io/AI-TOD-R/

## Iterative Noisy Annotation Correction（师生迭代框修正，农业实证）

- 出处：Frontiers in Plant Science, vol. 14, article 1238722, 2023，"An iterative noisy annotation correction model for robust plant disease detection"，Jiuqing Dong, Alvaro Fuentes, Sook Yoon, Hyongsuk Kim, Dong Sun Park，DOI: 10.3389/fpls.2023.1238722
- 等级：JCR Q1（Plant Sciences，IF≈5.6）
- 核心机制：迭代师生范式：教师模型修正退化（不准）的边界框，学生模型在修正框上学习更鲁棒的特征表示，多轮迭代精炼定位噪声；可自然扩展到半监督自动标注。
- 对 B3 的作用机理：可直接套用于 SSDC-UAV：以当前模型为教师对训练集重预测，用高置信预测框修正/补全 GT（迭代式），降低定位噪声与漏标对训练的污染；论文还分析了真实标注的定位噪声分布（小目标噪声更重），对甘蔗苗噪声建模有参考价值。这是少有的农业植物检测场景 SCI 实证。
- 场景实证：植物病害检测数据集（真实噪声 + 按真实分布合成的噪声），农业场景。
- YOLO 集成形态：数据流程（标注修正循环）
- 预训练兼容性：高（纯数据/训练流程侧）
- 预期收益指标与量级：原文摘要：Faster R-CNN 在噪声数据集上性能提升 26%（"achieves a 26% performance improvement on the noisy dataset"）；仅 1% 标签时达到全监督约 75% 的性能。
- 实现成本：中（需搭建迭代修正管线；算法本身简单，官方代码开源）
- 参考链接：
  - https://doi.org/10.3389/fpls.2023.1238722
  - https://github.com/JiuqingDong/TS_OAMIL-for-Plant-disease-detection

---

## 补充条目（出处等级不完全满足 CCF A/B / JCR Q1-Q2 限定，但机制与 B3 高度相关，供备选）

### Unbiased Teacher（伪标签偏差校正 + 类别平衡损失）

- 出处：ICLR 2021，"Unbiased Teacher for Semi-Supervised Object Detection"，Yen-Cheng Liu, Chih-Yao Ma, Zijian He, Chia-Wen Kuo, Kan Chen, Peizhao Zhang, Bichen Wu, Zsolt Kira, Peter Vajda，arXiv:2102.09480
- 等级：ICLR（未列入 CCF 目录的顶级 ML 会议；因机制契合列出，采用与否自行裁量）
- 核心机制：学生与缓慢演进的教师联合训练（EMA 互益），class-balance loss 下调过度自信的伪标签权重，消除伪标签偏差。
- 对 B3 的作用机理："下调过度自信伪标签"的思想可直接迁移为对高分背景误检的抑制：训练中降低模型过度自信但缺乏 GT 支持的样本权重，防止置信度失控（对应 332 个高分背景误检）。
- 场景实证：COCO-standard、COCO-additional、VOC。
- YOLO 集成形态：训练策略
- 预训练兼容性：高
- 预期收益指标与量级：原文摘要：1% COCO 标签下相对 SOTA 绝对提升 6.8 mAP；0.5/1/2% 标签下相对监督基线约 +10 mAP。
- 实现成本：高（双模型训练循环）
- 参考链接：
  - https://arxiv.org/abs/2102.09480
  - https://github.com/facebookresearch/unbiased-teacher

### EDL + Hierarchical Uncertainty Aggregation（检测用证据深度学习与不确定性选样）

- 出处：ICLR 2023，"Active Learning for Object Detection with Evidential Deep Learning and Hierarchical Uncertainty Aggregation"，Younghyun Park, Wonjeong Choi, Soyeong Kim, Dong Jun Han, Jaekyun Moon
- 等级：ICLR（未列入 CCF 目录；作为 EDL 检测变体参考列出）
- 核心机制：Model Evidence Head（MEH）将证据深度学习（EDL）引入检测，估计每个框的认知不确定性，并以层级不确定性聚合（HUA）计算图像级信息量用于主动学习选样。
- 对 B3 的作用机理：EDL 不确定性可区分"高分但高不确定性"的背景误检与真正可靠预测，为标注复核选样与高分 FP 压制提供独立于置信度的第二判据；可作为温度缩放/BPC 之外的替代路线。
- 场景实证：目标检测主动学习基准。
- YOLO 集成形态：模块（证据头）+ 数据流程（不确定性选样）
- 预训练兼容性：中（需改检测头输出为证据参数）
- 预期收益指标与量级：论文报告在相同标注预算下选样效率优于基线（具体 AP-预算曲线见原文）。
- 实现成本：中-高
- 参考链接：
  - https://openreview.net/forum?id=MnEjsw-vj-X

---

## 附录：已检索但丢弃的候选（出处/等级不达标或与 B3 相关性弱）

| 候选 | 出处 | 丢弃原因 |
|---|---|---|
| RobustDet（"Adversarially-Aware Robust Object Detector"，ECCV 2022 Oral，arXiv:2207.06202） | ECCV 2022（CCF B） | 针对对抗攻击鲁棒性，非标注噪声/背景误检问题，与 B3 相关性弱 |
| FMG-Det（"Foundation Model Guided Robust Object Detection"，IEEE ICIP 2025，DOI: 10.1109/ICIP55913.2025.11084306） | ICIP 2025（CCF C） | 机制契合（基础模型引导修正噪声框）但等级不达标 |
| DLD（"Dynamic Loss Decay based Robust Oriented Object Detection on Remote Sensing Images with Noisy Labels"，arXiv:2405.09024，Springer LNCS/PRCV 2024） | LNCS（PRCV） | 机制契合（遥感旋转检测+标签噪声、early-learning 思想）但等级不达标 |
| "Effect of Annotation Errors on Drone Detection with YOLOv3"（Koksal, Ince, Alatan，arXiv:2004.01059） | 仅 arXiv | 无人机标注错误影响实证与 B3 高度相关，但未查到正式会议/期刊出处，仅作背景参考 |

---

## 检索说明

- 全部条目均通过 WebSearch 核实 venue/年份/作者/链接（arXiv、IEEE Xplore、NeurIPS Proceedings、OpenReview、JAIR、ScienceDirect、Frontiers/PMC、dblp、项目主页交叉验证）。
- 时间范围 2019–2026；优先 2023–2026 条目（ELDET 2025、DN-TOD 2026、DCFL TPAMI 2025/2026、IBS-Net TGRS 2024、SSOD-AT GRSL 2024、ReFocal GRSL 2025、BPC CVPR 2023、ECCV 2024 校准、Frontiers 2023）。
- 与 B3 最直接相关的三类方案：(a) 标注审计/修正（Confident Learning、SSOD-AT、Iterative Noisy Annotation Correction、DN-TOD 的 CLC）；(b) 高分 FP 置信度压制（BPC、On Calibration PS/IR、Unbiased Teacher 的 class-balance 思想）；(c) 小目标分配/度量纠偏（NWD-RKA、DCFL、ReFocal）。
