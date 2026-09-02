# B2「边界框回归精度」改进方案文献调研

- 调研日期：2026-09-01
- 项目背景：SSDC-UAV 甘蔗苗无人机检测，Ultralytics YOLO12s，COCO 预训练微调 150 epoch，imgsz=640。
- 基线：mAP50=0.883，mAP75=0.609，mAP50-95=0.556。
- B2 定义：9073 个 TP 中 3347 个（36.9%）IoU 卡在 [0.5, 0.75)；Oracle 完美定位可使 AP75 从 0.607→0.880（+27.2）。目标多为小尺寸、近圆/椭圆、密集相邻的甘蔗苗框，相邻框易粘连。
- 本项目已试方案（增益均在噪声地板 σ≈0.10 mAP50 附近）：Focaler-CIoU（+0.389 mAP50，mAP75 61.14）、PIoU2（+0.272，mAP75 61.55）、WIoUv3（+0.166）、Slide-Loss（-0.093）、DySample（+0.104 / dyscope +0.175）、DyHead（+0.002）、FreqFusion（最好 +0.14，最差 -0.64）。

## 检索与收录说明

- 时间范围 2019–2026（优先 2023–2026）；来源限定 CCF A/B 会议期刊或 JCR Q1/Q2 交叉刊。
- 所有条目均经 WebSearch/arXiv 官方页面核实出处；查不到可靠出处的候选（JSCD、UniBox 等）已丢弃，见文末「检索记录（已排除）」。
- 例外标注：MPDIoU / Inner-IoU / Shape-IoU / SIoU 四个方法目前仅有 arXiv 预印本、未找到同行评审出处。因其与 B2 机理高度相关、且本项目已试的 Focaler-IoU/PIoU2/WIoU/Slide-Loss 同样源自 arXiv，故保留收录并显式标注「arXiv 预印本」，采用与否请自行权衡。
- 「对 B2 的作用机理」均针对：小圆框/椭圆框、密集粘连框、IoU≥0.75 命中率（AP75）提升。
- 建议：所有新方案以 mAP75 为主选指标（而非 mAP50），并至少 2 个随机种子以对照 σ≈0.10 的噪声地板。

---

## 1. EIoU / Focal-EIoU（Efficient IoU）

- 方法名（英文）：EIoU / Focal-EIoU
- 出处：Neurocomputing（Elsevier），2022，vol. 506，标题原文 "Focal and Efficient IOU Loss for Accurate Bounding Box Regression"（Yi-Fan Zhang, Weiqiang Ren, Zhang Zhang, Zhen Jia, Liang Wang, Tieniu Tan），DOI: 10.1016/j.neucom.2022.07.042；arXiv:2101.08158
- 等级：JCR Q1（Neurocomputing，中科院二区）
- 核心机制：EIoU 删除 CIoU 的宽高比惩罚项 v，将重叠面积、中心点距离、宽/高边长差三个几何因子显式分开度量（w、h 各自独立惩罚）；Focal-EIoU 再叠加回归版 focal loss（Effective Example Mining），让回归聚焦高质量 anchor。
- 对 B2 的作用机理：甘蔗苗框近圆/椭圆，预测框与 GT 宽高比相同但尺寸不同的情形极常见——此时 CIoU 的 v 项≈0、丧失梯度（"宽高比退化"），回归停滞在 IoU 0.5–0.75 带。EIoU 的独立 w/h 惩罚在该退化情形下仍提供梯度，直接推动高 IoU 精修；focal 版进一步把优化资源集中到高 IoU 样本，与 AP75 目标一致。
- 场景实证：SRW-YOLOv8n 穴盘辣椒苗主茎检测（Frontiers in Plant Science 2026）损失消融：EIoU mAP@0.5 92.6% vs CIoU 91.7%（+0.9pp）、DIoU 91.2%、GIoU 91.6%——苗株场景中 EIoU 为最优经典损失。
- YOLO 集成形态：损失（替换 CIoU，ultralytics 中改 `bbox_loss` 分支即可）
- 预训练兼容性：高（纯损失替换，不动结构与权重）
- 预期收益指标与量级：原文摘要称收敛速度与定位精度均优于 GIoU/DIoU/CIoU（合成+真实数据集，未给统一 AP 数）；苗株实证 +0.9pp mAP@0.5（见上）。对本项目预期 mAP75 方向为正，量级需实测。
- 实现成本：低（~30 行损失函数，无超参）
- 参考链接：https://doi.org/10.1016/j.neucom.2022.07.042 ; https://arxiv.org/abs/2101.08158

## 2. Alpha-IoU（α-IoU 幂族损失）

- 方法名（英文）：Alpha-IoU（α-DIoU / α-CIoU / α-GIoU）
- 出处：NeurIPS 2021，标题原文 "Alpha-IoU: A Family of Power Intersection over Union Losses for Bounding Box Regression"（Jiabo He, Sarah Erfani, Xingjun Ma, James Bailey, Ying Chi, Xian-Sheng Hua），arXiv:2110.13675
- 等级：CCF A
- 核心机制：把 IoU 损失推广为幂族形式（IoU^α + 幂正则项），单参数 α 调节损失/梯度重加权强度，保持序关系（order preserving）。
- 对 B2 的作用机理：密集粘连框的标注边界本身模糊（相邻苗框边缘互相侵入），属"noisy bbox"场景；原文明确 α-IoU 对小数据集和噪声框更鲁棒（α<1 时压低离群/低质量样本梯度，避免回归被粘连框的脏标注拖偏）。在保持高 IoU 样本序关系的前提下减少噪声梯度，有利于把 [0.5,0.75) 带样本稳定推向 0.75+。
- 场景实证：原文在 Pascal VOC / MS COCO 多个检测器上一致超过基线 IoU 损失；AIP Advances 2022（入水点水柱检测）将 α 与 EIoU 组合为 α-EIoU 使用。
- YOLO 集成形态：损失（在 CIoU/DIoU 外套幂变换）
- 预训练兼容性：高（纯损失）
- 预期收益指标与量级：原文摘要"surpass existing IoU-based losses by a noticeable performance margin"，且"more robust to small datasets and noisy bboxes"（未给统一 AP 数）；α 建议从 0.5/0.75/1.0/1.5 网格扫。
- 实现成本：低（损失外包一层，1 个超参 α）
- 参考链接：https://proceedings.neurips.cc/paper/2021/hash/a8f15eda80c50adb0e71943adc8015cf-Abstract.html ; https://arxiv.org/abs/2110.13675

## 3. MPDIoU（Minimum Point Distance IoU）

- 方法名（英文）：MPDIoU
- 出处：arXiv 预印本 2023（未检索到同行评审发表），标题原文 "MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression"（Siliang Ma, Yong Xu），arXiv:2307.07662
- 等级：arXiv 预印本（无 CCF/JCR 等级，收录理由见文首说明）
- 核心机制：用预测框与 GT 框的左上、右下两个角点欧氏距离构造相似度度量，统一覆盖重叠面积、中心点距离、宽高偏差三类信息；即使两框不重叠或宽高比完全相同也能给出有效优化梯度。
- 对 B2 的作用机理：原文明确指出的失效场景——"预测框与 GT 宽高比相同但宽高数值不同"时多数 IoU 损失失去梯度——正是近圆甘蔗苗框（宽高比≈1）的典型卡点。MPDIoU 以角点距离直接优化尺寸与位置，在高 IoU 区仍有稳定梯度，利于精修到 0.75+；对粘连框，角点度量比宽高比项更不易被相邻框干扰。
- 场景实证：原文在 YOLOv7/YOLACT + PASCAL VOC / MS COCO / IIIT5k 上超过现有损失（摘要未给统一数字）；应用实证：YOLOv8n 夜间车辆检测（ACM 2024）以 MPDIoU 替换 CIoU 得 AP +2.5%、参数 -6.7%；CCP-YOLO 钢板表面缺陷检测（Sensors 2026）采用 Wise-Inner-MPDIoU 组合。
- YOLO 集成形态：损失
- 预训练兼容性：高（纯损失）
- 预期收益指标与量级：二次应用证据 AP +2.5%（夜间车辆，YOLOv8n vs CIoU 基线）；原文声称全面优于 CIoU/DIoU/GIoU/SIoU 等。
- 实现成本：低（公式简单，无超参）
- 参考链接：https://arxiv.org/abs/2307.07662

## 4. Inner-IoU（辅助框 IoU）

- 方法名（英文）：Inner-IoU
- 出处：arXiv 预印本 2023（未检索到同行评审发表），标题原文 "Inner-IoU: More Effective Intersection over Union Loss with Auxiliary Bounding Box"，arXiv:2311.02877
- 等级：arXiv 预印本（收录理由见文首说明）
- 核心机制：用比例因子 ratio（<1）把预测框与 GT 框同时缩小为"辅助内框"再计算 IoU 与损失：高 IoU 样本缩放后 IoU 更高、损失梯度被放大继续精修；低 IoU 样本被相对抑制。可套在任意 IoU 损失上（Inner-CIoU、Inner-WIoU 等）。
- 对 B2 的作用机理：B2 的核心是 3347 个 TP 卡在 IoU [0.5,0.75)。Inner-IoU 是少数在机理上直接"放大高 IoU 样本回归信号"的损失——辅助框使已经较准的框获得更强梯度继续逼近 GT，正对应把 [0.5,0.75) 样本推过 0.75 阈值；与已试的 WIoUv3/Focaler 的聚焦机制互补（可叠加：Wise-Inner-IoU）。
- 场景实证：CCP-YOLO（Sensors 2026，钢板缺陷）Wise-Inner-MPDIoU 组合提升回归质量；FCMI-YOLO（PLOS ONE 2025，火灾检测）用 Inner-DIoU 显著提升定位精度；CEBW-YOLO11（Applied Sciences 2026，锂电池极片缺陷）采用 Wise-Inner-IoU。
- YOLO 集成形态：损失（wrapper，可与现有 WIoU/Focaler 代码叠加）
- 预训练兼容性：高（纯损失）
- 预期收益指标与量级：原文摘要称"更快、更有效的回归"（仿真+对比实验，未给统一 AP 数）；与本项目已试 WIoUv3（+0.166 mAP50）叠加后若机理互补，有机会超过噪声地板，需实测。
- 实现成本：低（损失 wrapper，1 个超参 ratio，建议 0.5–0.9 扫描）
- 参考链接：https://arxiv.org/abs/2311.02877

## 5. Shape-IoU（形状/尺度感知 IoU）

- 方法名（英文）：Shape-IoU
- 出处：arXiv 预印本 2023（未检索到同行评审发表），标题原文 "Shape-IoU: More Accurate Metric considering Bounding Box Shape and Scale"（Zhang & Zhang），arXiv:2312.17663
- 等级：arXiv 预印本（收录理由见文首说明）
- 核心机制：分析发现框自身的形状与尺度会影响回归结果，Shape-IoU 按框的形状/尺度动态加权惩罚项（关注 shape 与 scale 的度量），面向密集目标检测设计。
- 对 B2 的作用机理：甘蔗苗框形状先验强（近圆/椭圆、尺度集中），CIoU 对所有形状一视同仁的惩罚会把梯度浪费在无关方向；Shape-IoU 按形状/尺度自适应加权，理论上对"形状一致、只差精确定位"的小圆框更对症，且原文定位场景即密集检测。
- 场景实证：AITP-YOLO 番茄成熟度检测（PLOS ONE 2025）以 Shape-IoU 替换 CIoU；社区复现"结合 NWD 的 Shape-IoU 助力 YOLOv5 涨点"（掘金技术博客，非正式出处）。
- YOLO 集成形态：损失
- 预训练兼容性：高（纯损失）
- 预期收益指标与量级：原文摘要"有效提升检测性能"（大量对比实验，未给统一 AP 数）。
- 实现成本：低（损失替换，含形状/尺度相关超参需按目标形状先验设置）
- 参考链接：https://arxiv.org/abs/2312.17663

## 6. SIoU（Scylla-IoU，角度感知）

- 方法名（英文）：SIoU
- 出处：arXiv 预印本 2022（未检索到同行评审发表），标题原文 "SIoU Loss: More Powerful Learning for Bounding Box Regression"（Zhora Gevorgyan），arXiv:2205.12740
- 等级：arXiv 预印本（收录理由见文首说明）
- 核心机制：在距离/宽高惩罚外引入预测框与 GT 框中心连线的角度惩罚（angle cost），让回归先沿轴向对齐方向再收敛，减少训练中游走（wandering）。
- 对 B2 的作用机理：小框中心点偏移 1–2 px 即显著掉 IoU；角度项加速中心对齐、缩短收敛路径，间接给高 IoU 精修留更多训练预算。但对"宽高比退化"这一 B2 核心卡点无专门设计，对症度中等，作为低成本对照项。
- 场景实证：SRW-YOLOv8n 辣椒苗（Frontiers in Plant Science 2026）：SIoU mAP@0.5 +0.6pp vs CIoU；注意另有同名缩写但不同的 "Scale-Sensitive IoU (SIOU)"（Du et al.，遥感多尺度）被 MTD-YOLO（CEAG 2024，见第 18 节）与 DCYOLO 城市密集街道检测（Scientific Reports 2024）采用，勿混淆。
- YOLO 集成形态：损失
- 预训练兼容性：高（纯损失）
- 预期收益指标与量级：苗株实证 +0.6pp mAP@0.5（Frontiers Plant Sci 2026）；原文仅声称训练速度与推理精度提升（无统一数字）。
- 实现成本：低
- 参考链接：https://arxiv.org/abs/2205.12740

## 7. NWD（Normalized Gaussian Wasserstein Distance）

- 方法名（英文）：NWD
- 出处：初版 arXiv:2110.13389（2021），标题原文 "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"（Jinwang Wang, Chang Xu, Wen Yang, Lei Yu）；扩展版正式发表于 ISPRS Journal of Photogrammetry and Remote Sensing，2022，vol. 190，pp. 79–93，标题原文 "Detecting tiny objects in aerial images: A normalized Wasserstein distance and a new benchmark"，DOI: 10.1016/j.isprsjprs.2022.06.002
- 等级：JCR Q1（ISPRS J P&RS，遥感顶刊，IF≈11）
- 核心机制：把边界框建模为 2D 高斯分布，用归一化 Wasserstein 距离度量框相似度；对微小位移平滑不敏感（IoU 在小目标上 1 px 偏移即剧变，NWD 不会）。可无侵入替换 anchor-based 检测器中 assignment、NMS、loss 三处的 IoU。
- 对 B2 的作用机理：B2 的本质就是"IoU 对小框位移过度敏感"——3347 个 TP 只差几个像素就进不了 0.75。NWD 用平滑的高斯距离做损失/分配度量，使高 IoU 附近的梯度不再因 IoU 饱和/突变而消失；用于 NMS 时粘连小框不易因微小重叠被误判为重复而互相抑制。与已试的 Focaler/PIoU（仍在 IoU 框架内）相比，NWD 是度量层面的换血，机理上更针对小目标。
- 场景实证：AI-TOD 微小目标基准（平均目标远小于 COCO）：比标准微调基线 +6.7 AP，比当时 SOTA +6.0 AP（原文摘要）；ISPRS 2022 扩展版新增航空小目标 benchmark 并系统验证；已被大量遥感小目标检测论文采用。
- YOLO 集成形态：损失（最简：以 NWD 距离构造回归损失）｜分配（TAL 中 IoU 计算替换）｜NMS（可选）
- 预训练兼容性：高（不动结构；若只改损失则完全兼容）
- 预期收益指标与量级：原文 AI-TOD +6.7 AP（vs 微调基线）；本项目目标尺度（小苗框）介于 COCO 与 AI-TOD 之间，预期收益应打折，但方向与 B2 高度一致。
- 实现成本：低–中（损失版低；改 assignment 需动 TaskAlignedAssigner 的 IoU 计算，中）
- 参考链接：https://arxiv.org/abs/2110.13389 ; https://doi.org/10.1016/j.isprsjprs.2022.06.002 ; https://github.com/jwwangchn/NWD

## 8. GWD（Gaussian Wasserstein Distance Loss）

- 方法名（英文）：GWD
- 出处：ICML 2021（PMLR vol. 139），标题原文 "Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss"（Xue Yang, Junchi Yan, Qi Ming, Wentao Wang, Xiaopeng Zhang, Qi Tian），arXiv:2101.11952
- 等级：CCF A
- 核心机制：将（旋转）框转为 2D 高斯，用高斯 Wasserstein 距离近似不可微的旋转 IoU 损失；两框不重叠时仍有信息，兼具 GIoU/DIoU 的性质，并天然规避边界不连续与"方形问题"。
- 对 B2 的作用机理：机理参照项。水平框可视为旋转框角度=0 的特例，GWD 退化为中心距离+尺寸差的高斯度量，与 NWD 同源；其"高 IoU 区平滑、非重叠仍有梯度"的性质对粘连框友好。实操上水平框场景直接用 NWD（第 7 条）更直接，GWD 主要作为理论支撑与实现参考。
- 场景实证：原文在 DOTA、HRSC2016 等旋转框基准上一致提升（摘要未列统一数字）；mmrotate 标准组件。
- YOLO 集成形态：损失
- 预训练兼容性：高（纯损失）
- 预期收益指标与量级：原文为旋转框场景；水平小框场景建议以 NWD 为准。
- 实现成本：中（需实现高斯距离；若已实现 NWD 则边际成本≈0）
- 参考链接：https://proceedings.mlr.press/v139/yang21l.html ; https://arxiv.org/abs/2101.11952

## 9. KLD（Kullback-Leibler Divergence 框回归）

- 方法名（英文）：KLD
- 出处：NeurIPS 2021，标题原文 "Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence"（Xue Yang, Xiaojiang Yang, Jirui Yang, Qi Ming, Wentao Wang, Qi Tian, Junchi Yan），arXiv:2106.01883
- 等级：CCF A
- 核心机制：用两高斯分布的 KL 散度作回归损失；证明其尺度不变（scale invariant），且各参数梯度权重随目标宽高比动态自适应（高精度检测中，大长宽比目标的角度误差被自动加权）。
- 对 B2 的作用机理：机理参照项。尺度不变性对小框友好（小框绝对误差小、相对误差大，KLD 不因尺度缩小而梯度失衡）；"high-precision"设计目标与 AP75 一致。近圆框宽高比≈1 时 KLD 对 w/h 误差均衡加权，与 EIoU 的独立边长惩罚思路互补。同样，水平框场景更推荐 NWD 形式，KLD 作备选度量。
- 场景实证：原文在 DOTA、HRSC、DIOR-R 等高精度旋转框检测上一致优于 GWD/BCD 等（摘要未列统一数字）。
- YOLO 集成形态：损失
- 预训练兼容性：高
- 预期收益指标与量级：同 GWD，水平框场景以 NWD 为主。
- 实现成本：中
- 参考链接：https://papers.neurips.cc/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract.html ; https://arxiv.org/abs/2106.01883

## 10. GFLv2（Distribution-Guided Quality Predictor）

- 方法名（英文）：GFLv2 / DGQP
- 出处：CVPR 2021，标题原文 "Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection"（Xiang Li, Wenhai Wang, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang），arXiv:2011.12885
- 等级：CCF A
- 核心机制：发现 DFL 学到的框四边分布的统计量（峰度等）与真实定位质量高度相关：分布尖峰↔定位好。据此用极轻量的 DGQP 模块从分布统计预测 LQE 分数，参与排序/NMS。
- 对 B2 的作用机理：密集粘连框场景下，NMS/排序若只看分类置信度，会保留"置信度高但定位平庸"的框、压制"定位精准但置信度略低"的框。LQE 把真实定位质量注入排序，使高 IoU 框在 NMS 中存活——直接作用于 AP75。YOLO12 检测头本就输出 DFL 分布，DGQP 是即插即用的旁路分支，不改主干。
- 场景实证：COCO test-dev：GFLV2（ResNet-101）46.2 AP @14.6FPS，比同速 ATSS 基线（43.6 AP）绝对 +2.6 AP（原文摘要）。
- YOLO 集成形态：模块（检测头加 DGQP 分支）+ 损失（LQE 监督，用预测框与 GT 的 IoU 作目标）+ 推理排序
- 预训练兼容性：中（新增分支随机初始化，但主干/颈部权重全复用；需少量额外训练预算收敛新分支）
- 预期收益指标与量级：原文 +2.6 AP（ATSS→GFLV2，含 LQE 贡献）；对 YOLO 系移植的增益会小于该值，但 LQE 对 AP75 的定向性在密集场景证据充分。
- 实现成本：中（新增 1 个小分支 + 推理时分数融合；ultralytics 需改 Detect 头与验证逻辑）
- 参考链接：https://arxiv.org/abs/2011.12885 ; https://github.com/implus/GFocalV2

## 11. YOLOv10 一致性双分配（Consistent Dual Assignments）

- 方法名（英文）：YOLOv10 consistent dual assignments（one-to-one + one-to-many）
- 出处：NeurIPS 2024，标题原文 "YOLOv10: Real-Time End-to-End Object Detection"（Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding），arXiv:2405.14458
- 等级：CCF A
- 核心机制：训练时并行 one-to-many（富监督）与 one-to-one（唯一最优匹配）双分配，用一致匹配度量对齐两者；推理只用 one-to-one 头，免 NMS。
- 对 B2 的作用机理：密集粘连框是 NMS 的重灾区（相邻苗框互相抑制或残留重复框），NMS 的 IoU 阈值本身就会截断高 IoU 精度收益；one-to-one 分支强制每个目标输出唯一最优框，绕开 NMS 引入的定位退化，且 one-to-one 匹配以定位质量为核心目标，利于 AP75。
- 场景实证：原文：YOLOv10-S 与 RT-DETR-R18 相近 AP 下快 1.8×、参数/FLOPs 少 2.8×；YOLOv10-B 较 YOLOv9-C 同性能下延迟 -46%、参数 -25%（摘要）。
- YOLO 集成形态：分配/训练策略 + 双检测头
- 预训练兼容性：中（one-to-one 头为新增；COCO 预训练的 one-to-many 权重可复用，ultralytics 仓库已有 YOLOv10 实现可参考）
- 预期收益指标与量级：原文未单列定位指标增益（整体 AP 达 SOTA）；对密集场景的价值主要在免 NMS 的推理一致性与定位唯一性。
- 实现成本：中–高（双头 + 一致性匹配训练逻辑；可先评估直接用 YOLOv10 架构微调的可行性）
- 参考链接：https://arxiv.org/abs/2405.14458 ; https://github.com/THU-MIG/yolov10

## 12. TOOD / TAL（任务对齐分配，含可调定位权重）

- 方法名（英文）：TOOD（Task-aligned Head + Task Alignment Learning）
- 出处：ICCV 2021 Oral，标题原文 "TOOD: Task-aligned One-stage Object Detection"（Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R. Scott, Weilin Huang），arXiv:2108.07755
- 等级：CCF A
- 核心机制：T-Head 平衡任务交互/任务特定特征；TAL 用对齐度量 t = s^α · u^β（s=分类分，u=IoU）做样本分配与任务对齐损失，显式拉近分类与定位的最优样本。
- 对 B2 的作用机理：YOLO12 的 TaskAlignedAssigner 即源自 TAL——可操作点是：① 调大 β（或 box_weight）使分配更偏定位质量，让高 IoU 候选获得更多正样本资格；② 引入 T-Head 式任务交互缓解"分类强、定位弱"的空间错位（错位框正是卡 [0.5,0.75) 带的主体）。属零结构改动的训练策略微调。
- 场景实证：COCO：TOOD 单模单尺度 51.1 AP，大幅超 ATSS 47.7 / GFL 48.2 / PAA 49.0（原文摘要）；YOLOv8/v10/YOLO-World 均已采纳 TAL（TOOD 官方仓库新闻）。
- YOLO 集成形态：分配/训练策略（超参）｜可选模块（T-Head）
- 预训练兼容性：高（纯训练策略；T-Head 版需改头）
- 预期收益指标与量级：原文整体 +3~4 AP（vs ATSS/GFL 级基线）；本项目仅调 α/β 的预期收益较小但成本≈0，适合作为先行低成本试验。
- 实现成本：低（调 `tal_topk`/α/β 等分配超参）–中（T-Head）
- 参考链接：https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html ; https://arxiv.org/abs/2108.07755

## 13. Varifocal Loss（VFNet / IACS）

- 方法名（英文）：Varifocal Loss（VFL）/ VFNet
- 出处：CVPR 2021 Oral，标题原文 "VarifocalNet: An IoU-aware Dense Object Detector"（Haoyang Zhang, Ying Wang, Feras Dayoub, Niko Sünderhauf），arXiv:2008.13367
- 等级：CCF A
- 核心机制：训练分类分支预测 IACS（IoU-Aware Classification Score，物体置信度与定位质量的联合表示），VFL 对正负样本非对称加权；配合星形框特征做框精修。
- 对 B2 的作用机理：让置信度本身编码定位质量后，排序天然偏向高 IoU 框——密集粘连场景下减少"高置信、差定位"框挤占 NMS 名额；与 GFLv2 的 LQE 目标一致但实现更轻（只换分类损失）。农业实证 NVW-YOLOv8s（CEAG 2024）即在小目标/遮挡番茄场景用 VFL+WIoU 组合。
- 场景实证：COCO：VFNet 较 FCOS+ATSS 强基线一致提升约 2.0 AP（多骨干）；最佳 VFNet-X-1200（Res2Net-101-DCN）test-dev 55.1 AP（当时 SOTA，原文摘要）。
- YOLO 集成形态：损失（分类分支换 VFL）｜可选模块（星形采样+精修分支）
- 预训练兼容性：高（仅损失）–中（加精修分支）
- 预期收益指标与量级：原文约 +2.0 AP（整体）；仅 VFL 损失的移植收益预计更小，但对排序/高 IoU 保留有定向作用。
- 实现成本：低（VFL 损失）–中（完整 VFNet 组件）
- 参考链接：https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html ; https://arxiv.org/abs/2008.13367

## 14. Softer-NMS（KL-Loss，不确定性回归 + 邻框合并）

- 方法名（英文）：Softer-NMS / KL-Loss（IoU-uncertainty regression）
- 出处：CVPR 2019，标题原文 "Bounding Box Regression with Uncertainty for Accurate Object Detection"（Yihui He, Chenchen Zhu, Jianren Wang, Marios Savvides, Xiangyu Zhang），arXiv:1809.08545
- 等级：CCF A
- 核心机制：回归损失同时学习框变换与定位方差（不确定性）；学到的方差用于 NMS 时按置信度加权合并相邻框（Softer-NMS）。
- 对 B2 的作用机理：对高 IoU 阈值指标增益的直接证据（AP90 +6.2%）：不确定性建模让模型在"标注边界模糊"处（粘连苗框边缘）不强行拟合噪声，推理时用方差加权合并相邻候选框——对密集粘连框是专门的定位修正机制。
- 场景实证：MS-COCO：VGG-16 Faster R-CNN AP 23.6%→29.1%；ResNet-50-FPN Mask R-CNN AP +1.8%、AP90 +6.2%，显著超过此前的 bbox refinement 方法（原文摘要）。
- YOLO 集成形态：损失 + NMS（头需输出方差分支）
- 预训练兼容性：低–中（YOLO anchor-free 头需加 4 维方差输出并改 NMS；主干权重可复用）
- 预期收益指标与量级：原文 AP90 +6.2%（Mask R-CNN）——高 IoU 阈值指标上量级最大的已验证方案之一；移植到 one-stage YOLO 会打折。
- 实现成本：中–高（方差头 + 自定义 NMS；与免 NMS 路线冲突，需权衡）
- 参考链接：https://arxiv.org/abs/1809.08545 ; https://github.com/yihui-he/softer-NMS

## 15. Libra R-CNN（IoU-balanced sampling + Balanced L1）

- 方法名（英文）：IoU-balanced sampling / Balanced L1 Loss（Libra R-CNN 组件）
- 出处：CVPR 2019，标题原文 "Libra R-CNN: Towards Balanced Learning for Object Detection"（Jiangmiao Pang, Kai Chen, Jianping Shi, Huajun Feng, Wanli Ouyang, Dahua Lin），arXiv:1904.02701
- 等级：CCF A
- 核心机制：IoU-balanced sampling 按 IoU 区间均匀采负样本（纠正负样本集中在低 IoU 段的失衡）；Balanced L1 提升高 IoU 难样本的回归梯度权重。
- 对 B2 的作用机理：训练信号被大量低 IoU 易样本主导、高 IoU 样本梯度不足，是 [0.5,0.75) 带滞留的训练侧原因之一。Balanced L1 思想可直接迁移到 YOLO 的 DFL/IoU 损失加权（提高高 IoU 难样本权重）；IoU-balanced 思想对应调节 TAL 正样本的 IoU 分布。
- 场景实证：COCO：无 bells-and-whistles 下 Faster R-CNN +2.5 AP、RetinaNet +2.0 AP（原文摘要）。
- YOLO 集成形态：训练策略（损失加权/样本分布调节）
- 预训练兼容性：高（不动结构）
- 预期收益指标与量级：原文 +2.0~2.5 AP（两阶段/单阶段基线）；迁移为损失加权后预期为小幅但方向对准 AP75。
- 实现成本：低–中（Balanced L1 加权易实现；改采样分布需动 assigner）
- 参考链接：https://arxiv.org/abs/1904.02701 ; https://openaccess.thecvf.com/content_CVPR_2019/html/Pang_Libra_R-CNN_Towards_Balanced_Learning_for_Object_Detection_CVPR_2019_paper.html

## 16. RFLA（Gaussian Receptive Field Label Assignment）

- 方法名（英文）：RFLA（Receptive Field Distance + Hierarchical Label Assignment）
- 出处：ECCV 2022，标题原文 "RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection"（Chang Xu, Jinwang Wang, Wen Yang, Huai Yu, Lei Yu, Gui-Song Xia），arXiv:2208.08738
- 等级：CCF B
- 核心机制：特征点感受野近似高斯分布；用感受野距离（RFD，高斯距离）替代 IoU/center sampling 度量样本质量，并以 HLA 分层分配纠正"IoU 阈值与中心采样偏向大目标"的失衡。
- 对 B2 的作用机理：IoU 阈值式分配对小框天然苛刻（小框 IoU 对位移敏感，难达高阈值），导致小框高定位质量样本分配不足、学不出精定位；RFLA 的平滑高斯度量让小苗框获得更充分的高质量训练信号。与 NWD 同源（高斯度量），但作用于分配环节，与损失端 NWD 互补。
- 场景实证：AI-TOD：较 SOTA 竞争方法 +4.0 AP（原文摘要）；后续被 SAR 舰船检测等采用（HCA-RFLA，Electronics 2024）。
- YOLO 集成形态：分配（替换/混合 TaskAlignedAssigner 的度量）
- 预训练兼容性：中（训练策略级改动，不动权重结构；需调 HLA 阈值）
- 预期收益指标与量级：原文 AI-TOD +4.0 AP（tiny 基准）；本项目目标非极端 tiny，预期打折。
- 实现成本：中（需实现 RFD 度量并接入 ultralytics assigner）
- 参考链接：https://arxiv.org/abs/2208.08738 ; https://github.com/Chasel-Tsui/mmdet-rfla

## 17. CARAFE / CARAFE++（内容感知上采样）

- 方法名（英文）：CARAFE / CARAFE++
- 出处：① ICCV 2019 Oral，标题原文 "CARAFE: Content-Aware ReAssembly of FEatures"（Jiaqi Wang, Kai Chen, Rui Xu, Ziwei Liu, Chen Change Loy, Dahua Lin），arXiv:1905.02188；② 期刊扩展版 IEEE TPAMI 2022（vol. 44, no. 9），标题原文 "CARAFE++: Unified Content-Aware ReAssembly of FEatures"，arXiv:2012.04733
- 等级：CCF A（ICCV）；CCF A 期刊 / JCR Q1（TPAMI）
- 核心机制：内容感知动态核上采样：为每个目标位置在线生成自适应重组核，在大感受野内聚合上下文，替代固定核的双线性/反卷积上采样；轻量。
- 对 B2 的作用机理：P3/P4 上采样质量决定小框边界特征的保真度；双线性上采样的固定插值核会模糊苗框边缘，限制定位上限。CARAFE 的内容感知核在边缘处保留方向性细节，与已试 DySample（+0.104/+0.175）、FreqFusion（最好 +0.14）互为替代方案，提供独立证据线。
- 场景实证：原文（COCO 等）：CARAFE 带来检测 +1.2% AP、实例分割 +1.3%、语义分割 +1.8% mIoU、修复 +1.1 dB（摘要）；CARAFE++：+2.5% APbox、+2.1% APmask、+1.94% mIoU、+1.35 dB（摘要）。
- YOLO 集成形态：模块（替换 neck 中 nn.Upsample）
- 预训练兼容性：高（仅 neck 上采样层；主干/其余权重不受影响，新增核生成卷积随机初始化）
- 预期收益指标与量级：原文 +1.2% AP（检测，相对双线性基线）；在本项目与 DySample/FreqFusion 的对比框架下，预期同量级（±0.1–0.2 mAP50），需实测是否突破噪声地板。
- 实现成本：低–中（mmdetection 有官方实现；ultralytics 社区亦有 CARAFE 移植，需注册进 tasks.py 的 parse_model）
- 参考链接：https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_CARAFE_Content-Aware_ReAssembly_of_FEatures_ICCV_2019_paper.html ; https://arxiv.org/abs/2012.04733

## 18. 交叉场景实证（农业/遥感 SCI 刊，2022–2026）

本节汇总与甘蔗苗场景（小目标、密集、农业）最接近的已发表实证，用于校准上述方法的预期收益。

### 18.1 SRW-YOLOv8n（穴盘辣椒苗主茎检测，损失消融）

- 出处：Frontiers in Plant Science，2026，标题原文 "SRW-YOLOv8n: a high-precision method for main-stem detection and clamping-point positioning of plug pepper seedlings"，DOI: 10.3389/fpls.2026.1789467
- 等级：JCR Q1
- 关键数字（原文损失消融表，mAP@0.5/%）：CIoU 91.7、DIoU 91.2、GIoU 91.6、EIoU 92.6（+0.9）、SIoU 92.3（+0.6）；正文最终采用 WIoU（动态聚焦，处理宽高比相同失效问题）。
- 对 B2 的启示：真实苗株场景中 EIoU > SIoU > GIoU ≈ CIoU > DIoU 的排序，与"近圆框宽高比退化"假设一致；且该研究同样选中 WIoU 系（与本项目 WIoUv3 +0.166 呼应）。

### 18.2 NVW-YOLOv8s（番茄多成熟度检测+分割，VFL+WIoU）

- 出处：Computers and Electronics in Agriculture，2024，标题原文 "NVW-YOLOv8s: An improved YOLOv8s network for real-time detection and segmentation of tomato fruits at different ripeness stages"（Aichen Wang et al.），DOI: 10.1016/j.compag.2024.108833
- 等级：JCR Q1
- 关键数字：针对类别失衡、小目标、遮挡，采用 Varifocal Loss（分类）+ WIoU（回归），检测与分割 mAP@0.5 分别提升 4.8% 与约 5%（原文摘要）。
- 对 B2 的启示：VFL+WIoU 组合在农业小目标+遮挡/密集场景有整刊实证；VFL 可单独移植（见第 13 条）。

### 18.3 MTD-YOLO（樱桃番茄果穗多任务，Scale-Sensitive IoU）

- 出处：Computers and Electronics in Agriculture，2024，vol. 216，108533，标题原文 "MTD-YOLO: Multi-task deep convolutional neural network for cherry tomato fruit bunch maturity detection"
- 等级：JCR Q1
- 关键点：密集果穗（与甘蔗苗粘连类似的密集相邻目标），以 Scale-Sensitive IoU（SIOU，Du et al. 遥感尺度敏感版，注意与第 6 条 Scylla-SIoU 不同）替换 CIoU 提升识别精度。
- 对 B2 的启示：密集农业目标场景中"尺度敏感"的 IoU 变体有落地证据；尺度敏感思想与 Shape-IoU（第 5 条）相通。

---

## 检索记录（已排除/未收录条目）

- H2RBox（ICLR 2023，"H2RBox: Horizontal Box Annotation is All You Need for Oriented Object Detection"）：已核实出处（ICLR 2023，arXiv:2210.06742）。属弱监督旋转框检测（水平标注→旋转框），其视角一致性学习与本项目水平框 B2 瓶颈关联弱，收录价值低，排除。
- JSCD：多轮检索（"JSCD" + bbox loss/Gaussian/oriented detection）未找到任何边界框回归损失出处，命中均为通信领域 Joint Source Channel Decoding，按规则丢弃。
- UniBox（"Unified Box Regression"）：任务提示中提及，但经 arXiv API（标题/全文）与多轮网络检索均未找到可靠出处，疑似记忆错误条目，按规则丢弃。
- KFIoU（arXiv:2201.12558，"The KFIoU Loss for Rotated Object Detection"）：未检索到同行评审发表；机理与 GWD/KLD 同属高斯建模、且面向旋转框，不单列。
- IndexNet（ICCV 2019，"Indices Matter: Learning to Index for Deep Image Matting"）：出处已核实，但原文面向图像抠图，未检索到检测上采样方向的新实证，按主题 4"只收录有新证据"规则排除。
- SAIOU（TechRxiv 预印本）、Unified-IoU（arXiv:2408.06636）：均为未经评审预印本且机理与上文条目重叠，不收录。
- DySample（ICCV 2023）、FreqFusion（CVPR 2024）、DyHead、Focaler-IoU、PIoU2、WIoUv3、Slide-Loss：本项目已试，不重复收录。

## 优先级建议（供实验排期参考）

- 第一梯队（机理最对症 + 成本最低，纯损失替换）：Inner-IoU（可叠加于 WIoU）、MPDIoU、EIoU、NWD 损失版。四者分别针对"高 IoU 梯度放大""宽高比退化""边长独立回归""小框 IoU 过敏"，与 B2 的四个侧面一一对应。
- 第二梯队（损失/策略，低成本对照）：Shape-IoU、Alpha-IoU、SIoU、VFL（分类损失）、TAL α/β 调节。
- 第三梯队（结构/分配改动，成本中–高）：GFLv2-DGQP、RFLA、CARAFE、Softer-NMS、YOLOv10 双分配。
- 评估纪律：以 mAP75 为主选指标；每方案 ≥2 种子；增益 <0.2 mAP50 视为未超出噪声地板。
