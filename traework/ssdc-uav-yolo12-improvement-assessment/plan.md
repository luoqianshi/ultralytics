# Report Plan

## Meta
- Type: 算法改进综合评估报告（实验复盘 + 文献检索 + 下一步路线）
- Topic: SSDC-UAV 甘蔗幼苗检测 · YOLOv12s 改进实验评估与后续改进路线
- Audience: 研究者本人（目标：CEA 等顶刊论文，剩余 5 组 150-epoch 训练预算）
- Language: 中文

## Design System
- Palette: bg #F7F9F8 / surface #FFFFFF / ink #16241C / muted #5C6B63 / rule #E3E9E6 / accent #10B981 (Emerald) / accent2 #0EA5E9
- Typography: InstrumentSans (Latin标题) + 系统中文栈；JetBrainsMono 用于数据/代码；正文 15px/1.75
- Layout: 1080px 居中单列；Gradio 式极简卡片（1px 边框、无阴影堆叠）
- Components: h2 左侧色条 + 编号；表格细线；标注卡（accent 左边框）；无花哨动效

## Structure
1. 执行摘要（关键读数 + 核心结论卡）
2. 实验矩阵与协议（双基线问题：auto vs SGD）
3. 结果全景（总表 + Δ图 + P/R 散点 + 效率散点）
4. 数据集画像（尺寸分布按飞行高度）
5. 有效改进机理（权重保持律 / 损失双通道 / Focaler 实现瑕疵）
6. 无效改进机理（P2×9%小目标 / 注意力冗余+权重断裂 / SPDConv 工程事故）
7. 噪声底线与统计严谨性
8. 文献证据地图（CEA/TCSAE/TPAMI 等，带引用）
9. 推荐方案 A/B/C/D（SSFF+TFE、Soft-NMS、FreqFusion、计数监督/LSCD）
10. 五组实验执行路线（含决策门 + mermaid 流程）
11. 论文叙事建议（改进点分层、消融设计、计数验证、风险表）
12. 局限与诚实边界 + Sources

## Visuals
| Visual | Type | Tool |
|---|---|---|
| ΔmAP50-95 全方案 | 发散条形图 | ECharts |
| ΔP vs ΔR | 散点 | ECharts |
| 双基线对照（关键方案） | 分组柱状 | ECharts |
| 数据集尺寸分布（按高度） | 堆叠柱状 | ECharts |
| GFLOPs vs mAP50-95 | 散点 | ECharts |
| 5组实验决策路线 | 流程图 | Mermaid |

## Key Arguments
- 成功改进全部满足"保持 COCO 预训练权重完整"；失败改进几乎都伴随骨干/头部权重断裂
- 中目标 62% + 单类 → 定位质量是约束瓶颈，损失/分配器是高杠杆轴
- 现有损失增益 +0.05~+0.3 接近噪声底线，最终组合需多种子自证
- 下一步：SSFF+TFE（结构轴）+ Soft-NMS（后处理轴）与损失轴三轴正交，构成论文改进点
