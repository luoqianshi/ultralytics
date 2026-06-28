# YOLO12-NWD-P2-DySample 改进方案设计文档

> **日期：** 2026-06-27
> **目标数据集：** SSDC-UAV（Sugarcane Seedling Detection，甘蔗幼苗检测，单类 UAV 航拍）
> **基线模型：** YOLO12s（从随机初始化训练，无 COCO 预训练权重）
> **基线 mAP@.5:.95：** 0.5487（150 epoch）
> **目标：** mAP +1.5~2.5%，mAP_small +2~5%

---

## 0. 背景与失败根因

### 0.1 已有失败实验汇总

| 改进方案 | mAP@.5:.95 | Δ vs YOLO12s |
|---------|-----------|--------------|
| YOLO12s baseline (re0) | 0.5487 | 0 |
| YOLO12s baseline (300 Epoch) | 0.5504 | +0.17% |
| YOLO12s-SimAM | 0.5497 | +0.10% |
| YOLO12s-A2C2f_Mona | 0.5478 | -0.09% |
| YOLO12s-P2 | 0.5470 | -0.17% |
| YOLO12s-ESMoE | 0.5458 | -0.29% |
| YOLO12s-HPDown | 0.5433 | -0.54% |
| YOLO12s-AssemFormer-HSFPN | 0.5393 | -0.94% |

### 0.2 失败根因

**SSDC-UAV 是单类检测**（`nc: 1`），分类损失几乎为零，**整个训练瓶颈在 bbox 回归**。所有既有改进遵循"单模块插入"模式，未触及真痛点：

1. **YOLO12 的 A2C2f 已针对 from-scratch 调优**，叠加 SimAM/Mona/ESMoE 只是冗余 + 引入未训练参数
2. **CIoU 损失对小目标极不友好**：小目标 bbox 上几个像素偏差就让 IoU 从 0.5 跳到 0，梯度噪声极大
3. **P2 head 单独加无效**：之前 P2 实验用 CIoU 监督，新加的检测头仍学不好小目标
4. **HPDown 全替换破坏特征层次**：所有下采样都换让模型整体特征 hierarchy 重学

**核心结论：瓶颈是损失函数不匹配小目标场景，而不是架构缺东西。**

---

## 1. 总体架构

### 1.1 改动范围

3 处独立改动，可单独回滚：

| # | 改动 | 文件位置 | 性质 |
|---|------|---------|------|
| 1 | CIoU → NWD 损失 | `ultralytics/utils/loss.py` BboxLoss + `metrics.py` 加 `nwd()` + `default.yaml` 加开关 | **核心**，训练时生效，推理零成本 |
| 2 | 加轻量 P2 检测头 | 新建 `scripts/improved_yolo12/yolo12-NWD-P2-DySample.yaml` | 架构改，+~1.2M params |
| 3 | Upsample → DySample | 复制 DySample.py 到 `ultralytics/nn/modules/`，在 `tasks.py` 注册 | 架构改，+~0.1M params |

### 1.2 架构对比

```
YOLO12s baseline (9.3M / 21.7G)         YOLO12s-NWD-P2-DySample (~10.5M / ~25.3G)
┌────────────────────────────┐          ┌────────────────────────────────────┐
│ Backbone (P1-P5)  [不变]    │          │ Backbone (P1-P5)  [不变]            │
├────────────────────────────┤          ├────────────────────────────────────┤
│ Head                       │          │ Head                                │
│  P5 ─Upsample─→ P4融合     │          │  P5 ─DySample─→ P4融合  [改动3]      │
│  P4 ─Upsample─→ P3融合     │          │  P4 ─DySample─→ P3融合  [改动3]      │
│                            │          │  P3 ─DySample─→ P2融合  [改动2 新增] │
│  3-scale Detect(P3,P4,P5)  │          │  4-scale Detect(P2,P3,P4,P5) [改动2]│
├────────────────────────────┤          ├────────────────────────────────────┤
│ Loss: CIoU + BCE + DFL     │          │ Loss: NWD + BCE + DFL  [改动1 核心]  │
└────────────────────────────┘          └────────────────────────────────────┘
```

### 1.3 设计原则

- **改动 1 是真因**：NWD 解决小目标 bbox 回归梯度噪声问题
- **改动 2 依赖改动 1**：P2 head 之前失败正是因为 CIoU 监督下小目标学不好；有了 NWD 才能发挥
- **改动 3 是辅助**：DySample 让上采样保留更多高频细节，对 P2 路径尤其重要
- **可消融验证**：3 个改动都能单独开关，方便定位是哪个组件起作用

### 1.4 约束

| 项目 | 约束 |
|------|------|
| 模型架构 | ✅ 可改 |
| 损失函数 | ✅ 可改 |
| 训练超参 | ❌ 不动（epochs=150, imgsz=640, batch=16, SGD） |
| 数据增强 | ❌ 不动（保留默认 Mosaic/MixUp） |
| 预训练权重 | ❌ 不使用（从随机初始化训练） |
| 推理预算 | 10~50% 增加（≤ ~14M params / ~32 GFLOPs） |

---

## 2. NWD 损失实现

### 2.1 NWD 数学原理

把 bbox $(x_1,y_1,x_2,y_2)$ 建模为 2D 高斯分布：
- 均值 $\mu = (c_x, c_y)$ = bbox 中心
- 协方差 $\Sigma = \text{diag}((w/2)^2, (h/2)^2)$，其中 $w, h$ 是 bbox 宽高

两个高斯之间的 **Wasserstein-2 距离**：
$$W^2 = (\Delta c_x)^2 + (\Delta c_y)^2 + \left(\frac{\Delta w}{2}\right)^2 + \left(\frac{\Delta h}{2}\right)^2$$

**NWD** = $\exp\left(-\frac{\sqrt{W^2}}{C}\right)$，取值在 [0, 1]，类似 IoU 但**对小目标平移/尺寸微扰更平滑**。

### 2.2 CIoU vs NWD 梯度对比

| 场景 | CIoU 梯度 | NWD 梯度 |
|------|----------|---------|
| 16x16 目标偏移 2px | IoU 暴跌 ~0.15，梯度大噪声 | NWD 仅降 ~0.03，梯度平滑 |
| 4x4 目标偏移 2px | IoU 暴跌 ~0.4，几乎无法学 | NWD 仅降 ~0.08，仍可学 |
| 64x64 目标偏移 2px | IoU 降 ~0.04，正常 | NWD 降 ~0.01，正常 |

### 2.3 注入点

修改 `ultralytics/utils/loss.py` 的 `BboxLoss.forward()`（位于第 132 行）：

```python
# === 改前 ===
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

# === 改后（支持消融）===
if self.use_nwd:
    nwd_val = nwd(pred_bboxes[fg_mask], target_bboxes[fg_mask], self.nwd_c)
    if self.nwd_alpha < 1.0:
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        iou = self.nwd_alpha * nwd_val + (1 - self.nwd_alpha) * iou  # 加权融合
    else:
        iou = nwd_val  # 纯 NWD
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```

### 2.4 新增 `nwd()` 函数

加入 `ultralytics/utils/metrics.py`：

```python
def nwd(box1: torch.Tensor, box2: torch.Tensor, C: float = 12.0, eps: float = 1e-7) -> torch.Tensor:
    """Normalized Wasserstein Distance: 把 bbox 视为 2D 高斯，返回 NWD 相似度 ∈ [0,1]。
    Args:
        box1/box2: (N, 4) xyxy 格式
        C: 归一化常数，控制小目标敏感度。C 越小对小目标越敏感。
    """
    cx1, cy1 = (box1[..., 0] + box1[..., 2]) / 2, (box1[..., 1] + box1[..., 3]) / 2
    w1, h1 = box1[..., 2] - box1[..., 0], box1[..., 3] - box1[..., 1]
    cx2, cy2 = (box2[..., 0] + box2[..., 2]) / 2, (box2[..., 1] + box2[..., 3]) / 2
    w2, h2 = box2[..., 2] - box2[..., 0], box2[..., 3] - box2[..., 1]
    w_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) / 2) ** 2 + ((h1 - h2) / 2) ** 2
    return torch.exp(-torch.sqrt(w_dist + eps) / C)
```

### 2.5 配置开关

`ultralytics/cfg/default.yaml` 新增 3 个超参：

```yaml
nwd: False        # 是否启用 NWD 替代 CIoU
nwd_c: 10.0       # NWD 归一化常数，小目标场景典型 6~12
nwd_alpha: 1.0    # NWD 权重；1.0 = 纯 NWD，0.5 = NWD/CIoU 各半
```

`BboxLoss.__init__` 接收这 3 个参数，由 `v8DetectionLoss` 从 trainer 透传。

### 2.6 超参数选择

| 参数 | 默认值 | 选择理由 |
|------|--------|---------|
| `C` | **10.0** | NWD 原论文用 12.0（AI-TOD 平均目标 ~12px）；SSDC-UAV 幼苗可能更小，略降到 10.0。若 mAP_small 不达预期，可在 {6, 8, 10, 12} 网格搜索 |
| `alpha` | **1.0** | SSDC-UAV 是单类全小目标场景，纯 NWD 更对症。若 mAP_large 下降明显可降到 0.7 |

### 2.7 不动的部分

- **DFL loss** 完全不动（bbox 分布预测与 NWD 无关）
- **BCE cls loss** 完全不动（单类几乎为零）
- **TaskAlignedAssigner** 不动（label assignment 与 IoU/NWD 选择正交）
- **bbox_iou 函数保留**（仍用于评估指标计算和 NWD_alpha 混合模式）

### 2.8 风险点

1. **C 超参敏感**：C 过小会让大目标梯度过度平滑，C 过大就退化回普通 L2。**缓解**：默认 10.0 是保守值，且 `nwd_alpha` 可降级到混合模式
2. **NWD 对极小目标（<4px）仍可能失效**：SSDC-UAV 中如果有大量 <4px 目标，NWD 也救不了
3. **从零训练时 NWD 收敛速度未知**：NWD 文献多基于预训练微调，纯从零训练可能需要更多 epoch warmup。**缓解**：训练超参不能动，但若 50 epoch 仍未收敛则降低 nwd_alpha 到 0.5

---

## 3. 轻量 P2 head + DySample 集成

### 3.1 P2 head 结构对比

| 维度 | 失败的 yolo12-P2.yaml | 本设计（轻量版） |
|------|---------------------|----------------|
| P2 head 层数 | **2 层** A2C2f | **1 层** A2C2f |
| P2 通道数 | 128 (nominal) | 128 (nominal)，相同 |
| 监督方式 | CIoU（小目标梯度噪声大） | **NWD**（小目标友好） |
| 预算增加 | +~1.5M params | +~0.8M params |

**关键差异：** P2 head 加同样的容量，但有 NWD 监督 vs 没有 NWD 监督，效果天差地别。P2 head 是给 NWD 提供专门的小目标监督通道，不是单独起作用。

### 3.2 完整新 yaml 结构

文件路径：`scripts/improved_yolo12/yolo12-NWD-P2-DySample.yaml`

```yaml
nc: 1  # SSDC-UAV 单类
scales:
  s: [0.50, 0.50, 1024]

backbone:  # 完全不变
  - [-1, 1, Conv,  [64, 3, 2]]        # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2]]       # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]] # 2
  - [-1, 1, Conv,  [256, 3, 2]]       # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]] # 4
  - [-1, 1, Conv,  [512, 3, 2]]       # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]    # 6
  - [-1, 1, Conv,  [1024, 3, 2]]      # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]   # 8

head:
  # Top-down P5 → P4
  - [-1, 1, DySample, [512]]                  # 9  [替换 Upsample]
  - [[-1, 6], 1, Concat, [1]]                 # 10
  - [-1, 2, A2C2f, [512, False, -1]]          # 11

  # Top-down P4 → P3
  - [-1, 1, DySample, [256]]                  # 12 [替换 Upsample]
  - [[-1, 4], 1, Concat, [1]]                 # 13
  - [-1, 2, A2C2f, [256, False, -1]]          # 14

  # Top-down P3 → P2  [新增轻量 P2 路径]
  - [-1, 1, DySample, [128]]                  # 15 [新增]
  - [[-1, 2], 1, Concat, [1]]                 # 16 [新增] cat backbone P2
  - [-1, 1, A2C2f, [128, False, -1]]          # 17 [新增] 单层 P2 输出

  # Bottom-up P2 → P3
  - [-1, 1, Conv, [128, 3, 2]]                # 18 [新增]
  - [[-1, 14], 1, Concat, [1]]                # 19 [新增]
  - [-1, 2, A2C2f, [256, False, -1]]          # 20 (P3/8-small)

  # Bottom-up P3 → P4
  - [-1, 1, Conv, [256, 3, 2]]                # 21
  - [[-1, 11], 1, Concat, [1]]                # 22
  - [-1, 2, A2C2f, [512, False, -1]]          # 23 (P4/16-medium)

  # Bottom-up P4 → P5
  - [-1, 1, Conv, [512, 3, 2]]                # 24
  - [[-1, 8], 1, Concat, [1]]                 # 25
  - [-1, 2, C3k2, [1024, True]]               # 26 (P5/32-large)

  - [[17, 20, 23, 26], 1, Detect, [nc]]       # 27 Detect(P2, P3, P4, P5)
```

### 3.3 DySample 集成

**模块注册：**

1. 复制 `.lqs/improved_resource/AddModules/DySample.py` → `ultralytics/nn/modules/DySample.py`
2. 在 `ultralytics/nn/modules/__init__.py` 导出 `DySample`
3. 在 `ultralytics/nn/tasks.py` 的 `parse_model()` 注册表中，将 `DySample` 加入与 `nn.Upsample` 同一分支处理：
   ```python
   if m in {nn.Upsample, DySample}:
        c2 = ch[f]  # 输入通道 = 前层输出通道
        args = [c2, *args]  # 把 in_channels 注入到 args 头部
   ```

**替换点（3 处）：**
- Line 9: P5→P4 上采样（512 ch）
- Line 12: P4→P3 上采样（256 ch）
- Line 15: P3→P2 上采样（128 ch，**新增**）

**DySample 优势：** 用一个轻量 Conv 学到内容感知的采样偏移（offset），对边缘/纹理保留远好于 nearest；参数量极小（每组仅 2 个 filter）。

### 3.4 预算核算

| 组件 | 新增 params | 新增 GFLOPs |
|------|-----------|-------------|
| P2 head（A2C2f 单层 + Conv + DySample）| ~0.7M | ~2.1 |
| 2 处 Upsample → DySample（P5→P4, P4→P3）| ~0.1M | ~0.3 |
| Bottom-up P2→P3 路径 | ~0.4M | ~1.2 |
| **合计** | **~1.2M** | **~3.6** |
| **总模型** | **~10.5M** | **~25.3** |

**预算占用：** params +13%，FLOPs +17%，远低于 50% 上限，留有余量给后续叠加方案 B 的辅助头。

### 3.5 关键设计抉择

1. **P2 单层 A2C2f 而非 2 层**：失败版本用 2 层是为了"加容量"，但本设计的核心是 NWD 提供监督，容量并非瓶颈。单层省 0.4M params 且训练更快收敛
2. **Bottom-up 路径必须补回 P2→P3**：保证 P3/P4/P5 输出仍能融合 P2 的细节信息，不破坏原有 hierarchy
3. **DySample 仅替换 head 中 3 处上采样**：backbone 的下采样保持 Conv stride-2 不动（避免重蹈 HPDown 全替换覆辙）
4. **Detect 输出 4 scale**：训练时 NWD 同时监督 P2/P3/P4/P5 四个 head；推理时输出 4 scale 预测

### 3.6 风险点

1. **DySample 在 from-scratch 训练时可能学不动 offset**：DySample 的 offset Conv 从零初始化，前几个 epoch 可能退化成最近邻。**缓解**：DySample 内部已用 `normal_init(std=0.001)` 初始化，配合 `init_pos` 缓冲，前期行为接近 nearest，不会破坏训练
2. **P2 head 增加显存**：640 输入下 P2 特征图 160x160，比 P3 (80x80) 大 4 倍。batch=16 下可能显存吃紧。**缓解**：若 OOM，可降 P2 head 通道到 64
3. **4-scale Detect 改变 anchor/assignment**：YOLO 的 TaskAlignedAssigner 会自动适配 4 scale，无需手动调整

---

## 4. 消融实验设计 + 风险降级

### 4.1 消融实验矩阵

#### 阶段 1（必跑）：核心验证

| 实验 ID | 配置 | 训练成本 | 验证目标 |
|---------|------|---------|---------|
| **E0** | YOLO12s baseline | 0（已有结果 0.5487） | 对照基线 |
| **E1** | YOLO12s + **NWD only**（无 P2、无 DySample） | 1×150ep | **最关键**：NWD 单独是否有效？ |
| **E2** | YOLO12s + NWD + P2 + DySample（完整方案 A） | 1×150ep | 完整方案上限 |

**阶段 1 决策树：**

```
E1 vs E0:
├─ +0.5% 以上  → NWD 是真因，进入 E2 验证协同效应
├─ 持平 ~+0.5% → NWD 不够强，进入风险降级（4.3 节）
└─ 退化         → NWD 在 from-scratch 训练失效，立即切换方案 B

E2 vs E1:
├─ +0.5% 以上  → P2/DySample 协同有效，方案 A 确认
├─ 持平        → P2/DySample 无增益，采用 E1 配置即可
└─ 退化        → P2/DySample 破坏 NWD 收敛，回退到 E1
```

#### 阶段 2（条件触发，仅当 E2 成功且需归因时）

| 实验 ID | 配置 | 触发条件 |
|---------|------|---------|
| E3 | YOLO12s + NWD + P2（无 DySample） | E2 成功，需判断 DySample 是否必要 |
| E4 | YOLO12s + NWD + DySample（无 P2） | E2 成功，需判断 P2 是否必要 |

### 4.2 训练/验证协议

**训练（不变）：**
- epochs=150, imgsz=640, batch=16, optimizer=SGD, 默认 Mosaic/MixUp
- 训练命令增加 NWD 开关：`model.train(..., nwd=True, nwd_c=10.0, nwd_alpha=1.0)`

**验证（不变）：**
- 用已有 `scripts/coco_test/` 批量评估流程
- 输出 5 个指标：mAP@.5:.95, mAP50, mAP75, mAP_small, mAP_large
- 保存到 `runs/ssdc_uav_test/<exp_name>/predictions.json`

**关键监控点（训练中）：**
- **第 30 epoch 检查**：loss_iou 是否正常下降（不应明显大于 CIoU baseline 的 loss）
- **第 50 epoch 检查**：mAP_small 是否开始超过 baseline 的 0.3256
- **若第 50 epoch 仍不达 baseline 80%**：触发降级（见 4.3）

### 4.3 风险降级路径

| 风险场景 | 检测信号 | 降级动作 |
|---------|---------|---------|
| **NWD 纯替换收敛差** | E1 第 50 epoch mAP < baseline 80% | `nwd_alpha: 1.0 → 0.5`（NWD+CIoU 混合）重训 |
| **NWD 完全失效** | E1 退化 ≥ -0.5% | 切换方案 B（辅助头 + LSK + Wise-IoU） |
| **P2 head OOM** | 训练时 CUDA OOM | P2 通道 128→64，或 A2C2f 改为 C3k2 |
| **DySample 学不动** | E2 vs E1 持平或退化 | yaml 中 DySample 改回 nn.Upsample |
| **mAP_large 反而下降** | E2 mAP_large < 0.6476 | `nwd_alpha: 1.0 → 0.7`，给大目标保留 CIoU |
| **整体仅 +0~0.5%** | E2 vs E0 ≤ +0.5% | 叠加方案 B 的训练用辅助 P2 head |

### 4.4 成功标准

| 等级 | 标准（vs baseline 0.5487） | 后续动作 |
|------|---------------------------|---------|
| **失败** | ≤ +0.5% | 进入风险降级或切换方案 B |
| **及格** | +0.5% ~ +1.5% | 接受结果，记录分析；可选叠加方案 B |
| **达标** | +1.5% ~ +2.5%（预期） | ✅ 接受，发布为最终方案 |
| **超预期** | ≥ +2.5% | ✅ 强结果，可考虑写技术报告 |

**特别关注 mAP_small 提升：**
- baseline mAP_small = 0.3256
- **达标要求**：mAP_small ≥ 0.3456（+2%）
- **超预期**：mAP_small ≥ 0.3656（+4%）

### 4.5 与既有失败实验的隔离

**禁止混入失败组件：**
- ❌ 不使用 SimAM、Mona、ESMoE、AssemFormer、HSFPN（已验证失败）
- ❌ 不替换 backbone 下采样（HPDown 已验证失败）
- ❌ 不改 A2C2f 块本身（A2C2f_Mona 已验证失败）

**只引入新组件：**
- ✅ NWD（损失，未试过）
- ✅ DySample（上采样，未试过；且只替换 head 中 3 处，不动 backbone）
- ✅ P2 head（架构，单独试过失败，但配合 NWD 重试是新的组合）

---

## 5. 实现文件清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `ultralytics/utils/metrics.py` | 新增函数 | `nwd(box1, box2, C=12.0, eps=1e-7)` |
| `ultralytics/utils/loss.py` | 修改 BboxLoss | `__init__` 接收 nwd 参数；`forward` 中加入 NWD 分支 |
| `ultralytics/utils/loss.py` | 修改 v8DetectionLoss | 透传 nwd 参数给 BboxLoss |
| `ultralytics/cfg/default.yaml` | 新增 3 个键 | `nwd: False`, `nwd_c: 10.0`, `nwd_alpha: 1.0` |
| `ultralytics/nn/modules/DySample.py` | 新建文件 | 从 `.lqs/improved_resource/AddModules/DySample.py` 复制 |
| `ultralytics/nn/modules/__init__.py` | 导出 | `from .DySample import DySample` |
| `ultralytics/nn/tasks.py` | 修改 parse_model | 注册 DySample，与 nn.Upsample 同处理 |
| `scripts/improved_yolo12/yolo12-NWD-P2-DySample.yaml` | 新建 yaml | 完整 4-scale head 配置（见 3.2） |
| `scripts/improved_train/train_yolov12-NWD_ssdc-uav_re0.py` | 新建训练脚本 | E1 实验：YOLO12s + NWD only |
| `scripts/improved_train/train_yolov12-NWD-P2-DySample_ssdc-uav_re0.py` | 新建训练脚本 | E2 实验：完整方案 A |

---

## 6. 不在本次范围内（YAGNI）

以下不在本次实现范围，留待方案 A 验证失败后再考虑：

- 方案 B 的辅助训练头 + LSK + Wise-IoU
- 方案 C 的 HWD 频率下采样 + FreqFusion + ASFF head
- NWD 超参（C, alpha）的网格搜索（默认 10.0/1.0 起步，仅在未达标时搜索）
- 训练超参调整（imgsz, epochs, batch, optimizer 均不动）
- 数据增强调整（Mosaic/MixUp 不动）
- 预训练权重引入（明确从零训练约束）

---

**文档版本：** v1.0
**作者：** brainstorming session
**下一步：** 等待用户审阅本文档；批准后调用 writing-plans 技能生成详细实现计划。
