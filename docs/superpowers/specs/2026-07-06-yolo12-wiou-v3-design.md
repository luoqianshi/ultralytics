# YOLO12-WIoU-v3 改进方案设计文档

> **日期：** 2026-07-06
> **目标数据集：** SSDC-UAV（Sugarcane Seedling Detection，甘蔗幼苗检测，单类 UAV 航拍）
> **基线模型：** YOLO12s + 官方 COCO 预训练权重 + 300 epoch 微调
> **基线 mAP@.5:.95：** 0.5563（E0'，COCO pretrain + 300ep）
> **目标：** mAP +0.3% ~ +2.0%，mAP_small +1% ~ +2%

---

## 0. 背景与前置实验

### 0.1 训练约束变更

本方案对 project memory 既有 hard constraints 做如下调整（仅对本 WIoU v3 实验生效）：

| 项目 | 原约束（project memory） | 本方案约束 |
|------|------------------------|-----------|
| 预训练权重 | ❌ 必须从零训练 | ✅ 加载官方 `yolo12s.pt`（COCO 预训练） |
| 训练轮数 | epochs=150 | **epochs=300** |
| imgsz / batch / optimizer | 不动 | 仍不动（640 / 16 / SGD） |
| 数据增强 | 不动 | 仍不动（默认 Mosaic/MixUp） |

### 0.2 NWD 失败教训

前一版 NWD 实验在 **from-scratch + 150ep** 下失败：

| 实验 | 配置 | mAP@.5:.95 | Δ vs baseline |
|------|------|-----------|---------------|
| YOLO12s baseline (re0, 150ep, from-scratch) | CIoU | 0.5487 | 0 |
| YOLO12s baseline (300ep, from-scratch) | CIoU | 0.5504 | +0.17% |
| YOLO12s baseline (300ep, **COCO pretrain**) | CIoU | **0.5563** | +0.76% |
| YOLO12s-NWD (150ep, from-scratch) | NWD 纯替换 | 0.5079 | **−4.08%**（失败） |

**NWD 失败根因**：NWD 把 IoU 损失整体替换为 Wasserstein 距离，梯度景观剧变，from-scratch 训练无法收敛。

**WIoU v3 的差异**：保留 IoU 作为基础损失（梯度景观熟悉），仅叠加动态聚焦系数 `r` 对梯度做调制。COCO pretrain 下模型已收敛到不错的初始点，WIoU v3 的动态聚焦更易发挥。

### 0.3 COCO pretrain 下的既有改进汇总

| 改进方案 | mAP@.5:.95 | Δ vs E0'(0.5563) |
|---------|-----------|------------------|
| **E0' baseline** (COCO pretrain + 300ep) | **0.5563** | 0 |
| yolo12s-DySample | 0.5576 | +0.13% |
| yolo12s-A2C2f_Mona | 0.5551 | −0.12% |
| yolo12s-A2C2f_EMA | 0.5531 | −0.32% |
| yolo12s-A2C2f_SCSA | 0.5391 | −1.72% |
| yolo12s-MoCA | 0.5395 | −1.68% |
| yolo12s-SPDConv | 0.4473 | −10.90% |

**结论**：COCO pretrain 下 baseline 已很强，绝大多数改进在噪声范围内或退化。WIoU v3 需真实增益才能 beat E0'。

---

## 1. 架构与改动范围

### 1.1 改动文件清单（4 个文件，全部可单独回滚）

| # | 文件 | 操作 | 性质 |
|---|------|------|------|
| 1 | `ultralytics/utils/metrics.py` | 新增 `wiou_v3()` 函数（紧邻现有 `bbox_iou`，与 NWD 预案的 `nwd()` 同位置） | **核心**，纯函数，无状态 |
| 2 | `ultralytics/utils/loss.py` | 修改 `BboxLoss.__init__`（加 EMA buffer + 配置参数）和 `BboxLoss.forward`（加 WIoU 分支） | **核心**，训练时生效 |
| 3 | `ultralytics/utils/loss.py` | 修改 `v8DetectionLoss.__init__`（透传 wiou 配置给 BboxLoss） | 透传链路 |
| 4 | `ultralytics/cfg/default.yaml` | 新增 3 个键：`wiou`、`wiou_alpha`、`wiou_momentum` | 配置入口 |

### 1.2 架构对比

```
YOLO12s baseline (CIoU)                YOLO12s-WIoU v3 (方案A)
┌────────────────────────────┐        ┌────────────────────────────────┐
│ Backbone / Head  [完全不变]  │        │ Backbone / Head  [完全不变]      │
├────────────────────────────┤        ├────────────────────────────────┤
│ Loss:                       │        │ Loss:                           │
│   bbox_iou(CIoU=True)       │  →     │   if wiou:                      │
│   → iou_sim                 │        │     wiou_v3(pred, gt, iou_mean) │
│   loss = (1 - iou_sim)*w    │        │     + EMA update of iou_mean   │
│                             │        │   else:                         │
│                             │        │     [原 CIoU 路径，保留]         │
└────────────────────────────┘        └────────────────────────────────┘
```

### 1.3 关键不变量（约束边界）

- Backbone / Head / Detect yaml **完全不动**（不引入 P2 head、不替换 Upsample、不动 A2C2f）
- DFL loss / BCE cls loss / TaskAlignedAssigner **完全不动**
- `bbox_iou` 函数本身 **不修改**（评估指标、`tal.py` 等多处依赖它，保持纯净）；WIoU 是 **新增** 路径，不是覆盖
- 训练超参：epochs=300, imgsz=640, batch=16, SGD, Mosaic/MixUp **不动**
- 预训练：加载 `yolo12s.pt`（COCO pretrain）

### 1.4 与既有失败实验的隔离

- 不混入 SimAM/Mona/ESMoE/AssemFormer/HSFPN/HPDown（已验证失败）
- 不重蹈 NWD 纯替换覆辙：默认 `wiou_alpha=1.0` 纯 WIoU，但保留 `<1.0` 的 CIoU 混合回退路径
- 不引入 P2 head / DySample / 架构改动（E1 仅验证损失函数替换）

### 1.5 改动规模

约 +30 行新代码（`wiou_v3()` 函数）+ ~25 行 BboxLoss 改动 + 3 行 yaml + 5 行 v8DetectionLoss 透传。总影响面 < 65 行。

---

## 2. WIoU v3 实现细节

### 2.1 WIoU v3 数学结构

三层嵌套（v1 → v3）：

```
L_WIoU-v3  =  r  ×  L_WIoU-v1
              ↑      ↑
              │      └── 基础损失 = 1 − IoU + R_WIoU
              │          R_WIoU = exp( center_dist² / (W_g² + H_g²) )
              │          center_dist² = (cx−cx_gt)² + (cy−cy_gt)²
              │          W_g, H_g = 最小外接框宽高（detach，不回传梯度）
              │
              └── 动态聚焦系数 = β / (α·|IoU − IoU_mean|^γ + δ)
                  β = detach(1 − IoU_mean)（尺度因子，让 r 均值≈1）
                  IoU_mean = BboxLoss 维护的 EMA（detach）
                  α, γ, δ = 论文常数（α=1.9, γ=3, δ=0.5；实现时对照 WIoU 官方仓库核实）
```

### 2.2 关键 detach 点（梯度只走 IoU 和 center_dist，不走聚焦系数）

- `W_g² + H_g²` → detach（外接框尺寸不回传）
- `IoU_mean` → detach（EMA 不回传）
- `β` → detach（尺度因子不回传）
- 整个 `r` → detach（聚焦系数作为权重，本身不参与梯度）

**梯度本质**：v3 相比 v1 的唯一区别是 `r` 对"离群 anchor"（IoU 偏离 mean 多）降权、对"普通 anchor"（IoU 接近 mean）保持权重。这就是动态聚焦的核心。

### 2.3 `wiou_v3()` 函数签名

放在 `ultralytics/utils/metrics.py`，紧邻现有 `bbox_iou`：

```python
def wiou_v3(
    box1: torch.Tensor,
    box2: torch.Tensor,
    iou_mean: torch.Tensor,       # 标量 EMA，由 BboxLoss 传入（已 detach）
    xywh: bool = False,           # 默认 xyxy，与 BboxLoss 调用一致
    alpha: float = 1.9,           # 论文默认；实现时对照参考仓库核实
    gamma: float = 3.0,
    delta: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Wise-IoU v3：返回 IoU 相似度 ∈ (-∞, 1]（与 bbox_iou 同约定，越大越好）。

    返回值 = 1 − L_WIoU-v3
    BboxLoss 中 (1 − 返回值)·weight = L_WIoU-v3·weight，符合现有 loss 约定。

    Args:
        box1/box2: (N, 4) xyxy 格式（默认）或 xywh 格式。
        iou_mean: 标量 tensor，跨 batch 的 IoU EMA（由 BboxLoss 维护，已 detach）。
        alpha/gamma/delta: WIoU v3 动态聚焦常数（论文默认值）。
    """
```

**为什么返回相似度而非 loss**：与 `bbox_iou` 约定一致（BboxLoss 用 `1 − iou` 当 loss），便于 `wiou_alpha` 混合 CIoU 时统一公式：`iou = wiou_alpha · wiou_v3_sim + (1 − wiou_alpha) · ciou_sim`，`loss = (1 − iou) · weight`。

### 2.4 BboxLoss 状态管理

```python
class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, wiou=False, wiou_alpha=1.0, wiou_momentum=0.5):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.use_wiou = wiou
        self.wiou_alpha = wiou_alpha
        self.wiou_momentum = wiou_momentum
        # IoU_mean EMA buffer：初始 1.0（乐观初值，随训练降到真实水平）
        self.register_buffer("iou_mean", torch.tensor(1.0))
        # ↑ register_buffer：随 model.to(device) 自动迁移、随 checkpoint 保存/加载

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        if self.use_wiou:
            # 计算 WIoU v3 相似度
            wiou_sim = wiou_v3(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                               iou_mean=self.iou_mean)

            # EMA 更新 IoU_mean（detach，不回传梯度）
            with torch.no_grad():
                batch_iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                                     xywh=False)  # plain IoU，无 CIoU
                self.iou_mean.mul_(self.wiou_momentum).add_(
                    (1 - self.wiou_momentum) * batch_iou.mean())

            if self.wiou_alpha < 1.0:
                # 混合模式：CIoU fallback
                ciou_sim = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                                    xywh=False, CIoU=True)
                iou = self.wiou_alpha * wiou_sim + (1 - self.wiou_alpha) * ciou_sim
            else:
                iou = wiou_sim  # 纯 WIoU v3

            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        else:
            # 原路径，完全保留
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                           xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss 完全不动（保持原逻辑）
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl
```

### 2.5 关键设计抉择

1. **`register_buffer` 而非普通属性**：buffer 自动随 `model.to(device)` 迁移、随 `state_dict()` 保存、随 checkpoint 恢复。普通属性会丢状态。
2. **EMA 初值 = 1.0**（乐观初值）：训练初期 IoU 低，EMA 从 1.0 缓慢下降到真实水平，避免初期 `r` 爆炸（若初值=0，`(IoU−0)^γ` 会很大，r 很小，梯度被压没）。
3. **EMA momentum = 0.5**（论文 β 的标准做法）：半衰期约 1 batch，响应快。若训练不稳可降到 0.9（更平滑）。
4. **混合模式复用 NWD design 的 `alpha` 模式**：`wiou_alpha < 1.0` 时与 CIoU 混合，是 NWD 失败后的回退路径。
5. **`wiou_v3()` 是纯函数，状态在 BboxLoss**：与 NWD design 一致，`metrics.py` 无状态、易测试。

### 2.6 风险点

1. **EMA 初值 1.0 可能偏高**：若前几个 epoch IoU 在 0.3 附近，EMA 从 1.0 降到 0.3 需 ~5 batch（momentum=0.5）。期间 `|IoU − IoU_mean|` 偏大，r 偏小，梯度被压制。**缓解**：COCO pretrain 下初始 IoU 应该不低（预训练已学好 bbox），EMA 不会偏离太久。
2. **`r` 的 detach 链路**：必须确保 `iou_mean`、`β`、整个 `r` 都在 `torch.no_grad()` 或 `.detach()` 内，否则梯度会通过 EMA 反传到历史 batch（错误）。
3. **常数 α/γ/δ 的核实**：实现时对照 WIoU 论文原文（"Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism", 2023）或其官方代码仓库核实公式与常数，避免记忆偏差。实现阶段需 web 搜索确认参考仓库 URL。

---

## 3. 配置开关与透传链路

### 3.1 `default.yaml` 新增 3 个键

放在 `ultralytics/cfg/default.yaml` 的 loss gain 区块（line 102-110 附近）：

```yaml
# Loss gains
box: 7.5
cls: 0.5
dfl: 1.5
...
wiou: False          # (bool) 启用 Wise-IoU v3 替代 CIoU（bbox 回归损失）
wiou_alpha: 1.0       # (float) WIoU/CIoU 混合权重；1.0=纯 WIoU v3，0.5=各半
wiou_momentum: 0.5   # (float) IoU_mean EMA 动量；0.5=响应快，0.9=平滑
```

**为什么 `wiou_momentum` 也暴露**：NWD design 只暴露 2 键，但 WIoU v3 的 EMA momentum 直接影响 `r` 的稳定性，是除 `wiou_alpha` 外最重要的可调旋钮。暴露它不增加复杂度（一个 float），但能在不重训的情况下做 momentum 敏感性分析。

### 3.2 透传链路

```
default.yaml (wiou/wiou_alpha/wiou_momentum)
    ↓ trainer 启动时合并到 model.args
v8DetectionLoss.__init__(model)
    ↓ h = model.args 读取 3 个键
    ↓ self.bbox_loss = BboxLoss(m.reg_max, h['wiou'], h['wiou_alpha'], h['wiou_momentum'])
BboxLoss.__init__(reg_max, wiou, wiou_alpha, wiou_momentum)
    ↓ register_buffer("iou_mean", tensor(1.0))
BboxLoss.forward(...)
    ↓ if self.use_wiou: 调 wiou_v3()，更新 EMA
```

### 3.3 `v8DetectionLoss.__init__` 改动（loss.py line 368）

```python
# === 改前 ===
self.bbox_loss = BboxLoss(m.reg_max).to(device)

# === 改后 ===
self.bbox_loss = BboxLoss(
    m.reg_max,
    wiou=h.get("wiou", False),
    wiou_alpha=h.get("wiou_alpha", 1.0),
    wiou_momentum=h.get("wiou_momentum", 0.5),
).to(device)
```

用 `h.get(key, default)` 而非 `h[key]`：向后兼容旧 checkpoint / 旧配置（无 wiou 键时回退到 CIoU，与现状一致）。

### 3.4 训练/测试脚本

新建（参照 `scripts/train_coco_pretrain/train_yolov12_ssdc-uav_300Epoch.py` 结构）：

- `scripts/improved_train/coco_pretrained/train_yolov12-WIoUv3_ssdc-uav.py`
- `scripts/improved_test/coco_pretrained/test_yolov12-WIoUv3_ssdc_uav.py`

训练脚本关键配置：

```python
model = YOLO('yolo12s.pt')  # COCO 预训练权重
results = model.train(
    data=str(yaml_path),
    epochs=300, imgsz=640, batch=16, optimizer='SGD',
    wiou=True, wiou_alpha=1.0, wiou_momentum=0.5,  # 新增
    project='runs/ssdc_uav_train',
    name='yolo12s_WIoUv3_ssdc_uav_exp1',
    device='0', save=True,
)
```

---

## 4. 消融实验协议与风险降级

### 4.1 实验矩阵

| 实验 ID | 配置 | 训练成本 | mAP 参考 |
|---------|------|---------|---------|
| **E0'** | YOLO12s + COCO pretrain + 300ep（baseline） | 0（已有） | **0.5563** / mAP_small 0.3366 |
| **E1** | YOLO12s + COCO pretrain + 300ep + **WIoU v3**（wiou=True, alpha=1.0, momentum=0.5） | 1×300ep | 待跑 |

### 4.2 训练/验证协议

**训练（不变）**：epochs=300, imgsz=640, batch=16, SGD, 默认 Mosaic/MixUp，加载 `yolo12s.pt`（COCO pretrain）。

**验证（不变）**：用 `scripts/coco_test/` 批量评估流程，输出 5 指标：mAP@.5:.95, mAP50, mAP75, mAP_small, mAP_large。

**训练中监控点**：
- **第 30 epoch**：`loss_iou` 是否正常下降（不应显著大于 CIoU baseline 的 loss 值；若 >2× baseline，说明 `r` 压制过度）
- **第 50 epoch**：检查 `iou_mean` buffer 是否收敛到合理范围（COCO pretrain 下应 ~0.4-0.6；若仍接近 1.0 初值，EMA 更新有 bug）
- **第 100 epoch**：mAP_small 是否开始超过 baseline 的 0.3366

### 4.3 成功标准（vs E0' = 0.5563）

| 等级 | 标准（Δ mAP） | mAP_small 要求 | 后续动作 |
|------|--------------|---------------|---------|
| **失败** | ≤ +0.3% | < 0.3366 | 进入风险降级（4.4） |
| **及格** | +0.3% ~ +1.0% | ≥ 0.3366 | 接受，记录；可选叠加 DySample |
| **达标** | +1.0% ~ +2.0% | ≥ 0.3466（+1%） | ✅ 接受，发布 |
| **超预期** | ≥ +2.0% | ≥ 0.3566（+2%） | ✅ 强结果 |

**注意**：COCO pretrain 下 baseline 已很强（0.5563），DySample 也仅 +0.13%。WIoU v3 达到"+0.3% ~ +1.0% 及格"就算成功，不要求 NWD design 的 +1.5~2.5%（那是 from-scratch 下的预期）。

### 4.4 风险降级路径

| 风险场景 | 检测信号 | 降级动作 |
|---------|---------|---------|
| **WIoU v3 收敛慢但未退化** | E1 第 100 epoch mAP < E0' 95% 但 > 90% | 等 300ep 完整结果，不中途降级 |
| **`r` 压制梯度** | 第 30 epoch loss_iou > 2× baseline | `wiou_momentum: 0.5 → 0.9`（EMA 更平滑，r 更稳定）重训 |
| **纯 WIoU v3 退化** | E1 最终 mAP < E0' − 0.3% | `wiou_alpha: 1.0 → 0.7 → 0.5`（CIoU 混合）重训 |
| **mAP_large 下降** | E1 mAP_large < 0.6586（E0'） | `wiou_alpha: 1.0 → 0.7`，给大目标保留 CIoU 监督 |
| **灾难性失效（类 NWD）** | E1 退化 ≥ −2% | 立即回退 `wiou=False`，放弃 WIoU v3 路线 |

---

## 5. 实现文件清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `ultralytics/utils/metrics.py` | 新增函数 | `wiou_v3(box1, box2, iou_mean, xywh=False, alpha=1.9, gamma=3.0, delta=0.5, eps=1e-7)` |
| `ultralytics/utils/loss.py` | 修改 BboxLoss | `__init__` 接收 wiou/wiou_alpha/wiou_momentum + register_buffer("iou_mean")；`forward` 加 WIoU 分支 + EMA 更新 |
| `ultralytics/utils/loss.py` | 修改 v8DetectionLoss | `__init__` 透传 wiou 配置给 BboxLoss（line 368） |
| `ultralytics/cfg/default.yaml` | 新增 3 个键 | `wiou: False`, `wiou_alpha: 1.0`, `wiou_momentum: 0.5` |
| `scripts/improved_train/coco_pretrained/train_yolov12-WIoUv3_ssdc-uav.py` | 新建训练脚本 | E1 实验：YOLO12s + COCO pretrain + 300ep + WIoU v3 |
| `scripts/improved_test/coco_pretrained/test_yolov12-WIoUv3_ssdc_uav.py` | 新建测试脚本 | E1 评估 |

---

## 6. 不在本次范围内（YAGNI）

以下不在本次实现范围，留待 E1 验证后再考虑：

- WIoU v3 超参（α, γ, δ）网格搜索（默认论文值起步，未达标再搜索）
- WIoU v1 / v2 对比实验（只验证 v3，论文已证 v3 最优）
- WIoU v3 + DySample 组合（E1 成功后再考虑）
- WIoU v3 + P2 head 组合（E1 成功后再考虑）
- 训练超参调整（imgsz, batch, optimizer 不动）
- 数据增强调整（Mosaic/MixUp 不动）
- 从零训练对比（本方案明确使用 COCO pretrain）

---

**文档版本：** v1.0
**作者：** brainstorming session
**下一步：** 等待用户审阅本文档；批准后调用 writing-plans 技能生成详细实现计划。
