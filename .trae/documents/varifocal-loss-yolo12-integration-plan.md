# Varifocal Loss（VarifocalNet）提取并接入 Ultralytics YOLO12 方案

> 目标：从 [hyz-xmaster/VarifocalNet](https://github.com/hyz-xmaster/VarifocalNet)（CVPR 2021 Oral）提取 Varifocal Loss，按本仓库已有自定义损失（SlideLoss / WIoU v3 / SD Loss / Powerful-IoU）的接入惯例集成到 YOLO12 训练损失中，默认关闭、完全向后兼容。
>
> 背景衔接：20260901 瓶颈调研确认 B1 主瓶颈为"低分漏检"（94.7% FN 框已出但分数中位数仅 0.123），推荐首选 Varifocal Loss 修复分类打分与定位质量脱节的问题。

---

## 一、调研结论

### 1.1 VarifocalNet 原始实现（已核实源码）

损失定义在 `mmdet/models/losses/varifocal_loss.py`（raw 已抓取确认），核心公式：

```python
# pred/target: (N, C) 逐元素
pred_sigmoid = pred.sigmoid()
if iou_weighted:   # 官方默认 True
    focal_weight = target * (target > 0.0).float() + \
        alpha * (pred_sigmoid - target).abs().pow(gamma) * (target <= 0.0).float()
else:
    focal_weight = (target > 0.0).float() + \
        alpha * (pred_sigmoid - target).abs().pow(gamma) * (target <= 0.0).float()
loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
# 官方默认：alpha=0.75, gamma=2.0, iou_weighted=True
```

- **正样本**（target>0）：损失权重 = IACS 软标签值本身（预测框与 GT 框的 IoU）
- **负样本**（target=0）：权重 = α·p^γ（p 为 sigmoid 概率），非对称降权简单负样本
- 由于负样本 t=0，`|p−t|^γ ≡ p^γ`，与 mmdetection 后续移植版数学等价

IACS 目标构造（VFNet head 的 `get_targets`）：ATSS 分配出的每个正样本位置，在其 GT 类别通道上填入 **预测框（decode 后、detach）与 GT 框的 IoU**，其余通道为 0。这是 VFL 有效性的关键——分类分支被训练为"存在性 × 定位质量"的联合打分（动机实验：仅换打分口径 AP 56.1 → 74.7）。

### 1.2 本仓库现有损失接入惯例（已核实）

| 惯例要素 | 现状（以 SlideLoss / WIoU 为准） |
|---|---|
| 损失类位置 | `ultralytics/utils/loss.py`（**不放 AddModules**——AGENTS.md 的 AddModules 约定只针对 yaml 结构模块；损失函数一律走 utils/loss.py，SlideLoss/WIoU 均如此） |
| 代码标记 | `# @TODO Begin 引入XXX损失函数-YYYYMMDD ... # @TODO End` 注释块 + 中文说明 |
| 开关读取 | `v8DetectionLoss.__init__` 中 `h.get("xxx", False)`，`model.args` 为 `IterableSimpleNamespace`（带 `.get()`，训练时由 detect/train.py:147 `self.model.args = self.args` 注入） |
| 超参注册 | `ultralytics/cfg/default.yaml` 第 131-143 行 "Custom Loss Function Settings" 区块（wiou 有 1 个开关 + 2 个子参数的先例） |
| 启用提示 | `print("XXX 启用成功！请放心使用！")` |
| 单元测试 | `tests/test_wiou_v3.py`、`tests/test_bbox_loss_wiou.py`、`tests/test_piou.py` 先例 |

### 1.3 YOLO12 损失路径（已核实）

- `yolo12.yaml` → `DetectionModel.init_criterion()`（tasks.py:522）→ 非 end2end 时 `v8DetectionLoss`，end2end 时 `E2ELoss`（内部两个 `v8DetectionLoss` 实例）
- 分类损失唯一调用点：`v8DetectionLoss.get_assigned_targets_and_loss()`（loss.py:555-559），`v8SegmentationLoss` / `v8PoseLoss` / `PoseLoss26` / E2E 全部复用该方法 → 在此接入即可覆盖 yolo12 的 detect/seg/pose/E2E 全变体
- `v8OBBLoss` 有独立的 `loss()`（loss.py:1193 处自算 cls 损失），本次不改（OBB 非当前任务，可后续一行扩展）
- 注意尺度：调用点处 `pred_bboxes` 为特征图尺度、`target_bboxes` 为图像尺度（loss.py:546 佐证需乘 `stride_tensor`）

### 1.4 与仓库已有 `VarifocalLoss`（RT-DETR/DEIM 用）的关系

`ultralytics/utils/loss.py:23` 已有一个 `VarifocalLoss`，但它是 **RT-DETR 匹配器专用签名** `forward(pred_score, gt_score, label)` 且内部自带 `.mean(1).sum()` 归约，与 YOLO 稠密路径需要的"逐元素 (pred, target) 接口 + 外部 `/ target_scores_sum` 归约"不兼容。**不能复用、不能改名**（models/utils/loss.py:11 正在导入它），新类必须另取类名 `VarifocalNetLoss`。

---

## 二、方案设计

**核心决策：忠实移植 VFNet 原版损失 + IACS 软标签在线构造**（而非直接把 TAL 的 `target_scores` 当 VFL 目标）：

- 正样本软标签 = 该 anchor 解码预测框与被分配 GT 框的 **纯 IoU**（detach，不回传梯度）——与 VarifocalNet 原文、以及本仓库 RT-DETR/DEIM 分支的 VFL 目标口径一致
- TAL 的 `target_scores`（align_metric 归一化软标签）继续用于：标签分配、`target_scores_sum` 归一化、BboxLoss 正样本权重——**一律不动**，保证只改分类损失一项
- 归一化沿用现有 `.sum() / target_scores_sum`，损失量级与 BCE 基线可比，`hyp.cls` 增益可直接沿用
- 开关 `varifocal_loss` 默认 False，关闭时代码路径与现状完全一致（逐字节等价）

**覆盖范围**：YOLO12 detect / seg / pose / end2end（共享 `get_assigned_targets_and_loss`）；OBB 不接（如开启 flag 训练 OBB，cls 损失仍走 BCE，无副作用）。

---

## 三、具体改动（共 2 个文件修改 + 1 个新测试文件）

### 改动 1：`ultralytics/utils/loss.py` — 新增 `VarifocalNetLoss` 类

插入位置：SlideLoss 块之后（第 139 行 `# @TODO End` 与 BboxLoss 之间）：

```python
# @TODO Begin 引入Varifocal Loss损失函数（提取自VarifocalNet, CVPR 2021 Oral）-20260902
'''
Varifocal Loss，提取自 https://github.com/hyz-xmaster/VarifocalNet 的
mmdet/models/losses/varifocal_loss.py（去除 mmdet 注册器/weight_reduce_loss 依赖）。
正样本以 IACS 软标签（预测框与GT框的IoU）加权，负样本以 alpha*|p-t|^gamma 非对称降权，
用于修复"分类打分与定位质量脱节"导致的低分漏检（B1瓶颈）。
逐元素返回（等价 reduction='none'），归一化由调用处 .sum()/target_scores_sum 完成。
'''
class VarifocalNetLoss(nn.Module):
    """Varifocal Loss by Zhang et al. (VarifocalNet, CVPR 2021). Element-wise loss, no internal reduction.

    References:
        https://arxiv.org/abs/2008.13367
        https://github.com/hyz-xmaster/VarifocalNet
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, iou_weighted: bool = True):
        """Initialize with negative-part balance factor alpha and focusing parameter gamma (official defaults)."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute element-wise varifocal loss between predictions and IACS soft targets."""
        with autocast(enabled=False):  # 与本文件既有 VarifocalLoss 一致，AMP 下用 FP32 计算保数值稳定
            pred, target = pred.float(), target.float()
            assert pred.size() == target.size()
            pred_sigmoid = pred.sigmoid()
            if self.iou_weighted:
                focal_weight = target * (target > 0.0).float() + \
                    self.alpha * (pred_sigmoid - target).abs().pow(self.gamma) * (target <= 0.0).float()
            else:
                focal_weight = (target > 0.0).float() + \
                    self.alpha * (pred_sigmoid - target).abs().pow(self.gamma) * (target <= 0.0).float()
            return F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
# @TODO End 引入Varifocal Loss损失函数-20260902
```

（`autocast` 已在文件头 line 16 导入，无需新增 import。）

### 改动 2：`ultralytics/utils/loss.py` — `v8DetectionLoss` 两处修改

**(a) `__init__`（第 451 行 SlideLoss 块之后）新增开关读取：**

```python
        # @TODO 在向后兼容的基础上，引入Varifocal Loss（VarifocalNet）-20260902
        self.use_varifocal_loss = h.get("varifocal_loss", False)
        if self.use_varifocal_loss:
            if self.use_slide_loss:
                print("提示：slide_loss 与 varifocal_loss 同时开启，分类损失将优先使用 Varifocal Loss！")
            self.varifocal_loss = VarifocalNetLoss(
                alpha=h.get("varifocal_alpha", 0.75),
                gamma=h.get("varifocal_gamma", 2.0),
            )
            print("Varifocal Loss (VarifocalNet) 启用成功！请放心使用！")
```

**(b) 新增 IACS 软标签构造辅助方法**（放在 `get_assigned_targets_and_loss` 之后）：

```python
    # @TODO Begin 引入Varifocal Loss（VarifocalNet）-20260902
    def _build_varifocal_target(
        self,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        gt_labels: torch.Tensor,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """构造 VarifocalNet 式 IACS 分类软标签：正样本=预测框与GT框的IoU（detach），负样本=0。"""
        cls_target = torch.zeros_like(target_scores)
        if fg_mask.sum():
            # 图像尺度下计算被分配 anchor 的 预测框-GT框 IoU（不回传梯度）
            iou = bbox_iou(
                pred_bboxes[fg_mask] * stride_tensor[fg_mask], target_bboxes[fg_mask], xywh=False
            ).squeeze(-1).detach()
            batch_idx, anchor_idx = fg_mask.nonzero(as_tuple=True)
            pos_cls = gt_labels[batch_idx, target_gt_idx[batch_idx, anchor_idx], 0].long()  # 正样本GT类别
            cls_target[batch_idx, anchor_idx] = F.one_hot(pos_cls, self.nc) * iou.unsqueeze(-1)
        return cls_target.to(target_scores.dtype)
    # @TODO End 引入Varifocal Loss（VarifocalNet）-20260902
```

（`bbox_iou` 已在文件头 line 19 导入。`F.one_hot` 结果乘 `iou.unsqueeze(-1)` 自动广播到 nc 通道。）

**(c) `get_assigned_targets_and_loss` 分类损失调用点（原 555-559 行）改为 flag 分支：**

```python
        # Cls loss with optional class weighting
        # @TODO Begin 在向后兼容的基础上，引入Varifocal Loss（VarifocalNet）-20260902
        if self.use_varifocal_loss:
            cls_target = self._build_varifocal_target(
                target_scores, fg_mask, target_gt_idx, gt_labels, pred_bboxes, target_bboxes, stride_tensor
            )
            cls_loss = self.varifocal_loss(pred_scores, cls_target)  # VarifocalNet式IACS软标签
        else:
            cls_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc) 原始BCE路径
        # @TODO End
        if self.class_weights is not None:
            cls_loss *= self.class_weights
        loss[1] = cls_loss.sum() / target_scores_sum  # BCE/VFL
```

`target_gt_idx`、`gt_labels`、`pred_bboxes`、`stride_tensor` 均已在该函数作用域内（第 538/542/544 行），无额外计算。

### 改动 3：`ultralytics/cfg/default.yaml` — 注册超参（第 143 行 slide_loss 之后）

```yaml
# 6. Varifocal Loss (VarifocalNet, CVPR 2021)
varifocal_loss: False # (bool) enable VarifocalNet Varifocal Loss to replace BCE for classification loss (IACS soft labels)
varifocal_alpha: 0.75 # (float) balance factor for the negative part of Varifocal Loss
varifocal_gamma: 2.0 # (float) focusing parameter for the modulating factor
```

### 改动 4（新建）：`tests/test_varifocal_net_loss.py`

仿照 `tests/test_bbox_loss_wiou.py` 风格，覆盖：

1. `test_init_defaults` — 默认 alpha=0.75 / gamma=2.0 / iou_weighted=True
2. `test_output_shape` — 输出与输入逐元素同形状（bs, anchors, nc）
3. `test_negative_weight_formula` — 负样本（t=0）处 loss == α·p^γ·BCE(p,0)，与手算一致
4. `test_positive_weighted_by_target` — 正样本处 loss == t·BCE(p,t)（iou_weighted=True）；iou_weighted=False 时 == BCE(p,t)
5. `test_gradient_flow` — loss.backward() 后 pred.grad 非空且有限
6. `test_v8detection_loss_flag_off_backward_compat` — `DetectionModel("yolo12n.yaml")` + `model.args={}` → `use_varifocal_loss is False` 且 `self.bce` 仍为 `nn.BCEWithLogitsLoss`（向后兼容回归）
7. `test_v8detection_loss_flag_on` — `model.args={"varifocal_loss": True}` → `use_varifocal_loss is True` 且 `varifocal_loss` 为 `VarifocalNetLoss` 实例

---

## 四、向后兼容性说明

- `varifocal_loss` 默认 False：`h.get("varifocal_loss", False)` 对旧配置/旧 ckpt 恢复（train_args 里无此键）一律返回 False，走原 BCE 分支，**逐字节等价**于改动前代码
- 不修改任何现有类/函数签名：`VarifocalLoss`（RT-DETR 用）、`SlideLoss`、`BboxLoss`、TAL assigner、`target_scores_sum` 归一化、`hyp.cls` 增益全部不动
- EMA/断点续训（`load_checkpoint` 将 args 以 dict 附回，dict 同样支持 `.get()`）不受影响
- 仅当用户显式传 `varifocal_loss=True` 时才改变训练行为

## 五、使用方式

```bash
# CLI
yolo train model=yolo12n.pt data=your_data.yaml epochs=100 varifocal_loss=True

# Python API
from ultralytics import YOLO
model = YOLO("yolo12n.pt")
model.train(data="your_data.yaml", epochs=100, varifocal_loss=True)
```

超参默认取官方推荐（α=0.75, γ=2.0）；如需调参可用 `varifocal_alpha` / `varifocal_gamma`。分类增益 `cls=0.5` 建议先保持默认（VFL 与 BCE 的归一化口径一致，量级可比）。

## 六、验证步骤

1. **导入自检**：`python -c "from ultralytics.utils.loss import VarifocalNetLoss; print('ok')"`
2. **单元测试**：`python -m pytest tests/test_varifocal_net_loss.py -v`（全绿，含向后兼容回归项 6）
3. **冒烟训练（flag 开）**：`yolo train model=yolo12n.yaml data=coco8.yaml epochs=2 varifocal_loss=True imgsz=640` —— 确认打印"Varifocal Loss (VarifocalNet) 启用成功"，loss 正常下降无 NaN
4. **回归训练（flag 关）**：`yolo train model=yolo12n.yaml data=coco8.yaml epochs=2 imgsz=640` —— 与改动前基线 loss 曲线一致（不传新参即旧行为）
5. **正式实验**：在甘蔗检测数据集上以相同种子对比 baseline vs `varifocal_loss=True`，按调研标准验收（mAP50 ≥ +1.0 且 P/R 不同时降，Recall 优先）

## 七、假设与决策记录

| 决策点 | 选择 | 理由 |
|---|---|---|
| VFL 正样本目标 | 预测框-GT 框纯 IoU（VFNet 原版口径） | 忠实原文；与仓库 RT-DETR/DEIM 的 VFL 目标口径一致；避免 TAL align_metric 中混入 cls 分数带来的目标自指性 |
| 通用软标签方案（直接用 TAL `target_scores`） | 否决 | 那样正样本权重=归一化对齐度量而非定位质量，削弱"打分校准"这一核心收益；且与原仓库实现偏差更大 |
| IoU 目标是否 detach | 是 | 目标不应携带梯度；与 assigner 内 `pred_bboxes.detach()` 口径一致 |
| 归一化 | 沿用 `.sum() / target_scores_sum` | 与 BCE 基线量级可比，`hyp.cls` 无需重调；改动面最小 |
| 新类命名 | `VarifocalNetLoss` | 同文件已有 RT-DETR 用 `VarifocalLoss`，避免冲突 |
| OBB 分支 | 不接入 | 非当前任务；`v8OBBLoss.loss` 独立调用点保持 BCE，开启 flag 无副作用 |
| 提交代码 | 不自行 commit/push | 遵守 AGENTS.md：完成后报告改动，等用户确认 |
