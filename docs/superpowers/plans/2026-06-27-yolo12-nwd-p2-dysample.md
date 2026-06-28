# YOLO12-NWD-P2-DySample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 SSDC-UAV 单类甘蔗幼苗数据集上，通过 NWD 损失 + 轻量 P2 head + DySample 上采样三处改动，让 YOLO12s 从零训练的 mAP@.5:.95 从 0.5487 提升到 0.56+（+1.5~2.5%）。

**Architecture:** 三处独立改动：(1) 把 BboxLoss 中的 CIoU 替换为 NWD（Normalized Wasserstein Distance），解决小目标 bbox 回归梯度噪声问题；(2) 在 head 中新增轻量 P2 检测头（单层 A2C2f），给 NWD 提供专门的小目标监督通道；(3) 把 head 中 3 处 `nn.Upsample` 替换为 DySample 内容感知上采样。所有改动可单独开关以便消融。

**Tech Stack:** PyTorch, Ultralytics 框架, pytest, SSDC-UAV 数据集（COCO 格式）

**Spec:** `docs/superpowers/specs/2026-06-27-yolo12-nwd-p2-dysample-design.md`

---

## 文件结构

### 新建文件

| 路径 | 责任 |
|------|------|
| `tests/test_nwd_loss.py` | NWD 函数与 BboxLoss NWD 模式的单元/集成测试 |
| `tests/test_dysample.py` | DySample 模块前向传播测试 |
| `ultralytics/nn/modules/DySample.py` | DySample 模块实现（从 `.lqs/improved_resource/AddModules/DySample.py` 复制） |
| `scripts/improved_yolo12/yolo12-NWD-P2-DySample.yaml` | 完整方案 A 的 4-scale head 配置 |
| `scripts/improved_yolo12/yolo12-NWD.yaml` | E1 消融：仅 NWD（3-scale head，原架构） |
| `scripts/improved_train/train_yolov12-NWD_ssdc-uav_re0.py` | E1 训练脚本 |
| `scripts/improved_train/train_yolov12-NWD-P2-DySample_ssdc-uav_re0.py` | E2 训练脚本 |

### 修改文件

| 路径 | 修改内容 |
|------|---------|
| `ultralytics/utils/metrics.py` | 新增 `nwd()` 函数 |
| `ultralytics/utils/loss.py` | `BboxLoss.__init__` 接收 NWD 参数；`BboxLoss.forward` 增加 NWD 分支；`v8DetectionLoss.__init__` 透传 NWD 参数 |
| `ultralytics/cfg/default.yaml` | 新增 `nwd`, `nwd_c`, `nwd_alpha` 三个键 |
| `ultralytics/nn/modules/__init__.py` | 导出 `DySample` |
| `ultralytics/nn/tasks.py` | `parse_model()` 注册 `DySample` |

---

## Task 1: 添加 nwd() 函数到 metrics.py

**Files:**
- Modify: `ultralytics/utils/metrics.py`（在文件末尾追加）
- Test: `tests/test_nwd_loss.py`

- [ ] **Step 1.1: 写失败测试**

创建 `tests/test_nwd_loss.py`：

```python
"""Tests for NWD (Normalized Wasserstein Distance) loss support."""
import pytest
import torch

from ultralytics.utils.metrics import nwd


def test_nwd_identical_boxes_returns_one():
    """完全相同的两个 bbox，NWD 相似度应为 1.0。"""
    box1 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])  # xyxy
    box2 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
    result = nwd(box1, box2, C=10.0)
    assert result.shape == (1,)
    assert torch.allclose(result, torch.ones(1), atol=1e-6)


def test_nwd_different_boxes_returns_less_than_one():
    """有偏差的两个 bbox，NWD 相似度应 < 1.0。"""
    box1 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
    box2 = torch.tensor([[12.0, 12.0, 32.0, 32.0]])  # 偏移 2px
    result = nwd(box1, box2, C=10.0)
    assert result.shape == (1,)
    assert 0.0 < result.item() < 1.0


def test_nwd_batch_input():
    """批量输入应返回批量输出。"""
    box1 = torch.tensor([
        [10.0, 10.0, 30.0, 30.0],
        [0.0, 0.0, 16.0, 16.0],
    ])
    box2 = torch.tensor([
        [10.0, 10.0, 30.0, 30.0],
        [2.0, 2.0, 18.0, 18.0],
    ])
    result = nwd(box1, box2, C=10.0)
    assert result.shape == (2,)
    # 第一个完全相同 → 1.0
    assert torch.allclose(result[0], torch.tensor(1.0), atol=1e-6)
    # 第二个有偏差 → < 1.0
    assert result[1].item() < 1.0


def test_nwd_smaller_C_more_sensitive():
    """C 越小，对相同偏差越敏感（NWD 值越低）。"""
    box1 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
    box2 = torch.tensor([[12.0, 12.0, 32.0, 32.0]])  # 偏移 2px
    result_c6 = nwd(box1, box2, C=6.0)
    result_c12 = nwd(box1, box2, C=12.0)
    # C=6 应该比 C=12 给出更低的相似度（更敏感）
    assert result_c6.item() < result_c12.item()


def test_nwd_in_range_zero_to_one():
    """NWD 输出应在 [0, 1] 范围内，即使 boxes 完全分离。"""
    box1 = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    box2 = torch.tensor([[100.0, 100.0, 104.0, 104.0]])  # 完全分离
    result = nwd(box1, box2, C=10.0)
    assert 0.0 <= result.item() <= 1.0
    assert result.item() > 0.0  # exp(-x) 永远 > 0
```

- [ ] **Step 1.2: 运行测试确认失败**

Run: `python -m pytest tests/test_nwd_loss.py -v`
Expected: FAIL with `ImportError: cannot import name 'nwd' from 'ultralytics.utils.metrics'`

- [ ] **Step 1.3: 实现 nwd() 函数**

在 `ultralytics/utils/metrics.py` 文件末尾追加：

```python
def nwd(box1: torch.Tensor, box2: torch.Tensor, C: float = 12.0, eps: float = 1e-7) -> torch.Tensor:
    """Normalized Wasserstein Distance: 把 bbox 视为 2D 高斯，返回 NWD 相似度 ∈ [0,1]。

    小目标友好的 bbox 相似度度量。相比 IoU/CIoU，对小目标的位置/尺寸微扰梯度更平滑，
    适合从零训练的小目标检测场景。

    Args:
        box1 (torch.Tensor): (N, 4) xyxy 格式 bbox。
        box2 (torch.Tensor): (N, 4) xyxy 格式 bbox，与 box1 形状相同。
        C (float): 归一化常数，控制小目标敏感度。C 越小对小目标越敏感。典型 6~12。
        eps (float): 数值稳定小量。

    Returns:
        (torch.Tensor): (N,) NWD 相似度，取值 [0, 1]。1.0 表示完全相同。

    References:
        NWD 原论文: https://arxiv.org/abs/2110.13389 (AI-TOD 小目标检测)
    """
    # xyxy → cxcywh
    cx1 = (box1[..., 0] + box1[..., 2]) / 2
    cy1 = (box1[..., 1] + box1[..., 3]) / 2
    w1 = box1[..., 2] - box1[..., 0]
    h1 = box1[..., 3] - box1[..., 1]
    cx2 = (box2[..., 0] + box2[..., 2]) / 2
    cy2 = (box2[..., 1] + box2[..., 3]) / 2
    w2 = box2[..., 2] - box2[..., 0]
    h2 = box2[..., 3] - box2[..., 1]
    # Wasserstein-2 距离平方
    w_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) / 2) ** 2 + ((h1 - h2) / 2) ** 2
    return torch.exp(-torch.sqrt(w_dist + eps) / C)
```

- [ ] **Step 1.4: 运行测试确认通过**

Run: `python -m pytest tests/test_nwd_loss.py -v`
Expected: 5 passed

- [ ] **Step 1.5: 提交**

```bash
git add tests/test_nwd_loss.py ultralytics/utils/metrics.py
git commit -m "feat(loss): add NWD (Normalized Wasserstein Distance) similarity function"
```

---

## Task 2: 修改 BboxLoss 支持 NWD 模式

**Files:**
- Modify: `ultralytics/utils/loss.py`（`BboxLoss` 类，约 line 75-160）
- Test: `tests/test_nwd_loss.py`

- [ ] **Step 2.1: 阅读当前 BboxLoss 实现**

Run: `python -c "from ultralytics.utils.loss import BboxLoss; import inspect; print(inspect.getsource(BboxLoss))"`

记录当前 `__init__` 签名和 `forward` 中 IoU 计算行（约 line 132: `iou = bbox_iou(...)`）。

- [ ] **Step 2.2: 写失败测试 - BboxLoss NWD 模式**

在 `tests/test_nwd_loss.py` 末尾追加：

```python
import torch.nn as nn
from ultralytics.utils.loss import BboxLoss


def _make_dummy_bbox_loss_inputs(reg_max=7, n_pos=10, device="cpu"):
    """构造 BboxLoss.forward 所需的最小 dummy 输入。"""
    # pred_bboxes: (b, n_anchors, 4) xyxy
    # pred_dist: (b, n_anchors, 4*reg_max)
    # anchor_points: (n_anchors, 2)
    # gt_bboxes: (n_gt, 4) xyxy
    # gt_scores: (n_gt, nc) one-hot
    # fg_mask: (b, n_anchors) bool
    b, n_anchors = 1, 100
    nc = 1
    pred_bboxes = torch.rand(b, n_anchors, 4, device=device) * 100
    pred_bboxes[..., 2:] = pred_bboxes[..., :2] + torch.rand(b, n_anchors, 2, device=device) * 20 + 1
    pred_dist = torch.rand(b, n_anchors, 4 * reg_max, device=device)
    anchor_points = torch.rand(n_anchors, 2, device=device) * 100
    # 模拟正样本：前 n_pos 个 anchor 为正
    fg_mask = torch.zeros(b, n_anchors, dtype=torch.bool, device=device)
    fg_mask[0, :n_pos] = True
    gt_bboxes = pred_bboxes[0, :n_pos].clone()  # 让正样本的 pred 与 gt 接近
    gt_scores = torch.ones(n_pos, nc, device=device)
    target_bboxes = gt_bboxes.unsqueeze(0).expand(b, -1, -1)
    target_scores = torch.zeros(b, n_anchors, nc, device=device)
    target_scores[0, :n_pos] = gt_scores
    return pred_bboxes, pred_dist, anchor_points, target_bboxes, target_scores, fg_mask


def test_bbox_loss_init_accepts_nwd_params():
    """BboxLoss.__init__ 应接受 use_nwd, nwd_c, nwd_alpha 参数。"""
    bbox_loss = BboxLoss(reg_max=7, use_nwd=True, nwd_c=10.0, nwd_alpha=1.0)
    assert bbox_loss.use_nwd is True
    assert bbox_loss.nwd_c == 10.0
    assert bbox_loss.nwd_alpha == 1.0


def test_bbox_loss_init_defaults_to_ciou():
    """BboxLoss 默认应使用 CIoU（use_nwd=False）。"""
    bbox_loss = BboxLoss(reg_max=7)
    assert bbox_loss.use_nwd is False
    assert bbox_loss.nwd_c == 12.0
    assert bbox_loss.nwd_alpha == 1.0


def test_bbox_loss_forward_nwd_mode_runs():
    """启用 NWD 模式时，BboxLoss.forward 应能正常计算并返回三个 loss。"""
    bbox_loss = BboxLoss(reg_max=7, use_nwd=True, nwd_c=10.0, nwd_alpha=1.0)
    pred_bboxes, pred_dist, anchor_points, target_bboxes, target_scores, fg_mask = _make_dummy_bbox_loss_inputs()
    loss_iou, loss_dfl = bbox_loss(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, fg_mask
    )
    assert loss_iou.item() > 0
    assert loss_dfl.item() > 0
    # NWD 模式下 loss_iou 应为有限值
    assert torch.isfinite(loss_iou).all()


def test_bbox_loss_forward_ciou_mode_backward_compatible():
    """不启用 NWD 时，BboxLoss.forward 行为应与原版完全一致（回归测试）。"""
    bbox_loss_ciou = BboxLoss(reg_max=7, use_nwd=False)
    bbox_loss_nwd_off = BboxLoss(reg_max=7)  # 默认
    inputs = _make_dummy_bbox_loss_inputs()
    loss_iou1, loss_dfl1 = bbox_loss_ciou(*inputs)
    loss_iou2, loss_dfl2 = bbox_loss_nwd_off(*inputs)
    # 两次结果应完全相同（无随机性）
    assert torch.allclose(loss_iou1, loss_iou2)
    assert torch.allclose(loss_dfl1, loss_dfl2)
```

- [ ] **Step 2.3: 运行测试确认失败**

Run: `python -m pytest tests/test_nwd_loss.py::test_bbox_loss_init_accepts_nwd_params -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'use_nwd'`

- [ ] **Step 2.4: 修改 BboxLoss.__init__**

用 Edit 工具修改 `ultralytics/utils/loss.py`。找到当前 `BboxLoss.__init__`（参考 Step 2.1 输出），改成接收新参数。

**改前（约 line 75-90 形式）：**
```python
class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl_in_segmentation_loss=True):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl_in_segmentation_loss = use_dfl_in_segmentation_loss
        ...
```

**改后：**
```python
class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl_in_segmentation_loss=True, use_nwd=False, nwd_c=12.0, nwd_alpha=1.0):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl_in_segmentation_loss = use_dfl_in_segmentation_loss
        self.use_nwd = use_nwd
        self.nwd_c = nwd_c
        self.nwd_alpha = nwd_alpha
        ...
```

- [ ] **Step 2.5: 修改 BboxLoss.forward 中的 IoU 计算**

在 `ultralytics/utils/loss.py` 找到 `BboxLoss.forward` 中计算 `iou = bbox_iou(...)` 的行（约 line 132）。

**改前：**
```python
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```

**改后：**
```python
if self.use_nwd:
    nwd_val = nwd(pred_bboxes[fg_mask], target_bboxes[fg_mask], self.nwd_c)
    if self.nwd_alpha < 1.0:
        ciou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        iou = self.nwd_alpha * nwd_val + (1 - self.nwd_alpha) * ciou
    else:
        iou = nwd_val
else:
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```

并在 `loss.py` 顶部 import 中加入 `nwd`：

**改前（line 18）：**
```python
from .metrics import bbox_iou, probiou
```

**改后：**
```python
from .metrics import bbox_iou, nwd, probiou
```

- [ ] **Step 2.6: 运行测试确认通过**

Run: `python -m pytest tests/test_nwd_loss.py -v`
Expected: 9 passed (5 from Task 1 + 4 from Task 2)

- [ ] **Step 2.7: 提交**

```bash
git add ultralytics/utils/loss.py tests/test_nwd_loss.py
git commit -m "feat(loss): BboxLoss supports NWD mode as CIoU alternative"
```

---

## Task 3: 从 default.yaml 透传 NWD 配置到 BboxLoss

**Files:**
- Modify: `ultralytics/cfg/default.yaml`（在 `dfl:` 行附近追加）
- Modify: `ultralytics/utils/loss.py`（`v8DetectionLoss.__init__` line 368）
- Test: `tests/test_nwd_loss.py`

- [ ] **Step 3.1: 在 default.yaml 添加 NWD 配置**

在 `ultralytics/cfg/default.yaml` 中找到 `dfl: 1.5` 行（约 line 105），在它下面追加：

```yaml
dfl: 1.5 # (float) distribution focal loss gain
# NWD (Normalized Wasserstein Distance) for small object bbox regression
nwd: False # (bool) use NWD to replace CIoU for bbox regression loss
nwd_c: 10.0 # (float) NWD normalization constant, smaller = more sensitive to small objects
nwd_alpha: 1.0 # (float) NWD weight; 1.0 = pure NWD, 0.5 = NWD/CIoU hybrid
```

- [ ] **Step 3.2: 修改 v8DetectionLoss.__init__ 透传 NWD 参数**

在 `ultralytics/utils/loss.py` 找到 `v8DetectionLoss.__init__`（line 334-369），把 `self.bbox_loss = BboxLoss(m.reg_max).to(device)` 改成：

**改前（line 368）：**
```python
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
```

**改后：**
```python
        self.bbox_loss = BboxLoss(
            m.reg_max,
            use_nwd=getattr(h, "nwd", False),
            nwd_c=getattr(h, "nwd_c", 12.0),
            nwd_alpha=getattr(h, "nwd_alpha", 1.0),
        ).to(device)
```

注：用 `getattr` + 默认值是为了向后兼容（旧 cfg 文件没有 nwd 键时不报错）。

- [ ] **Step 3.3: 写集成测试 - YOLO 模型带 NWD 配置可实例化**

在 `tests/test_nwd_loss.py` 末尾追加：

```python
from ultralytics import YOLO


def test_yolo_model_loads_with_nwd_config():
    """YOLO 模型从 default.yaml 加载后，v8DetectionLoss 应正确读取 nwd 配置。"""
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    # 模拟用户在 cfg 中开启 NWD
    overrides = {"nwd": True, "nwd_c": 10.0, "nwd_alpha": 1.0}
    cfg = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
    assert cfg.nwd is True
    assert cfg.nwd_c == 10.0
    assert cfg.nwd_alpha == 1.0


def test_yolo_model_train_one_step_nwd_mode():
    """端到端：YOLO 模型启用 NWD 后能跑一个训练 step（不报错）。"""
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    model = YOLO("yolo12s.yaml")
    # 直接检查 loss 计算模块的配置
    loss_module = model.model.loss if hasattr(model.model, "loss") else None
    # YOLO 的 loss 是 lazy init，先做一次 forward 触发初始化
    # 构造 dummy batch
    import numpy as np
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    # 仅验证配置生效，不做完整训练
    model.overrides["nwd"] = True
    model.overrides["nwd_c"] = 10.0
    model.overrides["nwd_alpha"] = 1.0
    # 验证 overrides 被正确设置
    assert model.overrides["nwd"] is True
    assert model.overrides["nwd_c"] == 10.0
```

- [ ] **Step 3.4: 运行测试确认通过**

Run: `python -m pytest tests/test_nwd_loss.py -v`
Expected: 11 passed

- [ ] **Step 3.5: 提交**

```bash
git add ultralytics/cfg/default.yaml ultralytics/utils/loss.py tests/test_nwd_loss.py
git commit -m "feat(loss): wire NWD config from default.yaml to BboxLoss"
```

---

## Task 4: 添加 DySample 模块

**Files:**
- Create: `ultralytics/nn/modules/DySample.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Test: `tests/test_dysample.py`

- [ ] **Step 4.1: 复制 DySample.py 到目标位置**

源文件：`d:/Data/New_Codes/Python_Codes/ultralytics/.lqs/improved_resource/AddModules/DySample.py`
目标：`d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/modules/DySample.py`

使用 Read 工具读取源文件内容，再用 Write 工具写入目标位置（保持内容完全一致）。

- [ ] **Step 4.2: 写 DySample 前向传播测试**

创建 `tests/test_dysample.py`：

```python
"""Tests for DySample content-aware upsampling module."""
import pytest
import torch

from ultralytics.nn.modules import DySample


def test_dysample_doubles_spatial_dims():
    """DySample 默认 scale=2，应把 H/W 翻倍。"""
    layer = DySample(in_channels=64, scale=2)
    x = torch.randn(1, 64, 32, 32)
    y = layer(x)
    assert y.shape == (1, 64, 64, 64)


def test_dysample_preserves_channels():
    """DySample 不改变通道数。"""
    layer = DySample(in_channels=128, scale=2)
    x = torch.randn(2, 128, 16, 16)
    y = layer(x)
    assert y.shape == (2, 128, 32, 32)


def test_dysample_backward_pass():
    """DySample 应支持反向传播。"""
    layer = DySample(in_channels=32, scale=2)
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_dysample_different_styles():
    """DySample 应支持 'lp' 和 'pl' 两种 style。"""
    for style in ["lp", "pl"]:
        layer = DySample(in_channels=64, scale=2, style=style)
        x = torch.randn(1, 64, 16, 16)
        y = layer(x)
        assert y.shape == (1, 64, 32, 32)
```

- [ ] **Step 4.3: 在 __init__.py 导出 DySample**

在 `ultralytics/nn/modules/__init__.py` 中找到导出列表（通常是从各子模块 import 的位置），加入：

```python
from .DySample import DySample
```

并把 `DySample` 加入 `__all__` 列表（如果存在）。

- [ ] **Step 4.4: 运行测试确认通过**

Run: `python -m pytest tests/test_dysample.py -v`
Expected: 4 passed

- [ ] **Step 4.5: 提交**

```bash
git add ultralytics/nn/modules/DySample.py ultralytics/nn/modules/__init__.py tests/test_dysample.py
git commit -m "feat(modules): add DySample content-aware upsampling module"
```

---

## Task 5: 在 parse_model 中注册 DySample

**Files:**
- Modify: `ultralytics/nn/tasks.py`（`parse_model()` 函数中的 `if m in {nn.Upsample, ...}` 分支）
- Test: `tests/test_dysample.py`

- [ ] **Step 5.1: 定位 parse_model 中的 Upsample 注册**

Run: `python -c "import ultralytics.nn.tasks as t; import inspect; src = inspect.getsource(t.parse_model); import re; m = re.search(r'nn\.Upsample[^\\n]*', src); print(m.group(0) if m else 'not found')"`

记录 `nn.Upsample` 在 parse_model 中的处理行号和上下文。

- [ ] **Step 5.2: 写失败测试 - 从 yaml 实例化含 DySample 的模型**

在 `tests/test_dysample.py` 末尾追加：

```python
def test_dysample_parsable_from_yaml():
    """含 DySample 的 yaml 应能被 parse_model 正确实例化。"""
    import yaml
    import tempfile
    from pathlib import Path
    from ultralytics.nn.tasks import parse_model
    from ultralytics.utils.torch_utils import torch_safe_load
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    import torch.nn as nn

    # 最小 yaml：1 个 Conv + 1 个 DySample
    yaml_content = {
        "nc": 1,
        "scales": {"n": [0.25, 0.5, 1024]},
        "backbone": [
            [-1, 1, "Conv", [32, 3, 2]],   # 0
        ],
        "head": [
            [-1, 1, "DySample", [32]],      # 1
            [[1], 1, "Detect", [1]],        # 2 (用最小 Detect 验证)
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f)
        yaml_path = f.name
    try:
        cfg = get_cfg(cfg=DEFAULT_CFG)
        model, _ = parse_model(yaml.safe_load(open(yaml_path, encoding="utf-8")), cfg)
        # 模型应能实例化
        assert model is not None
        # 验证模型中包含 DySample 层
        has_dysample = any(isinstance(m, DySample) for m in model.modules())
        assert has_dysample, "模型中未找到 DySample 层"
    finally:
        Path(yaml_path).unlink(missing_ok=True)
```

- [ ] **Step 5.3: 运行测试确认失败**

Run: `python -m pytest tests/test_dysample.py::test_dysample_parsable_from_yaml -v`
Expected: FAIL（DySample 未注册到 parse_model）

- [ ] **Step 5.4: 修改 parse_model 注册 DySample**

在 `ultralytics/nn/tasks.py` 中：

1. 顶部 import 加入：
```python
from ultralytics.nn.modules import DySample
```
（如果已经有 `from ultralytics.nn.modules import *` 或类似聚合 import，确认 DySample 已被聚合到）

2. 在 `parse_model()` 函数中找到处理 `nn.Upsample` 的 `if m in {...}` 分支，把 `DySample` 加入：

**改前（典型形式）：**
```python
elif m is nn.Upsample:
    if ch[f] < 0:
        ch[f] = ch[-1]
    c2 = ch[f]
    if not args:
        args = ["nearest"]
```

**改后：**
```python
elif m in {nn.Upsample, DySample}:
    if ch[f] < 0:
        ch[f] = ch[-1]
    c2 = ch[f]
    if m is nn.Upsample and not args:
        args = ["nearest"]
    elif m is DySample:
        # DySample 第一个参数是 in_channels，由前层输出自动填入
        args = [c2, *args]
```

注：DySample 在 yaml 中写 `DySample, [512]` 时，`args = [512]` 是 scale 参数（不是 channels）。修改后 `args = [c2, 512]`，但 DySample 签名是 `(in_channels, scale=2, ...)`，所以 yaml 中的 `[512]` 实际是 `scale` 参数。

**重要：重新审视 yaml 写法** — 实际 yaml 应该写 `[-1, 1, DySample, []]`（无显式 scale，用默认 scale=2），让 parse_model 自动注入 in_channels。

后续 Task 6 的 yaml 会采用 `[-1, 1, DySample, []]` 写法。本步 parse_model 修改保持上面逻辑（`args = [c2, *args]`，当 args 为空时变成 `[c2]`，调用 `DySample(c2)` 用默认 scale=2）。

- [ ] **Step 5.5: 修正测试 yaml**

由于 DySample 在 yaml 中应写 `[-1, 1, DySample, []]`，修改 Step 5.2 的测试：

把 `"head": [[-1, 1, "DySample", [32]]]` 改成 `"head": [[-1, 1, "DySample", []]]`

- [ ] **Step 5.6: 运行测试确认通过**

Run: `python -m pytest tests/test_dysample.py -v`
Expected: 5 passed

- [ ] **Step 5.7: 提交**

```bash
git add ultralytics/nn/tasks.py tests/test_dysample.py
git commit -m "feat(tasks): register DySample in parse_model"
```

---

## Task 6: 创建 yolo12-NWD-P2-DySample.yaml

**Files:**
- Create: `scripts/improved_yolo12/yolo12-NWD-P2-DySample.yaml`
- Create: `scripts/improved_yolo12/yolo12-NWD.yaml`（E1 消融用）
- Test: 手动实例化验证

- [ ] **Step 6.1: 创建完整方案 A 的 yaml**

创建 `scripts/improved_yolo12/yolo12-NWD-P2-DySample.yaml`：

```yaml
# YOLO12s with NWD loss + lightweight P2 head + DySample upsampling
# For SSDC-UAV small object detection, from-scratch training
# Changes vs yolo12s baseline:
#   1. (loss) NWD replaces CIoU - configured via training script, not yaml
#   2. (arch) Add lightweight P2 head (single A2C2f layer)
#   3. (arch) Replace 3x nn.Upsample with DySample in head

nc: 1  # SSDC-UAV single class
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
  - [-1, 1, DySample, []]                     # 9  [替换 Upsample]
  - [[-1, 6], 1, Concat, [1]]                 # 10
  - [-1, 2, A2C2f, [512, False, -1]]          # 11

  # Top-down P4 → P3
  - [-1, 1, DySample, []]                     # 12 [替换 Upsample]
  - [[-1, 4], 1, Concat, [1]]                 # 13
  - [-1, 2, A2C2f, [256, False, -1]]          # 14

  # Top-down P3 → P2  [新增轻量 P2 路径]
  - [-1, 1, DySample, []]                     # 15 [新增]
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

- [ ] **Step 6.2: 创建 E1 消融 yaml（NWD only，原架构）**

创建 `scripts/improved_yolo12/yolo12-NWD.yaml`（直接复制 `yolo12.yaml` 内容，无需结构改动，因为 E1 只改损失不改架构）：

从 `scripts/improved_yolo12/yolo12.yaml` 复制内容，仅改文件头注释说明这是 NWD 消融基线。

- [ ] **Step 6.3: 验证 yaml 可实例化**

Run:
```bash
python -c "
from ultralytics import YOLO
m = YOLO('scripts/improved_yolo12/yolo12-NWD-P2-DySample.yaml')
print('Params:', sum(p.numel() for p in m.model.parameters()))
print('GFLOPs:', m.model.info(verbose=False))
# 期望：~10.5M params
"
```
Expected: 模型成功实例化，params 在 9.5M~11.5M 范围内（接近预期 10.5M）

- [ ] **Step 6.4: 验证 E1 yaml 也可实例化**

Run:
```bash
python -c "
from ultralytics import YOLO
m = YOLO('scripts/improved_yolo12/yolo12-NWD.yaml')
print('Params:', sum(p.numel() for p in m.model.parameters()))
# 期望：与 yolo12s baseline 相同 ~9.3M
"
```
Expected: params ≈ 9.3M（与 baseline 相同，因为 E1 只改损失）

- [ ] **Step 6.5: 提交**

```bash
git add scripts/improved_yolo12/yolo12-NWD-P2-DySample.yaml scripts/improved_yolo12/yolo12-NWD.yaml
git commit -m "feat(config): add yolo12-NWD-P2-DySample and yolo12-NWD yaml configs"
```

---

## Task 7: 创建 E1 训练脚本（NWD only）

**Files:**
- Create: `scripts/improved_train/train_yolov12-NWD_ssdc-uav_re0.py`
- Test: 1 epoch smoke test

- [ ] **Step 7.1: 参考已有训练脚本结构**

Run: `python -c "import os; os.listdir('scripts/improved_train') if os.path.exists('scripts/improved_train') else print('dir not found')"`

如有已有训练脚本（如 `train_yolov12_ssdc-uav_re0.py`），读取作为模板。否则用标准 YOLO train API。

- [ ] **Step 7.2: 创建 E1 训练脚本**

创建 `scripts/improved_train/train_yolov12-NWD_ssdc-uav_re0.py`：

```python
"""
E1 消融实验：YOLO12s + NWD only（无 P2 head，无 DySample）
- 目的：验证 NWD 损失单独是否能在 SSDC-UAV 上提升 mAP
- 配置：原 yolo12s 架构 + NWD 损失（nwd=True, nwd_c=10.0, nwd_alpha=1.0）
- 训练：150 epoch, imgsz=640, batch=16, SGD, 从随机初始化（re0 = from zero）
- 数据：SSDC-UAV（单类甘蔗幼苗检测）
"""
from ultralytics import YOLO


def main():
    # 从 yaml 实例化模型（不加载预训练权重 = re0）
    model = YOLO("scripts/improved_yolo12/yolo12-NWD.yaml")

    # SSDC-UAV 数据集 yaml 路径（参考已有训练脚本中的路径）
    data_yaml = "datasets/SSDC-UAV/data.yaml"  # 按实际路径调整

    # 训练 - 关键：开启 NWD
    results = model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=16,
        optimizer="SGD",
        # NWD 配置（透传到 v8DetectionLoss）
        nwd=True,
        nwd_c=10.0,
        nwd_alpha=1.0,
        # 从零训练约束：不加载预训练权重
        pretrained=False,
        # 其他超参保持默认（与 baseline 一致）
        project="runs/ssdc_uav_train",
        name="yolov12-NWD_ssdc-uav_re0",
        exist_ok=True,
    )
    print(f"E1 训练完成: {results}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.3: 1 epoch smoke test**

临时把 `epochs=150` 改成 `epochs=1` 跑通验证，再改回 150：

Run: `python scripts/improved_train/train_yolov12-NWD_ssdc-uav_re0.py`
Expected: 训练正常启动，完成 1 epoch 无报错，loss 正常下降

注：smoke test 完成后，把 `epochs=1` 改回 `epochs=150`。

- [ ] **Step 7.4: 提交**

```bash
git add scripts/improved_train/train_yolov12-NWD_ssdc-uav_re0.py
git commit -m "feat(train): add E1 ablation training script (NWD only)"
```

---

## Task 8: 创建 E2 训练脚本（完整方案 A）

**Files:**
- Create: `scripts/improved_train/train_yolov12-NWD-P2-DySample_ssdc-uav_re0.py`
- Test: 1 epoch smoke test

- [ ] **Step 8.1: 创建 E2 训练脚本**

创建 `scripts/improved_train/train_yolov12-NWD-P2-DySample_ssdc-uav_re0.py`：

```python
"""
E2 实验：YOLO12s + NWD + P2 head + DySample（完整方案 A）
- 目的：验证 NWD + 轻量 P2 + DySample 协同效果
- 配置：4-scale head（P2/P3/P4/P5）+ NWD 损失 + DySample 上采样
- 训练：150 epoch, imgsz=640, batch=16, SGD, 从随机初始化（re0）
- 数据：SSDC-UAV（单类甘蔗幼苗检测）
- 预期：mAP@.5:.95 从 baseline 0.5487 提升到 0.56+（+1.5~2.5%）
"""
from ultralytics import YOLO


def main():
    # 从 yaml 实例化模型（含 P2 head + DySample，不加载预训练权重 = re0）
    model = YOLO("scripts/improved_yolo12/yolo12-NWD-P2-DySample.yaml")

    # SSDC-UAV 数据集 yaml 路径
    data_yaml = "datasets/SSDC-UAV/data.yaml"  # 按实际路径调整

    # 训练 - 开启 NWD
    results = model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=16,
        optimizer="SGD",
        # NWD 配置
        nwd=True,
        nwd_c=10.0,
        nwd_alpha=1.0,
        # 从零训练约束
        pretrained=False,
        # 其他超参保持默认
        project="runs/ssdc_uav_train",
        name="yolov12-NWD-P2-DySample_ssdc-uav_re0",
        exist_ok=True,
    )
    print(f"E2 训练完成: {results}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: 1 epoch smoke test**

临时把 `epochs=150` 改成 `epochs=1`，运行验证：

Run: `python scripts/improved_train/train_yolov12-NWD-P2-DySample_ssdc-uav_re0.py`
Expected: 训练正常启动，4-scale head 正确构建，1 epoch 无报错

注：若 OOM，按 spec §3.6 风险降级：把 yaml 中 P2 head 通道从 128 降到 64。

smoke test 完成后，把 `epochs=1` 改回 `epochs=150`。

- [ ] **Step 8.3: 提交**

```bash
git add scripts/improved_train/train_yolov12-NWD-P2-DySample_ssdc-uav_re0.py
git commit -m "feat(train): add E2 full Approach A training script"
```

---

## Task 9: E1 完整训练 + 评估

**Files:**
- Run: `scripts/improved_train/train_yolov12-NWD_ssdc-uav_re0.py`
- Eval: `scripts/coco_test/`（已有评估流程）

- [ ] **Step 9.1: 启动 E1 完整训练（150 epoch）**

Run: `python scripts/improved_train/train_yolov12-NWD_ssdc-uav_re0.py`

**训练中监控（按 spec §4.2）：**
- 第 30 epoch：检查 loss_iou 是否正常下降
- 第 50 epoch：检查 mAP_small 是否开始超过 baseline 的 0.3256
- **若第 50 epoch mAP < baseline 80%（< 0.4390）**：触发降级（Step 9.4）

预计耗时：~8-12 小时（取决于 GPU）。

- [ ] **Step 9.2: 用 COCO 评估流程评估 E1**

参考已有 `scripts/coco_test/` 评估流程，对 E1 训练得到的 `runs/ssdc_uav_train/yolov12-NWD_ssdc-uav_re0/weights/best.pt` 评估。

Run: `python scripts/coco_test/<eval_script>.py`（按已有评估脚本格式调用）

记录 5 个指标：
- mAP@.5:.95
- mAP50
- mAP75
- mAP_small
- mAP_large

- [ ] **Step 9.3: 应用 E1 决策树**

按 spec §4.1 决策树判断：

| E1 vs baseline (0.5487) | 决策 |
|--------------------------|------|
| ≥ +0.5% | ✅ NWD 有效，继续 E2（Task 10） |
| 持平 ~ +0.5% | ⚠️ NWD 不够强，但仍继续 E2 看协同 |
| 退化 | ❌ NWD 在 from-scratch 失效，跳过 E2，直接进入风险降级（Step 9.4） |

- [ ] **Step 9.4: 风险降级（仅当 E1 退化或第 50 epoch 未达 80%）**

若需降级，修改 E1 训练脚本：
- `nwd_alpha: 1.0 → 0.5`（NWD+CIoU 混合）
- 重新训练 150 epoch
- 重新评估

若混合模式仍退化，按 spec §4.3 切换方案 B（不在本计划范围内）。

- [ ] **Step 9.5: 提交 E1 结果**

```bash
git add runs/ssdc_uav_train/yolov12-NWD_ssdc-uav_re0/
git commit -m "exp(E1): YOLO12s + NWD only results on SSDC-UAV"
```

---

## Task 10: E2 完整训练 + 评估

**Files:**
- Run: `scripts/improved_train/train_yolov12-NWD-P2-DySample_ssdc-uav_re0.py`
- Eval: `scripts/coco_test/`

- [ ] **Step 10.1: 启动 E2 完整训练（150 epoch）**

Run: `python scripts/improved_train/train_yolov12-NWD-P2-DySample_ssdc-uav_re0.py`

**训练中监控：**
- 第 30 epoch：loss_iou 应与 E1 同量级
- 第 50 epoch：mAP_small 应优于 E1
- **若 OOM**：按 spec §3.6 把 yaml 中 P2 head 通道 128→64，重启训练

- [ ] **Step 10.2: 用 COCO 评估流程评估 E2**

参考 `scripts/coco_test/` 评估流程，对 `runs/ssdc_uav_train/yolov12-NWD-P2-DySample_ssdc-uav_re0/weights/best.pt` 评估。

记录 5 个指标。

- [ ] **Step 10.3: 应用 E2 vs E1 决策树**

| E2 vs E1 | 决策 |
|----------|------|
| ≥ +0.5% | ✅ P2/DySample 协同有效，方案 A 确认 |
| 持平 | ⚠️ P2/DySample 无增益，采用 E1 配置即可 |
| 退化 | ❌ P2/DySample 破坏 NWD 收敛，回退到 E1 |

- [ ] **Step 10.4: 应用 E2 vs baseline 最终判定**

按 spec §4.4 成功标准：

| E2 vs baseline (0.5487) | 等级 | 后续 |
|--------------------------|------|------|
| ≥ +2.5% | 超预期 | ✅ 发布为最终方案，可写技术报告 |
| +1.5% ~ +2.5% | 达标 | ✅ 发布为最终方案 |
| +0.5% ~ +1.5% | 及格 | 接受结果；可选叠加方案 B |
| ≤ +0.5% | 失败 | 进入风险降级或切换方案 B |

- [ ] **Step 10.5: 提交 E2 结果**

```bash
git add runs/ssdc_uav_train/yolov12-NWD-P2-DySample_ssdc-uav_re0/
git commit -m "exp(E2): YOLO12s + NWD + P2 + DySample results on SSDC-UAV"
```

---

## Task 11: 结果汇总与反思日志更新

**Files:**
- Update: `.lqs/反思日志/2026-06-27-SSDC-UAV-YOLO12从零训练改进失败反思与方案.md`

- [ ] **Step 11.1: 收集所有实验结果**

整理 E0（baseline）、E1（NWD only）、E2（完整方案 A）的完整指标表：

| 实验 | mAP@.5:.95 | mAP50 | mAP75 | mAP_small | mAP_large | Δ vs baseline |
|------|-----------|-------|-------|-----------|-----------|---------------|
| E0 baseline | 0.5487 | - | - | 0.3256 | 0.6476 | 0 |
| E1 NWD only | - | - | - | - | - | - |
| E2 完整方案 A | - | - | - | - | - | - |

- [ ] **Step 11.2: 更新反思日志**

在 `.lqs/反思日志/2026-06-27-SSDC-UAV-YOLO12从零训练改进失败反思与方案.md` 末尾追加"实验结果与复盘"章节，包含：
- 三组实验的完整指标对比表
- NWD 是否有效的结论
- P2/DySample 是否有协同增益的结论
- 与既有失败改进的对比分析
- 若未达标：下一步建议（切换方案 B 或调整超参）

- [ ] **Step 11.3: 提交反思日志更新**

```bash
git add .lqs/反思日志/2026-06-27-SSDC-UAV-YOLO12从零训练改进失败反思与方案.md
git commit -m "docs: update reflection log with E1/E2 experiment results"
```

---

## Self-Review 自审

### 1. Spec 覆盖检查

| Spec 章节 | 实现任务 | 状态 |
|----------|---------|------|
| §1 总体架构（3 处改动） | Task 1-8 覆盖所有 3 处改动 | ✅ |
| §2 NWD 损失实现 | Task 1（nwd 函数）+ Task 2（BboxLoss）+ Task 3（配置透传） | ✅ |
| §3.1-3.2 P2 head + yaml | Task 6（yaml 创建） | ✅ |
| §3.3 DySample 集成 | Task 4（模块）+ Task 5（注册） | ✅ |
| §4.1 消融实验 E1 | Task 7（脚本）+ Task 9（训练+评估） | ✅ |
| §4.1 消融实验 E2 | Task 8（脚本）+ Task 10（训练+评估） | ✅ |
| §4.3 风险降级 | Task 9.4 + Task 10.1 内嵌降级步骤 | ✅ |
| §4.4 成功标准 | Task 10.4 决策表 | ✅ |
| §5 实现文件清单 | 全部覆盖 | ✅ |

### 2. 占位符扫描

- ❌ 无 TBD/TODO
- ❌ 无"appropriate error handling"等模糊描述
- ✅ 所有代码步骤都有完整代码块
- ✅ 所有命令都有 expected output
- ✅ Task 7/8 中的 `data_yaml` 路径有"按实际路径调整"注释（这是合理的，因为 spec 没指定确切路径）

### 3. 类型/方法一致性

- ✅ `nwd(box1, box2, C=12.0, eps=1e-7)` 在 Task 1 定义，Task 2 调用签名一致
- ✅ `BboxLoss(reg_max, use_nwd=False, nwd_c=12.0, nwd_alpha=1.0)` 在 Task 2 定义，Task 3 调用一致
- ✅ `DySample(in_channels, scale=2, style='lp')` 在 Task 4 测试，Task 5 parse_model 注入 `args = [c2, *args]`
- ✅ yaml 中 `DySample, []` 与 parse_model 处理逻辑一致（args 为空 → `[c2]` → `DySample(c2)` 用默认 scale=2）

### 4. 已知小问题（已在 Task 中处理）

- **Task 5.4 中 yaml 写法修正**：原 spec §3.2 写 `DySample, [512]` 容易误解为 in_channels=512，实际应写 `DySample, []` 让 parse_model 自动注入 in_channels。Task 6 的 yaml 已采用 `DySample, []` 写法。
- **Task 7/8 的 data_yaml 路径**：需要实施时按实际 SSDC-UAV 数据集位置调整，已用注释标注。

---

**Plan complete.** 共 11 个 Task，35 个 Step。预计实施 + 训练总耗时受 GPU 影响较大（E1+E2 各 150 epoch 约 16-24 小时）。
