# 计划：将最佳模型保存指标从 mAP50-95 切换为 mAP50（向后兼容）

## 摘要

在训练框架中新增 `best_metric` 配置项，允许用户通过 `best_metric=mAP50` 让 best.pt 的保存、早停判定都以 mAP50 为准；默认值 `mAP50-95` 保持现有行为，完全向后兼容。

**已确认决策**：fitness 采用纯 mAP50（不加权混合）。

## 当前状态分析

### 现有机制（已探索确认）
- **fitness 计算**：[metrics.py L1208-1211](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/utils/metrics.py#L1208-L1211)，`Metric.fitness()` 硬编码权重 `w = [0.0, 0.0, 0.0, 1.0]`（对应 `[P, R, mAP50, mAP50-95]`），即 fitness = mAP50-95
- **best.pt 判定**：[trainer.py L689](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/trainer.py#L689)，`save_model()` 中 `if self.best_fitness == self.fitness` 时写入 best.pt；`best_fitness` 在 [validate() L776](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/trainer.py#L760-L778) 中更新
- **早停**：`EarlyStopping`（torch_utils.py）同样基于 fitness 判断
- **faster-coco-eval 路径**：[detect/val.py L525](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/models/yolo/detect/val.py#L524-L525)，`save_json=True` 时 fitness = 0.9×AP_all + 0.1×AP_50，会覆盖常规 fitness
- **Validator 构造**：[detect/val.py L60](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/models/yolo/detect/val.py#L60)，`DetectionValidator.__init__` 中创建 `self.metrics = DetMetrics()`；`DetMetrics.fitness` 委托给 `self.box.fitness()`（Metric 实例）
- **配置校验**：`check_cfg()` 只校验在 `CFG_FLOAT/INT/BOOL_KEYS` 中的键，字符串键直接放行；`check_dict_alignment` 要求键存在于 default.yaml —— 因此只需在 default.yaml 添加即可被接受

### 用户场景
基于 SSDC-UAV 数据集在 YOLO12 上做检测改进，目标是提升低分漏检对象，提高 P、R、F1、mAP50。将 best 模型选择指标切换为 mAP50 可直接服务于该目标。

## 修改方案（共 3 个文件）

### 1. `ultralytics/cfg/default.yaml` — 新增配置项

在 `# Val/Test settings` 区域（`plots: True` 之后，约第 58 行）添加：

```yaml
best_metric: mAP50-95 # (str) metric used to select best model: mAP50-95 or mAP50
```

默认值 `mAP50-95` → 不传参时行为与现在完全一致。

### 2. `ultralytics/utils/metrics.py` — fitness 权重可配置

- `Metric.__init__()`（L1119-1127）末尾添加：`self._fitness_weights = [0.0, 0.0, 0.0, 1.0]`
- `Metric.fitness()`（L1208-1211）改为读取 `self._fitness_weights`：

```python
def fitness(self) -> float:
    """Return model fitness as a weighted combination of metrics."""
    w = self._fitness_weights  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())
```

默认权重与原硬编码值相同 → 未配置时行为不变。

### 3. `ultralytics/models/yolo/detect/val.py` — 按配置设置权重

**a) `DetectionValidator.__init__()`（L60 之后）**：创建 `DetMetrics()` 后，根据 `self.args.best_metric` 设置 `self.metrics.box._fitness_weights`：

```python
self.metrics = DetMetrics()
# 按 best_metric 配置选择最佳模型判定指标（默认 mAP50-95，向后兼容）
if getattr(self.args, "best_metric", "mAP50-95") == "mAP50":
    self.metrics.box._fitness_weights = [0.0, 0.0, 1.0, 0.0]  # fitness = mAP50
```

**b) faster-coco-eval 路径（L524-525）**：使该路径同样尊重配置：

```python
# update fitness
if getattr(self.args, "best_metric", "mAP50-95") == "mAP50":
    stats["fitness"] = val.stats_as_dict["AP_50"]
else:
    stats["fitness"] = 0.9 * val.stats_as_dict["AP_all"] + 0.1 * val.stats_as_dict["AP_50"]
```

**不改动**：
- LVIS 特殊分支（L532-533）保持原样（用户不使用 LVIS）
- `SegmentValidator`/`PoseValidator`/`OBBValidator` 不改（用户任务为 detect；若将来需要，同样模式可扩展）
- `TASK2METRIC`（tuner/benchmarks 用）不改，避免影响超参搜索等外围功能

## 向后兼容性保证

| 场景 | 保证 |
|------|------|
| 不传 `best_metric` | default.yaml 默认 `mAP50-95`，权重 `[0,0,0,1]` 与原硬编码一致 |
| 直接实例化 `Metric()`/`DetMetrics()` | `_fitness_weights` 有默认值，行为不变 |
| 旧 checkpoint resume | `getattr(self.args, "best_metric", "mAP50-95")` 安全取值；且 resume 时 args 从 ckpt 恢复后也会经 `get_cfg` 合并默认值 |
| 独立 `yolo val` | 同样读取配置，不传则默认行为 |
| 构造函数签名 | 均未改动 |

## 使用方式

```bash
# 默认行为（不变）
yolo train model=yolo12.yaml data=ssdc-uav.yaml

# best.pt / 早停按 mAP50 选择
yolo train model=yolo12.yaml data=ssdc-uav.yaml best_metric=mAP50
```

## 验证步骤

1. `python -c "from ultralytics import YOLO"` 确认无导入错误
2. 快速冒烟：用小数据/少 epoch 训练，不传 `best_metric`，确认日志中 fitness == mAP50-95 值
3. 传 `best_metric=mAP50` 冒烟训练，确认：
   - 每个 epoch 的 fitness 值等于 mAP50 值（对比 results.csv）
   - best.pt 在 mAP50 创新高时更新
4. 单元级验证（可选）：构造 `Metric` 实例，`update` 模拟结果，分别检查默认与设置权重后的 `fitness()` 返回值

## 附注（针对用户研究目标）

切换 best_metric 只改变"选哪个 checkpoint"，不改变模型本身的学习行为。要真正提升低分漏检对象的 P/R/F1/mAP50，还需配合算法改进（如损失函数、注意力模块、数据增强等）。本仓库已有 `cls_pw`、`slide_loss` 等自定义机制可参考。此部分不在本计划范围内，如需要可另行规划。
