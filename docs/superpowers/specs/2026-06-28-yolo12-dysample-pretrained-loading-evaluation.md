# YOLO12-DySample 结构微调后预训练权重加载评估与修复设计

**日期**: 2026-06-28
**评估对象**:
- `scripts/improved_yolo12/yolo12-DySample.yaml` (DySample 改进架构)
- `scripts/improved_train/coco_pretrained/train_yolov12-DySample_ssdc-uav.py` (COCO 预训练微调脚本)

**评估目标**: 分析在微调 YOLO12 模型结构 (用 DySample 替换 nn.Upsample) 后，当前脚本能否有效实现预训练权重部分加载并正常训练；若存在缺陷，评估官方框架提供的替代方案。

**结论**: 当前脚本的权重加载方式**不可行** (2 处致命缺陷)，需改用官方 `pretrained=` 参数机制。本设计给出完整评估与修复方案。

---

## 1. 权重加载机制兼容性分析

### 1.1 当前脚本处理方式

```python
# train_yolov12-DySample_ssdc-uav.py 第 64-69 行
model = YOLO(r'...yolo12-DySample.yaml')
pretrained_dict = torch.load('yolo12s.pt', map_location='cpu')                  # 缺陷 1
model.load_state_dict(pretrained_dict['model'].state_dict(), strict=False)      # 缺陷 2
```

### 1.2 兼容性评估结论: 不可行 (3 处缺陷)

| # | 缺陷 | 后果 | 根因 |
|---|------|------|------|
| 1 | `torch` 未导入 | `NameError` 立即终止 | 脚本顶部仅 `from ultralytics import YOLO`，无 `import torch` |
| 2 | `strict=False` 不跳过形状不匹配键 | `RuntimeError: size mismatch` 在 Detect 头崩溃 | COCO `nc=80` vs SSDC-UAV `nc=1`，`cv2.*.weight` 键存在于两侧但形状不同；PyTorch 的 `strict=False` 仅跳过缺失/多余键，对形状冲突仍报错 |
| 3 | 绕过官方 `intersect_dicts` 机制 | 丢失首层卷积通道适配、无加载日志 | `BaseModel.load()` 的 `intersect_dicts` 要求 `v.shape == db[k].shape` 才保留键，能自动过滤 Detect 头；手动方式无此保护 |

### 1.3 关键技术依据

官方 `intersect_dicts` (`ultralytics/utils/torch_utils.py:555-566`) 核心逻辑:

```python
return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}
```

它以**形状为过滤条件**做交集，是官方处理「结构微调后部分加载」的标准机制。当前脚本的手动写法完全绕过了它。

`strict=False` 的真实语义 (PyTorch 官方文档): 仅忽略 `missing_keys` 和 `unexpected_keys`，对**键名相同但形状不同**的张量仍抛出 `RuntimeError`。这是当前脚本最致命的误解。

---

## 2. 模型结构修改对参数映射的影响

### 2.1 DySample 替换 nn.Upsample 的参数映射分类

将 head 中两处 `nn.Upsample` (索引 9、12) 替换为 `DySample` 后，参数键的变化分三类:

| 类别 | 键名示例 | 在 yolo12s.pt 中 | 在 DySample 模型中 | 处理方式 |
|------|---------|------------------|-------------------|----------|
| **A. 完全匹配** (backbone + head 共享层) | `model.0.conv.weight`、`model.6.cv1.0.weight`、`model.10.cv1.0.weight`、`model.14.cv1.0.weight`、`model.17.cv1.0.weight`、`model.20.cv1.0.weight` 等 | 存在 | 存在，形状一致 | `intersect_dicts` 保留，成功加载 |
| **B. DySample 新增参数** (无对应预训练键) | `model.9.offset.weight`、`model.9.offset.bias`、`model.9.init_pos`、`model.12.offset.weight`、`model.12.offset.bias`、`model.12.init_pos` | 不存在 | 存在 | `intersect_dicts` 过滤 (键不在 da)，保留 DySample 的 `normal_init(std=0.001)` / `constant_init(0)` / `_init_pos()` 初始化 |
| **C. Detect 头形状冲突** | `model.21.cv2.0.0.weight` (`[80,256,1,1]` vs `[1,256,1,1]`)、`model.21.cv2.0.1.bias` (`[80]` vs `[1]`) 等 | 存在 | 存在，形状不同 | `intersect_dicts` 过滤 (`v.shape != db[k].shape`)，保留随机初始化；后续由 `bias_init()` 重新初始化 |

### 2.2 层索引映射完整性证明

对比两个 YAML 的 head 结构 (backbone 0-8 完全相同):

```
标准 yolo12.yaml                      yolo12-DySample.yaml
head:                                 head:
  9: nn.Upsample [None, 2, "nearest"]   9: DySample []           ← 替换，索引不变
 10: Concat                            10: Concat
 11: A2C2f [512, ...]                  11: A2C2f [512, ...]      ← 键名一致
 12: nn.Upsample [None, 2, "nearest"]  12: DySample []           ← 替换，索引不变
 13: Concat                            13: Concat
 14: A2C2f [256, ...]                  14: A2C2f [256, ...]      ← 键名一致
 ... (17、20、21 同理)                 ... (17、20、21 同理)
```

由于 `nn.Upsample` 本身无可学习参数，而 `DySample` 仅新增参数 (不改变后续层的索引)，所有 A 类共享层的键名 `model.{10,11,13,14,15,16,17,18,19,20,21}.*` 与 yolo12s.pt 完全一致。

### 2.3 预期加载覆盖率

- yolo12s.pt 约 9.28M 参数 (272 层)
- DySample 模型参数量略多 (DySample 新增约 2×(512×8×1×1 + 8 + 256×8×1×1 + 8) ≈ 6K 参数，可忽略)
- 预期 `Transferred` 数量 ≈ 总参数键数 − DySample 新增键(6) − Detect 头形状冲突键(约 12-18 个)
- **骨干网络、颈部 A2C2f/C3k2/Conv 全部成功迁移**，这是迁移学习的核心价值所在

### 2.4 结论

DySample 的结构性修改对参数映射的影响是**最小化且可自动处理**的:
- 不破坏任何共享层的键名 (索引保留)
- 新增参数有合理的轻量初始化 (`std=0.001`，接近零初始化的偏移学习)
- Detect 头由 `intersect_dicts` 自动跳过，不影响骨干迁移

**唯一前提**: 必须使用官方 `intersect_dicts` 机制，而非手动 `strict=False`。

---

## 3. 训练过程错误风险及解决方案

### 3.1 风险矩阵

| 风险等级 | 风险点 | 触发条件 | 影响范围 | 解决方案 |
|---------|--------|---------|---------|---------|
| 致命 | `NameError: name 'torch' is not defined` | 运行脚本 | 立即终止 | 方案1 删除手动 `torch.load`，改用 `pretrained=` 参数，无需 `import torch` |
| 致命 | `RuntimeError: size mismatch` on Detect head | `strict=False` 遇到 `cv2.*.weight` 形状冲突 | 训练前崩溃 | 方案1 走 `intersect_dicts`，自动过滤形状冲突键 |
| 中等 | `yolo12s.pt` 文件不存在/路径错误 | 工作目录非项目根 | 加载失败 | 使用官方 `pretrained='yolo12s.pt'`，框架自动调用 `attempt_download_asset` 从官方仓库下载 |
| 中等 | DySample `offset` 初始化过大导致训练初期上采样失真 | `normal_init(std=0.001)` 实际很小，风险低 | 早期 epoch loss 波动 | 当前 `std=0.001` 已足够小，`offset * 0.25 + init_pos` 保证初始接近双线性上采样，无需额外处理 |
| 低 | `SaveLastNCheckpointsCallback` 回调与官方 `pretrained` 路径冲突 | 回调读取 `trainer.last` | 回调失效 | 回调逻辑独立于权重加载，无冲突；保留现有回调实现 |
| 低 | `intersect_dicts` 加载日志未显示 | 用户想确认加载量 | 调试不便 | 官方 `BaseModel.load()` 自动打印 `Transferred X/Y items`，方案1 自带日志 |
| 低 | EMA 初始化使用了未加载预训练的随机权重 | `get_model` 返回已加载模型后，trainer 的 `_setup_train` 才初始化 EMA | EMA 早期偏差 | 官方流程保证 EMA 在 `get_model` 之后初始化，顺序正确 |

### 3.2 方案1 下的训练流程时序 (验证无副作用)

```
1. YOLO('yolo12-DySample.yaml')          → 从 YAML 构建 DySample 模型 (随机初始化)
2. model.add_callback(...)               → 注册 SaveLastNCheckpointsCallback
3. model.train(data=..., pretrained='yolo12s.pt', ...)
   ├─ model.py:786-789  weights, _ = load_checkpoint('yolo12s.pt')
   ├─ trainer.get_model(cfg=yaml, weights=ckpt)
   │   ├─ DetectionModel(cfg, nc=1)              → 重建 nc=1 的 DySample 模型
   │   └─ model.load(ckpt)                       → BaseModel.load()
   │       ├─ intersect_dicts(csd, self.state_dict())  → 形状过滤
   │       ├─ load_state_dict(filtered, strict=False)  → 安全加载
   │       └─ LOGGER.info("Transferred X/Y items")     → 加载日志
   ├─ trainer._setup_train() → 初始化 optimizer / EMA / scheduler
   └─ trainer.train()        → 开始 150 epoch 训练
       └─ 每个 epoch 末: on_train_epoch_end 回调保存 epoch_X.pt
```

### 3.3 关键安全性论证

**为何 `pretrained='yolo12s.pt'` 不会与 `YOLO(yaml)` 冲突?**

`model.py:789` `self.trainer.model = self.trainer.get_model(weights=weights, cfg=self.model.yaml)` —— trainer 用 `self.model.yaml` (即 DySample yaml) 重建模型，再用 `model.load(weights)` 加载 yolo12s.pt 权重。

即: **架构以 yaml 为准 (DySample)，权重以 .pt 为来源 (过滤后部分加载)**。两者职责分离，不会用 yolo12s.pt 的架构覆盖 DySample 架构。

**为何 `nc=80→1` 的 Detect 头不会出错?**

- `get_model` 第 181 行 `DetectionModel(cfg, nc=self.data["nc"])` 使用数据集的 nc=1 构建 Detect 头
- yolo12s.pt 中 nc=80 的 Detect 头参数被 `intersect_dicts` 过滤 (形状不匹配)，Detect 头保持 nc=1 的随机初始化 + `bias_init()`
- 这是官方标准的迁移学习行为，与 `YOLO('yolov8s.pt').train(data=custom.yaml)` 的工作方式完全一致

### 3.4 结论

方案1 下所有已识别风险均有官方机制兜底，**无新增副作用**。当前脚本的 2 个致命风险 (缺 import、形状崩溃) 在方案1 中被结构性消除。

---

## 4. 推荐实施方案

### 4.1 方案选择

**方案1: 官方 `pretrained` 参数 (推荐，已采纳)**

在 `model.train()` 调用中传入 `pretrained='yolo12s.pt'`，让 trainer 通过官方 `intersect_dicts` 机制加载。

理由:
- 最简洁，符合官方设计
- 自动处理形状过滤 (解决 Detect 头 nc 冲突)
- 自动处理首层卷积通道适配
- 自带加载日志 (`Transferred X/Y items`)
- 自动下载缺失权重 (`attempt_download_asset`)

### 4.2 脚本修改清单

| 位置 | 操作 | 理由 |
|------|------|------|
| 第 65-69 行 | 删除 `@TODO` 注释、`pretrained_dict = torch.load(...)`、`model.load_state_dict(...)` | 移除手动加载，改由官方机制处理 |
| `model.train(...)` 调用 | 新增 `pretrained='yolo12s.pt'` 参数 | 触发官方 `intersect_dicts` 部分加载路径 |
| 脚本顶部 import | 无需新增 `import torch` | 方案1 不再直接使用 torch |
| `SaveLastNCheckpointsCallback` | 保持不变 | 与权重加载机制正交，无冲突 |
| `yaml_path`、`epochs=150`、`imgsz=640`、`batch=16`、`optimizer='SGD'`、`project`、`name`、`device`、`save` | 保持不变 | 符合公平对比约束 |

### 4.3 修改后代码 (核心部分)

```python
# 2. 加载模型: 从 YAML 构建 DySample 架构 (随机初始化)
#    预训练权重通过 train(pretrained=...) 交给官方 intersect_dicts 机制加载
model = YOLO(r'D:\Data\New_Codes\Python_Codes\ultralytics\scripts\improved_yolo12\yolo12-DySample.yaml')

# 注册自定义回调函数: 保存最近 3 个 epoch 的权重
save_callback = SaveLastNCheckpointsCallback(n=3)
model.add_callback("on_train_epoch_end", save_callback.on_train_epoch_end)

# 3. 开始训练
#    pretrained='yolo12s.pt' 触发 trainer.get_model() → BaseModel.load()
#    → intersect_dicts 自动过滤形状不匹配键 (Detect 头 nc=80→1)
#    → DySample 新增参数保留其 normal_init 初始化
#    → 骨干 + 颈部共享层成功迁移
results = model.train(
    data=str(yaml_path),
    epochs=150,               # [对齐] 与 DETR 实验保持一致
    imgsz=640,                # [对齐] 输入图像尺寸
    batch=16,
    project='runs/ssdc_uav_train',
    name='yolo12s_DySample_ssdc_uav_exp01',
    device='0',
    save=True,
    optimizer='SGD',
    pretrained='yolo12s.pt',  # 官方迁移学习入口
)
```

### 4.4 验证标准

修复后运行脚本，预期看到以下日志 (按顺序):

1. 模型构建日志: `... DySample ...` 出现在层列表中，参数量约 9.29M (yolo12s + DySample 新增约 6K)
2. 权重迁移日志: `Transferred XXX/YYY items from pretrained weights` (XXX 应接近 YYY，差额 ≈ DySample 新增键 6 + Detect 头冲突键约 12-18)
3. 训练正常启动: `Epoch GPU_mem box_loss cls_loss dfl_loss ...`
4. 回调生效: `weights/epoch_1.pt`、`epoch_2.pt`、`epoch_3.pt` 依次生成

### 4.5 与项目记忆约束的对齐

| 项目记忆约束 | 本方案是否满足 |
|-------------|---------------|
| imgsz=640 / epochs=150 / batch=16 / SGD 不可改 | 满足，全部保留 |
| 不使用已失败组件 (SimAM/Mona/ESMoE/AssemFormer/HSFPN/HPDown) | 满足，仅用 DySample (不在失败列表) |
| 自定义模块放 `nn/AddModules/` | 满足，DySample 已在该目录 |
| 「必须从头训练」硬约束 | **需更新**: 用户已确认转向 COCO 预训练微调路线，此约束已过时 |

### 4.6 后续建议 (非本次修改范围)

1. **更新 project_memory**: 将「Must use from-scratch training」约束修订为「已转向 COCO 预训练微调路线，使用官方 `pretrained=` 参数加载 yolo12s.pt」
2. **消融对照**: 保留 `coco_pretrained/` 与 `from_scratch/` 两个目录的实验对照，便于对比两条路线在 SSDC-UAV 上的 mAP 差异

---

## 附录: 官方权重加载机制参考

### A.1 BaseModel.load() (ultralytics/nn/tasks.py:302-325)

```python
def load(self, weights, verbose=True):
    model = weights["model"] if isinstance(weights, dict) else weights
    csd = model.float().state_dict()                          # checkpoint state_dict as FP32
    updated_csd = intersect_dicts(csd, self.state_dict())     # 形状匹配交集
    self.load_state_dict(updated_csd, strict=False)           # 安全加载
    len_updated_csd = len(updated_csd)
    first_conv = "model.0.conv.weight"
    state_dict = self.state_dict()
    if first_conv not in updated_csd and first_conv in state_dict:
        # 首层卷积通道适配 (多通道训练场景)
        c1, c2, h, w = state_dict[first_conv].shape
        cc1, cc2, ch, cw = csd[first_conv].shape
        if ch == h and cw == w:
            c1, c2 = min(c1, cc1), min(c2, cc2)
            state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
            len_updated_csd += 1
    if verbose:
        LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")
```

### A.2 DetectionTrainer.get_model() (ultralytics/models/yolo/detect/train.py:170-184)

```python
def get_model(self, cfg=None, weights=None, verbose=True):
    model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
    if weights:
        model.load(weights)    # 调用 BaseModel.load()
    return model
```

### A.3 训练入口权重传递路径 (ultralytics/engine/model.py:770-790)

```python
pretrained = kwargs.get("pretrained", overrides.get("pretrained", True) if kwargs.get("cfg") else True)
...
self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
if not args.get("resume") and self.ckpt:
    weights = None if pretrained is False else self.model
    if isinstance(pretrained, (str, Path)):
        weights, _ = load_checkpoint(pretrained)              # 加载 yolo12s.pt
    self.trainer.model = self.trainer.get_model(weights=weights, cfg=self.model.yaml)  # 用 DySample yaml 重建 + load
    self.model = self.trainer.model
```
