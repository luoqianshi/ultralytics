# DyHead 检测头集成评估与修复计划

## 一、结论摘要

**DyHead 替换原始 Detect 检测头的集成链路是正确的，可以正常训练。** 模块注册、yaml 参数映射、父类签名、nc 对齐、预训练权重加载、stride 探测、损失函数兼容性等 10 项关键检查全部通过。

仅发现 **1 个实际 bug**（自定义回调的时序 off-by-one，导致 epoch_N.pt 实际保存的是第 N-1 个 epoch 的权重）和 **2 个低风险提示项**。均为训练脚本层面的小改动，不涉及 DyHead 模块本身。

---

## 二、集成链路验证结果（全部通过）

| # | 检查项 | 结果 | 依据 |
|---|--------|------|------|
| 1 | 模块导出 | ✅ | [AddModules/\_\_init\_\_.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/__init__.py#L22) 已启用 `from .DyHead import *`，经 tasks.py 的 `from .AddModules import *` 聚合导入 |
| 2 | parse_model 注册 | ✅ | [tasks.py#L1804](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1801-L1822) 检测头 frozenset 已包含 DyHead，`m.legacy = legacy` 正确设置 |
| 3 | yaml 参数映射 | ✅ | yaml args `[nc, 128, 1]` + parse_model 注入 `[reg_max, end2end, ch]` → `DyHead(nc, hidc=128, block_num=1, reg_max=16, end2end=None, ch=[128,256,512])`，与签名完全匹配（`nc` 字符串经 tasks.py#L1761-1763 的 `locals()` 解析为数值） |
| 4 | 父类签名匹配 | ✅ | [head.py#L89](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/modules/head.py#L89) `Detect.__init__(nc, reg_max, end2end, ch)` 与 DyHead 的 `super().__init__(nc, reg_max, end2end, ch)` 位置参数一一对应 |
| 5 | legacy 标志 | ✅ | backbone 的 C3k2/A2C2f 在解析到 head 之前已将 `legacy=False`，DyHead 重建的 cv3 采用 DWConv 非 legacy 结构，与 YOLO12 原生头一致 |
| 6 | nc 对齐 | ✅ | 模型 yaml `nc: 1` = 数据集 [ssdc-uav.yaml](file:///d:/Data/New_Codes/Python_Codes/ultralytics/datasets/SSDC-UAV_yolo/ssdc-uav.yaml) `nc: 1`（Sugarcane Seedling 单类）。即使不一致，trainer 也会以 data nc 重建模型 |
| 7 | 预训练权重加载 | ✅ | `YOLO(yaml)` 构建时 `ckpt=None` → 走 [trainer.py setup_model#L739-743](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/trainer.py#L725-L744)：`load_checkpoint('yolo12s.pt')` → `DetectionModel(cfg, nc=1)` → `model.load()`（intersect_dicts + strict=False）。backbone 21 层全部迁移；头部 cv2[0]/cv3[0] 第一层通道（128→64/128→128）恰好匹配也会部分迁移，控制台会打印 "Transferred X/Y items" |
| 8 | scale 推断 | ✅ | 文件名 `yolo12s-DyHead.yaml` 被 `guess_model_scale` 的正则匹配为 `s`（depth=0.5, width=0.5, max_channels=1024），与 yolo12s.pt 预训练权重 backbone 宽度一致 |
| 9 | stride 探测 / CPU 前向 | ✅ | [tasks.py#L414-420](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L414-L420) 新版 Detect 训练态返回 dict，`_forward` 提取 `feats` 计算 stride；构建期 CPU 前向由 DyDCNv2 的普通卷积回退兜底（规避 torchvision 0.16 CPU 访问冲突），GPU 训练走真 DCNv2 算子 |
| 10 | 损失/输出格式兼容 | ✅ | DyHead 仅重写 `__init__`/`forward`，训练态返回 `dict(boxes, scores, feats)`、bias_init、DFL 解码、EMA、fuse 全部继承新版 Detect，与 v8DetectionLoss 完全兼容 |

---

## 三、发现的问题与修复方案

### 问题 1（实际 bug，建议修复）：回调时序 off-by-one

**现象**：回调挂在 `on_train_epoch_end`，但该事件在 [trainer.py#L531](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/trainer.py#L531) 触发，而框架的 `save_model()`（写入 last.pt）在其后的 [trainer.py#L553](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/trainer.py#L553) 才执行。

**后果**：
- `epoch_N.pt` 里实际是第 **N-1** 个 epoch 的权重（滞后一个 epoch）
- 第 1 个 epoch 永远不会保存（此时 last.pt 尚不存在，`os.path.exists` 直接跳过）

**修复**：改挂 `on_fit_epoch_end`（trainer.py#L566，在 save_model 之后执行）。注意训练结束后的 `final_eval()`（trainer.py#L848-865）会以 `epoch+1` 额外触发一次该事件，需加防重入守卫。

**修改文件**：[train_yolov12-DyHead_ssdc-uav.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_train/coco_pretrained/train_yolov12-DyHead_ssdc-uav.py)

回调方法改为：

```python
def on_fit_epoch_end(self, trainer):
    current_epoch = trainer.epoch + 1
    # 防重入：final_eval() 会以 epoch+1 再触发一次 on_fit_epoch_end，跳过超出总轮数的调用
    if current_epoch > trainer.epochs:
        return
    ...（其余逻辑不变）
```

注册处改为：

```python
model.add_callback("on_fit_epoch_end", save_callback.on_fit_epoch_end)
```

### 问题 2（低风险提示）：`pretrained='yolo12s.pt'` 为相对路径

setup_model 中按相对路径解析，若不从仓库根目录启动脚本，会触发联网下载或报错。建议改为绝对路径（与脚本中其他路径风格一致）：

```python
pretrained=r'D:\Data\New_Codes\Python_Codes\ultralytics\yolo12s.pt',  # 若权重在仓库根目录
```

### 问题 3（观察项，无需改动）：AMP 与 DCNv2

默认 `amp=True`，DCNv2 在 fp16 下训练通常没问题；但 [save_model](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/trainer.py#L651-L654) 检测到 EMA 含 NaN/Inf 会**跳过保存**（有 warning 日志）。若训练早期出现 loss=NaN 或 last.pt 长期不更新，先用 `amp=False` 排查是否为 DCN 偏移量精度问题。

---

## 四、非错误的设计确认（无需改动）

- `hidc=128, block_num=1`：对 s 规模合理（原论文 block_num=2 偏重）；cv2/cv3 内部通道（c2=64, c3=128）与原生 s 级 Detect 一致，最大化预训练迁移
- `epochs=150, imgsz=640, batch=16, SGD`：超参无冲突，使用框架默认学习率（lr0=0.01）
- yaml 的 `end2end` 未定义 → None → 正确走 one2many 常规训练路径（无 one2one 头）

## 五、执行步骤

1. 修改训练脚本：回调事件 `on_train_epoch_end` → `on_fit_epoch_end` + 防重入守卫（问题 1）
2. （可选）`pretrained` 改绝对路径（问题 2）
3. 冒烟验证：以 `epochs=2` 短跑一次，确认：
   - 控制台打印 "Transferred X/Y items"（backbone 权重迁移）
   - 模型摘要中第 22 层为 DyHead、参数量合理（约 9M+DCN 增量）
   - `weights/` 下 `epoch_1.pt`、`epoch_2.pt` 时间戳与 last.pt 同步更新（验证 off-by-one 已修复）
   - 训练 loss 正常下降、无 NaN
