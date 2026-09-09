# YOLO12s-NeckSimAM-DySample-DyHead 训练脚本与网络配置评审及修复计划

## 一、评审结论总览

对以下三个文件进行了逐行核实（对照 `tasks.py` parse_model 解析分支、`trainer.py` 训练循环时序、各模块源码签名）：

1. [yolo12-NeckSimAM_DySample_DyHead.yaml](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_yolo12/yolo12-NeckSimAM_DySample_DyHead.yaml)
2. [train_yolov12-NeckSimAM_DySample_DyHead_ssdc-uav_re0_300Epoch.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_train/from_scratch/train_yolov12-NeckSimAM_DySample_DyHead_ssdc-uav_re0_300Epoch.py)
3. [AddModules/__init__.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/__init__.py)

**总体结论：网络结构配置文件撰写正确，可直接使用；训练脚本存在 1 个回调时序 bug 和 2 处文案与实际不符，需小修。**

## 二、已验证正确的部分（无需改动）

### 2.1 YAML 网络结构（正确）

| 检查项 | 结论 | 依据 |
|---|---|---|
| 层拓扑与跳连索引 | 正确。P3/P4/P5（层 14/17/20）→ SimAM（21/22/23）→ DyHead（24），DySample 替代两层上采样（9/12），Concat 跳连索引（6/4/11/8）全部指向正确 | 手动推演 s scale 通道流：P3=128、P4=256、P5=512，无错位 |
| DySample 解析 | 正确。`[2, 'lp', 4, False]` 匹配签名 `(in_channels, scale=2, style='lp', groups=4, dyscope=False)`；parse_model 专属分支注入 in_channels | [tasks.py:1869-1872](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1869-L1872) |
| SimAM 解析 | 正确（走兜底 else 分支）。`c2 = ch[f]` 通道保持不变，`args=[]` 无参构造，SimAM `__init__` 全默认参数且参数无关（无参模块），恰好兼容 | [tasks.py:1891-1892](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1891-L1892) |
| DyHead 解析 | 正确。args `[nc, 128, 1]` 经 detect 头分支 extend 为 `[nc, hidc=128, block_num=1, reg_max, end2end, ch_list]`，与 `DyHead.__init__(nc, hidc, block_num, reg_max, end2end, ch)` 位置参数完全匹配；继承原生 Detect（DFL/stride/bias_init/导出全部继承） | [tasks.py:1801-1818](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1801-L1818)、[DyHead.py:248-253](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/DyHead.py#L248-L253) |
| 任务识别 | 正确。`guess_model_task` 识别输出层名 "dyhead" → detect 任务 | [tasks.py:1963-1965](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1963-L1965) |
| nc 一致性 | 正确。YAML `nc: 1` 与 ssdc-uav.yaml `nc: 1`（Sugarcane Seedling）一致 | [ssdc-uav.yaml](file:///d:/Data/New_Codes/Python_Codes/ultralytics/datasets/SSDC-UAV_yolo/ssdc-uav.yaml) |
| scale 解析 | 正确。文件名 `yolo12s-...` 解析出 s scale（width=0.5） | guess_model_scale 正则匹配 |

### 2.2 DyHead hidc=128 的通道兼容性（正确）

s scale 下三层输入通道 `[128, 256, 512]`，统一对齐到 hidc=128 后：`GroupNorm(16, 128)` 整除、`DyReLU` squeeze=128//4=32 整除、`cv2` c2=max(16, 32, 64)=64、`cv3` c3=max(128, min(1,100))=128，均无整除问题。

### 2.3 AddModules/__init__.py（正确）

- DySample、SimAM、DyHead 三个启用模块的文件均已定义 `__all__`（DySample.py→`["DySample"]`、SimAM.py→`["SimAM","A2C2f_SimAM"]`、DyHead.py→`["DyHead"]`），不泄漏内部重定义的 Conv/Bottleneck/C3 等官方同名类，**无命名空间遮蔽风险**，符合项目 AGENTS.md 规约。
- 未启用的模块（EMA 注意力、BiFPN、FreqFusion 等虽在 import 列表中，但其 `__all__` 均已约束）。

### 2.4 训练超参（正确）

imgsz=640、epochs=300、batch=16、optimizer=SGD，与项目硬约束（SSDC-UAV 基线对齐 E0': 300ep）一致。从零训练位于 `from_scratch/` 目录属刻意实验设计（从零训练可完全规避 DyHead/SimAM/DySample 新增模块的预训练权重迁移问题）。

## 三、发现的问题与修复方案

### 问题 1（bug，必修）：检查点回调存在 off-by-one 时序错误

**现象**：`SaveLastNCheckpointsCallback` 挂在 `on_train_epoch_end` 事件，但 [trainer.py:531](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/trainer.py#L531) 中该事件在 `save_model()`（[trainer.py:553](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/trainer.py#L553)）**之前**触发：

- 第 1 个 epoch 结束：`last.pt` 尚不存在 → `os.path.exists` 静默跳过，epoch_1.pt 永远不生成；
- 第 N 个 epoch 结束：复制到的是**上一个 epoch** 的 last.pt，却命名为 epoch_N.pt（内容与文件名错位一个 epoch）；
- 最后一个 epoch（300）：epoch_300.pt 实际是 epoch 299 的权重。

**修复**：回调改挂 `on_model_save` 事件——该事件在 `save_model()` 成功写入 last.pt 后立即触发（[trainer.py:553-554](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/trainer.py#L553-L554)），此时 `trainer.last` 内容与刚完成的 epoch 一致，且 `trainer.epoch` 为 0-indexed，`trainer.epoch + 1` 命名正确。NaN epoch 时 save_model 返回 False 不触发回调，行为也自洽。

修改文件：[train_yolov12-NeckSimAM_DySample_DyHead_ssdc-uav_re0_300Epoch.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_train/from_scratch/train_yolov12-NeckSimAM_DySample_DyHead_ssdc-uav_re0_300Epoch.py)

```python
# 注册自定义回调函数：保存最近 3 个 epoch 的权重
# 注意：必须挂 on_model_save（save_model() 成功后触发）；
# 若挂 on_train_epoch_end 会早于 last.pt 写入，导致 epoch_X.pt 内容错位一个 epoch
save_callback = SaveLastNCheckpointsCallback(n=3)
model.add_callback("on_model_save", save_callback.on_model_save)
```

同步将回调类的钩子方法改名 `on_train_epoch_end` → `on_model_save`（内部逻辑不变）。

### 问题 2（文案不符，修注释）：docstring 声称 Powerful-CIoU2 但实际走默认 CIoU

脚本 docstring 写「损失: Powerful-CIoU2 训练脚本」，但 `model.train()` 未传 `powerful_iou=True`（default.yaml 中默认 False，走标准 CIoU+DFL）。**用户已确认本实验使用默认 CIoU**，docstring 系从 PIoU2 脚本复制的残留文案。

修改：将 docstring 中两处「Powerful-CIoU2」改为「默认 CIoU+DFL（未启用 powerful_iou）」的准确描述。

### 问题 3（文案不符，修注释）：YAML 注释与实际模块不符

[yolo12-NeckSimAM_DySample_DyHead.yaml](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_yolo12/yolo12-NeckSimAM_DySample_DyHead.yaml) 第 47 行注释写「在 DyHead 前对颈部三个输出各接一个 EMA 注意力」，但实际插入的是 SimAM（文件头注释也残留 EMA 描述）。

修改：更正注释为 SimAM，避免后续实验追溯时误导。

### 提醒（不改动，仅告知用户）

- **SimAM 属项目记忆失败清单组件**：此前失败场景是 backbone A2C2f 块内使用（tasks.py:1713 注释「try1 没用」）。本次是在 Neck 检测头前做特征增强，位置不同，属新实验设计，风险由用户自行评估。
- **从零训练与项目记忆「必须使用 COCO 预训练权重」约束不符**：但脚本位于 `from_scratch/` 目录且命名 re0_300Epoch，属刻意的从零训练对照实验（与同目录 PIoU2 脚本一致），且从零训练规避了新模块权重迁移问题，符合该实验track的设定。

## 四、实施步骤

1. 修改训练脚本：
   - 回调类钩子方法 `on_train_epoch_end` 改名为 `on_model_save`（内部逻辑不变）
   - `model.add_callback("on_train_epoch_end", ...)` 改为 `model.add_callback("on_model_save", ...)`
   - docstring 两处「Powerful-CIoU2」改为「默认 CIoU+DFL」
2. 修改 YAML 注释：第 47 行及文件头注释中 EMA 描述改为 SimAM
3. 验证（见下）

## 五、验证步骤

1. **YAML 构建验证**（确认 parse_model 全链路无误）：
   ```powershell
   python -c "from ultralytics import YOLO; m = YOLO(r'scripts\improved_yolo12\yolo12s-NeckSimAM_DySample_DyHead.yaml'); m.info()"
   ```
   预期：正常打印层数/参数量/GFLOPs，最后一层为 DyHead（Detect 子类），无报错。
2. **前向传播验证**（确认 DySample/SimAM/DyHead 数据流形状正确）：
   ```powershell
   python -c "import torch; from ultralytics import YOLO; m = YOLO(r'scripts\improved_yolo12\yolo12s-NeckSimAM_DySample_DyHead.yaml').model.eval(); x = torch.randn(1,3,640,640); [m(torch.tensor(x))] and print('forward OK')"
   ```
   注：DyHead 的 DCN 在 CPU 上会降级为普通卷积（DyDCNv2 内已处理），仅验证形状链路。若本机 GPU 可用，可在 `device='cuda'` 上验证真 DCN 路径。
3. **回调修复代码审查**：确认 `on_model_save` 是 BaseTrainer 默认回调事件之一（trainer.py:554 `run_callbacks("on_model_save")`），add_callback 经 [model.py:783](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/model.py#L783) `_callbacks=self.callbacks` 传入 trainer，注册机制有效。
4. **语法检查**：`python -m py_compile` 训练脚本。
