# Ultralytics 代码仓库深度使用指南

> 面向以 YOLO12 为主的目标检测算法研究员
>
> 本文基于本地仓库 `D:\Data\New_Codes\Python_Codes\ultralytics` 的当前源码（含本地改进定制）逐文件调研撰写，文中行号均为撰写时快照，后续更新可能漂移，请以"函数/类名 + 关键字"定位为准。
>
> 撰写日期：2026-08-25

---

## 目录

1. [仓库总览与目录结构](#1-仓库总览与目录结构)
2. [安装与快速上手](#2-安装与快速上手)
3. [框架核心设计思想](#3-框架核心设计思想)
4. [配置系统详解](#4-配置系统详解)
5. [YOLO12 模型配置文件详解](#5-yolo12-模型配置文件详解)
6. [模型构建流程：从 yaml 到可训练网络](#6-模型构建流程从-yaml-到可训练网络)
7. [YOLO12 网络结构深度剖析（主干 / 颈部 / 检测头）](#7-yolo12-网络结构深度剖析)
8. [损失函数与标签分配](#8-损失函数与标签分配)
9. [数据管线详解](#9-数据管线详解)
10. [训练流程详解](#10-训练流程详解)
11. [验证流程详解](#11-验证流程详解)
12. [推理流程详解](#12-推理流程详解)
13. [面向 YOLO12 的算法改进逐步指南](#13-面向-yolo12-的算法改进逐步指南)
14. [常用命令速查](#14-常用命令速查)
15. [本仓库的本地定制清单](#15-本仓库的本地定制清单)
16. [常见问题与注意事项](#16-常见问题与注意事项)

---

## 1. 仓库总览与目录结构

### 1.1 仓库根目录

```
ultralytics/                  # 核心 Python 包（本文主角）
datasets/                     # 数据集默认存放目录（自动下载的数据集会放这里）
runs/                         # 训练/验证/推理结果输出目录
scripts/                      # 辅助脚本（含 improved_yolo12 等改进实验配置副本）
tests/                        # 单元测试
docs/  mkdocs.yml             # 官方文档源
examples/                     # 官方示例
docker/                       # 容器化环境
pyproject.toml                # 打包与依赖定义
yolo12n.pt / yolo12s.pt       # YOLO12 预训练权重（可直接加载微调）
yolo11n.pt / yolo26n.pt / yolov8n.pt / yolov5nu.pt / yolo26s.pt  # 其它预训练权重
AGENTS.md                     # 本项目的 AI 协作规约（自定义模块放置规约、git 规约）
```

### 1.2 `ultralytics/` 包顶层目录一览

| 目录 | 职责 | 研究员关注度 |
|---|---|---|
| `cfg/` | 配置中心：默认超参 `default.yaml`、模型结构 yaml（`models/12/yolo12.yaml` 等）、数据集 yaml（`datasets/`）、跟踪器配置 | ★★★★★ |
| `nn/` | 神经网络：`tasks.py`（yaml→模型的解析与构建）、`modules/`（所有网络模块）、`autobackend.py`（推理后端调度）、`backends/`（各推理后端实现）、`AddModules/`（本仓库自定义改进模块目录） | ★★★★★ |
| `engine/` | 引擎层：`trainer.py`（训练循环）、`validator.py`（验证）、`predictor.py`（推理）、`model.py`（YOLO 基类）、`results.py`（推理结果封装）、`exporter.py`（导出） | ★★★★★ |
| `models/` | 任务入口：`yolo/model.py` 定义 `YOLO` 类；`yolo/detect|segment|pose|classify|obb/` 各有 `train/val/predict/world.py`；另有 `rtdetr/`、`sam/`、`fastsam/`、`nas/` 等 | ★★★★☆ |
| `data/` | 数据管线：数据集构建、增强、标签解析、推理数据源加载器 | ★★★★☆ |
| `utils/` | 工具库：`loss.py`（损失）、`tal.py`（标签分配）、`metrics.py`（指标）、`nms.py`（NMS）、`ops.py`（坐标运算）、`torch_utils.py`（EMA/设备/早停）、`callbacks/`（回调）、绘图、下载等 | ★★★★★ |
| `optim/` | 优化器扩展（含 MuSGD 等） | ★★☆☆☆ |
| `trackers/` | 多目标跟踪（BoT-SORT / ByteTrack） | ★★☆☆☆ |
| `solutions/` | 应用级方案（计数、车牌识别等） | ★☆☆☆☆ |
| `hub/` | Ultralytics HUB 云平台对接 | ★☆☆☆☆ |
| `assets/` | 示例图片等资源 | ★☆☆☆☆ |

### 1.3 关键文件功能速查（按研究员日常使用频率排序）

| 文件 | 功能 |
|---|---|
| `ultralytics/cfg/default.yaml` | 全部训练/验证/推理/导出默认超参数（148 行），可被命令行或 API 覆盖 |
| `ultralytics/cfg/models/12/yolo12.yaml` | YOLO12 检测网络结构定义（backbone + head + scales） |
| `ultralytics/models/yolo/model.py` | `YOLO` 类：用户最常接触的入口，`train/val/predict/export` 都从这里分派 |
| `ultralytics/engine/model.py` | `Model` 基类：加载 `.pt`/`.yaml`、`task_map` 机制、参数合并逻辑 |
| `ultralytics/nn/tasks.py` | 模型构建核心（约 2000 行）：`parse_model`、`BaseModel`、`DetectionModel`、`fuse`、权重加载 |
| `ultralytics/nn/modules/block.py` | 结构块：`Conv` 家族之外的所有块，含 YOLO12 核心 `AAttn`（区域注意力）、`ABlock`、`A2C2f`、`C3k2`、`C2f`、`SPPF`、`DFL`、PSA 家族 |
| `ultralytics/nn/modules/conv.py` | 卷积基元：`Conv`、`DWConv`、`Concat`、`CBAM`、`RepConv` 等 |
| `ultralytics/nn/modules/head.py` | 检测头：`Detect`、`Segment`、`Pose`、`OBB`、`Classify` 等 |
| `ultralytics/engine/trainer.py` | `BaseTrainer`：训练主循环、优化器构建、调度器、AMP、EMA、checkpoint |
| `ultralytics/models/yolo/detect/train.py` | `DetectionTrainer`：检测任务的 `get_model/preprocess_batch/build_dataset` 等 |
| `ultralytics/utils/loss.py` | `v8DetectionLoss`（分类/框/DFL 三部分损失）、`BboxLoss`、`DFLoss`、`E2ELoss` |
| `ultralytics/utils/tal.py` | `TaskAlignedAssigner`（动态正样本分配）、`make_anchors`、`dist2bbox`、`bbox2dist` |
| `ultralytics/utils/nms.py` | `non_max_suppression`（本仓库已从 `ops.py` 独立成文件） |
| `ultralytics/utils/metrics.py` | `box_iou`、`bbox_iou`（CIoU 等）、`ap_per_class`、`DetMetrics`（mAP 统计） |
| `ultralytics/engine/predictor.py` | `BasePredictor`：推理主循环（预处理→前向→后处理） |
| `ultralytics/models/yolo/detect/predict.py` | `DetectionPredictor`：检测推理后处理（NMS 调用、坐标还原） |
| `ultralytics/engine/results.py` | `Results`/`Boxes`：推理结果数据结构（`xyxy/xywh/xyxyn/xywhn` 等） |
| `ultralytics/data/augment.py` | 全部数据增强类：`Mosaic`、`MixUp`、`CutMix`、`RandomPerspective`、`LetterBox`、`Format` 等 |
| `ultralytics/data/dataset.py` | `YOLODataset`：标签扫描、变换管线装配、`collate_fn` |
| `ultralytics/data/build.py` | `build_yolo_dataset`、`build_dataloader`、`InfiniteDataLoader`、推理源工厂 |
| `ultralytics/nn/AddModules/` | 本仓库自定义改进模块目录（SimAM、EMA、DySample、FreqFusion 等 16 个文件） |

---

## 2. 安装与快速上手

### 2.1 安装

仓库根目录包含 `pyproject.toml`，推荐以可编辑模式安装（便于边改边研究）：

```bash
cd D:\Data\New_Codes\Python_Codes\ultralytics
pip install -e .
```

主要依赖：`torch`、`torchvision`、`opencv-python`、`numpy`、`matplotlib`、`pandas`、`pillow`、`pyyaml`、`tqdm`（仓库自带零依赖实现）。可选：`albumentations`（数据增强）、`onnx`/`onnxruntime`（导出）、`tensorboard` 等。

### 2.2 最小可运行示例（Python API）

```python
from ultralytics import YOLO

# ① 从零构建并训练（结构来自 cfg/models/12/yolo12.yaml，scale=n）
model = YOLO("yolo12n.yaml")
model.train(data="coco8.yaml", epochs=100, imgsz=640, batch=16, device=0)
#   coco8.yaml 是 8 张图的玩具数据集，会自动下载，适合冒烟测试

# ② 加载预训练权重微调（仓库根目录自带 yolo12n.pt / yolo12s.pt）
model = YOLO("yolo12n.pt")
model.train(data="my_dataset.yaml", epochs=100)

# ③ 验证：返回 precision / recall / mAP50 / mAP50-95 / fitness
metrics = model.val()
print(metrics.box.map50, metrics.box.map)   # mAP50, mAP50-95

# ④ 推理
results = model.predict("bus.jpg", conf=0.25, iou=0.7)
for r in results:
    print(r.boxes.xyxy)      # 原图坐标 (N,4)
    print(r.boxes.conf)      # 置信度
    print(r.boxes.cls)       # 类别索引

# ⑤ 导出
model.export(format="onnx", imgsz=640, half=False)
```

### 2.3 命令行（CLI）

安装后可用 `yolo` 命令，语法为 `yolo 任务 模式 key=value`：

```bash
yolo train   model=yolo12n.yaml data=coco8.yaml epochs=100 imgsz=640 batch=16
yolo val     model=runs/detect/train/weights/best.pt data=coco8.yaml
yolo predict model=yolo12n.pt source=bus.jpg conf=0.25
yolo export  model=yolo12n.pt format=onnx
```

所有结果默认保存在 `runs/<任务>/<实验名>/`（权重在 `weights/last.pt`、`weights/best.pt`，曲线在 `results.csv`，日志在 `args.yaml`）。`project=` 与 `name=` 可自定义输出位置。

> 本仓库的默认任务/模型映射（`cfg/__init__.py` 的 `TASK2MODEL`）指向较新的 yolo26 系列，因此**做 YOLO12 研究时务必显式写 `model=yolo12*.yaml` 或 `model=yolo12*.pt`**。

---

## 3. 框架核心设计思想

### 3.1 "任务 × 模式"二维分派

框架把一切操作抽象为 **任务（task）× 模式（mode）**：

- 任务 `TASKS = {detect, segment, classify, pose, obb, semantic}`（`cfg/__init__.py` L60）
- 模式 `MODES = {train, val, predict, export, track, benchmark}`（L59）

`YOLO` 类（`models/yolo/model.py` L27）继承 `engine/model.py` 的 `Model` 基类，通过 `task_map`（L86-126）把每个任务映射到四件套：

```python
"task_map":
    "detect": {
        "model":     DetectionModel,          # nn/tasks.py，网络本体
        "trainer":   DetectionTrainer,        # models/yolo/detect/train.py
        "validator": DetectionValidator,      # models/yolo/detect/val.py
        "predictor": DetectionPredictor,      # models/yolo/detect/predict.py
    },
```

调用 `model.train()` 时，`Model.train()`（engine/model.py L715）通过 `_smart_load("trainer")` 查表取出 `DetectionTrainer` 并实例化。**研究员改进网络时只需要关心 `model`（yaml + nn 模块），训练/验证/推理逻辑自动复用。**

### 3.2 yaml 驱动的网络定义

网络结构不写死在 Python 里，而是由 yaml 的 `backbone`/`head` 列表描述，每行 `[from, repeats, module, args]`。`nn/tasks.py` 的 `parse_model`（约 L1640-1890）逐行解析：

- `from`：输入来自哪些层（-1 = 上一层；列表 = 多输入拼接）
- `repeats`：重复次数（受 `depth` 缩放）
- `module`：模块类名（字符串，运行时用 `globals()` 解析——**这是自定义模块能直接被 yaml 引用的原因**）
- `args`：构造参数（输出通道受 `width` 缩放）

这带来两大好处：改结构只改 yaml；同一套解析器支持 v3~v26 所有版本。

### 3.3 **端到端数据流总览**

```
                     cfg/models/12/yolo12.yaml
                                │ yaml_model_load（提取 scale n/s/m/l/x）
                                ▼
YOLO("yolo12n.yaml") ──> DetectionModel ──> parse_model 逐层实例化 ──> nn.Sequential
                                │                        （dummy forward 计算 stride=8/16/32）
cfg/datasets/coco8.yaml         │
      │ check_det_dataset       ▼
      ▼                  BaseTrainer._do_train()
YOLODataset ─DataLoader─> batch ──> 前向 → v8DetectionLoss → 反向 → EMA 更新
                                │
                        每 epoch validate() ──> fitness(mAP50-95) ──> best.pt / last.pt
                                │
                        model.predict(): AutoBackend(fuse) → Detect → NMS → Results
```

### 3.4 回调系统

训练/验证/推理的关键节点都会触发回调（`run_callbacks`），默认回调定义在 **`utils/callbacks/base.py`**（注意：本仓库已把旧版 `default.py` 改名为 `base.py`）。常用事件：

| 事件 | 触发位置 |
|---|---|
| `on_pretrain_routine_end` | `_setup_train` 结束 |
| `on_train_epoch_start/end`、`on_train_batch_start/end` | 训练循环 |
| `on_fit_epoch_end` | 每个 epoch 验证后（写 TensorBoard 等） |
| `on_model_save` | checkpoint 保存后 |
| `on_train_end` | 训练结束 |
| `on_val_start/end`、`on_val_batch_end` | 验证过程 |
| `on_predict_start/postprocess/end` | 推理过程 |

TensorBoard、MLflow、Comet 等集成都是以回调形式挂载的（`add_integration_callbacks`，base.py L195-235）。研究员可以用 `model.add_callback("on_fit_epoch_end", my_fn)` 注入自定义逻辑（如自定义指标记录）。

---

## 4. 配置系统详解

### 4.1 `cfg/` 目录结构

```
cfg/
├── __init__.py        # 配置加载/校验/CLI 入口（约 1080 行）
├── default.yaml       # 全局默认超参数（148 行）
├── datasets/          # 42 个数据集配置（coco.yaml、coco8.yaml、VOC.yaml、dota8.yaml …）
├── models/            # 模型结构 yaml
│   ├── 11/            # yolo11 系列
│   ├── 12/            # YOLO12 系列（yolo12.yaml / -seg / -pose / -obb / -cls）★
│   ├── 26/            # yolo26 系列（含 p2/p6 变体，可作多层级改进的参考）
│   ├── v3/ v5/ v6/ v8/ v9/ v10/  rt-detr/
└── trackers/          # botsort.yaml / bytetrack.yaml
```

### 4.2 `default.yaml` 全量超参数（按分组）

**顶层（L6-7）**：`task: detect`、`mode: train`。

**训练设置（L9-40）**：

| 参数 | 默认 | 含义 |
|---|---|---|
| `model` / `data` | 空 | 模型文件（.pt 或 .yaml）/ 数据集 yaml |
| `epochs` | 100 | 训练轮数；`time` 设置后按小时数覆盖 |
| `patience` | 100 | 早停：验证指标 N 轮不提升即停 |
| `batch` | 16 | 整数固定批；0.0–1.0 浮点为 AutoBatch 显存占比 |
| `imgsz` | 640 | 输入尺寸 |
| `save` / `save_period` | True / -1 | 保存 checkpoint；每 N 轮额外存一次 |
| `cache` | False | 图像缓存到 `ram` 或 `disk` |
| `device` | 空 | `0`、`0,1,2,3`、`cpu`、`-1`（自动选空闲卡） |
| `workers` | 8 | 数据加载进程数 |
| `project` / `name` / `exist_ok` | 空/空/False | 结果目录控制 |
| `pretrained` | True | 布尔=使用官方权重；字符串=指定权重路径 |
| `optimizer` | auto | SGD/MuSGD/Adam/Adamax/AdamW/NAdam/RAdam/RMSProp/auto |
| `seed` / `deterministic` | 0 / True | 随机种子与确定性 |
| `single_cls` | False | 全部类别合并为 1 类 |
| `rect` | False | 矩形批训练（减少填充）；验证默认开启 |
| `cos_lr` | False | 余弦退火学习率（默认线性衰减） |
| `close_mosaic` | 10 | 最后 N 个 epoch 关闭 mosaic |
| `resume` | False | 断点续训 |
| `amp` | True | 自动混合精度 |
| `fraction` | 1.0 | 训练集抽样比例（快速验证时很有用） |
| `freeze` | 空 | 冻结前 N 层或指定层索引列表 |
| `multi_scale` | 0.0 | 多尺度训练幅度（相对 imgsz） |
| `compile` | False | torch.compile 加速 |

**验证/推理/导出（L49-92）**：`val`、`split`、`save_json`、`conf`（predict 默认 0.25，val 默认 0.001）、`iou: 0.7`、`max_det: 300`、`half`、`plots`、`end2end`；predict 侧 `source`、`vid_stride`、`visualize`、`augment`（TTA）、`agnostic_nms`、`classes`、`retina_masks`、`embed`；可视化 `show`、`save_txt`、`save_conf`、`save_crop`、`show_labels/conf/boxes`、`line_width`；导出 `format`（onnx/openvino/engine/tflite/…）、`int8`、`dynamic`、`simplify`、`opset`、`workspace`、`nms`。

**训练超参数（L94-128）——算法研究员最常调的部分**：

| 分组 | 参数（默认值） |
|---|---|
| 优化器/学习率 | `lr0: 0.01`、`lrf: 0.01`（最终 lr = lr0×lrf）、`momentum: 0.937`、`weight_decay: 0.0005`、`warmup_epochs: 3.0`、`warmup_momentum: 0.8`、`warmup_bias_lr: 0.1` |
| 损失权重 | `box: 7.5`、`cls: 0.5`、`cls_pw: 0.0`（类别不平衡幂）、`dfl: 1.5`、`pose: 12.0`、`kobj: 1.0`、`angle: 1.0`、`nbs: 64`（名义批，用于损失归一化与梯度累积） |
| 数据增强 | `hsv_h: 0.015`、`hsv_s: 0.7`、`hsv_v: 0.4`、`degrees: 0.0`、`translate: 0.1`、`scale: 0.5`、`shear: 0.0`、`perspective: 0.0`、`flipud: 0.0`、`fliplr: 0.5`、`bgr: 0.0`、`mosaic: 1.0`、`mixup: 0.0`、`cutmix: 0.0`、`copy_paste: 0.0`、`copy_paste_mode: flip`、分类专用 `auto_augment: randaugment`、`erasing: 0.4` |

**本仓库特有的自定义损失开关（L130-142）**：`wiou`（Wise-IoU v3，配 `wiou_alpha`、`wiou_momentum`）、`focaler_ciou`、`sd_loss`、`powerful_iou`、`slide_loss`（替换分类 BCE）。详见第 8 章与第 15 章。

**其它（L144-148）**：`cfg:`（用自定义 yaml 整体覆盖默认配置）、`tracker: botsort.yaml`。

### 4.3 配置加载与校验（`cfg/__init__.py`）

- `get_cfg(cfg, overrides)`（L304-348）：加载基础配置 → 与用户覆盖项合并（用户优先）→ `check_dict_alignment` 拒绝未知键（带近似拼写提示）→ `check_cfg` 类型校验 → 返回 `IterableSimpleNamespace`。
- 类型键集合：`CFG_FLOAT_KEYS`（L175-188）、`CFG_FRACTION_KEYS`（L189-215，必须 0~1）、`CFG_INT_KEYS`（L216-230）、`CFG_BOOL_KEYS`（L231-267）。**新增自定义超参时必须把键登记进对应集合，否则类型校验会失败或按字符串处理。**
- `entrypoint()`（L877-1045）：CLI 入口，解析 `key=value`，最终 `getattr(model, mode)(**overrides)`。
- `yolo copy-cfg`：把 default.yaml 复制到当前目录供整体定制。

### 4.4 数据集配置格式（`cfg/datasets/*.yaml`）

以 `coco8.yaml` 为例：

```yaml
path: coco8            # 数据集根目录（相对路径会到 datasets/ 下查找/自动下载）
train: images/train    # 训练图像目录（相对 path；可为列表或 .txt 清单）
val: images/val        # 验证图像目录
test:                  # 可选
names:                 # 类别字典：索引 → 类名
  0: person
  1: bicycle
  ...
download: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip
```

- `check_det_dataset`（`data/utils.py` L453-546）负责解析：校验 `train/val/names`，把相对路径拼到 `datasets/` 下，不存在时按 `download` 字段自动下载（zip URL / bash 脚本 / 内联 python）。
- 关键点数据集（`coco8-pose.yaml`）额外有 `kpt_shape: [17, 3]`、`flip_idx`（水平翻转时关键点左右对称索引置换）。
- 自定义数据集只需：按 `images/train`、`images/val`、`labels/train`、`labels/val` 组织目录，每张图对应同名 `.txt`（YOLO 格式：每行 `cls x_center y_center w h`，全部归一化到 0~1），再写一个如上格式的 yaml 即可。

---

## 5. YOLO12 模型配置文件详解

### 5.1 文件清单

`ultralytics/cfg/models/12/` 下共 5 个文件，**没有**按 n/s/m/l/x 分文件——规模由同一文件的 `scales` 字典控制（`model=yolo12n.yaml` 会被正则归一化为 `yolo12.yaml` 并取 scale `n`）：

| 文件 | 任务 | 末层模块 |
|---|---|---|
| `yolo12.yaml` | 目标检测 | `Detect [nc]` |
| `yolo12-seg.yaml` | 实例分割 | `Segment [nc, 32, 256]`（32 个掩膜系数 + 256 通道 proto） |
| `yolo12-pose.yaml` | 姿态估计 | `Pose [nc, kpt_shape]`，yaml 内含 `kpt_shape: [17, 3]` |
| `yolo12-obb.yaml` | 旋转框检测 | `OBB [nc, 1]` |
| `yolo12-cls.yaml` | 图像分类 | `Classify [nc]`，nc=1000 |

### 5.2 `yolo12.yaml` 全文与逐层解析（检测主文件，49 行）

```yaml
# Ultralytics YOLO12 object detection model with P3/8 - P5/32 outputs
# nc: number of classes. Scales: n, s, m, l, x
nc: 80   # 训练时会被数据集 names 的数量自动覆盖

# [depth, width, max_channels]
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# YOLO12n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]     # 6   ← 注意力核心（P4，area=4）
  - [-1, 1, Conv, [1024, 3, 2]]        # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]    # 8   ← 注意力核心（P5，area=1 全局）

# YOLO12n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 9
  - [[-1, 6], 1, Concat, [1]]                       # 10  cat backbone P4
  - [-1, 2, A2C2f, [512, False, -1]]                # 11  ← neck P4（纯卷积）

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 12
  - [[-1, 4], 1, Concat, [1]]                       # 13  cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]]                # 14  ← P3/8 输出（小目标）

  - [-1, 1, Conv, [256, 3, 2]]                      # 15
  - [[-1, 11], 1, Concat, [1]]                      # 16  cat head P4
  - [-1, 2, A2C2f, [512, False, -1]]                # 17  ← P4/16 输出（中目标）

  - [-1, 1, Conv, [512, 3, 2]]                      # 18
  - [[-1, 8], 1, Concat, [1]]                       # 19  cat head P5
  - [-1, 2, C3k2, [1024, True]]                     # 20  ← P5/32 输出（大目标）

  - [[14, 17, 20], 1, Detect, [nc]]                 # 21  Detect(P3, P4, P5)
```

要点：

- **格式约定**：每行 `[from, repeats, module, args]`；`from=-1` 取上一层输出，`from=[a,b]` 取多层（通常配合 `Concat`）。
- **无 anchors、无 stride 字段**：YOLO12 是 anchor-free 检测，步幅由下采样层隐式决定，建模时用一次假前向自动算出 `[8, 16, 32]`。
- **depth 缩放**：`repeats × depth` 后取整（最小 1），作用于堆叠型模块（`C3k2`、`A2C2f`）。n/s/m 的 depth=0.5，即 `repeats=4` 变 2；l/x 的 depth=1.0。
- **width 缩放**：`args[0]`（输出通道）× width 后向 8 对齐，且不超过 `max_channels`。n 的 width=0.25，所以 `[64,...]` 实际是 16 通道。
- **官方规模参数**（注释中的 summary）：

| scale | depth/width/max_ch | 层数 | 参数量 | GFLOPs |
|---|---|---|---|---|
| n | 0.50 / 0.25 / 1024 | 272 | 2,602,288 | 6.7 |
| s | 0.50 / 0.50 / 1024 | 272 | 9,284,096 | 21.7 |
| m | 0.50 / 1.00 / 512 | 292 | 20,199,168 | 68.1 |
| l | 1.00 / 1.00 / 512 | 488 | 26,450,784 | 89.7 |
| x | 1.00 / 1.50 / 512 | 488 | 59,210,784 | 200.3 |

### 5.3 YOLO12 的结构特色（相对 YOLO11）

- 主干浅层（P2/P3 段）仍用卷积块 `C3k2`；**深层 P4/P5 换成了注意力模块 `A2C2f`**（Area-Attention 增强的 C2f），这是 YOLO12 "attention-centric" 设计的核心。
- P4 层 `A2C2f [512, True, 4]`：`True`=启用注意力，`4`=把特征序列切成 4 个区域做局部注意力（降低复杂度）；P5 层 `[1024, True, 1]`：`area=1` 即全局注意力（特征图小，承受得起）。
- 颈部（FPN+PAN）三路分支用 `A2C2f [c, False, -1]`：`False`=退化为纯卷积 C3k 路径（大特征图上不做注意力以省算力），`-1` 是 area 占位（不生效）。
- 大输出（P5）分支末层用了 `C3k2 [1024, True]`（`attn=True`，内部为 Bottleneck+PSABlock）。
- 检测头沿用 v8 风格解耦头（回归 + 分类两支），`reg_max=16` 的 DFL 分布回归。

### 5.4 变体差异

四个变体的 backbone 与 head 前 20 层与检测版**完全一致**，仅最后一行不同（见 5.1 表格）。因此**对检测的改进经验可以直接迁移到 seg/pose/obb 任务**。

---

## 6. 模型构建流程：从 yaml 到可训练网络

### 6.1 总链路

```
YOLO("yolo12n.yaml")
 └─ Model.__init__（engine/model.py L81-147）
     └─ _new(model)（L226-257，yaml 分支）
         ├─ yaml_model_load(cfg)（nn/tasks.py L1893-1913）
         │    ├─ 正则把 "yolo12n.yaml" 归一化为 "yolo12.yaml" 并加载
         │    ├─ d["scale"] = guess_model_scale(path)   # 提取 'n'
         │    └─ d["yaml_file"] = path
         ├─ guess_model_task(cfg) → "detect"（head 末层模块名判定）
         └─ DetectionModel(cfg_dict)（nn/tasks.py L365-522）
             ├─ _initialize_yolo_model（L346-362）
             │    ├─ model.yaml["channels"] = ch(=3)；nc 覆盖
             │    ├─ model.model, model.save = parse_model(yaml, ch)   ★ 核心
             │    └─ model.names = {0:'0', ...}
             ├─ 假前向计算 stride：输入 (1,3,256,256)，
             │    m.stride = 256 / 各输出层边长 → [8,16,32]（L420）
             ├─ m.bias_init()（head.py L196-208，头偏置先验初始化）
             ├─ initialize_weights（BN eps=1e-3、激活 inplace）
             └─ model.info()（打印 "YOLO12 summary: 272 layers, 2602288 parameters, 6.7 GFLOPs"）
```

加载 `.pt` 权重则走 `_load` → `load_checkpoint`（tasks.py L1602-1638）：`torch_safe_load` 读盘 → **优先取 `ckpt["ema"]`**（EMA 权重）→ `guess_model_task` → `fuse()`（Conv+BN 融合）→ `eval().to(device)`。

### 6.2 `parse_model` 逐分支详解（tasks.py 约 L1640-1890）

前置（L1656-1676）：从 yaml 读 `nc, scales, reg_max(默认16), end2end`；按 `scale` 解包 `depth, width, max_channels = scales[scale]`；`ch = [ch]` 记录各层输出通道。

主循环逐层解析（L1752 起）：

1. **模块类解析**（L1753-1759）：`"nn."` 前缀 → `getattr(torch.nn, ...)`；否则 `globals()[m]`。**`globals()` 能查到自定义模块，是因为 tasks.py L102-103 有 `from .AddModules import *`。**
2. **字符串参数替换**：`"nc"` 等被替换为实际值。
3. **depth 乘法**：`n = max(round(n * depth), 1)`（L1764）。
4. **分支路由**：

| 分支 | 适用模块 | 通道计算 |
|---|---|---|
| `m in base_modules`（L1677-1731 的 frozenset：Conv、C2f、C3k2、A2C2f、SPPF、ADown、C2PSA 等 30+ 类） | 常规结构块 | `c1=ch[f]`；`c2 = make_divisible(min(args[0], max_channels) * width, 8)`；`args=[c1,c2,*args[1:]]`；若同时在 `repeat_modules`（BottleneckCSP、C2f、C3k2、A2C2f 等），`repeats` 插入为构造参数 |
| `m is Concat` | 拼接 | `c2 = sum(ch[x] for x in f)` |
| 检测头族 `{Detect, Segment, Pose, OBB, ...}`（L1801-1820） | 头部 | `args.extend([reg_max, end2end, [ch[x] for x in f]])` —— yaml 里只写 `[nc]`，其余全部自动注入 |
| `m is DySample`（L1860-1863） | 动态上采样 | `c2=ch[f]`；`args=[c2,*args]`（注入输入通道，不改变通道数） |
| `m in {FreqFusion}`（L1849）/ `FreqFusionUpsample`（L1854） | 频率融合 | 多输入/单输入的注入式分支 |
| `m is BiFPN`（L1872） | 加权融合 | 只注入输入流数量 |
| `else`（如 `nn.Upsample`） | 透传 | `c2 = ch[f]` |

5. **A2C2f 特判**（L1781-1784）：`legacy=False`；**scale 为 l/x 时自动 `args.extend((True, 1.2))`**，即大模型开启残差 γ 并把 mlp_ratio 降到 1.2。
6. **收尾**（L1879-1890）：实例化（`n>1` 时用 `nn.Sequential` 堆叠）；记录 `m_.i/m_.f/m_.type`；把被引用的层索引加入 `save`（前向时缓存这些中间输出）；`ch.append(c2)`。

### 6.3 `BaseModel` 前向：`_predict_once`（tasks.py L165-193）

```python
def _predict_once(self, x, profile=False, visualize=False, embed=None):
    y = []  # 缓存中间输出（仅 save 列表中的层）
    for m in self.model:
        if m.f != -1:                      # 非顺序连接
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in self.save else None)
    return x
```

训练时 `forward` 收到 dict 输入会转去算损失（`self.loss(x)`），推理时走 `predict`。

### 6.4 各任务模型类（tasks.py）

| 类 | 行号 | 损失函数（`init_criterion`） |
|---|---|---|
| `DetectionModel` | L365-522 | `v8DetectionLoss`（end2end 时 `E2ELoss`） |
| `SegmentationModel` | L557-586 | `v8SegmentationLoss` |
| `PoseModel` | L661-699 | `v8PoseLoss` |
| `OBBModel` | L525-554 | `v8OBBLoss` |
| `ClassificationModel` | L702-791 | `v8ClassificationLoss` |

YOLO12 检测复用 `DetectionModel`（框架按任务而非按版本分类）。`BaseModel.load`（L302-325）支持**部分权重迁移**：`intersect_dicts` 只加载形状匹配的参数，因此改了网络结构后仍可从官方权重热启动（首卷积通道数不匹配也有专门处理）。

---

## 7. YOLO12 网络结构深度剖析

模块全部位于 `ultralytics/nn/modules/`：`conv.py`（卷积基元）、`block.py`（结构块与注意力）、`head.py`（检测头）、`transformer.py`（Transformer 组件）、`utils.py`（初始化等）。**注意：仓库中没有独立的 `attention.py`，区域注意力 `AAttn` 在 `block.py`；也没有名为 `AreaAttention` 的类。**

### 7.1 卷积基元（conv.py）

- **`Conv`**（L39-89）：`Conv2d(bias=False) + BatchNorm2d + SiLU`。类属性 `default_act = nn.SiLU()`（L49）是全局默认激活；`autopad`（L30-36）实现 same-padding。`forward = act(bn(conv(x)))`；`fuse()` 后走 `forward_fuse`（BN 权重已吸收进卷积）。
- **`DWConv`**（L185-199）：深度可分离卷积，`g = gcd(c1, c2)`。
- **`Concat`**（L616-641）：`torch.cat(x, dim)`，默认通道维。
- 其它：`Conv2`（简化 RepConv）、`ConvTranspose`、`Focus`、`GhostConv`、`RepConv`、`ChannelAttention`、`SpatialAttention`、`CBAM`、`Index`。

### 7.2 结构块（block.py）

- **`Bottleneck`**（L457-481）：双卷积瓶颈 `cv1(1x1或3x3) → cv2`，`shortcut and c1==c2` 时加残差。
- **`C2f`**（L288-319）：YOLOv8/11/12 的融合基块。`cv1` 输出 `2c` 后 chunk 两支，n 个 Bottleneck **链式**传递，最后把**所有中间分支一起 concat** 过 `cv2`（梯度路径更丰富）：

```python
def forward(self, x):
    y = list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))
```

- **`C3k2`**（L1069-1106）：`C2f` 的子类，内部块可切换：`c3k=True` 用串联小卷积 `C3k`；`attn=True` 用 `Bottleneck + PSABlock`（卷积自注意力）；否则普通 `Bottleneck`。YOLO12 用它做浅层与 P5 输出分支。
- **`SPPF`**（L208-237）：同一 5×5 MaxPool 串行 3 次再拼接，等效感受野 5/9/13。
- **`ADown`**（L940-962）：avg-pool + max-pool 双路下采样（信息损失更小）。
- **`DFL`**（L58-80）：见 7.6。
- PSA 家族：`Attention`（L1271-1328，1x1 qkv 卷积式自注意力 + 3x3 DW 位置编码）、`PSABlock`、`PSA`、`C2PSA`、`C2fPSA`。

### 7.3 YOLO12 核心：区域注意力 `AAttn`（block.py L1641-1727）

```python
def __init__(self, dim, num_heads, area=1):
    self.area = area
    self.head_dim = head_dim = dim // num_heads
    self.qkv  = Conv(dim, head_dim*num_heads*3, 1, act=False)      # 1x1 卷积出 Q/K/V
    self.proj = Conv(head_dim*num_heads, dim, 1, act=False)
    self.pe   = Conv(dim, dim, 7, 1, 3, g=dim, act=False)          # 7x7 DW 卷积位置编码
```

`forward` 的关键逻辑（L1691-1727）：

```python
B, _, H, W = x.shape
N = H * W
qkv = self.qkv(x).flatten(2).transpose(1, 2)          # (B, N, 3*dim)
if self.area > 1:
    qkv = qkv.reshape(B * self.area, N // self.area, ...)   # ① 序列等分为 area 段
q, k, v = ...按头拆分...
attn = (q.transpose(-2, -1) @ k) * (head_dim ** -0.5)       # ② 每区域内独立注意力
attn = attn.softmax(dim=-1)
x = v @ attn.transpose(-2, -1)
if self.area > 1:
    x = x.reshape(B // self.area, N * self.area, ...)        # ③ 恢复原批维
x = x.reshape(B, H, W, dim)... + self.pe(v)                  # ④ V 上叠加 7x7 DW 位置编码
return self.proj(x)
```

**机制本质**：把展平后的 `N=H*W` 序列**等分成 area 段**（折进批维），注意力复杂度从 O(N²) 降为 area×O((N/area)²)；区域划分是按展平序列切的（并非规则 2D 分块），因此要求 `N` 能被 `area` 整除。7×7 DW 卷积 `pe` 弥补切区域后的局部位置信息。**改进思路**：若想把区域划分改成 2D 网格、或让区域大小自适应，改动点就在 ①③ 两处 reshape。

### 7.4 `ABlock` 与 `A2C2f`

**`ABlock`**（L1730-1792）：标准 Transformer 式双残差块：`x = x + AAttn(x)`；`x = x + MLP(x)`（MLP 为两段 1x1 卷积，隐层 ×mlp_ratio，默认 2.0）。

**`A2C2f`**（L1795-1874）——YOLO12 主干的核心：

```python
def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False,
             mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
    c_ = int(c2 * e)
    assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv((1 + n) * c_, c2, 1)
    self.gamma = nn.Parameter(0.01 * torch.ones(c2)) if a2 and residual else None
    self.m = nn.ModuleList(
        nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
        if a2 else C3k(c_, c_, 2, shortcut, g)
        for _ in range(n))

def forward(self, x):
    y = [self.cv1(x)]
    y.extend(m(y[-1]) for m in self.m)      # C2f 式链式多分支
    y = self.cv2(torch.cat(y, 1))
    if self.gamma is not None:
        return x + self.gamma.view(-1, c2, 1, 1) * y   # 可学习缩放残差，稳定深层注意力
    return y
```

要点：

- 参数表：`c1` 输入通道，`c2` 输出通道，`n` 子块数（repeats×depth），`a2` 是否启用注意力，`area` 区域数，`residual` 是否加 γ 残差（l/x 规模自动开启），`mlp_ratio`。
- **约束**：隐层 `c_` 必须是 32 的倍数（`num_heads = c_//32`）——改 width 缩放或自定义通道时要注意。
- yaml 与代码的对应关系：

| yaml 写法 | 实际含义 |
|---|---|
| 主干层6 `A2C2f, [512, True, 4]` | a2=True, area=4（P4 特征图 40×40，序列 1600 切 4 段） |
| 主干层8 `A2C2f, [1024, True, 1]` | a2=True, area=1（P5 特征图 20×20，全局注意力） |
| 颈部 `A2C2f, [c, False, -1]` | a2=False → 纯卷积 C3k 路径（大特征图省算力） |

- **改进范式**：仓库 `nn/AddModules/` 里的 `A2C2f_EMA`、`A2C2f_MCA`、`A2C2f_SCSA` 等都是"保持 A2C2f 的 C2f 骨架、把内部 ABlock 的注意力换成自研注意力"的封装，可直接参考。

### 7.5 检测头 `Detect`（head.py L37-262）

YOLO12 的 `Detect` 直接继承 `nn.Module`（本版本无 BaseDetect 基类）；`Segment`、`Pose`、`OBB` 等均在它之上扩展。

```python
self.nc = nc
self.nl = len(ch)                     # 输出层数 = 3（P3/P4/P5）
self.reg_max = reg_max                # DFL 分布桶数，默认 16
self.no = nc + reg_max * 4            # 每个锚点输出维度（无 objectness）
self.stride = torch.zeros(self.nl)    # 建模时由假前向填为 [8,16,32]
# cv2：回归分支 3x3→3x3→1x1，输出 4*reg_max 通道
# cv3：分类分支 (DWConv3x3→1x1)×2→1x1，输出 nc 通道
self.dfl = DFL(reg_max)
```

`forward`（L157-171）按训练/推理分流：

```python
preds = self.forward_head(x, **self.one2many)   # 三路特征图各自过 cv2/cv3，展平后拼接
if self.training:
    return preds            # dict(boxes=(B,4*16,N), scores=(B,nc,N), feats)
y = self._inference(preds)  # 推理：DFL 解码 → dist2bbox → 乘 stride → 拼 sigmoid 分类分
return y if self.export else (y, preds)
```

- `make_anchors`（tal.py L415-428）：按各级特征图尺寸生成网格中心点（+0.5）与对应步长。
- `_get_decode_boxes`（L186-194）：`dbox = decode_bboxes(dfl(boxes), anchors) * strides`，其中 `dist2bbox`（tal.py L431-440）把 (l,t,r,b) 距离换算成中心点+宽高。
- `bias_init`（L196-208）：回归偏置置 2.0；分类偏置按 `log(5/nc/(640/stride)²)` 初始化（目标密度先验），加速收敛。
- 640 输入时锚点总数 = 80² + 40² + 20² = **8400**，推理输出形状 `(B, 4+nc, 8400)`。

### 7.6 `DFL` 模块（Distribution Focal Loss 的推理实现，block.py L58-80）

```python
self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
self.conv.weight.data[:] = nn.Parameter(torch.arange(c1).view(1, c1, 1, 1))  # 权重固定 [0..15]
def forward(self, x):
    return self.conv(x.view(b, 4, c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
```

即对每条边预测的 16 个分布桶先 softmax，再用固定权重 `[0,1,...,15]` 求期望，把"分布"积分成标量距离。训练时对应的损失是 `utils/loss.py` 的 `DFLoss`。

### 7.7 YOLO12n（640 输入）的特征尺寸流

| 层 | 模块 | 输出尺寸（n 规模通道） | 备注 |
|---|---|---|---|
| 0 | Conv 64→16 | 320×320 | P1/2 |
| 1 | Conv 128→32 | 160×160 | P2/4 |
| 2 | C3k2 256→64 ×1 | 160×160 | repeats 2×0.5=1 |
| 3 | Conv 256→64 | 80×80 | P3/8 |
| 4 | C3k2 512→128 ×1 | 80×80 | |
| 5 | Conv 512→128 | 40×40 | P4/16 |
| 6 | A2C2f 512→128, area=4 ×2 | 40×40 | 主干注意力 |
| 7 | Conv 1024→256 | 20×20 | P5/32 |
| 8 | A2C2f 1024→256, area=1 ×2 | 20×20 | 主干注意力 |
| 9-11 | Upsample+Concat(P4)+A2C2f | 40×40 | FPN P4 |
| 12-14 | Upsample+Concat(P3)+A2C2f | 80×80 | FPN P3（小目标） |
| 15-17 | Conv↓+Concat+ A2C2f | 40×40 | PAN P4 |
| 18-20 | Conv↓+Concat+ C3k2 | 20×20 | PAN P5（大目标） |
| 21 | Detect(from=[14,17,20]) | 8400 锚点 | |

（通道数为 n 规模：width=0.25 缩放且受 max_channels=1024 限制后的实际值。）

---

## 8. 损失函数与标签分配

### 8.1 损失入口

训练态下 `model(batch)` 会走到 `BaseModel.loss`（tasks.py L327-339），首次调用时通过 `init_criterion()` 创建损失对象。`DetectionModel.init_criterion`（L520-522）：

```python
return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
```

从纯 `yolo12.yaml` 构建时 `end2end=False`（yaml 无该字段），所以 **YOLO12 标准训练走 `v8DetectionLoss`**。

### 8.2 三部分损失（`utils/loss.py` v8DetectionLoss，L438-602）

| 组成 | 实现 | 增益（default.yaml） |
|---|---|---|
| 分类损失 | `BCEWithLogitsLoss`，目标为 TAL 归一化后的**软标签**（非 one-hot） | `cls: 0.5` |
| 框回归损失 | `1 - CIoU`（按对齐度加权，仅前景） | `box: 7.5` |
| 分布焦点损失 | `DFLoss`：把连续距离拆到相邻两个 bin 的加权交叉熵 | `dfl: 1.5` |

核心流程（`get_assigned_targets_and_loss`，L520-582）：

```python
anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)      # 8400 个锚点
pred_bboxes = self.bbox_decode(anchor_points, pred_distri)           # DFL softmax·期望 → dist2bbox
_, target_bboxes, target_scores, fg_mask, _ = self.assigner(         # ★ 动态正样本分配
    pred_scores.detach().sigmoid(),
    (pred_bboxes.detach() * stride_tensor),
    anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
target_scores_sum = max(target_scores.sum(), 1)
loss[1] = self.bce(pred_scores, target_scores).sum() / target_scores_sum   # cls
if fg_mask.sum():
    loss[0], loss[2] = self.bbox_loss(...)                                   # box + dfl
loss[0] *= self.hyp.box; loss[1] *= self.hyp.cls; loss[2] *= self.hyp.dfl
return loss * batch_size, loss.detach()
```

返回值是 `(按批缩放的损失向量, 其 detach 副本)`：前者参与 `loss.sum().backward()`，后者用于日志与验证时的损失累积。

### 8.3 TaskAlignedAssigner（`utils/tal.py`）——动态正样本分配

构造参数（由 `v8DetectionLoss` 传入）：`topk=10, alpha=0.5, beta=6.0, stride=[8,16,32]`。

1. **候选筛选** `select_candidates_in_gts`（L304-333）：锚点中心必须落在 GT 框内。**本仓库定制**：若某 GT 宽高 < 8 像素（小目标），临时扩到 16 再判断，保证小目标能选到锚点（L320-327）。
2. **对齐度计算** `get_box_metrics`（L180-211）：

   ```python
   align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)   # 分类分^0.5 × IoU^6.0
   ```

3. **topk 选择** `select_topk_candidates`（L232-261）：每个 GT 选对齐度最高的 10 个锚点作正样本——正样本数量与位置随预测质量**逐迭代动态变化**。
4. **冲突消解** `select_highest_overlaps`（L335-370）：一个锚点被多个 GT 选中时，只保留 IoU 最大的 GT。
5. **软标签归一化**（L141-151）：`target_scores = one_hot × (对齐度×IoU / 最大值)`，作为分类损失的软目标。

辅助函数：`make_anchors`（L415-428）、`dist2bbox`（L431-440）、`bbox2dist`（L443-449，DFL 目标生成，clamp 到 `reg_max-0.01`）。

### 8.4 BboxLoss 与 DFLoss（loss.py L90-255）

- 前景回归权重 = TAL 归一化对齐度：`weight = target_scores.sum(-1)[fg_mask]`。
- 默认 `bbox_iou(pred, target, xywh=False, CIoU=True)`（`utils/metrics.py`）；`loss_iou = ((1-iou)*weight).sum() / target_scores_sum`。
- DFL 目标：`target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max-1)`；`DFLoss` 把目标距离拆到左右两个 bin（`tl=floor, tr=tl+1`，权重按距离线性）做交叉熵加权和。
- `reg_max=1`（关闭 DFL）时退化为按图像尺寸归一化的 L1（YOLO26 风格）。

### 8.5 本仓库的损失扩展开关（default.yaml L130-142）

| 开关 | 效果 | 位置 |
|---|---|---|
| `wiou: True` | Wise-IoU v3（含异常值动量聚焦，`wiou_alpha`、`wiou_momentum` 可调） | BboxLoss L189-208 |
| `powerful_iou: True` | PIoU2 | BboxLoss L210-214 |
| `sd_loss: True` | SDIoU | BboxLoss L217-218 |
| `focaler_ciou: True` | IoU 重标定 (iou-0)/(0.95-0) 后再算损失 | BboxLoss L224-233 |
| `slide_loss: True` | 分类损失由 BCE 换为 SlideLoss | v8DetectionLoss L448-451 |

这些开关同时影响 `TaskAlignedAssigner.iou_calculation`（对齐度里的 IoU 也同步替换），是**学习"如何注册自定义损失"的现成范例**。

### 8.6 E2ELoss（端到端，了解即可）

`E2ELoss`（loss.py L1291-1323）：one2many（topk=10）+ one2one（topk=7, topk2=1）两套损失加权，one2many 权重从 0.8 线性衰减到 0.1，每 epoch 调 `criterion.update()`。仅在使用带 one2one 头的模型（如 YOLO26）时激活；YOLO12 标准训练不涉及。

---

## 9. 数据管线详解

### 9.1 文件职责

| 文件 | 职责 |
|---|---|
| `data/build.py` | `build_yolo_dataset`、`build_dataloader`、`InfiniteDataLoader`（worker 永久复用）、推理源工厂 `load_inference_source` |
| `data/base.py` | `BaseDataset`：图片发现（`get_img_files`）、读取与缩放（`load_image`）、ram/disk 缓存、rect 批形状预计算（`set_rectangle`） |
| `data/dataset.py` | `YOLODataset`：标签扫描与缓存（`.cache` 文件）、变换管线装配（`build_transforms`）、`collate_fn` |
| `data/augment.py` | 全部增强类（约 3100 行）：`Mosaic`、`MixUp`、`CutMix`、`RandomPerspective`、`RandomHSV`、`RandomFlip`、`LetterBox`、`CopyPaste`、`Albumentations`、`Format`、`v8_transforms` |
| `data/utils.py` | `img2label_paths`、`verify_image_label`、`check_det_dataset`（数据 yaml 解析与自动下载）、缓存读写 |
| `data/loaders.py` | 推理输入源：`LoadImagesAndVideos`、`LoadStreams`（摄像头/RTSP）、`LoadScreenshots`、`LoadPilAndNumpy`、`LoadTensor` |
| `data/converter.py` | COCO/DOTA 等格式互转 |

### 9.2 数据集解析链路

```
data="coco8.yaml"
 └─ check_det_dataset（data/utils.py L453-546）
     ├─ 校验 train/val/names 字段
     ├─ path 相对路径 → datasets/ 目录
     └─ 数据不存在时按 download 字段自动下载解压
 └─ build_yolo_dataset(cfg=self.args, img_path, batch, data, mode, rect, stride)（build.py L230-275）
 └─ YOLODataset(...)
     ├─ get_img_files：glob 图片（按 IMG_FORMATS 过滤），可按 fraction 抽样
     ├─ get_labels：img2label_paths 把 .../images/train/x.jpg 映射为 .../labels/train/x.txt
     │    └─ verify_image_label 并行校验（5 列、归一化 ≤1.01、类别号 < nc、损坏图修复）
     │    └─ 结果写入 labels.cache（下次秒开）
     └─ build_transforms(hyp)：装配训练/验证管线
```

### 9.3 训练增强管线（`v8_transforms`，augment.py L2692-2763）

训练模式下的完整顺序：

```
Mosaic(p=hyp.mosaic, 默认1.0)
 → [CopyPaste(p=hyp.copy_paste)，仅分割任务]
 → RandomPerspective(degrees/translate=0.1/scale=0.5/shear/perspective, size=(imgsz,imgsz))
 → MixUp(p=hyp.mixup, 默认0.0)
 → CutMix(p=hyp.cutmix, 默认0.0)
 → Albumentations（可选；默认仅 Blur/ToGray/CLAHE 等以 0.01 概率生效）
 → RandomHSV(h/s/v)
 → RandomFlip(vertical, p=hyp.flipud=0)
 → RandomFlip(horizontal, p=hyp.fliplr=0.5)
 → Format（BGR→RGB、HWC→CHW、标签归一化、生成 batch_idx 占位）
```

关键实现细节：

- **Mosaic**（L422-759）：4 图拼在 `2*imgsz` 画布上（114 灰边填充），随机中心；标签随之平移拼接并裁剪。9 图模式可选。
- **RandomPerspective**（L1036-1397）：仿射矩阵按 `M = T @ S @ R @ P @ C`（平移→剪切→旋转缩放→透视→中心化）组合，把 `2*imgsz` 马赛克缩回 `imgsz×imgsz`；框的 4 角同样变换后重取外接框，并用 `wh_thr=2px, ar_thr=100, area_thr=0.1` 过滤过度扭曲的框。
- **MixUp**：混合系数 `r ~ Beta(32,32)`（集中在 0.5），图像线性混合，标签直接拼接（两图框都保留）。
- **验证管线**（`augment=False`）：仅 `LetterBox(scaleup=False)` + `Format`——只缩小不放大，保证 mAP 公平。
- **Format**（L2187-2416）：不做 /255（归一化在训练器 `preprocess_batch` 完成）。

### 9.4 `collate_fn` 与 batch 结构（dataset.py L284-312）

图像 `torch.stack` 成 `(B,3,H,W)`；所有目标沿 0 维 `cat`；第 `i` 张图的 `batch_idx` 占位张量整体加 `i`。最终：

```
batch["img"]      : (B, 3, 640, 640) uint8
batch["cls"]      : (N, 1)
batch["bboxes"]   : (N, 4)   归一化 xywh
batch["batch_idx"]: (N,)     每个目标属于批中第几张图
```

损失函数里 `targets = cat((batch_idx, cls, bboxes), 1)` 即经典 6 列 targets。

### 9.5 增强超参传递路径

`default.yaml` → `get_cfg` 合并命令行覆盖 → `trainer.args` → `build_yolo_dataset(hyp=args)` → `YOLODataset.build_transforms(hyp)` → `v8_transforms` 按属性读取（`hyp.mosaic`、`hyp.scale`…）。**因此命令行 `yolo train mosaic=0.5 scale=0.9` 就能直接改增强，无需动代码**；新增增强超参则需在 `default.yaml` + `cfg/__init__.py` 的类型键集合中登记。

### 9.6 DataLoader 细节（build.py）

- `InfiniteDataLoader`（L43-118）：用 `_RepeatSampler` 无限重复产出索引，**worker 进程跨 epoch 复用**，避免每轮重启开销。
- 实际进程数 `nw = min(cpu_count // gpu_count, workers)`；`prefetch_factor=4`；`pin_memory` 仅 GPU 可用时开。
- 验证集 `rect=True` 时按宽高比排序并计算每批最小矩形形状（`batch_shapes`），LetterBox 按批填充，减少无效计算。

---

## 10. 训练流程详解

### 10.1 调用链

```
YOLO("yolo12n.yaml").train(data=..., epochs=100)
 └─ Model.train（engine/model.py L715-803）
     └─ trainer = DetectionTrainer(overrides=args, _callbacks=...)   # 按 task_map 查表
 └─ BaseTrainer.train（engine/trainer.py L219-246）
     └─ 单卡: _do_train()；多卡: 生成 DDP 命令起子进程
```

`BaseTrainer.__init__`（L117-204）完成：`get_cfg` 合并参数、`select_device`、建 `save_dir`（`runs/detect/<name>/`）与 `args.yaml`、`get_dataset()`（`check_det_dataset`）、加载回调。

### 10.2 `_setup_train`（L294-374）

| 步骤 | 内容 |
|---|---|
| 模型 | `setup_model` → `DetectionTrainer.get_model`：`DetectionModel(cfg, nc=data_nc, ch=channels)`，有预训练权重则 `model.load(weights)`（按形状交集迁移） |
| 冻结 | `freeze=10` 冻结前 10 层或按索引列表；**`.dfl` 层永久冻结**（L311） |
| AMP | RANK∈{-1,0} 上 `check_amp` 实测后决定；`scaler = torch.amp.GradScaler(enabled=amp)` |
| 步长 | `gs = max(model.stride.max(), 32)`；`check_imgsz` 对齐 |
| 数据 | `_build_train_pipeline`：训练/验证 DataLoader（验证 batch 翻倍）；`accumulate = round(nbs/batch)`（nbs=64）；`weight_decay *= batch*accumulate/nbs` |
| 优化器 | `build_optimizer`（L994-1078）：参数分 4 组（带衰减权重/BN/偏置/矩阵参数）；`auto` 时：迭代数 > 10000 选 **MuSGD**(lr=0.01)，否则 **AdamW**（`lr_fit = 0.002*5/(4+nc)`） |
| 调度器 | 默认**线性衰减** `1→lrf`；`cos_lr=True` 用余弦（`one_cycle`） |
| EMA | `ModelEMA`（torch_utils.py L647-708）：`decay = 0.9999*(1-exp(-updates/2000))`，每次 optimizer_step 更新；**验证与保存的都是 EMA 权重** |
| 早停 | `EarlyStopping(patience=100)` |

### 10.3 训练主循环 `_do_train`（L376-589）

```python
nw = round(warmup_epochs * nb)          # warmup 迭代数（约 3 个 epoch）
for epoch:
    scheduler.step()                     # epoch 级学习率
    if epoch == epochs - close_mosaic:   # ★ 最后 10 轮关马赛克
        dataset.mosaic = 0; train_loader.reset()
    for i, batch in pbar:
        # warmup：前 nw 个迭代线性插值
        #   accumulate: 1 → nbs/batch
        #   bias lr: warmup_bias_lr(0.1) → lr0；其余组 0 → lr0*lf(epoch)
        #   momentum: warmup_momentum(0.8) → 0.937
        with autocast(self.amp):
            batch = self.preprocess_batch(batch)      # 搬设备 + /255 + 多尺度 resize
            loss, loss_items = self.model(batch)      # 训练态前向直接返回损失
            self.loss = loss.sum()
            self.tloss = 滑动平均(loss_items)          # 用于日志
        self.scaler.scale(self.loss).backward()
        if ni - last_opt_step >= self.accumulate:      # 梯度累积到位
            self.optimizer_step()                      # unscale → clip_grad_norm(max_norm=10)
                                                       # → scaler.step → zero_grad → ema.update
    # epoch 末：ema.update_attr → validate()（需要时）→ EarlyStopping → save_model()
final_eval()   # strip_optimizer 后用 best.pt 复验
```

要点：

- `preprocess_batch`（detect/train.py L107-137）：张量搬设备、`img.float()/255`；`multi_scale>0` 时随机尺寸并对齐到 stride 倍数。
- OOM 自动恢复：显存不足时自动降 batch 重建管线（首 epoch、单卡、最多 3 次）。
- `set_class_weights`（detect/train.py L151-168）：`cls_pw>0` 时按类频逆加权分类损失。

### 10.4 close_mosaic 机制

默认最后 10 个 epoch 关闭马赛克（`close_mosaic: 10`）：`dataset.close_mosaic(hyp)` 把 `mosaic/copy_paste/mixup/cutmix` 全部置 0 并重建变换管线（退化为普通增强）。断点续训时若已超过该边界会立即关闭。

### 10.5 checkpoint 与恢复

`save_model`（L647-693）：保存 **EMA 权重的半精度副本**（`ckpt["ema"]`）+ optimizer/scaler/args/训练指标；`"model": None`（一切从 EMA 派生）。`best_fitness == fitness` 时写 `best.pt`；`save_period>0` 时写 `epoch{N}.pt`。`final_eval` 会对 last/best 执行 `strip_optimizer` 瘦身并用 best.pt 复验。`resume=True` 恢复 optimizer/scaler/EMA/best_fitness；损失出现 NaN 时有自动回滚机制（最多 3 次）。

### 10.6 结果目录产物

`runs/detect/<name>/` 下：`weights/{last,best,epochN}.pt`、`results.csv`、`results.png`（损失与指标曲线）、`confusion_matrix.png`、`labels.jpg`（标签分布）、`train_batch*.jpg`（增强后样本可视化）、`val_batch*.jpg`、`args.yaml`（完整超参快照，复现依据）。

---

## 11. 验证流程详解

### 11.1 调用与执行

`model.val()` → `DetectionValidator`（models/yolo/detect/val.py）。训练中验证使用 **EMA 模型**，且 AMP 训练时强制半精度验证；独立验证默认 `conf=0.001`（几乎不筛，保证 PR 曲线完整）、`iou=0.7`。

每批四段计时：`preprocess`（/255）→ `inference` → `loss`（累积验证损失）→ `postprocess`（**NMS，`multi_label=True`**——验证需要一个框对多类都算，与推理不同）。

### 11.2 mAP 计算（utils/metrics.py）

- `iouv = torch.linspace(0.5, 0.95, 10)`：10 个 IoU 阈值。
- `match_predictions`：对每个阈值做贪心匹配（IoU 降序、每个检测框与每个 GT 只匹配一次），得到 `(N, 10)` 的 TP 矩阵。
- `ap_per_class`（L995-1088）：全部检测按置信度排序 → 逐类累积 TP/FP → 在 1000 个置信度采样点上插值出 P-R 曲线 → **对 10 个 IoU 阈值分别用 COCO 官方 101 点插值算 AP** → 得到 `ap (nc, 10)`。
- 指标：`mAP50 = ap[:,0].mean()`、`mAP50-95 = ap.mean()`；**fitness = mAP50-95**（权重 [0,0,0,1]），决定 best.pt 与早停。
- `save_json=True` 时输出 COCO 格式 JSON，可用 pycocotools 复核。

---

## 12. 推理流程详解

### 12.1 调用链

```
model.predict("bus.jpg", conf=0.25, iou=0.7, save=True)
 └─ Model.predict（engine/model.py L479-537）：合并参数（方法默认含 rect:True, conf:0.25, batch:1）
     └─ DetectionPredictor(overrides=args)；predictor.setup_model(model)
 └─ BasePredictor.__call__（engine/predictor.py L210-228）
     └─ stream_inference（生成器；stream=False 时物化为 list）
```

`setup_model`（predictor.py L390-417）：`AutoBackend(model, device, fp16=half, fuse=True)`——`AutoBackend`（nn/autobackend.py）按文件后缀路由到 21 种后端（`.pt`→`PyTorchBackend`，另有 onnx/engine/openvino/tflite/…）；pt 加载即第 6 章的 `load_checkpoint`（EMA 权重 + Conv-BN 融合）+ `eval()`。

### 12.2 推理主循环（stream_inference，L279-388）

```python
setup_source(source)                    # load_inference_source 按类型选加载器
warmup(...)                             # 一次空输入前向（含 NMS 路径预热），只做一次
for batch in dataset:                   # (paths, imgs_bgr_list, info)
    with profiler0: im = self.preprocess(im0s)      # LetterBox + BGR→RGB + BCHW + /255
    with profiler1: preds = self.inference(im)      # AutoBackend → DetectionModel
    with profiler2: results = self.postprocess(preds, im, im0s)   # NMS
    for i in range(n):
        results[i].speed = {...}        # 三段毫秒数
        write_results(i, ...)           # plot / save / save_txt / show
    yield from results
```

**预处理细节**（L154-204）：`pre_transform` 逐图做 `LetterBox`——单图 + `rect=True`（predict 方法级默认）+ pt 格式时 `auto=True`，只补齐到 stride 的整数倍（最小矩形）；否则填充到完整方形。**预测时默认不走训练那种固定 640² 填充**。

### 12.3 后处理：NMS（`utils/nms.py`，本仓库已从 ops.py 独立）

`DetectionPredictor.postprocess`（detect/predict.py L33-80）调用：

```python
preds = nms.non_max_suppression(preds, conf, iou, classes, agnostic_nms,
                                max_det=300, nc=0, end2end=...)
```

`non_max_suppression`（L13-165）完整逻辑：

1. **输入**：pt 模型返回 `(推理结果, 损失输出)` 二元组，取 `[0]`，形状 `(B, 4+nc, 8400)`。
2. **向量化粗筛**：`xc = prediction[:, 4:4+nc].amax(1) > conf_thres`，一次过滤全批。
3. **转置 + 格式转换**：`transpose(-1,-2)` → `(B, 8400, 4+nc)`；`xywh2xyxy`（中心点+宽高 → 角点）。
4. **逐图循环**（每图输出数量不定，无法批并行）：
   - 应用粗筛掩码；拆为 `box / cls / mask` 三段；
   - `multi_label=False`（predict 默认）：每框只取最高类 `cls.max(1)`；`True`（val 用）：一个框可保留多个超阈值类别；
   - 按 `classes` 过滤 → 超 30000 个候选时按置信度截断；
   - **类别偏移技巧**：`boxes += cls_id * max_wh(7680)`，把不同类的框推到不相交的空间区域，等效**逐类 NMS**；`agnostic_nms=True` 则不偏移、所有类一起抑制；
   - 执行 NMS：已导入 `torchvision` 时用 `torchvision.ops.nms`，否则纯 torch 实现 `TorchNMS.nms`；
   - `i[:max_det]`（默认 300）截断。
5. **输出**：每图 `(n, 6)` = `(x1, y1, x2, y2, conf, cls)`（letterbox 坐标系）。

### 12.4 坐标还原与结果封装

`construct_result`（detect/predict.py L109-122）：

```python
pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
return Results(orig_img, path, names, boxes=pred[:, :6])
```

`scale_boxes`（ops.py L102-141）与 LetterBox 严格互逆：反推 `gain = min(lb_h/orig_h, lb_w/orig_w)` 与居中 `pad`，执行 `(xy − pad) / gain`，最后 `clip_boxes` 钳制到原图边界。

`Results`（engine/results.py）：

| 属性/方法 | 说明 |
|---|---|
| `r.boxes.xyxy` | 原图像素坐标 `(N,4)` |
| `r.boxes.xywh` | 中心点+宽高 |
| `r.boxes.xyxyn` / `xywhn` | 按原图尺寸归一化（对齐标签格式，可直接写标注） |
| `r.boxes.conf` / `cls` / `id` | 置信度 / 类别索引 / 跟踪 ID |
| `r.plot()` | 返回画好框的 ndarray |
| `r.save_txt(path)` | 保存 `cls x_c y_c w h [conf]`（归一化 xywh） |
| `r.speed` | `{preprocess, inference, postprocess}` 毫秒 |
| `r.cpu()/numpy()/cuda()/to()` | 设备迁移 |

### 12.5 常用推理姿势

```python
results = model.predict(source, conf=0.25, iou=0.7, imgsz=640,
                        device=0, half=False,        # 半精度
                        classes=[0, 2],              # 只看 person、car
                        agnostic_nms=False, max_det=300,
                        save=True, save_txt=False, save_crop=False,
                        stream=False,                # 视频/摄像头建议 True
                        augment=False)               # TTA 测试时增强
```

输入支持：图片、目录、`.txt` 图片列表、视频、摄像头序号 `'0'`、RTSP/HTTP URL、PIL/ndarray、URL。

---

## 13. 面向 YOLO12 的算法改进逐步指南

本章是全指南的重点，按"从易到难"给出可操作的改进路径。所有改动遵循仓库根目录 `AGENTS.md` 的规约：**自定义模块必须放在 `ultralytics/nn/AddModules/`，不得放入 `ultralytics/nn/modules/`（该目录仅存框架原生模块）；未经明确允许不得执行 `git commit`/`git push`。**

### 13.1 添加一个全新的网络模块（标准五步法）

以添加一个名为 `MyAttention` 的即插即用注意力为例。

**第 1 步：创建模块文件** `ultralytics/nn/AddModules/MyAttention.py`：

```python
import torch
import torch.nn as nn

class MyAttention(nn.Module):
    """即插即用注意力：输入输出形状不变。"""
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // ratio), nn.ReLU(),
            nn.Linear(channels // ratio, channels), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x).view(x.shape[0], -1, 1, 1)
        return x * w
```

约定：模块签名尽量写成 `__init__(self, c1/c2/in_channels, ...)` 风格，便于 `parse_model` 注入通道数；若与 `Conv` 同签名（`c1, c2, k, s, ...`）可直接套用通道缩放分支。

**第 2 步：启用导出**。编辑 `ultralytics/nn/AddModules/__init__.py`，添加：

```python
from .MyAttention import *
```

注意该文件的"按需启用"约定：多个模块文件内部会重复定义 `Conv/Bottleneck/C3` 等类，星号导入会互相覆盖，**同一时间只启用正在改的那一组**。

**第 3 步：让 `parse_model` 认识它**。`tasks.py` 已有 `from .AddModules import *`（约 L102），yaml 里的类名通过 `globals()[m]` 自动可见。剩下按模块的"通道行为"选一种注册方式（`parse_model` 内，约 L1677-1877）：

| 模块类型 | 注册方式 | 现有参考 |
|---|---|---|
| 透传型（形状/通道不变，如注意力、上采样替换） | 不注册也能跑（走 `else: c2 = ch[f]`），但显式加 `elif m is MyAttention: c2 = ch[f]; args = [c2, *args]` 更稳妥，可注入通道数 | `DySample`（L1860-1863）、`FreqFusionUpsample`（L1854-1857） |
| 常规结构块（需要 width 缩放输出通道） | 把类名加入 `base_modules` frozenset（L1677-1731；L1713-1729 有预留注释位） | `A2C2f`、`C2f` |
| 带 repeats 的堆叠块 | 同时加入 `repeat_modules`（L1732-1751，L1749 有预留位） | `C3k2` |
| 多输入型（from 为列表，如融合模块） | 自定义 `elif`，从 `[ch[x] for x in f]` 计算通道 | `FreqFusion`（L1849-1851）、`BiFPN`（L1872-1874） |
| 下采样/上采样（改变尺寸不改通道） | 透传分支 + 在 yaml 中注意后续层的 `from` | `HPDown`、`SPDConv`（AddModules 中） |

**第 4 步：在 yaml 中使用**。复制 `cfg/models/12/yolo12.yaml` 为新文件（如 `yolo12-myattn.yaml`，建议放在 `cfg/models/12/` 或自己的实验目录），在目标位置插入：

```yaml
backbone:
  ...
  - [-1, 4, A2C2f, [1024, True, 1]]    # 8
  - [-1, 1, MyAttention, []]           # 9 ← 新增：P5 特征后接注意力（注意后续 from 索引全部 +1）
```

**注意**：插入层后，后面所有 `from` 索引都要同步偏移（`Concat` 引用的 `[6]`、`[4]`、`[8]` 等）。

**第 5 步：验证与训练**：

```python
from ultralytics import YOLO
model = YOLO("yolo12-myattn.yaml")     # 会打印逐层表 + summary（层数/参数量/GFLOPs）
model.info()
import torch
model.model(torch.zeros(1, 3, 640, 640, device="cpu"))  # 冒烟前向
model.train(data="coco8.yaml", epochs=3, fraction=1.0)   # 小数据冒烟训练
```

### 13.2 常见改进方向与落点速查

| 改进方向 | 操作文件 | 说明 |
|---|---|---|
| 换主干下采样（保留更多小目标信息） | 新模块 + yaml | 用 `HPDown`、`SPDConv`（AddModules 已有）替换 `Conv [*,3,2]`；注意 stride=2 的语义要保留 |
| 替换/增强主干注意力 | `AddModules/` | 参考 `EMA.py`、`MCA.py`、`SCSA.py` 的 `A2C2f_X` 封装范式：保持 A2C2f 的 C2f 骨架，替换内部注意力；然后改 yaml 层6/层8 的模块名 |
| 颈部上采样升级 | yaml 直接改 | `nn.Upsample` → `DySample`（已注册，写法 `[-1, 1, DySample, [2]]`）或 `FreqFusionUpsample`（即插即用，自动用 nearest 合成高频流） |
| 颈部特征融合加权 | yaml + parse_model（已注册） | `BiFPN`（多输入加权）、`FreqFusion`（高低频双流融合）、`HSFPN`（`ChannelAttention_HSFPN`+`Multiply`+`Add`，parse 分支当前被注释，需先在 tasks.py L1839-1847 恢复） |
| 增加小目标检测层 P2 | yaml | 参考 `cfg/models/v8/yolov8-p2.yaml`：从主干层2（160×160）引一路进 head，`Detect` 的 from 变 4 个，stride 自动算出 `[4,8,16,32]`；注意显存与计算量显著增加 |
| 增加大目标检测层 P6 | yaml | 参考 `cfg/models/v8/yolov8-p6.yaml`、`cfg/models/26/yolo26-p6.yaml` |
| 改检测头 | `nn/modules/head.py` + yaml | 改 `Detect.cv2/cv3` 结构（如解耦更深、加共享卷积分支）；`reg_max` 可在 yaml 顶层加 `reg_max: N`；`reg_max: 1` 即关闭 DFL（损失自动退化为 L1） |
| 改损失函数 | `utils/loss.py`、`utils/metrics.py` | 新 IoU：在 `metrics.py` 的 `bbox_iou` 加分支 → `loss.py` `BboxLoss.forward` 与 `TaskAlignedAssigner.iou_calculation` 两处接线 → `default.yaml` 加开关 → `cfg/__init__.py` 的 `CFG_BOOL_KEYS`/`CFG_FLOAT_KEYS` 登记。仓库现有的 `wiou`/`sd_loss`/`powerful_iou`/`focaler_ciou`/`slide_loss` 就是完整范例 |
| 改标签分配策略 | `utils/tal.py` | `TaskAlignedAssigner` 的 `topk`、`alpha`、`beta` 由 `v8DetectionLoss` 构造时传入（L468-477），可加超参透传；`select_candidates_in_gts` 的小目标扩框逻辑（L320-327）也可改造 |
| 改数据增强 | `data/augment.py` + `default.yaml` | 新增强类继承 `BaseTransform`（实现 `get_params/apply_image/apply_instances`），在 `v8_transforms`（L2692-2763）中插入；新概率超参登记到 default.yaml 与类型键集合 |
| 改优化器/学习率策略 | `engine/trainer.py` | `build_optimizer`（L994-1078）、`_setup_scheduler`（L248-254） |
| 冻结/迁移学习 | 仅命令行 | `freeze=10`（前 10 层）或 `freeze=[0,1,2,3,4,5,6,7,8]`；`pretrained=yolo12n.pt` 或 `False` |

### 13.3 完整示例：YOLO12 + DySample 颈部（可直接运行）

`DySample` 已在本仓库注册并启用（`AddModules/__init__.py` L17；`parse_model` L1860-1863）。新建 `yolo12-dysample.yaml`，只需把 head 的两处 `nn.Upsample` 换掉：

```yaml
head:
  - [-1, 1, DySample, [2]]                          # 9  原: [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]                       # 10 cat backbone P4
  - [-1, 2, A2C2f, [512, False, -1]]                # 11

  - [-1, 1, DySample, [2]]                          # 12
  - [[-1, 4], 1, Concat, [1]]                       # 13 cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]]                # 14

  - [-1, 1, Conv, [256, 3, 2]]                      # 15
  - [[-1, 11], 1, Concat, [1]]                      # 16
  - [-1, 2, A2C2f, [512, False, -1]]                # 17

  - [-1, 1, Conv, [512, 3, 2]]                      # 18
  - [[-1, 8], 1, Concat, [1]]                       # 19
  - [-1, 2, C3k2, [1024, True]]                     # 20

  - [[14, 17, 20], 1, Detect, [nc]]                 # 21
```

（backbone 与 scales 照抄原文件。）然后：

```bash
yolo train model=yolo12-dysample.yaml data=coco8.yaml epochs=100 imgsz=640 name=yolo12-dysample
```

### 13.4 自定义注意力的"封装范式"示例

参照 `AddModules/EMA.py` 的写法（保持 A2C2f 骨架、替换内部注意力）：

```python
# AddModules/MyAttn.py
from ..modules import Conv, C3k
from ..modules.block import A2C2f, ABlock  # 参考用

class MyAttnBlock(nn.Module):
    """用自研注意力替换 ABlock 中的 AAttn。"""
    def __init__(self, dim, num_heads, area=1, mlp_ratio=2.0):
        ...
    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)

class A2C2f_MyAttn(A2C2f):
    def __init__(self, c1, c2, n=1, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2=False, residual=residual, e=e, g=g, shortcut=shortcut)
        c_ = int(c2 * e)
        assert c_ % 32 == 0
        self.m = nn.ModuleList(
            nn.Sequential(*(MyAttnBlock(c_, c_ // 32, area, mlp_ratio) for _ in range(2)))
            for _ in range(n))
```

注册：加入 `base_modules` 与 `repeat_modules`（因为要享受 width/depth 缩放与 repeats 注入）；`parse_model` 可仿照 `A2C2f` 特判（L1781-1784）加一段 `legacy=False`。yaml 中把层6/层8 的 `A2C2f` 换成 `A2C2f_MyAttn` 即可。

### 13.5 实验管理与消融建议

- 每个改进用独立 `name=`（如 `yolo12-dysample-v1`），`runs/detect/<name>/args.yaml` 是完整实验记录。
- 快速验证：`fraction=0.1`（10% 数据）、`epochs=10`、`coco8.yaml`；确认能收敛再上全量。
- 公平对比：固定 `seed=0 deterministic=True imgsz=640 batch=16 optimizer=auto`，同一数据集同一轮数；关注 `best.pt` 的 mAP50-95（fitness）与参数量（`model.info()`）。
- 从预训练权重热启动改结构模型：`model = YOLO("yolo12-dysample.yaml"); model.load("yolo12n.pt")`（只迁移形状匹配的参数，新增层随机初始化）。
- 改完代码后先 `git diff` 自查，**按仓库规约不要自行提交**；测试文件请用 `from ultralytics.nn.AddModules import Xxx` 导入。

### 13.6 调试技巧

- 建模时会自动打印逐层表（`from / n / params / module / arguments`），通道对不上会直接在这一步报错。
- `model.model` 是 `nn.Sequential`，可按索引取出任意层做单测；`model.model[6]` 即主干层6。
- 前向传 `visualize=True`（predict）可保存特征图可视化（`feature_visualization`，torch_utils）。
- 推理时想要中间层特征：`model.predict(..., embed=[6, 8, 14])` 返回指定层嵌入。
- 显存评估：`model.info()` 输出层数/参数/梯度数/GFLOPs；训练 OOM 时框架会自动降 batch 重试（仅首 epoch）。

---

## 14. 常用命令速查

```bash
# 训练
yolo train model=yolo12n.yaml data=coco8.yaml epochs=100 imgsz=640 batch=16 device=0
yolo train model=yolo12n.pt data=my.yaml epochs=300 patience=50 optimizer=AdamW lr0=0.001
yolo train resume=True project=runs/detect name=exp1      # 断点续训
yolo train model=yolo12n.yaml data=my.yaml freeze=10      # 冻结主干前 10 层
yolo train ... close_mosaic=0 mosaic=0.5 mixup=0.1        # 调整增强
yolo train ... cos_lr=True warmup_epochs=5

# 验证
yolo val model=runs/detect/exp1/weights/best.pt data=coco8.yaml split=test
yolo val model=yolo12n.pt data=coco.yaml save_json=True   # COCO 官方格式评测

# 推理
yolo predict model=yolo12n.pt source=path/to/imgs conf=0.25 save=True
yolo predict model=yolo12n.pt source=0                    # 摄像头
yolo predict model=yolo12n.pt source=rtsp://... stream=True

# 导出
yolo export model=yolo12n.pt format=onnx imgsz=640 dynamic=True simplify=True
yolo export model=yolo12n.pt format=engine half=True      # TensorRT FP16

# 配置
yolo copy-cfg                                             # 复制 default.yaml 到当前目录
```

Python API 等价写法：`YOLO(...).train/val/predict/export(...)`，所有 `key=value` 与命令行一一对应。

---

## 15. 本仓库的本地定制清单

本仓库是 Ultralytics 主线的**深度定制分支**（含 YOLO26/MuSGD/semantic 任务等新特性），做研究前需了解以下本地差异：

### 15.1 `nn/AddModules/` 自定义模块（16 个文件）

**已启用**（`AddModules/__init__.py`）：`DySample`（动态上采样）、`BiFPN`（加权双向融合）、`FreqFusion`（频率融合 + `FreqFusionUpsample` 即插即用版）。

**被注释（按需启用）**：

| 文件 | 核心类 | 用途 |
|---|---|---|
| `SimAM.py` | `SimAM`、`A2C2f_SimAM` | 无参数注意力 |
| `EMA.py` | `EMA`、`A2C2f_EMA` | 高效多维注意力（注释称替换主干 x2 有轻量化收益） |
| `MCA.py` | `MCA`、`A2C2f_MCA` | 标准差池化门控跨通道/空间注意力 |
| `SCSA.py` | `SCSA`、`A2C2f_SCSA` | 空间-通道选择性注意力 |
| `MoCAttention.py` | `MoCAttention`、`A2C2f_MoCA` | Mixture-of-Convolutions 注意力 |
| `Mona.py` | `Mona`、`A2C2f_Mona` | 多尺度邻域注意力 |
| `AssemFormer.py` | `AssemFormer`、`A2C2f_AssemFormer` | 线性自注意力（MobileViTv2 风格） |
| `FBRT_YOLO.py` | `FCM`、`A2C2f_FCM` | 频域通道调制 |
| `ESMoE.py` | `ESMoE` | 极简专家混合（多种专家+路由器） |
| `HPDown.py` | `HPDown` | 高性能下采样（替换 stride-2 Conv） |
| `SPDConv.py` | `SPDConv` | 空间到深度下采样 |
| `HSFPN.py` | `ChannelAttention_HSFPN`、`Multiply`、`Add` | HSFPN 颈部（parse 分支当前被注释） |
| `MultiScaleGateAttn.py` | `MultiScaleGatedAttn` | 多尺度门控注意力（多输入） |

`parse_model` 中对应的注册位（约 L1713-1729、L1749、L1839-1847）大多也以注释形式预留，启用模块时需同步取消注释。

### 15.2 其它本地差异（影响研究复现）

- 自定义损失开关：`wiou / focaler_ciou / sd_loss / powerful_iou / slide_loss`（default.yaml L130-142）。
- `utils/callbacks/base.py`（旧版叫 `default.py`）；进度条是自研零依赖 `TQDM`（`utils/tqdm.py`）。
- `non_max_suppression` 独立为 `utils/nms.py`；推理后端拆分到 `nn/backends/*.py`（`AutoBackend` 只做路由）。
- `TaskAlignedAssigner.select_candidates_in_gts` 含小目标扩框逻辑（tal.py L320-327），非官方行为。
- `optimizer=auto` 在长训练（>10000 迭代）时选 **MuSGD**（Muon+SGD 混合）。
- checkpoint 中 `"model": None`，推理权重一律从 EMA 派生。
- 默认任务映射 `TASK2MODEL` 指向 yolo26 系列；用 YOLO12 必须显式指定模型。

---

## 16. 常见问题与注意事项

**Q1：yaml 里写了自定义模块名，报 `KeyError` 或"模块不存在"？**
检查三点：① `AddModules/__init__.py` 是否启用了该文件的导入；② 模块文件里的类名与 yaml 完全一致；③ 是否被其它已启用文件的同名类覆盖（星号导入冲突，遵循"一次只启用一组"约定）。

**Q2：`A2C2f` 报 "Dimension of ABlock must be a multiple of 32"？**
隐层通道 `c_ = int(c2 * 0.5)` 必须是 32 的倍数（`num_heads = c_//32`）。调整 yaml 通道数或 width 缩放时注意；小模型（n）的最小可用 c2 是 64。

**Q3：改了结构，还能用官方预训练权重吗？**
能。`model.load("yolo12n.pt")` 走 `intersect_dicts`，只迁移形状匹配的参数并打印迁移摘要；新增/修改的层保持随机初始化。首卷积通道数不匹配（如多光谱输入）也有专门兼容。

**Q4：训练用的权重和保存的权重是同一份吗？**
不是。训练中验证与 `best.pt` 保存的都是 **EMA 权重**（半精度），反向传播更新的是在线权重。这是 mAP 的常规做法，复现时不要混淆。

**Q5：多卡训练怎么用？**
`device=0,1,2,3`（或 `yolo train ... device=0,1`），框架自动生成 DDP 子进程；`workers` 是每卡进程数。注意 DDP 下 `batch` 是总批大小。

**Q6：如何完全关闭某个数据增强？**
命令行覆盖即可：`mosaic=0 mixup=0 fliplr=0 hsv_h=0 ...`；`close_mosaic=0` 可让马赛克全程开启（配合更大 `epochs`）。

**Q7：推理结果坐标和标签格式怎么对应？**
`results.boxes.xywhn` 直接就是标签格式（归一化中心点+宽高），`results.boxes.cls` 是类别索引；`save_txt=True` 保存的文件可直接作为伪标签回灌训练。

**Q8：YOLO12 的 `end2end`（免 NMS）模式？**
纯 `yolo12.yaml` 训练出的模型 `end2end=False`，推理走 NMS。one2one 头与 `E2ELoss` 目前主要服务于 YOLO26 系模型；`Detect.fuse()` 与 `postprocess`（内置 top-k）的相关代码在 head.py 中可参考。

**Q9：在哪里看每个超参的实际生效值？**
`runs/detect/<name>/args.yaml`——训练启动时完整快照，复现与论文附录的可靠来源。

**Q10：行号对不上怎么办？**
本指南行号为 2026-08-25 快照。仓库仍在活跃改动（`scripts/improved_yolo12/` 等目录持续更新），建议以"类名/函数名 + 关键字"搜索定位；核心结构（`parse_model`、`A2C2f`、`v8DetectionLoss`、`TaskAlignedAssigner`）的函数签名在版本间通常稳定。

---

## 附录：YOLO12 研究快速索引

| 想做什么 | 去哪里 |
|---|---|
| 看/改网络结构 | `cfg/models/12/yolo12.yaml` |
| 看/改模块实现 | `nn/modules/block.py`（A2C2f/AAttn）、`conv.py`、`head.py` |
| 加自定义模块 | `nn/AddModules/` + `nn/tasks.py parse_model` |
| 看/改损失 | `utils/loss.py`、`utils/tal.py`、`utils/metrics.py` |
| 看/改训练逻辑 | `engine/trainer.py`、`models/yolo/detect/train.py` |
| 看/改增强 | `data/augment.py`、`data/dataset.py` |
| 看/改推理与 NMS | `engine/predictor.py`、`utils/nms.py`、`engine/results.py` |
| 调超参 | `cfg/default.yaml` + 命令行 `key=value` |
| 看实验结果 | `runs/detect/<name>/`（args.yaml、results.csv、weights/） |

> 完。祝改进顺利，mAP 节节高。

