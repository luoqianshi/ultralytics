# YOLO12s × FreqFusion 集成评估与向后兼容方案

> 评估对象：
> - [FreqFusion.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/FreqFusion.py)
> - [yolo12-FreqFusion.yaml](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_yolo12/yolo12-FreqFusion.yaml)
> - [tasks.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py)（parse_model 中 FreqFusion 分支，L1849）

---

## 一、评估结论摘要

**接口适配、维度链路、前向逻辑基本正确**（代码为官方 FreqFusion 仓库 seg/FPN 变体的忠实移植，且已正确适配 Ultralytics 的 list 输入约定），但存在 **2 个 P0 级阻断问题**：当前代码在本地环境**构建即崩溃**，且 yaml 命名会导致**静默按 n 缩放构建而非 s**。另有一处影响实验公平性的参数偏离（用户已确认对齐）。

### 1.1 验证为正确的部分

| 项 | 结论 |
|---|---|
| parse_model 分支（[tasks.py L1849-1851](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1849-L1851)） | 正确。`c2 = ch[f[0]]` 输出通道=HR 输入通道（FreqFusion 返回 `hr_feat + lr_feat`，二者等通道）；`args = [[ch[x] for x in f], *args]` 中 `ch[-1]` 正确解析为上一层（ch 列表按层追加，`ch[-1]` == 上一层输出） |
| forward 接口适配 | 正确。`BaseModel._predict_once` 对多输入层传 list `[y[f0], x]`，`forward(self, x)` 解包 `hr_feat, lr_feat = x` 与之匹配 |
| yaml 拓扑与维度 | 正确。layer 12: FreqFusion(P4[128ch,HR], P5[128ch,LR])→128ch@P4；layer 14: FreqFusion(P3[128,HR], P4融合[128,LR])→128ch@P3；[HR, LR] 顺序符合官方约定；carafe 上采样因子 2 与 P5→P4、P4→P3 匹配；Detect 输入 128/256/512ch 与官方 yolo12s 一致 |
| carafe 调用约定 | 一致。所有调用遵循 mask `[N, k²·group, up·H, up·W]` 约定（semi_conv 路径 mask 生成于 HR 分辨率） |
| backbone | 与官方 yolo12.yaml 完全一致（layer 0-8）→ yolo12s.pt backbone 权重可迁移 |
| 默认超参 | semi_conv=True / comp_feat_upsample=True / use_high_pass=True / hr_residual=True / feature_resample=False，为论文推荐默认配置，yaml 无需传参 |

### 1.2 问题清单（按严重度）

**P0-1【构建即崩溃】mmcv 依赖缺失**
- 本地环境实测：torch 2.2.2+cpu，**mmcv 未安装**。[FreqFusion.py L6-9](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/FreqFusion.py#L6-L9) 的 `except ImportError: pass` 导致 `xavier_init`、`carafe` 未定义 → `DetectionModel.__init__` 调用 `init_weights()`（L145 `xavier_init`）时直接 NameError。
- 更深层的冲突：即使 GPU 机器安装 mmcv-full，其 carafe 算子为 **CUDA-only**，而 Ultralytics 模型初始化时在 **CPU 上跑一次前向计算 stride**（`DetectionModel.__init__`），必然崩溃（社区已有大量同类报错记录）。
- 官方 FreqFusion 仓库已于 2024-10-24 加入自实现纯 PyTorch CARAFE 算子解决此问题，用户本地文件早于该更新。

**P0-2【静默构建错误缩放】yaml 文件名缺 scale 字母**
- `guess_model_scale`（[tasks.py L1910](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1910)）用正则 `yolo(e-)?[v]?\d+([nslmx])` 从文件名提取缩放档。`yolo12-FreqFusion.yaml` 无匹配 → scale="" → [parse_model L1663-1665](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1663-L1665) 静默取第一个 key **'n'**（仅一条 warning）→ 实际构建 **YOLO12n**（width 0.25，约 2.6M 参数）而非 YOLO12s（width 0.5，约 9.3M 参数）。实验结论将完全失效。
- 修复：重命名为 `yolo12s-FreqFusion.yaml`（与磁盘上已有的 `yolo12s-NWD.yaml` 命名惯例一致）。

**P1-1【混淆变量】head 中 A2C2f 参数偏离官方基线**（用户已确认对齐）
- 官方 [yolo12.yaml](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/cfg/models/12/yolo12.yaml) head 中 A2C2f 均为 `[c2, False, -1]`（a2=False 纯卷积路径）；当前 yaml 用默认参数（a2=True 区域注意力）。属"FreqFusion+区域注意力"双重变量，违背单变量归因原则（记忆中已有"双基线问题"教训）。

**P1-2【对齐基线】layer 21 C3k2 参数**
- 当前 `C3k2, [1024]`（c3k=False 默认）vs 官方 `C3k2, [1024, True]`。对齐。

**P1-3【预期管理】预训练迁移范围缩小**
- head 全重构（layer 9-21 全新）→ `model.load('yolo12s.pt')` 仅迁移 backbone（layer 0-8），迁移率显著低于 DySample 方案（其保留 head 主体）。需在训练日志确认 "Transferred X/Y items" 并记入实验记录，属已知代价而非缺陷。

**P2-1【潜伏 bug】dysampler 输入通道错误**
- [FreqFusion.py L120](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/FreqFusion.py#L120) `LocalSimGuidedSampler(in_channels=compressed_channels, ...)` 使用形参（默认 64），而实际压缩通道为 `self.compressed_channels = (hr+lr)//8 = 32` → 若开启 `feature_resample=True` 必然通道不匹配崩溃。默认 False 故当前不触发，顺手修复。

**P2-2【说明项】GFLOPs 低估**：`get_flops`/thop 不统计 unfold/interpolate/carafe 等函数式算子 → 报告 GFLOPs 时需注明口径。

**P2-3【已知限制】**：semi_conv 路径硬编码 2× 上采样（仅适用 P5→P4→P3，未来 P6 结构需扩展）；ONNX/TensorRT 导出本次不做（纯 torch 算子可导出性尚可但未验证）；纯 torch carafe 的 reflect padding 与 mmcv zero-padding 数值有微小差异（本仓库无 mmcv 历史权重，无兼容负担）。

---

## 二、修改方案（决策已完成，可直接执行）

### 2.1 [编辑] `ultralytics/nn/AddModules/FreqFusion.py`

**改动 1：删除 mmcv 依赖，替换为纯 PyTorch 实现**（移植自官方仓库 2024-10-24 自实现，去掉 debug print，加防护 assert）

将 L6-9 的 try/except 块删除，在现有 `normal_init` 定义之前插入：

```python
def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """纯 PyTorch 实现，替代 mmcv.ops.carafe.xavier_init"""
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def carafe(x, normed_mask, kernel_size, group=1, up=1):
    """纯 PyTorch CARAFE（移植自官方 FreqFusion 仓库自实现，替代 mmcv.ops.carafe 的 CUDA-only 算子）。

    Args:
        x: [N, C, H, W] 输入特征
        normed_mask: [N, kernel_size**2 * group, up*H, up*W] 已归一化重组核
    """
    assert group == 1, 'pure-PyTorch carafe 仅支持 group=1（FreqFusion 默认 up_group=1）'
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape
    assert m_c == kernel_size ** 2
    assert m_h == up * h
    assert m_w == up * w
    pad = kernel_size // 2
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
    unfold_x = unfold_x.reshape(b, c * kernel_size ** 2, h, w)
    unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
    unfold_x = unfold_x.reshape(b, c, kernel_size ** 2, m_h, m_w)
    normed_mask = normed_mask.reshape(b, 1, kernel_size ** 2, m_h, m_w)
    res = (unfold_x * normed_mask).sum(dim=2).reshape(b, c, m_h, m_w)
    return res
```

文件头部 import 区变为：`torch / torch.nn / torch.nn.functional / checkpoint / warnings / numpy`，无第三方依赖。现有本地 `normal_init`、`constant_init` 保留不动。

**改动 2：修复 dysampler 潜伏通道 bug**（L120）

```python
# 修改前
self.dysampler = LocalSimGuidedSampler(in_channels=compressed_channels, ...)
# 修改后
self.dysampler = LocalSimGuidedSampler(in_channels=self.compressed_channels, ...)
```

**改动 3（可选加固）：** `__init__` 中 `hr_channels, lr_channels = channels` 之后加一行 `assert hr_channels == lr_channels, 'FreqFusion 要求 HR/LR 特征通道数一致（forward 末尾逐元素相加）'`，将运行期形状错误提前为构建期清晰报错。

### 2.2 [重命名+编辑] `scripts/improved_yolo12/yolo12-FreqFusion.yaml` → `yolo12s-FreqFusion.yaml`

backbone 不动；head 修改三处 A2C2f 参数 + 一处 C3k2 参数，其余保持：

```yaml
head:
  - [4, 1, Conv, [256]] # 9-P3/8（1x1 横向压缩）
  - [6, 1, Conv, [256]] # 10-P4/16
  - [8, 1, Conv, [256]] # 11-P5/32

  - [[10, -1], 1, FreqFusion, []] # 12 FreqFusion(P4-HR, P5-LR)，替代 Upsample+Concat
  - [-1, 2, A2C2f, [256, False, -1]] # 13

  - [[9, -1], 1, FreqFusion, []] # 14 FreqFusion(P3-HR, P4-LR)
  - [-1, 2, A2C2f, [256, False, -1]] # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, A2C2f, [512, False, -1]] # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 11], 1, Concat, [1]] # cat head P5（横向输入为 1x1 压缩后的 P5）
  - [-1, 2, C3k2, [1024, True]] # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

同时更新文件头注释：删除误导性的 "YOLO12n backbone" 字样，注明"基线 yolo12s、head A2C2f 参数与官方对齐、FreqFusion 为唯一变量"。

### 2.3 [零改动] `ultralytics/nn/tasks.py`

L1849-1851 的 FreqFusion 分支经验证正确，**不修改**。`AddModules/__init__.py` 中 `from .FreqFusion import *` 已启用，**不修改**。

> 重要耦合约束（写入实验记录）：tasks.py 出现 `elif m in {FreqFusion}` 后，`AddModules/__init__.py` 的 FreqFusion 导入**必须保持启用**，否则 `parse_model` 执行到该行时 NameError 会炸掉所有模型构建（含 baseline）。未来若要停用该改进，需两处同步回退（注释 AddModules 导入 + 注释 tasks.py 分支）。

### 2.4 [新建] 训练/测试脚本（沿用仓库既有惯例）

**`scripts/improved_train/coco_pretrained/train_yolov12-FreqFusion_ssdc-uav.py`**
- 复制 [train_yolov12-DySample_ssdc-uav.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_train/coco_pretrained/train_yolov12-DySample_ssdc-uav.py) 模板
- 仅改：model 路径 → `scripts/improved_yolo12/yolo12s-FreqFusion.yaml`；name → `yolo12s_FreqFusion_ssdc_uav_exp01`
- 保持与基线严格对齐：`model.load('yolo12s.pt')`、SGD、epochs=150、imgsz=640、batch=16、device='0'、同数据 yaml、SaveLastNCheckpointsCallback(n=3)

**`scripts/improved_test/coco_pretrained/test_yolov12-FreqFusion_ssdc_uav.py`**
- 复制 [test_yolov12-DySample_ssdc_uav.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_test/coco_pretrained/test_yolov12-DySample_ssdc_uav.py) 模板
- 改：weights 路径、实验名、Model='YOLO12s-FreqFusion'
- 增：FPS/延迟指标输出——从 `metrics.speed`（dict: preprocess/inference/postprocess，ms/im）读取 inference 耗时并换算 FPS，加入打印与 CSV 列（'Inference (ms/im)', 'FPS (val, batch=16)'），注明测速口径

---

## 三、向后兼容方案

1. **配置文件设计**：官方 `ultralytics/cfg/models/12/yolo12.yaml` 及全部原生模块零改动（本方案已满足）；改进 yaml 一律放 `scripts/improved_yolo12/`（已满足）；**文件名强制携带 scale 字母**（`yolo12s-*.yaml`），杜绝 P0-2 类静默缩放错误；yaml 头注释标明基线与实验变量。
2. **条件导入机制**：沿用 `AddModules/__init__.py` 按需启用约定（FreqFusion 已启用）；FreqFusion.py 本身做到**零第三方依赖**（仅 torch/numpy），从根本上消除 mmcv 环境差异导致的导入失败；遵守 tasks.py 分支与 AddModules 导入的同步启用/停用约束（见 2.3）。
3. **版本控制策略**：改动收敛在 4 个文件（2 编辑 + 1 重命名 + 2 新建脚本，tasks.py/AddModules/\_\_init\_\_.py 零改动）；建议在独立分支（如 `feature/freqfusion`）提交，commit message 说明"移植官方纯 torch CARAFE + scale 修复 + head 参数对齐"；**遵守 AGENTS.md：AI 不自行 commit/push，完成修改后报告并等待用户确认**；runs/ 与 CSV 实验产物不入库。
4. **兼容性测试流程**：见下节阶段 0（含 baseline 回归测试——因 AddModules 聚合导入，FreqFusion.py 的任何导入错误都会波及所有模型，回归测试必须覆盖）。

---

## 四、验证指标与测试流程

### 阶段 0：本地 CPU 冒烟（改完立即执行，全部通过才算完成）

| # | 测试 | 通过标准 |
|---|---|---|
| T0.1 | 模块单元测试：`FreqFusion(channels=[128,128])`，输入 `(1,128,40,40)+(1,128,20,20)` 前向 | 输出 `(1,128,40,40)`；反传梯度全部 finite；half 精度前向不报错 |
| T0.2 | carafe 语义抽检：均匀 mask（全 1/k²）时 | 输出 ≈ 以 `floor(pos/up)` 为中心的 k×k 邻域均值，与理论一致 |
| T0.3 | 整机构建：`YOLO('scripts/improved_yolo12/yolo12s-FreqFusion.yaml')` | 日志**不出现** "no model scale passed"（确认 s 缩放生效）；`model.info()` 正常打印层数/参数量/GFLOPs；640×640 前向输出 P3/P4/P5 三尺度 |
| T0.4 | 回归测试：`YOLO('yolo12s.yaml')` 构建 + 前向 + `yolo12s.pt` 单张 CPU 推理 | 全部正常（确认 AddModules 改动对原生路径零回归） |
| T0.5 | 权重迁移：`model.load('yolo12s.pt')` | 打印 "Transferred X/Y items"；X 对应 backbone（预期迁移率低于 DySample 方案，属已知代价） |

### 阶段 1：GPU 训练冒烟

- SSDC-UAV 上 1 epoch（或小子集）：loss 正常下降、无 NaN、batch=16@640 显存可接受（AMP 默认开启）。
- 可选：单机多卡 10 iter DDP 冒烟（纯 torch 算子理论 DDP 安全）。

### 阶段 2：完整训练（与基线严格对齐，杜绝"双基线问题"）

- 150 epochs / SGD / batch 16 / imgsz 640 / 同数据配置 / 同增广，对照已有 yolo12s baseline 与 yolo12s-DySample 结果。

### 阶段 3：评测指标（test 脚本自动采集，沿用 CSV 汇总惯例）

| 指标 | 采集方式 | 判定标准 |
|---|---|---|
| mAP50 / mAP75 / mAP50-95 | `model.val(split='test')` | **主指标**：ΔmAP50-95(vs baseline) ≥ +0.5 判定有效（论文检测 FPN 上 +0.5~1.5 AP） |
| Precision / Recall / F1 | 同上 | 辅助诊断（小目标漏检/误检变化） |
| Params (M) | `sum(parameters)` | 预期 +0.1~0.3M（新增 3 个 1×1 压缩卷积 + 2 个 FreqFusion，减去 Concat 后通道减半的 A2C2f） |
| GFLOPs@640 | `get_flops` | 如实报告，**注明不含 unfold/carafe 等函数式算子（低估口径）** |
| Inference ms / FPS | `metrics.speed['inference']` 换算 | 相对 baseline 减速 ≤ 15% 判定可接受（纯 torch carafe 有 unfold 开销） |

若 ΔmAP<0：先检查收敛曲线与 T0.5 迁移日志，再考虑消融（use_high_pass=False 仅 ALPF、semi_conv=False）或回退，不直接判定模块无效。

---

## 五、假设与决策记录

- **D1**（用户已确认）：采用纯 PyTorch CARAFE，彻底移除 mmcv 依赖。理由：本地无 mmcv 且 CPU torch；mmcv carafe CUDA-only 与 Ultralytics CPU 初始化前向结构性冲突；官方仓库同方向演进；各机器数值完全一致，利于复现。
- **D2**（用户已确认）：head A2C2f 对齐官方 `[c2, False, -1]`，隔离 FreqFusion 单一变量。
- **D3**：tasks.py 与 AddModules/\_\_init\_\_.py 零改动（现有分支验证正确）。
- **D4**：仅支持 2× 上采样融合路径（当前 yaml 的 P5→P4→P3 恰好全部 2×）。
- **D5**：训练在 GPU 机器执行（device='0'），本地仅做 CPU 冒烟。
- **D6**：ONNX/TensorRT 导出不在本次范围（记录为已知限制）。

## 六、执行清单

1. 编辑 `ultralytics/nn/AddModules/FreqFusion.py`（改动 1/2/3）
2. 重命名 + 编辑 yaml → `scripts/improved_yolo12/yolo12s-FreqFusion.yaml`
3. 新建 `scripts/improved_train/coco_pretrained/train_yolov12-FreqFusion_ssdc-uav.py`
4. 新建 `scripts/improved_test/coco_pretrained/test_yolov12-FreqFusion_ssdc_uav.py`
5. 执行阶段 0 全部冒烟测试（本地 CPU）并汇报结果
6. 汇报改动摘要，等待用户确认（不自行 git commit）
