# FreqFusion 公平对比集成方案 + 非结构化改进方案清单

## Summary

两部分工作：
1. **Q1（需实施）**：新建「自输入式 FreqFusion-Upsample」配置——把 FreqFusion 以 `from=[-1,-1]`（同一输入喂 HR/LR 两路）放在**原 nn.Upsample 的位置**，保留 Concat 及官方 head 全部结构与层索引。改动量与 DySample 完全同级（仅换上采样算子），预训练迁移预期与 DySample 一致（全部可迁移项加载），从而实现 FreqFusion vs DySample 的同构公平对比。**零框架代码改动**，仅新建 3 个脚本文件。
2. **Q2（仅文档，不建脚本）**：不改网络结构、保证 COCO 预训练完整迁移的性能提升方案清单（按预期收益分级），内容即本文件 Part 2。

---

## Part 1（Q1）：自输入式 FreqFusion-Upsample

### 1.1 设计原理（为何可行且公平）

- **机制**：[tasks.py:1849-1851](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1849-L1851) 的 FreqFusion 分支已支持列表 from：`c2 = ch[f[0]]`，`args = [[ch[x] for x in f], *args]`。`f=[-1,-1]` 时得 `FreqFusion(channels=[512,512])`（s 尺度 P5 位置）/ `([256,256])`（P4 位置），满足模块内 `hr_channels == lr_channels` 断言；BaseModel 对列表 from 的处理 `x = [y[j] if j != -1 else x for j in m.f]` 会把当前特征喂两次 → `FF(x, x)`，输出通道=输入通道、空间 ×2（semi_conv 模式 carafe up=2），**精确复刻 nn.Upsample(scale=2) 的位置与张量形状**。
- **语义**：HR/LR 同源时，AHPF 对 x 自身做高频增强、ALPF 由自身引导生成抗混叠上采样核，输出 = HP(x) + LP_up(x)——即把"融合模块"忠实适配为"频率感知上采样器"，与 DySample（单输入动态点采样上采样器）信息量完全对齐（都只看上采样输入流，不偷看跳连），构成论文式同位算子对比。
- **迁移**：head 层索引 9–21 与官方 yaml 完全一致（DySample 运行已验证该索引对齐可获得满额迁移）；model.9/12 的 FreqFusion 参数在 ckpt 中无对应（原 Upsample 无参数），intersect_dicts 跳过，其余全部加载——预期 `Transferred` 日志与 DySample 运行一致（远大于重构版的 342/729）。
- **开销**：预估参数 ~+0.22M（两个 FF 实例的 compressor+encoder），GFLOPs ~+0.2，与 DySample（+~25K 参数）同量级，均远小于 9.25M 基线。具体以冒烟测试输出为准。

### 1.2 变更清单（3 个新文件，零框架代码改动）

**文件 1（新建）**：`scripts/improved_yolo12/yolo12s-FreqFusion_up.yaml`
- 内容 = 官方 [yolo12.yaml](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_yolo12/yolo12.yaml) 的 backbone 原样 + head 中第 9/12 层由 `nn.Upsample` 换成 `[[-1, -1], 1, FreqFusion, []]`，其余层（含 Concat 来源、A2C2f/C3k2 参数、Detect）逐行不变；`nc: 1`、scales 块与现有 yaml 一致。
- 关键行示例：

```yaml
head:
  - [[-1, -1], 1, FreqFusion, []] # 9 FreqFusion_up(P5)：自输入式，替代 Upsample
  - [[-1, 6], 1, Concat, [1]] # 10 cat backbone P4
  - [-1, 2, A2C2f, [512, False, -1]] # 11
  - [[-1, -1], 1, FreqFusion, []] # 12 FreqFusion_up(P4)
  - [[-1, 4], 1, Concat, [1]] # 13 cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]] # 14
  # 15–21 与官方 yolo12.yaml 逐行相同（Conv 下采样/Concat/A2C2f/C3k2/Detect）
```

- 注意：文件名必须携带 `s`（否则 parse_model 静默按 n 缩放，见现有 yaml 注释规约）。

**文件 2（新建）**：`scripts/improved_train/coco_pretrained/train_yolov12-FreqFusion_up_ssdc-uav.py`
- 逐行镜像 [train_yolov12-DySample_ssdc-uav.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_train/coco_pretrained/train_yolov12-DySample_ssdc-uav.py)（对比对象就是 DySample，协议必须逐项对齐）：
  - 保留 `SaveLastNCheckpointsCallback(n=3)`
  - `model = YOLO(r'...\scripts\improved_yolo12\yolo12s-FreqFusion_up.yaml')`
  - `model.load('yolo12s.pt')`（与 DySample 脚本相同机制；不在 train() 里另传 pretrained=）
  - train 参数原样：`epochs=150, imgsz=640, batch=16, optimizer='SGD', device='0', project='runs/ssdc_uav_train', name='yolo12s_FreqFusion_up_ssdc_uav_exp01'`

**文件 3（新建）**：`scripts/improved_test/coco_pretrained/test_yolov12-FreqFusion_up_ssdc_uav.py`
- 复制现有 [test_yolov12-FreqFusion_ssdc_uav.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/improved_test/coco_pretrained/test_yolov12-FreqFusion_ssdc_uav.py)，仅改 3 处：`weights_path` 指向 `runs/ssdc_uav_train/yolo12s_FreqFusion_up_ssdc_uav_exp01/weights/best.pt`、`name='yolo12s_FreqFusion_up_ssdc_uav_test_exp1'`、CSV 中 `Model='YOLO12s-FreqFusion_up'`。

### 1.3 验证步骤（实施后由代理执行；训练本身由用户启动）

1. **冒烟测试**（只读性质，不启动训练）：

```python
from ultralytics import YOLO
import torch
m = YOLO('scripts/improved_yolo12/yolo12s-FreqFusion_up.yaml')
m.load('yolo12s.pt')                      # 预期打印 Transferred 数与 DySample 一致（~697 档，非 342 档）
out = m.model(torch.zeros(1, 3, 640, 640))  # 前向无错，输出 3 尺度检测特征
print(sum(p.numel() for p in m.model.parameters()))
```

2. 检查构建摘要：层 9/12 为 FreqFusion、参数 ~9.4–9.6M、GFLOPs ~21.0–21.5、层索引与官方对齐。
3. 用户启动训练；完成后运行文件 3 的测试脚本，结果自动追加至 `runs/test_result/SSDC-UAV_Test_Result.csv`，与 DySample（55.823）/baseline（55.814）同表对比。

### 1.4 判读标准

- 若 FreqFusion_up ≥ DySample（≥55.8）：算子在该基准有效，此前失败归因于 head 重构+迁移损失（呼应上一轮分析）。
- 若 FreqFusion_up ≈ 或 < DySample：在 SSDC-UAV 协议下上采样算子选择无显著差异，论文叙事转向"结构级重构才是瓶颈"。

---

## Part 2（Q2）：不改网络结构、保全预训练迁移的性能提升方案清单

> 前提：所有方案均不改动网络结构与权重形状，`yolo12s.pt` 迁移率不变。基于本仓库已有证据排序：300e 延长训练无效（55.68 vs 150e 的 55.81）；损失类改进（WIoU/Focaler/PIoU/SD）增益 ≤0.15 —— 预算应投向以下方向。

### 第一梯队（UAV 小目标场景预期收益最大）

| # | 方案 | 具体配置 | 依据/预期 |
|---|---|---|---|
| 1 | **提高输入分辨率** | `imgsz=1280`（训练+测试一致），显存不足则 batch 16→8；折中档 960 | 小目标最大单一杠杆，典型 +1.5~4 mAP50-95；conv 权重尺寸无关、迁移率不变。注意：属计算预算变化，论文对比需同分辨率基线或明确声明 |
| 2 | **空域增强组合（航拍先验）** | `flipud=0.5`；俯视视角可再加 `degrees=90`（或 180）；`copy_paste=0.3` + `copy_paste_mode='flip'`（本仓库 [augment.py:2733](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/data/augment.py#L2733) 已支持检测框）；`close_mosaic=30` + `cos_lr=True` | 俯视航拍图像翻转/旋转不变是合法先验；copy_paste 复制粘贴小目标实例直接提升小目标召回；更长 mosaic 关闭精调期+余弦退火是稳定免费增益。**先抽查 val_batch 图片确认视角**（斜视则去掉 degrees） |
| 3 | **测试时增强 TTA** | 测试脚本 `model.val(..., augment=True)`（[validator.py:154](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/engine/validator.py#L154) 已支持） | 零训练成本，多尺度+翻转融合，典型 +0.5~1.5；只改一行，可立即在现有 best.pt 上验证 |

### 第二梯队（零/低成本微调技巧）

| # | 方案 | 具体配置 | 依据/预期 |
|---|---|---|---|
| 4 | **权重平均（Model Soup）** | 平均已保存的 `epoch_148/149/150.pt`（SaveLastNCheckpointsCallback 正是为此准备）后 val | 一次性脚本，通常 +0.1~0.4；对"close_mosaic 后过拟合"（上轮发现 e140–150 val loss 单调上升）尤其对症 |
| 5 | **低学习率精调（polish）** | 从 best.pt 续训 20–30e：`lr0=0.002, lrf=0.1, mosaic=0.0, cos_lr=True`（用 `model=best.pt` 直接 resume 型训练） | 关闭强增强后小学习率收敛，常 +0.2~0.5 |
| 6 | **多尺度训练** | `multi_scale=True` | 提升尺度鲁棒性，显存约 +30%；与 #1 二选一即可 |

### 第三梯队（工程/部署级，改变推理管线，需评估论文协议是否允许）

| # | 方案 | 说明 |
|---|---|---|
| 7 | **SAHI 切片推理** | 640 重叠滑窗 + 合并 NMS/WBF，UAV 小目标标准做法、召回提升显著；但非端到端 640 协议，对比实验需单独一组 |
| 8 | **多模型预测融合（WBF）** | 不同 seed/改进组的 best.pt 预测融合；训练成本高 |
| 9 | **知识蒸馏** | 同数据训练 yolo12m/l 作教师蒸馏 yolo12s 学生；学生结构不变、迁移率不受影响 |

### 推荐执行顺序

1. 立即可做（零训练）：#3 TTA（现有各 best.pt 各加一行）→ #4 Soup（一次性脚本）。
2. 下一轮训练：#1 imgsz=1280 与 #2 空域增强组合各一组（可先单独消融再合并）。
3. 视结果叠加：#5 polish；#6/#7/#8/#9 按论文叙事需要选用。

---

## Assumptions & Decisions

- 自输入式变体为 Q1 唯一实施对象（用户已选定；跳连引导式不做）。
- Q2 仅交付本清单文档，不创建任何训练/测试脚本（用户已选定）。
- FreqFusion 模块 kwargs 保持默认（lowpass_kernel=5, highpass_kernel=3, semi_conv=True 等），与 DySample 一样不调参，保证协议公平。
- 训练与正式测试为长任务，由用户启动；代理只做构建/迁移冒烟验证。
- 新文件均遵循 AGENTS.md 规约：yaml 带 scale 字母；本方案不新增 nn 模块、不触碰 tasks.py 与 AddModules。
- Git：仅创建文件，不执行任何 commit/push（遵守仓库规约，等用户确认）。

---

## 实施记录（2026-08-22，已完成）

**设计修正**：原方案"from=[-1,-1] 纯自输入、零代码改动"在冒烟测试中被证伪——FreqFusion 内部（semi_conv 路径）要求 HR 空间尺寸 = 2× LR（[FreqFusion.py:227](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/FreqFusion.py#L227) carafe 断言 m_h == up*h），同尺寸自输入直接断言失败。已按最小正确方式修正：
- 在 [FreqFusion.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/FreqFusion.py) 中新增 `FreqFusionUpsample` 适配器（单输入 x：hr 流 = nearest 2× 初始上采样合成，与原 nn.Upsample 行为一致，再交 FreqFusion 做 AHPF+ALPF；输出通道不变、空间 ×2），并加入 `__all__`（`AddModules/__init__.py` 的 `from .FreqFusion import *` 已启用，自动聚合导出）
- 在 [tasks.py:1853-1858](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1853-L1858) 注册 `FreqFusionUpsample` 解析分支（仿 DySample：注入 in_channels，c2 = ch[f]）
- yaml 第 9/12 层改为 `[-1, 1, FreqFusionUpsample, []]`（单输入，from=-1）

**实际交付文件**：
1. `scripts/improved_yolo12/yolo12s-FreqFusion_up.yaml`（新建）
2. `scripts/improved_train/coco_pretrained/train_yolov12-FreqFusion_up_ssdc-uav.py`（新建，逐行镜像 DySample 脚本协议）
3. `scripts/improved_test/coco_pretrained/test_yolov12-FreqFusion_up_ssdc_uav.py`（新建，改 weights/name/Model 三处）
4. `ultralytics/nn/AddModules/FreqFusion.py`（新增 FreqFusionUpsample 类）
5. `ultralytics/nn/tasks.py`（新增解析分支）

**冒烟测试结果（全部通过）**：
- 构建：层 9/12 = FreqFusionUpsample，stride [8,16,32] 正确
- 迁移：`Transferred 685/711 items`——匹配项 685 与 DySample 初始化加载（685/697）完全一致；分母差 14 = 新模块参数/缓冲项数差（FF 每实例 10 项 vs DySample 3 项：697−6+20=711，精确吻合），即可迁移项 100% 加载
- 前向：eval 输出 (1,5,8400)、train 模式前向均正常
- 规模：参数 9.477M（+0.23M vs 基线，符合预估）；GFLOPs 22.51（高于预估 21.0–21.5，carafe/unfold 算子计入所致，vs DySample 21.6 仍同量级）

**下一步（由用户执行）**：运行 `train_yolov12-FreqFusion_up_ssdc-uav.py`（150e/SGD/640/batch16），完成后运行对应 test 脚本，与 DySample（55.823）/baseline（55.814）同表对比判读。
