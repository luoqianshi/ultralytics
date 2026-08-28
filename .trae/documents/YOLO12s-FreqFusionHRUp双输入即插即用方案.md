# YOLO12s-FreqFusionHRUp：HR 引导双输入即插即用 FreqFusion 上采样方案

## 1. 概要（Summary）

现有两种 FreqFusion 集成路线各有硬伤：

- **结构重构版**（`yolo12-FreqFusion.yaml` + `FreqFusion`）：HR/LR 双输入契合原始设计，但 head 前置 3 个 1x1 Conv 拉齐通道，层索引 9-21 全部偏移，head 权重无法从 `yolo12s.pt` 迁移 → 精度受损。
- **单输入即插即用版**（`yolo12s-FreqFusion_up.yaml` + `FreqFusionUpsample`）：原位替换 `nn.Upsample`，迁移率与 DySample 一致，但 HR 流由 LR 自身 nearest 2x 合成，丢失了原始设计"HR 来自主干高分辨率特征"的核心优势。

本方案新增**第三种适配器 `FreqFusionHRUp`**：双输入 `[HR(主干跳连), LR(前序层)]`，原位替换 head 第 9/12 层的 `nn.Upsample`，输出张量形状与 `nn.Upsample(scale=2)` 完全一致（通道=LR 通道、空间=HR 分辨率），从而**同时**满足：

1. 即插即用：Concat 及其后所有层与官方 `yolo12.yaml` 逐行相同，层索引 9-21 对齐，预训练迁移率与 DySample 版一致；
2. 双输入保真：AHPF 从真实主干 HR 特征提取高频细节、ALPF 由真实 HR/LR 联合生成抗混叠核，契合 FreqFusion 原始设计思路。

向后兼容：不改动 `FreqFusion`、`FreqFusionUpsample` 的任何现有代码与现有 3 个 yaml，仅做纯增量添加。

## 2. 现状分析（Current State）

关键代码事实（均来自实际阅读）：

- [FreqFusion.py L127](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/FreqFusion.py#L127)：`FreqFusion` 断言 `hr_channels == lr_channels`（forward 末尾 `hr_feat + lr_feat` 逐元素相加）。而 yolo12s 颈部 HR/LR 通道天然不等：P4(主干层6)=256 vs P5(层8)=512；P3(层4)=128 vs head-P4(层11)=256 → **适配器内部必须做 HR→LR 的 1x1 通道投影**。
- [FreqFusion.py L275-296](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/FreqFusion.py#L275-L296)：`FreqFusionUpsample` 的既有模式——`__init__(channels)` 内组合 `FreqFusion`，`forward` 单输入合成 HR。新适配器沿用同一组合模式，仅改为双输入 + 通道对齐。
- [tasks.py L1849-1858](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1849-L1858)：`parse_model` 中 `FreqFusion` 分支用 `[ch[x] for x in f]` 解析多输入通道（`ch[-1]` 即前序层通道）；`FreqFusionUpsample` 分支仿 DySample 注入单通道。新分支介于两者之间。
- [tasks.py L182](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L182)：前向时 `f` 为列表则按 `[x if j == -1 else y[j] for j in m.f]` 组包传入，模块 `forward` 收到张量列表——与 `FreqFusion.forward` 的 `hr_feat, lr_feat = x` 解包约定一致，新适配器照用。
- [tasks.py L1885](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1885)：`save.extend(x % i for x in f if x != -1)` —— yaml 中 `from=[[6,-1],...]` / `[[4,-1],...]` 会自动把层 6/4 加入 savelist，无需额外处理。
- `AddModules/__init__.py` 第 19 行 `from .FreqFusion import *` 已启用 → 只需在 `FreqFusion.py` 的 `__all__` 中追加新类名即自动聚合导出，无需动 `__init__.py` 与 tasks.py 顶部 import。
- 通道核对（s scale，width=0.5）：层4=256、层6=256、层8=512。官方 head：层9 Upsample 输出 512ch@P4；层10 Concat(512+256)=768；层12 Upsample 输出 256ch@P3；层13 Concat(256+256)=512。新适配器输出必须分别 = 512ch@P4、256ch@P3（= LR 通道 @ HR 分辨率），下游形状逐字节不变。（注：P3 处 HR=层4=256 与 LR=层11=256 等通道，align 退化为 Identity；仅 P4 处 HR=256→LR=512 需要 1x1 投影。）

## 3. 拟议改动（Proposed Changes）

### 3.1 `ultralytics/nn/AddModules/FreqFusion.py`（增量修改）

- `__all__` 由 `['FreqFusion', 'FreqFusionUpsample']` 改为 `['FreqFusion', 'FreqFusionUpsample', 'FreqFusionHRUp']`。
- 在 `FreqFusionUpsample` 类之后追加新类：

```python
class FreqFusionHRUp(nn.Module):
    """FreqFusion 的 HR 引导即插即用上采样适配器（双输入版）。

    与单输入 FreqFusionUpsample 的唯一区别：HR 流不再由 nearest 2x 合成，
    而是真实来自主干网络的高分辨率跳连特征，契合 FreqFusion 原始双输入设计
    （AHPF 从真实 HR 提取高频细节，ALPF 由真实 HR/LR 联合生成抗混叠上采样核）。

    用法（原位替代 head 中 nn.Upsample(scale=2)，保留后续 Concat）：
        - [[hr_idx, -1], 1, FreqFusionHRUp, []]
    输出 = FreqFusion(align(HR), LR)：通道 = LR 通道、空间 = HR 分辨率，
    与原 nn.Upsample(scale=2) 输出形状完全一致，head 层索引与官方 yolo12.yaml 对齐。

    通道对齐：颈部 HR/LR 通道一般不等（如 yolo12s P4=256 vs P5=512），而
    FreqFusion 要求等通道（forward 末尾逐元素相加），故内置 1x1 Conv 将 HR
    投影到 LR 通道；等通道时退化为 Identity（零额外参数、零行为变化）。

    Args:
        hr_channels (int): HR（主干跳连）输入通道数。
        lr_channels (int): LR（前序层）输入通道数 = 本层输出通道数。
        kwargs: 透传给 FreqFusion 的可选参数（默认即与原模块一致）。
    """

    def __init__(self, hr_channels, lr_channels, **kwargs):
        super().__init__()
        self.align = nn.Conv2d(hr_channels, lr_channels, 1) if hr_channels != lr_channels else nn.Identity()
        self.ff = FreqFusion([lr_channels, lr_channels], **kwargs)

    def forward(self, x):
        hr_feat, lr_feat = x
        return self.ff((self.align(hr_feat), lr_feat))
```

不改动 `FreqFusion`、`FreqFusionUpsample` 及其余任何已有代码行。

### 3.2 `ultralytics/nn/tasks.py`（增量修改）

在 [L1858](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1858) `FreqFusionUpsample` 分支之后插入新分支（带与现有风格一致的 @TODO Begin/End 注释标记）：

```python
        # @TODO Begin 20260827 FreqFusion HR 引导双输入即插即用上采样适配器
        elif m is FreqFusionHRUp:
            # FreqFusionHRUp: 双输入 [HR(主干跳连), LR(前序层)]，原位替代 nn.Upsample
            c2 = ch[f[-1]]  # 输出通道 = LR 通道（与原 nn.Upsample 输出一致）
            args = [ch[f[0]], ch[f[-1]], *args]  # 注入 (hr_channels, lr_channels)
        # @TODO End 20260827 FreqFusion HR 引导双输入即插即用上采样适配器
```

既有 `FreqFusion` / `FreqFusionUpsample` 分支原样保留。

### 3.3 新建 `scripts/improved_yolo12/yolo12s-FreqFusion_hrup.yaml`

以 `yolo12s-FreqFusion_up.yaml` 为底（backbone 与官方完全一致；head 仅第 9/12 层与官方不同），文件名携带 scale 字母 `s`：

```yaml
# Parameters
nc: 1
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# YOLO12 backbone（与官方 yolo12.yaml 完全一致）
backbone:
  - [-1, 1, Conv,  [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]] # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]] # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]] # 8

# YOLO12s-FreqFusion_hrup head（仅第 9/12 层替换 Upsample，其余与官方逐行相同）
head:
  - [[6, -1], 1, FreqFusionHRUp, []] # 9 HR=backbone P4(256), LR=P5(512) → 512ch@P4，替代 Upsample
  - [[-1, 6], 1, Concat, [1]] # 10 cat backbone P4（与官方一致，768ch）
  - [-1, 2, A2C2f, [512, False, -1]] # 11

  - [[4, -1], 1, FreqFusionHRUp, []] # 12 HR=backbone P3(256), LR=head P4(256) → 256ch@P3
  - [[-1, 4], 1, Concat, [1]] # 13 cat backbone P3（与官方一致，512ch）
  - [-1, 2, A2C2f, [256, False, -1]] # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]] # 16 cat head P4
  - [-1, 2, A2C2f, [512, False, -1]] # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]] # 19 cat head P5
  - [-1, 2, C3k2, [1024, True]] # 20 (P5/32-large)

  - [[14, 17, 20], 1, Detect, [nc]] # 21 Detect(P3, P4, P5)
```

文件头部按仓库惯例写设计注释（设计动机、与两种既有方案的差异、迁移率预期、scale 字母提醒）。

## 4. 假设与决策（Assumptions & Decisions）

1. **命名**：类名 `FreqFusionHRUp`、yaml 名 `yolo12s-FreqFusion_hrup.yaml`——明确区分于单输入 `FreqFusionUpsample` / `yolo12s-FreqFusion_up.yaml`，"HR" 表明高分辨率流来自主干。
2. **通道对齐放在适配器内而非 FreqFusion 内**：保持 `FreqFusion` 的等通道断言与既有行为一字不动，避免影响结构重构版 yaml；对齐 1x1 Conv 仅出现于 P4 处（HR 256→LR 512：131k 参数），P3 处 HR/LR 等通道（256）退化为 Identity（0 参数），属预期新参数，与 Upsample 无参数一样不影响旧键迁移。
3. **HR/LR 顺序约定**：yaml 写 `[[hr_idx, -1], ...]`（HR 在前），与既有 `FreqFusion` 分支及官方 FreqFusion 仓库的 `(hr, lr)` 顺序一致。
4. **输出含 HR 残差是特性而非冗余**：输出 = ALPF_up(LR) + align(HR) + HP(align(HR))，与 `FreqFusion` 默认 `hr_residual=True` 行为一致；后续 Concat 再拼原始 HR 与官方拓扑相同。FreqFusion 全部超参保持默认，与另两个版本公平可比。
5. **不新建训练/测试脚本**：用户仅要求 yaml + 适配器类；训练脚本（仿 `train_yolov12-FreqFusion_up_ssdc-uav.py`）可作为后续任务。
6. **预期迁移率**（已实测确认）：未 fuse 状态下 `Transferred 685/713`，与 `FreqFusion_up` 版（685/711）迁移条数完全相同；未迁移项仅为 model.9/12 新参数（22 个）、nc=1 分类头（6 个形状不匹配，预期）及仓库版 A2C2f 的 `pe.conv.bias` 版本差异（8 个，对所有变体一视同仁，与本次改动无关）。

## 5. 验证步骤（Verification）

按顺序执行（均为只读构建/推理检查，不启动完整训练）：

1. **构建冒烟**：`YOLO('scripts/improved_yolo12/yolo12s-FreqFusion_hrup.yaml')`，确认：
   - 第 9/12 层类型为 `FreqFusionHRUp`，`model.9.align` 为 Conv2d(256,512,1)、`model.12.align` 为 Conv2d(128,256,1)；
   - 层 10/13 为 Concat、层 21 为 Detect，总层数 22，stride = [8,16,32]。
2. **前向形状**：640×640 随机输入前向，Detect 输出形状与官方 yolo12s 一致；层 9 输出 [B,512,80,80]、层 12 输出 [B,256,160,160]（imgsz=1280 时相应翻倍）。
3. **迁移率**：`model.load('yolo12s.pt')`（或 train 的 pretrained 路径），确认日志 "Transferred 697/697 items"（与 DySample/FreqFusion_up 版相同数值）。
4. **回归检查（向后兼容）**：依次构建 `yolo12-FreqFusion.yaml`、`yolo12s-FreqFusion_up.yaml`、`yolo12-DySample.yaml` 并跑一次前向，确认三种既有方案行为不变。
5. **可选**：`model.info()` 记录参数量/FLOPs 并与 `FreqFusion_up` 版对比存档，供实验记录引用。
