# SSDC-UAV YOLO12 改进实验「权重迁移失败」分析计划

## Summary

扫描 `runs/ssdc_uav_train/` 下全部 41 个实验的 `train.log`，基于两类证据评估每个实验是否存在：
1. **网络结构大改导致预训练权重迁移失败**（以 `Transferred X/Y` 迁移率为主指标）；
2. **AddModules 模块缺少 `__all__` 约束导致官方 `Conv` 等类被遮蔽（shadowing）**（以模型结构表中 `ultralytics.nn.AddModules.<模块>.Conv` 泄漏为证据）。

最终产出一份 markdown 报告，存放于 `D:\Data\New_Codes\Python_Codes\ultralytics\.lqs\失败分析\`（目录已存在）。

**本任务为纯只读分析 + 写一个新报告文件，不修改任何代码。**

## Current State Analysis（探索已获得的事实）

### 数据源
- 41 个 `train.log`：`runs/ssdc_uav_train/{marked,size-s-base}/*/train.log` 及 `runs/ssdc_uav_train/*/train.log`
- 每个实验目录均有 `results.csv`（best mAP50 / mAP50-95 已提取）
- 无任何 log 中出现 `size mismatch` / `Traceback` / `RuntimeError`（加载走 `intersect_dicts + strict=False`，静默丢弃不匹配项，见 [tasks.py load()](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L302-L325)）

### 迁移率判读规则（已验证）
- 部分脚本（如 A2C2f_MCA）先显式 `model.load('yolo12s.pt')` 打印第一条 `Transferred`，trainer 内部再打印第二条（第二条是内存模型自拷贝，恒为 N/N，**无意义**）。**以第一条（对官方 yolo12s.pt 的迁移）为准**。
- 基线参照：`yolo12s` = 685/691 ≈ **99.1%**（缺的 6 项是 nc=80→1 的检测头参数，属正常）。

### 问题1：结构大改 → 迁移严重受损的实验（迁移率 < 85%）

| 实验 | Transferred | 迁移率 | 结构改动 | best mAP50-95 |
|---|---|---|---|---|
| yolo12s_FreqFusion | 342/729 | 46.9% | 颈部整体重排（FreqFusion 融合改变层编号） | 0.5665 |
| yolo12_A2C2f_FCM | 335/719 | 46.6% | FBRT 模块替换颈部 | 0.5708 |
| yolo12_EMA | 342/697 | 49.1% | EMA 插入 layer 9 后全部层错位 | 0.5688 |
| yolo12s_BiFPN | 414/774 | 53.5% | BiFPN 加权双向颈部重构 | 0.5709 |
| yolo12_P2_DySample | 469/854 | 54.9% | 增加 P2 检测层（且 epoch 1 即中断，saved=0） | 0.5722* |
| yolo12_P2 | 469/845 | 55.5% | 增加 P2 检测层，颈部全部错位 | 0.5714 |
| yolo12_A2C2f_SCSA | 298/485 | 61.4% | SCSA 替换颈部 A2C2f | 0.5609 |
| yolo12_MultiScaleGatedAttn | 570/850 | 67.1% | 多尺度门控注意力插入 | 0.5747 |
| yolo12_A2C2f_MoCA | 292/381 | 76.6% | MoCA 替换主干 A2C2f | 0.5577 |
| yolo12_Mona | 370/475 | 77.9% | Mona 替换多处（层数 185 vs 265） | 0.5645 |
| yolo12_A2C2f_Mona | 445/579 | 76.9% | A2C2f_Mona 替换 | 0.5740 |
| yolo12s_A2C2f_MCA | 421/523 | 80.5% | A2C2f_MCA 替换 | 0.5729 |
| yolo12_A2C2f_EMA | 421/523 | 80.5% | A2C2f_EMA 替换 | 0.5754 |

轻度受损（90–98%，属预期内的头部替换代价）：Detect-DyHead 649/673 (96.4%)、DyHead 671/726 (92.4%)、PIoU2-DySample(_plus)_DyHead 671/732、734。
健康（≥98%）：DySample 系列、纯 Loss 改动（Focaler-CIoU/PIoU2/SD-Loss/WIoUv3/Slide/Varifocal，均 685/691）、FreqFusion_up/hrup（685/711、713）、SPDConv（684/691）。
特例：**SPDConv 迁移率 99% 但 mAP50-95 仅 0.4643**（stem 语义改变，权重虽"形状兼容"但已失效）——报告中单独标注为"迁移成功≠迁移有效"。

### 问题2：缺少 `__all__` 导致 Conv 被错误引入（日志泄漏证据）

机制：`tasks.py` 顶部 `from .AddModules import *`；若某 AddModules 文件复制了官方 `Conv/Bottleneck/C3` 且无 `__all__`，则遮蔽官方类，yaml 里写 `Conv` 实际解析到自定义版本。日志中表现为结构表出现 `ultralytics.nn.AddModules.<X>.Conv`。

受影响实验（log 中 leakConv>0，共 12 个）：
- `yolo12_A2C2f_EMA`、`yolo12_EMA` → EMA.Conv
- `yolo12_A2C2f_FCM` → FBRT_YOLO.Conv
- `yolo12_A2C2f_MoCA`、`marked/yolo12s_WIoUv3` → MoCAttention.Conv
- `yolo12_A2C2f_Mona`、`yolo12_Mona`、`yolo12_P2` → Mona.Conv（注意：P2 实验根本不含 Mona 模块，却被 Mona 泄漏污染）
- `yolo12s_A2C2f_MCA` → MCA.Conv
- `marked/yolo12s_ssdc_uav_exp02_300Epoch` → SCSA.Conv（**基线复跑也被污染**）
- `yolo12_MultiScaleGatedAttn` → MultiScaleGateAttn.Conv

关键结论（需写入报告）：
1. 泄漏的自定义 Conv 均为官方 Conv 的逐行拷贝（已抽查 Mona/SCSA/MoCAttention/MCA/FBRT/EMA/MultiScaleGateAttn，签名与参数量一致，如 model.0 均为 928 params），state_dict 键名不变 → **对权重迁移率本身无额外伤害**，问题1 的低迁移率均源于结构重排导致的层号错位/模块替换。
2. 真正的风险是**静默语义漂移**：日后若在某个 AddModules 文件里修改其私有 `Conv`（如加注意力），所有 yaml 中的 `Conv` 都会跟着变，且无任何报错。
3. 无 `__all__` 且重定义 Conv/Bottleneck/C3 的高危文件（现存）：`AssemFormer.py, FBRT_YOLO.py, HPDown.py, MCA.py, MoCAttention.py, Mona.py, MultiScaleGateAttn.py, SCSA.py, SimAM.py`；已修复：`EMA.py`（2026-09-02 补 `__all__`）、`FreqFusion.py`、`Detect_*.py`、`ESMoE.py`、`ASFF.py`。

## Proposed Changes

仅创建一个新文件：

- **`D:\Data\New_Codes\Python_Codes\ultralytics\.lqs\失败分析\YOLO12改进实验_权重迁移失败分析.md`**

报告结构：
1. **概述**：分析范围（41 个实验）、数据来源、判读方法（Transferred 第一条为准、基线 99.1% 参照、leakConv 检测正则）
2. **问题1 判定表**：全部实验按迁移率分档（严重受损 <85% / 轻度 90–98% / 健康 ≥98%），列：实验名、Transferred、迁移率、缺失原因（结构改动类型）、best mAP50/mAP50-95、相对基线差值
3. **问题2 判定表**：12 个 Conv 泄漏实验 + 泄漏来源模块 + 是否同时命中问题1；说明"拷贝一致故未直接破坏迁移，但存在静默改结构风险"
4. **交叉结论**：问题1∩问题2 的实验（A2C2f_EMA/FCM/MoCA/Mona/MCA/SCSA、P2、MultiScaleGatedAttn、WIoUv3、基线 exp02）；SPDConv 特例（迁移成功但效果崩溃）
5. **建议**（只列不改）：为 9 个高危文件补 `__all__`；结构重排型改进（BiFPN/FreqFusion/P2/EMA）改用时注意层号对齐或接受冷启动；P2_DySample 中断需复跑

## Assumptions & Decisions

- 报告语言：中文；文件放 `.lqs\失败分析\`（目录已存在，无需新建）
- `marked/` 与 `size-s-base/` 下的实验一并纳入分析
- 迁移率阈值：85% / 98% 为分档线（基线 99.1%，头部替换类 92–96% 视为可接受）
- 不修改任何训练脚本、AddModules 源码或 `__init__.py`（用户只要求"整理信息"）
- yolo12_P2_DySample 训练在 epoch 1 中断（log 尾部为 grid_sampler 确定性警告后截止），其指标来自后续单独 val

## Verification

1. 报告中每个 `Transferred X/Y` 数字可回溯到对应 `train.log` 行号（报告附行号）
2. leakConv 判定可用命令复现：`Select-String -Pattern "ultralytics\.nn\.AddModules\.\w+\.(Conv|Bottleneck|C3)\b"`
3. 打开生成的 markdown 确认表格渲染正常、路径正确
