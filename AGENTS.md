# Agents.md

本文件记录项目中需要遵守的规约，供 AI 代理（Agent）和开发者参考。

---

## 自定义改进模块放置规约

新增的自定义改进模块（如 DySample、SimAM、Mona、ESMoE、AssemFormer、HSFPN、HPDown 等）必须放在：

```
ultralytics/nn/AddModules/
```

**不要**放在 `ultralytics/nn/modules/`（该目录仅用于 Ultralytics 框架原生模块）。

### 集成步骤

1. 在 `ultralytics/nn/AddModules/` 下创建 `<ModuleName>.py` 实现模块
2. 在 `ultralytics/nn/AddModules/__init__.py` 中启用导出：
   ```python
   from .<ModuleName> import *
   ```
3. `ultralytics/nn/tasks.py` 已有 `from .AddModules import *`（约第 104 行），会自动聚合导入，无需在 tasks.py 顶部显式 import
4. 如需在 `parse_model()` 中注册新的 yaml 解析分支，直接在 tasks.py 的 `parse_model` 函数内引用模块类名即可（类名已通过 AddModules 聚合导入到 tasks 命名空间）

### 注意事项

- `AddModules/__init__.py` 的约定是"按需启用"：默认所有 import 都注释掉，改谁的时候就启用谁，避免导入所有模块导致不必要的依赖
- 测试文件中应使用 `from ultralytics.nn.AddModules import <ModuleName>` 或 `from ultralytics.nn.AddModules.<ModuleName> import <ModuleName>` 导入
- `ultralytics/nn/modules/__init__.py` 的 `__all__` 中不应包含自定义模块名

### 【强制】模块文件必须定义 `__all__`（防止遮蔽官方类）

许多从第三方仓库移植的模块文件（如 Mona、SCSA、MCA、MoCAttention、FBRT_YOLO、SimAM、AssemFormer、HPDown、MultiScaleGateAttn 等）会**复制一份官方 `Conv/Bottleneck/C3/C3k` 定义**作为内部依赖。由于 `tasks.py` 顶部有 `from .AddModules import *`，一旦该模块在 `__init__.py` 中被启用且**没有 `__all__` 约束**，这些同名副本就会**静默遮蔽官方类**——yaml 里写 `Conv` 实际构建的是自定义版本，导致网络结构被悄悄改变、预训练权重行为异常，且无任何报错。

**规则：`AddModules/` 下每个模块文件顶部必须定义 `__all__`，只导出真正的改进模块（如 `__all__ = ["Mona", "A2C2f_Mona"]`），绝不导出 `Conv/Bottleneck/C3/C3k` 等与 ultralytics/torch 内置重名的符号，也不导出内部辅助类（如 `MonaOp`、`StdPool`、`BaseConv2d`）。**

参考写法（见 `EMA.py`、`Mona.py` 等文件头部注释）：

```python
# 本文件重定义了 Conv/Bottleneck/C3/C3k（与 ultralytics 内置同名但实现不同），
# 若不加 __all__，tasks.py 中 `from .AddModules import *` 会遮蔽官方模块，静默改变整个网络结构
__all__ = ["<ModuleName>", "A2C2f_<ModuleName>"]
```

**AI 代理审核义务**：每当用户引入新模块、启用 `__init__.py` 中某个模块、或修改 AddModules 文件时，必须审核该文件是否定义了 `__all__`、导出名单是否泄漏了官方同名类；发现缺失或泄漏应主动提醒并修复。历史教训：2026-09-03 之前的分析（`.lqs/失败分析/YOLO12改进实验_权重迁移失败分析.md`）发现 12 个实验（含基线复跑）曾被无 `__all__` 的模块污染。

---

## Git 提交规约

- **没有用户的明确允许，AI 代理不得自行执行 `git commit`、`git push` 等写操作**。完成代码修改后，应向用户报告改动内容并等待用户确认后再提交。
- 仅允许执行只读 git 命令（如 `git status`、`git diff`、`git log`、`git branch` 等）用于了解仓库状态。
- 若用户在某个任务中明确授权提交（如"提交这些改动"），则仅对该次任务生效，不延续到后续任务。
