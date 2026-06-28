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
