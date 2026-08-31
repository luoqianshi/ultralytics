# 训练脚本与配置评估：YOLO12s + DySample plus + DyHead + Powerful-CIoU2（SSDC-UAV）

## 总结（最终评估结论）

**检查无误，可以直接执行训练。**

三个改进点（DySample plus、DyHead、Powerful-CIoU2）的模块代码、注册、yaml 解析、损失启用、预训练加载链路全部验证通过。

关于 yaml 文件名的初步疑点（脚本引用 `yolo12s-DySample-plus_DyHead.yaml` 而磁盘上是 `yolo12-DySample-plus_DyHead.yaml`）：经确认 **不构成问题**。`yaml_model_load`（[tasks.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1916-L1917)）会先把文件名归一化（`re.sub(r"(\d+)([nslmx])(.+)?$", ...)` 去掉 scale 字符），`check_yaml(unified_path, hard=False)` 优先命中现存的 `yolo12-DySample-plus_DyHead.yaml` 并加载其内容；`guess_model_scale` 再从原始文件名解析出 `scale='s'`，模型按 s 尺度构建，与 yolo12s.pt 预训练权重对齐。这正是 Ultralytics 官方 `yolov8s.yaml → yolov8.yaml` 的同一机制。

## 逐项验证结果

### 1. Powerful-CIoU2（PIoU2）启用链路 ✓
- `powerful_iou` 已在 [default.yaml](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/cfg/default.yaml#L140) 注册，`powerful_iou=True` 是合法训练参数
- [loss.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/utils/loss.py#L210-L214)：`piou(..., PIoU2=True)` 正是 Powerful-CIoU v2 路径，启动时会打印 "Powerful-IoU 启用成功！请放心使用！"
- [loss.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/utils/loss.py#L476) 与 [tal.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/utils/tal.py#L227)：回归损失与标签分配两处同步读取该参数，链路完整

### 2. 模块注册与 yaml 解析 ✓
- [AddModules/\_\_init\_\_.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/__init__.py#L17-L22)：DySample、DyHead 均已启用导出
- [tasks.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1801-L1822)：DyHead 走 Detect 同款分支，args 注入 `[reg_max, end2end, ch]`，与 `DyHead.__init__(nc, hidc, block_num, reg_max, end2end, ch)` 签名完全匹配（end2end=None 为 falsy，无害）
- [tasks.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1868-L1871)：DySample 正确注入 in_channels；yaml args `[2, 'lp', 4, True]` → `DySample(512, 2, 'lp', 4, dyscope=True)`，dyscope=True 即 DySample plus 变体
- [tasks.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L1956-L1958)：`guess_model_task` 将 "dyhead" 归入 detect 任务
- yaml 网络结构层索引（DyHead 取 [14, 17, 20]）与标准 yolo12 head 一致，DySample 替换 Upsample 后通道数不变，不影响后续层

### 3. 预训练权重加载 ✓
- [trainer.py](file:///d:/Data/New_Codes/Python_Codes/Python_Codes/ultralytics/ultralytics/engine/trainer.py#L739-L740)：`pretrained='yolo12s.pt'`（字符串）经 `setup_model` → `load_checkpoint` 加载；yolo12s.pt 存在于仓库根目录
- 权重迁移靠 `intersect_dicts` 形状匹配：backbone/neck 全部可迁移；DyHead 新增的 conv/dyhead 及重建的 cv2/cv3 与原 Detect 头形状或键名不匹配会被自动跳过，属预期行为
- scale='s' 解析正确（见上文归一化说明），backbone 权重形状与 yolo12s.pt 对齐

### 4. DyHead 模块本身 ✓
- stride 探测（[tasks.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/tasks.py#L407)）`isinstance(m, Detect)` 包含子类 DyHead；bias_init 兼容重建后的 cv2/cv3
- legacy 链路正确：backbone 的 A2C2f 将 legacy 置 False → `DyHead.legacy=False`，与重建的 DWConv 结构一致
- [DyHead.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/ultralytics/nn/AddModules/DyHead.py#L141-L151)：CPU 路径 deform_conv2d 访问冲突已降级规避，构建期 stride 探测/FLOPs 统计安全；GPU 训练走真 DCNv2 算子

### 5. 数据集配置 ✓
- ssdc-uav.yaml 存在，nc=1（Sugarcane Seedling），与模型 yaml `nc: 1` 一致，train/val/test 路径齐全

## 次要提示（不影响执行，仅供参考）

- **SaveLastNCheckpointsCallback 的 epoch 编号错位 1 拍**：`on_train_epoch_end` 回调触发时本 epoch 的 last.pt 尚未写入（保存发生在验证之后），因此 `epoch_X.pt` 实际复制的是上一 epoch 的权重；首个 epoch 时 `trainer.last` 为 None 会抛 TypeError（被 except 捕获吞掉，无实质影响）。与其他实验脚本行为一致，不影响训练本身
- **启动目录**：`pretrained='yolo12s.pt'` 是相对路径，需从仓库根目录启动脚本（与其他已跑通实验的用法一致）
- **训练启动时的确认点**：日志应出现 "Powerful-IoU 启用成功！"，且不应出现 "no model scale passed. Assuming scale='n'" 警告；参数量应为 ~10M 级（yolo12s 9.3M + DyHead/DySample 增量）

## 结论

无需任何代码或配置修改，可直接运行：
```
python scripts/improved_train/coco_pretrained/train_yolov12-PIoU2_DySample-plus_DyHead_ssdc-uav.py
```
