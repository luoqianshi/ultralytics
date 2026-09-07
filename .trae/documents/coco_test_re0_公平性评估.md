# coco_test_re0.py 公平性评估报告与修复计划

## 一、任务摘要

评估 `scripts/coco_test_re0.py` 是否能在 Ultralytics 框架上**公平**地评测各目标检测算法在 SSDC-UAV 测试集上的性能。评估方式：静态审查脚本逻辑 + 实际核验 5 个实验的 `predictions.json`、GT 标注、生成这些预测的测试脚本（`scripts/test/*.py`）与运行日志。

## 二、当前状态核验结果（基于实际数据）

| 核验项 | 结果 |
|---|---|
| GT 文件 | `datasets/SSDC-UAV_coco/annotations/test.json`，1703 张图，单类别（Sugarcane Seedling, id=1），无 iscrowd=1 |
| COCO test.json vs YOLO test split | 文件名 1703/1703 完全一致，无差集 |
| 预测文件 | 5 个 `predictions.json`，全部由 `model.val(save_json=True)` 生成；均覆盖全部 1703 张图；min_score=0.001（默认 conf=0.001）；max_det/img≤300（默认 max_det=300） |
| image_id 匹配 | 脚本通过 `file_name` 映射，5 个文件均 100% 匹配，无丢弃 |
| 推理参数 | 6 个测试脚本的 `model.val()` 参数完全一致：`split='test', batch=16, device='0'`，均未覆盖 conf/iou/imgsz/half（即全部使用相同默认值） |
| 权重选择 | 全部测试脚本统一使用 `best.pt` |
| 评估器 | 标准 `pycocotools.COCOeval(cocoGt, cocoDt, 'bbox')`，evaluate/accumulate/summarize 全流程，无自定义 IoU/面积过滤 |

## 三、公平性结论

**结论：在当前数据与流程下，该脚本的评测是公平的、符合 COCO 官方评测协议。** 依据：

1. **统一 GT、统一测试集**：所有模型在同一份 test.json（1703 图）上评测，且与训练时使用的 YOLO test split 完全一致，无数据泄漏或集合不一致。
2. **统一评测工具与协议**：使用官方 pycocotools，12 项标准指标，AP 计算方式与 COCO benchmark 一致；预测保留 conf≥0.001、每图最多 300 框，是标准 COCO 协议（与 Ultralytics 内部 `eval_json` 等价）。
3. **统一推理协议**：所有预测由相同参数的 `model.val()` 生成（同 conf/iou/imgsz/batch/device，无 half），预测文件格式、坐标系（原图尺度）、类别映射一致。
4. **统一权重选择标准**：均用 best.pt。
5. **单类别数据集**：不存在类别不平衡导致的 AR/AP 偏置问题；image_id 映射 100% 正确，无静默丢框。

## 四、发现的问题（不影响当前结果公平性，但存在隐患）

按严重度排序：

1. **【中】无推理参数一致性校验**：脚本不检查各实验目录的 `args.yaml`。当前各实验目录未保存 args.yaml（val 输出被移动到 `runs/ssdc_uav_test_re0` 时未带出），若未来某次测试误用不同 conf/iou/imgsz/half，评测会**静默不公平**。建议：脚本在评估前扫描各实验目录，若存在 args.yaml 则校验关键参数一致，不一致时警告。
2. **【中】不匹配预测被静默丢弃**：`evaluate_coco` 中未匹配的预测直接 `pass`，无计数无警告。当前 100% 匹配无实际影响，但未来文件名不一致时会静默丢框导致指标偏低且难以察觉。建议：记录并打印丢弃数量，丢弃比例 >0 时醒目警告。
3. **【低】exp_name 仅取父目录名**：若不同子目录下出现同名实验目录，xlsx 中会出现重复行名。建议：用相对 `runs_dir` 的路径（如 `from_scratch/exp01`）作为名称。
4. **【低】TeeLogger 无异常保护**：`sys.stdout = logger` 之后若 main 中抛异常，stdout 不会恢复、日志未落盘。建议：try/finally 包裹。
5. **【低】评估顺序不稳定**：`rglob` 返回顺序不确定，xlsx 行顺序每次可能不同。建议：`sorted(pred_files)`。
6. **【低】无测试集覆盖率校验**：未验证每个预测文件是否覆盖全部 1703 张 GT 图（缺图会被 COCOeval 按漏检计，压低 recall）。当前全覆盖，建议加一行断言/警告。
7. **【提示】xlsx 列序与打印摘要列序不一致**（AP75/AP50 顺序对调），非错误，可顺手统一。

## 五、拟议修改（仅改 `scripts/coco_test_re0.py`）

1. `main()`：`pred_files = sorted(runs_dir.rglob('predictions.json'))`；用 try/finally 保证 `logger.close()`。
2. `evaluate_coco()`：
   - 统计并打印未匹配预测数量（>0 时警告）；
   - 打印预测覆盖的图像数 vs GT 图像数（不一致时警告）；
   - 若实验目录存在 `args.yaml`，读取并校验 `conf/iou/imgsz/half/max_det` 与首个实验一致，不一致打印警告。
3. `main()`：`exp_name` 改为 `pred_file.parent.relative_to(runs_dir)`（保留子目录层级，避免重名）。
4. （可选）统一 xlsx 表头列序与摘要打印顺序一致。

不改动任何训练/测试脚本，不改动数据集。

## 六、假设与决定

- 假设用户希望保持现有评测流程（外部 pycocotools 评测），因为数据集 yaml 未配置 COCO 标注路径，Ultralytics 内部 `eval_json` 不会自动触发。
- 假设"公平评测"指：同一测试集、同一推理协议、同一评测工具与协议下比较各模型——本脚本满足。
- 仅做健壮性增强，不改变任何指标计算逻辑（当前计算逻辑本身正确）。

## 七、验证步骤

1. 修改后运行 `python scripts/coco_test_re0.py`。
2. 确认 5 个实验全部成功评估，匹配率 100%、覆盖率 1703/1703 打印正常。
3. 对比 xlsx 中 mAP50-95 与各实验 `result.log` 中 Ultralytics 内部 mAP50-95（如 exp01 为 0.549），两者应基本一致（差异 <0.002，来自内部评估的 conf 截断差异属正常）。
4. 确认日志 txt 与 xlsx 正常生成于 `scripts/coco_test/`。
