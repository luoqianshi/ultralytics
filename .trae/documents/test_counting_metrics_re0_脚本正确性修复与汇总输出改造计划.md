# test_counting_metrics_re0_300Epoch.py 正确性修复与汇总输出改造计划

## 一、概要

对 [test_counting_metrics_re0_300Epoch.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/test_counting_metrics_re0_300Epoch.py) 做两件事：

1. **正确性修复**：当前脚本在 `runs\ssdc_uav_test_re0` 上完全无法产出结果（目录遍历层级错误），另存在指标计算偏差（inner merge 丢图）和列名匹配大小写 bug。
2. **汇总输出改造**：将所有模型的计数结果集中输出到 `scripts/counting_test_re0/` 文件夹，并生成完整汇总 xlsx（Sheet1 指标总表 + Sheet2 每图计数对比矩阵）。

---

## 二、现状分析（正确性审查结论）

### 致命问题：目录遍历层级错误（脚本跑不出任何结果）

- 脚本 L293：`model_dirs = [d for d in runs_path.iterdir() if d.is_dir()]` 只遍历**一层**子目录。
- 实际 `runs\ssdc_uav_test_re0` 是**两层嵌套**结构：

```
runs\ssdc_uav_test_re0\
├── from_scratch\
│   ├── yolo12s_ssdc_uav_re0_exp01\            → predictions.json (23.2 MB)
│   └── yolo12s_ssdc_uav_re0_exp02_300Epoch\   → predictions.json (20.3 MB)
└── from_scratch_new_improve\
    ├── yolo12s_DySample-plus_ssdc_uav_exp02_300Epoch\        → predictions.json (22.9 MB)
    └── yolo12s_PIoU2-DySample_DyHead_ssdc_uav_exp1_300Epoch\ → predictions.json (21.1 MB)
```

- 结果：脚本把 `from_scratch`、`from_scratch_new_improve` 两个**分组目录**当作模型目录，其下没有直接的 `predictions.json`，4 个真实模型**全部触发 L101-102 "跳过"**。整个 runs 树中无任何 counting 产物（无 `*_counting_results.csv` / `*_metrics.txt` / `final_original_predictions.json`）佐证此脚本从未成功运行。

### 指标计算偏差：inner merge 丢弃无预测图像

- L235：`pd.merge(gt_df, count_df, on='merge_key', how='inner')`。
- `mapped_predictions` 只包含**至少有 1 个 conf≥0.5 预测**的原图。若某原图的所有预测都被置信度阈值过滤（模型完全未检出），该图不会出现在 `count_df` 中，inner merge 后被**静默丢弃**。
- 后果：MAE/MSE/R² 基于不全的样本计算，**指标偏乐观**。GT 共 30 张测试图，正确做法是以 GT 为基准 left merge，缺失计数填 0。

### 潜在 bug：列名匹配大小写不一致

- L219/L221：`col_lower = str(col).lower()` 后与 `['File_Name', ...]`、`['Count', ...]` 比较，列表项含大写字母永远无法匹配小写的 `col_lower`。
- 当前 GT 文件第三列恰好是 Count 列，靠 L225-226 的 `columns[2]` fallback **碰巧**得到正确结果；列顺序一旦变化即出错。

### 次要问题（不影响结果，顺手清理）

- L4：`import glob` 未使用。
- L153：`original_preds_json = []` 声明后从未使用（死代码）。
- L232：`gt_df['merge_key'] = ...` 原地修改共享 DataFrame，多模型循环时重复写入（覆盖同列，无报错但不干净）。

### 确认正确的部分（不动）

- 切片文件名解析正则 `(.*)_tile_\d+_x(\d+)_y(\d+)$`，与实际文件名（如 `DJI_20250511173415_0290_D_tile_0000_x0_y0`）匹配。
- 坐标映射链路：COCO bbox `[x, y, w, h]` → 加切片偏移 → XYXY → NMS → XYWH 存储，转换正确。
- 跨切片 NMS 去重核心逻辑（conf≥0.5 预过滤 + IoU 0.5 NMS）正确，这是脚本的核心目的。
- `predictions.json` 的 `image_id` 为字符串 stem（已实际验证样例），`f"{image_id}.jpg"` 构造文件名后 `Path.stem` 还原，逻辑成立。
- R² 的 `ss_tot==0` 边界处理、`merge_key` 用 stem 去扩展名匹配，均正确。
- `NMS_IOU_THRESHOLD=0.5`、`CONF_THRESHOLD=0.5` 属用户研究决策，不修改。

---

## 三、修改方案

只修改一个文件：[scripts/test_counting_metrics_re0_300Epoch.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/test_counting_metrics_re0_300Epoch.py)（参照 [coco_test_re0.py](file:///d:/Data/New_Codes/Python_Codes/ultralytics/scripts/coco_test_re0.py) 的 rglob + 汇总 xlsx 惯例）。

### 改动 1：配置区新增集中输出目录（L12-29 区域）

```python
# 计数结果汇总输出目录（所有模型结果集中于此）
OUTPUT_DIR = r'D:\Data\New_Codes\Python_Codes\ultralytics\scripts\counting_test_re0'
```

同时删除 `import glob`（L4），新增 `from datetime import datetime`。

### 改动 2：新增 GT 列识别独立函数

将 L213-228 的启发式列名匹配逻辑提取为模块级函数 `identify_gt_columns(gt_df) -> (gt_name_col, gt_count_col)`，并把匹配列表全部改为小写：`['file_name', 'image', 'file', 'id']`、`['count', 'num', 'amount', 'label']`。fallback 逻辑（`columns[1]` / `columns[2]`）保留。main 中调用一次，结果作为参数传递，避免每个模型重复识别。

### 改动 3：process_model_predictions 改造

**签名**：`process_model_predictions(model_dir, gt_df, output_dir, gt_name_col, gt_count_col) -> (metrics_dict, count_map) or None`

**函数体修改点**：

1. 开头 `gt_df = gt_df.copy()`（避免原地污染共享 DataFrame）；merge_key 在 main 中已预先计算好（见改动 5），函数内删除 L232 的重复赋值。
2. L235：`how='inner'` → `how='left'`。
3. L244 前补：`y_pred = merged_df['predicted_count'].fillna(0).values`（无预测的图计数为 0）。
4. L196-205 三个输出路径改到集中目录（文件名加模型名前缀防冲突）：
   - `output_dir / f'{model_name}_final_original_predictions.json'`
   - `output_dir / f'{model_name}_counting_results.csv'`
   - `output_dir / f'{model_name}_metrics.txt'`
5. 删除 L153 死变量 `original_preds_json = []`。
6. 计算完指标后，返回 `(metrics_dict, count_map)`：
   - `metrics_dict = {'Model': model_name, 'MAE': mae, 'MSE': mse, 'R2': r2}`
   - `count_map = dict(zip(merged_df['merge_key'], y_pred))`（已含 fillna(0) 的全部 GT 图）

### 改动 4：新增汇总 xlsx 生成函数

```python
def save_summary_xlsx(all_metrics, all_counts, gt_df, gt_name_col, gt_count_col, output_dir):
```

- 输出路径：`output_dir / f'SSDC-UAV-Re0-Counting_{datetime.now():%Y%m%d}.xlsx'`（命名参考 coco_test_re0.py 的 `SSDC-UAV-Re0_{YYYYMMDD}.xlsx` 惯例）。
- **Sheet1 "Metrics"**：每模型一行，列为 `Model / MAE / MSE / R2`，数值保留 4 位小数（与 txt 一致）。
- **Sheet2 "PerImage"**：行 = GT 全部 30 张原图，列为 `Image / GT_Count / {各模型名}`；每模型计数取 `count_map.get(key, 0)`，即无预测填 0，与指标计算口径一致。
- 使用 `pd.ExcelWriter(engine='openpyxl')` 写入两个 sheet（pandas/openpyxl 均为已验证可用依赖）。

### 改动 5：main() 重写（L270-300）

```python
def main():
    # 依赖检查、GT 加载逻辑保持不变
    ...
    runs_path = Path(RUNS_DIR)
    if not runs_path.exists():
        ...
    # 修复致命问题：递归发现所有 predictions.json（兼容两层嵌套）
    pred_files = sorted(runs_path.rglob('predictions.json'))
    if not pred_files:
        print("未找到任何 predictions.json")
        return

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GT 列识别 + merge_key 只计算一次
    gt_name_col, gt_count_col = identify_gt_columns(gt_df)
    gt_df['merge_key'] = gt_df[gt_name_col].astype(str).apply(lambda x: Path(x).stem)

    all_metrics, all_counts = [], {}
    for pred_file in pred_files:
        result = process_model_predictions(pred_file.parent, gt_df, output_dir, gt_name_col, gt_count_col)
        if result:
            metrics_dict, count_map = result
            all_metrics.append(metrics_dict)
            all_counts[pred_file.parent.name] = count_map

    if all_metrics:
        save_summary_xlsx(all_metrics, all_counts, gt_df, gt_name_col, gt_count_col, output_dir)
```

---

## 四、假设与决策

| # | 假设/决策 | 依据 |
|---|---|---|
| 1 | 4 个模型目录名互不相同，文件名前缀不会冲突 | 已实际列出 4 个目录名验证 |
| 2 | GT xlsx 为 3 列结构（Order/File_Name/Count），stem 匹配有效 | 探索确认 + 脚本注释 |
| 3 | 不修改 `CONF_THRESHOLD`/`NMS_IOU_THRESHOLD` | 用户研究决策，非代码缺陷 |
| 4 | 只改 re0 版脚本，不动 `test_counting_metrics_pretrain_150Epoch.py` 及 coco_test 系列 | 用户仅要求此脚本 |
| 5 | 汇总形式采用用户确认的"文件夹 + 完整汇总 xlsx（指标总表 + 每图对比矩阵）" | AskUserQuestion 确认 |
| 6 | 不引入 TeeLogger 日志（coco_test 有，但计数脚本输出量小，终端即够） | 最小改动原则 |

## 五、验证步骤

1. 运行 `python scripts/test_counting_metrics_re0_300Epoch.py`。
2. 终端确认：**无"跳过"输出**，4 个模型（exp01、exp02_300Epoch、DySample-plus、PIoU2-DySample_DyHead）均打印 MAE/MSE/R²。
3. 检查 `scripts/counting_test_re0/` 产出清单：4 组模型文件（`*_counting_results.csv` / `*_metrics.txt` / `*_final_original_predictions.json`）+ 1 个 `SSDC-UAV-Re0-Counting_{YYYYMMDD}.xlsx`。
4. 打开 xlsx 验证：Metrics sheet 有 4 行指标；PerImage sheet 有 **30 行**（GT 全量图像，验证 left merge 修复生效）。
5. 抽查任意模型 CSV：行数 = 30（若存在无预测图像，计数为 0 而非缺行）。
6. 指标 sanity check：MAE ≥ 0，R² ≤ 1，各模型指标量级合理。
