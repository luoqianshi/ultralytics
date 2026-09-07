import os
import json
import re
import glob
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torchvision.ops import nms
from typing import Dict, List, Tuple

# ================= 配置区域 =================
# 真实值 CSV/Excel 文件路径
GT_PATH = r'D:\Data\New_Codes\Python_Codes\ultralytics\datasets\SSDC-UAV_Original\count_original_images_testdataset.xlsx'

# 模型测试结果根目录
# @TODO 这里更新为150 Epoch + YOLO12s + 预训练配置在SSDC-UAV数据集上的测试结果文件夹
RUNS_DIR = r'D:\Data\New_Codes\Python_Codes\ultralytics\runs\ssdc_uav_test_pretrain'

# 原始图片目录 (用于确认图片尺寸等信息，如果需要)
ORIGINAL_IMAGES_DIR = r'D:\Data\New_Codes\Python_Codes\ultralytics\datasets\SSDC-UAV_Original\images'

# NMS (非极大值抑制) 的 IoU 阈值
NMS_IOU_THRESHOLD = 0.5

# 预测框的置信度阈值 (低于此分数的框将被过滤)
CONF_THRESHOLD = 0.5

# ===========================================

def parse_tile_filename(filename: str) -> Tuple[str, int, int]:
    """
    从子图文件名中解析原始图像名称和偏移量。
    假设文件名格式为: {original_name}_tile_{id}_x{x_off}_y{y_off}.ext
    
    Args:
        filename: 子图文件名 (包含扩展名)
        
    Returns:
        tuple: (original_name, x_offset, y_offset)
    """
    # 去除扩展名
    stem = Path(filename).stem
    
    # 使用正则表达式匹配 _tile_{id}_x{x}_y{y}
    # 示例: DJI_20250511173415_0290_D_tile_0000_x0_y0
    pattern = r"(.*)_tile_\d+_x(\d+)_y(\d+)$"
    match = re.search(pattern, stem)
    
    if match:
        original_name = match.group(1)
        x_off = int(match.group(2))
        y_off = int(match.group(3))
        return original_name, x_off, y_off
    else:
        # 如果匹配失败，可能不是切片图像，或者是原图
        print(f"警告: 无法解析文件名中的坐标信息: {filename}")
        return stem, 0, 0

def load_ground_truth(gt_path: str) -> pd.DataFrame:
    """
    加载真实值数据。支持 CSV 和伪装成 CSV 的 Excel 文件。
    
    Args:
        gt_path: 文件路径
        
    Returns:
        pd.DataFrame: 包含 'image_name' (或类似) 和 'count' 的 DataFrame
    """
    print(f"正在加载真实值文件: {gt_path}")
    try:
        # 尝试作为 CSV 读取
        df = pd.read_csv(gt_path)
    except Exception as e_csv:
        print(f"CSV 读取失败，尝试作为 Excel 读取: {e_csv}")
        try:
            # 尝试作为 Excel 读取 (即使扩展名是 .csv)
            df = pd.read_excel(gt_path, engine='openpyxl')
        except Exception as e_excel:
            raise ValueError(f"无法读取真实值文件，请确保安装了 pandas 和 openpyxl。\nCSV 错误: {e_csv}\nExcel 错误: {e_excel}")
            
    # 标准化列名
    # 实际上，第一列为序号(Order)，第二列为图片文件名（File_Name），第三列为数量（Count）
    # 这里我们打印前几行供用户调试，并尝试自动识别
    print("真实值文件前 5 行:")
    print(df.head())
    
    return df

def process_model_predictions(model_dir: Path, gt_df: pd.DataFrame):
    """
    处理单个模型的预测结果。
    
    Args:
        model_dir: 模型结果目录
        gt_df: 真实值 DataFrame
    """
    model_name = model_dir.name
    predictions_json_path = model_dir / 'predictions.json'
    
    if not predictions_json_path.exists():
        print(f"跳过 {model_name}: 找不到 predictions.json")
        return

    print(f"\n开始处理模型: {model_name}")
    
    # 1. 加载预测结果
    with open(predictions_json_path, 'r') as f:
        predictions = json.load(f)
        
    print(f"已加载 {len(predictions)} 个预测框")
    
    # 2. 映射回原图坐标
    # 结构: {original_image_name: {'boxes': [], 'scores': [], 'categories': []}}
    mapped_predictions = {}
    
    for pred in predictions:
        # 过滤低置信度
        if pred['score'] < CONF_THRESHOLD:
            continue
            
        # 解析文件名 (predictions.json 通常包含 image_id，这里假设 image_id 就是文件名或能映射到文件名)
        # 注意：Ultralytics 导出的 COCO json 中，image_id 可能是数字或字符串
        # 如果是数字，我们需要 image_id 到 file_name 的映射，这通常在 predictions.json 同级的 info 中，或者我们需要自己推断
        # 但是根据用户描述，predictions.json 中应该包含文件名信息，或者 image_id 本身就是文件名
        # 通常 Ultralytics 的 predictions.json 列表项包含 'image_id'
        
        # 这里假设 image_id 是字符串格式的文件名 (如 DJI_..._tile_... )
        # 如果 image_id 是数字，这步会比较麻烦，需要额外的 json 文件来映射 id 到文件名
        # 根据之前的 `prepare_lafssdc.py`，image_id 在数据集中是字符串。
        
        image_id = pred['image_id']
        file_name = f"{image_id}.jpg" # 构造文件名用于解析
        
        original_name, x_off, y_off = parse_tile_filename(file_name)
        
        if original_name not in mapped_predictions:
            mapped_predictions[original_name] = {'boxes': [], 'scores': [], 'categories': []}
            
        # COCO bbox 格式: [x_min, y_min, width, height]
        x, y, w, h = pred['bbox']
        
        # 映射坐标
        x_mapped = x + x_off
        y_mapped = y + y_off
        
        # 存储为 XYXY 格式以便 NMS 使用 (x_min, y_min, x_max, y_max)
        mapped_predictions[original_name]['boxes'].append([x_mapped, y_mapped, x_mapped + w, y_mapped + h])
        mapped_predictions[original_name]['scores'].append(pred['score'])
        mapped_predictions[original_name]['categories'].append(pred['category_id'])

    # 保存映射后的原始预测 (可选)
    original_preds_json = []
    
    # 3. 执行 NMS 并统计
    final_preds_json = []
    counting_results = []
    
    print("正在执行 NMS 并统计数量...")
    
    for img_name, data in mapped_predictions.items():
        if not data['boxes']:
            count = 0
        else:
            boxes = torch.tensor(data['boxes'], dtype=torch.float32)
            scores = torch.tensor(data['scores'], dtype=torch.float32)
            
            # 执行 NMS
            keep_indices = nms(boxes, scores, NMS_IOU_THRESHOLD)
            
            final_boxes = boxes[keep_indices]
            final_scores = scores[keep_indices]
            final_cats = torch.tensor(data['categories'])[keep_indices]
            
            count = len(final_boxes)
            
            # 收集最终预测结果用于保存 json
            for i in range(count):
                # 转换回 XYWH 格式用于 COCO JSON
                x1, y1, x2, y2 = final_boxes[i].tolist()
                w, h = x2 - x1, y2 - y1
                
                final_preds_json.append({
                    "image_id": img_name,
                    "bbox": [x1, y1, w, h],
                    "score": float(final_scores[i]),
                    "category_id": int(final_cats[i])
                })
        
        counting_results.append({
            'image_name': img_name,
            'predicted_count': count
        })
        
    # 保存去重后的预测结果
    final_json_path = model_dir / 'final_original_predictions.json'
    with open(final_json_path, 'w') as f:
        json.dump(final_preds_json, f)
    print(f"已保存去重后的预测结果: {final_json_path}")
    
    # 4. 保存计数结果 CSV
    count_df = pd.DataFrame(counting_results)
    count_csv_path = model_dir / f'{model_name}_counting_results.csv'
    count_df.to_csv(count_csv_path, index=False)
    print(f"已保存计数结果: {count_csv_path}")
    
    # 5. 计算性能指标
    # 需要将预测结果与真实值合并
    # 真实值 CSV 的第一列是序号(Order)，第二列是图片文件名（File_Name），第三列是数量（Count）
    # 我们需要先清理真实值 DataFrame 的列名
    
    # 尝试找到匹配的列
    gt_name_col = None
    gt_count_col = None
    
    # 简单的启发式搜索列名
    for col in gt_df.columns:
        col_lower = str(col).lower()
        if any(x in col_lower for x in ['File_Name', 'image', 'file', 'id']):
            gt_name_col = col
        if any(x in col_lower for x in ['Count', 'num', 'amount', 'label']):
            gt_count_col = col
            
    # 如果没找到，默认使用第二列和第三列
    if not gt_name_col: gt_name_col = gt_df.columns[1]
    if not gt_count_col: gt_count_col = gt_df.columns[2]
    
    print(f"使用真实值列: 图片名='{gt_name_col}', 数量='{gt_count_col}'")
    
    # 确保图片名格式一致 (去掉扩展名进行匹配，或者保留)
    # 假设 GT 中的图片名不带扩展名，或者带了。这里做一个简单的处理：统一去掉扩展名比较
    gt_df['merge_key'] = gt_df[gt_name_col].astype(str).apply(lambda x: Path(x).stem)
    count_df['merge_key'] = count_df['image_name'].astype(str).apply(lambda x: Path(x).stem)
    
    merged_df = pd.merge(gt_df, count_df, on='merge_key', how='inner')
    
    if len(merged_df) == 0:
        print("警告: 预测结果与真实值合并后为空！请检查图片名称是否一致。")
        print(f"GT 示例: {gt_df['merge_key'].iloc[0] if not gt_df.empty else 'Empty'}")
        print(f"Pred 示例: {count_df['merge_key'].iloc[0] if not count_df.empty else 'Empty'}")
        return

    y_true = merged_df[gt_count_col].values
    y_pred = merged_df['predicted_count'].values
    
    # 计算指标
    # MAE计算
    mae = np.mean(np.abs(y_true - y_pred))
    # MSE计算
    mse = np.mean((y_true - y_pred) ** 2)
    
    # 决定系数 R2 计算
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"--- {model_name} 计数性能指标 ---")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"MSE (均方误差): {mse:.4f}")
    print(f"R^2 (决定系数): {r2:.4f}")
    
    # 将指标保存到文件
    metrics_path = model_dir / f'{model_name}_metrics.txt'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")

def main():
    # 检查依赖
    try:
        import pandas
        import openpyxl
    except ImportError:
        print("错误: 缺少必要的库。请运行: pip install pandas openpyxl scikit-learn")
        # 即使报错也继续尝试，可能部分功能可用
    
    # 加载真实值
    if not os.path.exists(GT_PATH):
        print(f"错误: 找不到真实值文件 {GT_PATH}")
        return
        
    gt_df = load_ground_truth(GT_PATH)
    
    # 遍历所有模型目录
    runs_path = Path(RUNS_DIR)
    if not runs_path.exists():
        print(f"错误: 找不到测试结果目录 {RUNS_DIR}")
        return
        
    # 查找所有子目录 (假设每个子目录是一个模型的结果)
    model_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("未找到模型结果目录")
        return
        
    for model_dir in model_dirs:
        process_model_predictions(model_dir, gt_df)

if __name__ == '__main__':
    main()
