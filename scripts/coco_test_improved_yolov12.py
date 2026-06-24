import os
import json
import glob
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_coco(gt_path, pred_path, model_name):
    """
    使用 pycocotools 评估单个模型的预测结果
    
    参数:
        gt_path (str): COCO 格式的 Ground Truth 标注文件路径 (test.json)
        pred_path (str): 模型生成的预测结果文件路径 (predictions.json)
        model_name (str): 模型名称，用于日志输出
    """
    print(f"\n{'='*20} 正在评估模型: {model_name} {'='*20}")
    print(f"标注文件: {gt_path}")
    print(f"预测文件: {pred_path}")

    try:
        # 1. 加载 Ground Truth
        # 为了避免重复加载导致内存占用过高，可以在外部加载，但这里为了函数独立性，先在函数内加载
        # 如果需要批量评估，建议在外部加载一次 COCO 对象传入
        cocoGt = COCO(gt_path)

        # 构建文件名到 ID 的映射
        # Ultralytics 生成的 predictions.json 中 image_id 可能是文件名字符串
        # 而 COCO GT 中通常是整数 ID，需要进行映射转换
        filename_to_id = {img['file_name']: img['id'] for img in cocoGt.dataset['images']}
        # 同时也建立 stem (不带扩展名) 到 ID 的映射，以防万一
        stem_to_id = {os.path.splitext(img['file_name'])[0]: img['id'] for img in cocoGt.dataset['images']}

        # 2. 加载预测结果并修正 image_id
        with open(pred_path, 'r') as f:
            preds = json.load(f)
        
        valid_preds = []
        for p in preds:
            # 尝试通过 file_name 匹配
            f_name = p.get('file_name')
            img_id = None
            
            if f_name in filename_to_id:
                img_id = filename_to_id[f_name]
            # 尝试通过 image_id (如果是字符串且是文件名stem) 匹配
            elif isinstance(p.get('image_id'), str) and p['image_id'] in stem_to_id:
                img_id = stem_to_id[p['image_id']]
            
            if img_id is not None:
                p['image_id'] = img_id
                valid_preds.append(p)
            else:
                # 找不到对应的图片ID，可能是多余的预测或文件名不匹配
                pass

        if not valid_preds:
            print(f"[错误] 无法匹配任何预测结果到 Ground Truth 图片。请检查文件名是否一致。")
            return

        print(f"成功加载并匹配 {len(valid_preds)} / {len(preds)} 条预测记录。")

        # loadRes 方法会自动处理 JSON 对象列表
        cocoDt = cocoGt.loadRes(valid_preds)

        # 3. 初始化评估对象
        # 'bbox' 表示检测任务
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        
        # 4. 执行评估
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # 5. 提取并打印关键指标
        # cocoEval.stats 是一个包含 12 个指标的数组
        # 0: mAP (IoU=0.50:0.95)
        # 1: mAP50 (IoU=0.50)
        # 2: mAP75 (IoU=0.75)
        # 3: mAPs (small)
        # 4: mAPm (medium)
        # 5: mAPl (large)
        stats = cocoEval.stats
        
        print("\n[COCO 评估结果摘要]")
        print(f"{'指标':<30} | {'值':<10}")
        print("-" * 45)
        print(f"{'mAP (IoU=0.50:0.95)':<30} | {stats[0]:.4f}")
        print(f"{'mAP50 (IoU=0.50)':<30} | {stats[1]:.4f}")
        print(f"{'mAP75 (IoU=0.75)':<30} | {stats[2]:.4f}")
        print(f"{'mAP_small (area < 32^2)':<30} | {stats[3]:.4f}")
        print(f"{'mAP_medium (32^2 < area < 96^2)':<30} | {stats[4]:.4f}")
        print(f"{'mAP_large (area > 96^2)':<30} | {stats[5]:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"[错误] 评估 {model_name} 时发生异常: {e}")

def main():
    # ================= 配置路径 =================
    
    # 1. Ground Truth 路径 (COCO 格式的 test.json)
    # @TODO 根据之前的分析，路径在 datasets/SSDC-UAV_coco/annotations/test.json
    gt_json_path = Path(r'D:\Data\New_Codes\Python_Codes\ultralytics\datasets\SSDC-UAV_coco\annotations\test.json')
    
    # 2. @TODO 更换测试结果根目录
    runs_dir = Path(r'D:\Data\New_Codes\Python_Codes\ultralytics\runs\ssdc_uav_test')
    
    # ===========================================

    if not gt_json_path.exists():
        print(f"[错误] 未找到标注文件: {gt_json_path}")
        return

    if not runs_dir.exists():
        print(f"[错误] 未找到测试结果目录: {runs_dir}")
        return

    print("开始批量评估 COCO 指标...")
    
    # 查找所有的 predictions.json 文件
    # 使用 rglob 递归查找
    pred_files = list(runs_dir.rglob('predictions.json'))
    
    if not pred_files:
        print(f"[警告] 在 {runs_dir} 下未找到任何 predictions.json 文件。")
        print("请确保您在运行 val/test 时设置了 save_json=True。")
        return

    print(f"找到 {len(pred_files)} 个预测文件，准备开始评估。\n")

    # 遍历评估
    for pred_file in pred_files:
        # 获取实验名称 (父目录名)
        exp_name = pred_file.parent.name
        
        # 执行评估
        evaluate_coco(str(gt_json_path), str(pred_file), exp_name)

if __name__ == "__main__":
    main()
