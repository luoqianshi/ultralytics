import sys
import os

# 将项目根目录添加到 sys.path，以确保能导入 ultralytics
# 假设脚本位于 scripts/ 目录下，根目录为 scripts/ 的上一级
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
from pathlib import Path
import numpy as np
import csv
from datetime import datetime

# 在这里统一记录关键的测试参数
Model = 'YOLOv12s-WIoUv3-300e/194e'  # 300 Epoch 训练, WIoU v3 替代 CIoU
Epoch = 300
Type = 'coco_pretrain'


def test():
    """
    SSDC-UAV 数据集测试脚本 (E1: YOLO12s + WIoU v3)

    功能：
    1. 加载训练好的 YOLO12s + WIoU v3 模型权重。
    2. 使用指定的配置文件在测试集上进行评估。
    3. 输出评估指标 (mAP, Precision, Recall, F1等)。
    4. 与 baseline E0' (COCO pretrain + 300ep, CIoU, mAP=0.5563) 对比。
    """

    # =========================================================================
    # 1. 路径配置
    # =========================================================================

    # 权重文件路径 (E1 训练好的 best.pt)
    weights_path = Path(r'D:\Data\New_Codes\Python_Codes\ultralytics\runs\ssdc_uav_train\yolo12s_WIoUv3_ssdc_uav_exp1\weights\best.pt')

    # 数据集配置文件路径
    dataset_yaml_path = Path(r'D:\Data\New_Codes\Python_Codes\ultralytics\datasets\SSDC-UAV_yolo\ssdc-uav.yaml')

    print("=" * 50)
    print(f"{Model} {Epoch} Epoch {Type} SSDC-UAV 测试脚本启动")
    print("=" * 50)

    # =========================================================================
    # 2. 检查文件存在性
    # =========================================================================

    if not weights_path.exists():
        print(f"[错误] 权重文件未找到: {weights_path}")
        print("请检查路径是否正确，或确认训练是否已完成并生成了 best.pt。")
        print("训练脚本: scripts/improved_train/coco_pretrained/train_yolov12-WIoUv3_ssdc-uav.py")
        return

    if not dataset_yaml_path.exists():
        print(f"[错误] 数据集配置文件未找到: {dataset_yaml_path}")
        return

    print(f"权重文件: {weights_path}")
    print(f"数据集配置: {dataset_yaml_path}")

    # =========================================================================
    # 3. 加载模型与执行测试
    # =========================================================================

    try:
        print("\n[信息] 正在加载模型...")
        model = YOLO(weights_path)

        print(f"[信息] 开始在测试集 (split='test') 上进行评估...")

        # [新增] 计算模型参数量和 GFLOPs
        n_params = sum(x.numel() for x in model.model.parameters())
        flops = get_flops(model.model, imgsz=640)

        # 运行验证/测试
        # split='test' 指示使用 yaml 中定义的 test 数据集
        # save_json=True 保存结果用于 COCO 格式评估
        # plots=True 保存混淆矩阵、PR曲线等图表
        # verbose=False 关闭 ultralytics 默认的打印，避免重复输出，我们将在最后统一输出
        metrics = model.val(
            data=str(dataset_yaml_path),
            split='test',
            save_json=True,
            plots=True,
            device='0', # 默认使用第一个 GPU
            batch=16,   # 根据显存调整
            project='runs/ssdc_uav_test', # 测试结果保存路径
            name='yolo12s_WIoUv3_ssdc_uav_exp1',
            exist_ok=True, # 允许覆盖同名实验目录
            verbose=False
        )

        # =========================================================================
        # 4. 输出结果
        # =========================================================================

        print("\n" + "="*60)
        print("测试集评估完成。综合指标如下:")
        print("="*60)

        # 基础指标
        map50 = metrics.box.map50
        map75 = metrics.box.map75
        map5095 = metrics.box.map
        precision = metrics.box.mp
        recall = metrics.box.mr

        # F1-Score (计算所有类别的平均 F1)
        # metrics.box.f1 是每个类别的 F1 数组
        f1_score = np.mean(metrics.box.f1) if len(metrics.box.f1) > 0 else 0.0

        print(f"{'指标 (Metric)':<30} | {'值 (Value)':<15}")
        print("-" * 50)
        print(f"{'Parameters (M)':<30} | {n_params / 1e6:.2f}")
        print(f"{'GFLOPS (imgsz=640)':<30} | {flops:.2f}")
        print(f"{'Precision':<30} | {precision:.5f}")
        print(f"{'Recall':<30} | {recall:.5f}")
        print(f"{'F1-Score':<30} | {f1_score:.5f}")
        print(f"{'mAP50 (IoU=0.50)':<30} | {map50:.5f}")
        print(f"{'mAP75 (IoU=0.75)':<30} | {map75:.5f}")
        print(f"{'mAP50-95 (IoU=0.50:0.95)':<30} | {map5095:.5f}")

        # COCO mAP 指标
        # 如果 save_json=True 且满足 COCO 评估条件，pycocotools 会输出 mAP small/medium/large
        # 这里提示用户查看 pycocotools 的输出，或者如果我们需要捕获它们，通常需要更复杂的逻辑
        # 因为 model.val() 返回的 metrics 对象中不包含 COCO 的 s/m/l 指标。
        print("-" * 50)
        print("[提示] COCO mAP (small/medium/large) 指标通常由 pycocotools 在上方直接输出。")
        print("       如果在上方日志中未看到 'Average Precision ... (area= small)' 等信息，")
        print("       可能是因为数据集不包含 COCO 格式的 JSON 标注文件，或未触发 eval_json。")

        print("-" * 50)
        print(f"详细测试结果 (图表、预测结果) 已保存至: {metrics.save_dir}")
        print("="*60)

        # =========================================================================
        # 与 baseline E0' 对比
        # =========================================================================
        baseline_map = 0.5563  # E0': COCO pretrain + 300ep, CIoU
        baseline_map_small = 0.3366  # E0' mAP_small
        delta = map5095 - baseline_map

        print("\n" + "=" * 60)
        print("与 baseline E0' 对比 (COCO pretrain + 300ep, CIoU)")
        print("=" * 60)
        print(f"{'Baseline mAP@.5:.95':<30} | {baseline_map:.5f}")
        print(f"{'E1 (WIoU v3) mAP@.5:.95':<30} | {map5095:.5f}")
        print(f"{'Δ mAP':<30} | {delta:+.5f}")
        # 成功标准判定 (per spec §4.3)
        if delta >= 0.020:
            verdict = "超预期 (≥ +2.0%)"
        elif delta >= 0.010:
            verdict = "达标 (+1.0% ~ +2.0%)"
        elif delta >= 0.003:
            verdict = "及格 (+0.3% ~ +1.0%)"
        else:
            verdict = "失败 (≤ +0.3%)"
        print(f"{'结论':<30} | {verdict}")
        print("=" * 60)

        # =========================================================================
        # 5. 保存结果到 CSV
        # =========================================================================
        csv_dir = Path(r'D:\Data\New_Codes\Python_Codes\ultralytics\runs\test_result')
        csv_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = csv_dir / f'SSDC-UAV_Test_Result_{timestamp}.csv'

        csv_header = [
            'Model', 'Epoch', 'Type',
            'Parameters (M)', 'GFLOPS (imgsz=640)',
            'Precision', 'Recall', 'F1-Score',
            'mAP50 (IoU=0.50)', 'mAP75 (IoU=0.75)', 'mAP50-95 (IoU=0.50:0.95)'
        ]
        csv_row = [
            Model, Epoch, Type,
            f"{n_params / 1e6:.2f}", f"{flops:.2f}",
            f"{precision * 100:.3f}", f"{recall * 100:.3f}", f"{f1_score * 100:.3f}",
            f"{map50 * 100:.3f}", f"{map75 * 100:.3f}", f"{map5095 * 100:.3f}"
        ]

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerow(csv_row)

        print(f"\n[信息] 测试结果已保存至 CSV 文件: {csv_path}")

        # =========================================================================
        # 6. 追加结果到汇总 CSV
        # =========================================================================
        summary_csv_path = csv_dir / 'SSDC-UAV_Test_Result.csv'
        file_exists = summary_csv_path.exists()
        with open(summary_csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(csv_header)
            else:
                # 先换行，确保新数据另起一行
                f.write('\n')
            writer.writerow(csv_row)

        print(f"[信息] 测试结果已追加至汇总 CSV 文件: {summary_csv_path}")

    except Exception as e:
        print(f"\n[异常] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test()
