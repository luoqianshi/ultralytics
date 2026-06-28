"""
E1 消融实验：YOLO12s + NWD only（无 P2 head，无 DySample）
- 目的：验证 NWD 损失单独是否能在 SSDC-UAV 上提升 mAP
- 配置：原 yolo12s 架构（3-scale P3/P4/P5）+ NWD 损失（nwd=True, nwd_c=10.0, nwd_alpha=1.0）
- 训练：150 epoch, imgsz=640, batch=16, SGD, 从随机初始化（re0 = from zero）
- 数据：SSDC-UAV（单类甘蔗幼苗检测）
- Baseline: yolo12s mAP@.5:.95 = 0.5487
"""
from ultralytics import YOLO
from pathlib import Path
import os
import shutil


class SaveLastNCheckpointsCallback:
    """
    自定义回调函数：保存最近 N 个 epoch 的模型权重。
    Ultralytics 默认只保存 last.pt 和 best.pt。
    此回调会在每个 epoch 结束时将 last.pt 复制为 epoch_X.pt，并维护最多 N 个历史文件。
    """

    def __init__(self, n=3):
        self.n = n
        self.saved_epochs = []

    def on_train_epoch_end(self, trainer):
        current_epoch = trainer.epoch + 1
        weights_dir = os.path.join(trainer.save_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        target_path = os.path.join(weights_dir, f"epoch_{current_epoch}.pt")
        if os.path.exists(trainer.last):
            try:
                shutil.copy2(trainer.last, target_path)
                self.saved_epochs.append(target_path)
                if len(self.saved_epochs) > self.n:
                    to_remove = self.saved_epochs.pop(0)
                    if os.path.exists(to_remove):
                        os.remove(to_remove)
            except Exception as e:
                print(f"保存检查点时出错: {e}")


def train():
    """E1: YOLO12s + NWD only 训练脚本。"""
    # 1. 数据集配置路径（与现有实验对齐）
    yaml_path = Path(r"D:\Data\New_Codes\Python_Codes\ultralytics\datasets\SSDC-UAV_yolo\ssdc-uav.yaml")
    print(f"使用的配置文件路径: {yaml_path}")

    # 2. 从 yaml 实例化模型（不加载预训练权重 = re0）
    model = YOLO(r"scripts/improved_yolo12/yolo12s-NWD.yaml")

    # 注册自定义回调：保存最近 3 个 epoch 的权重
    save_callback = SaveLastNCheckpointsCallback(n=3)
    model.add_callback("on_train_epoch_end", save_callback.on_train_epoch_end)

    # 3. 训练 - 关键：开启 NWD
    print(f"开始 E1 训练: YOLO12s + NWD only")
    results = model.train(
        data=str(yaml_path),
        epochs=150,               # 与 baseline 及 DETR 实验对齐
        imgsz=640,                # 输入图像尺寸
        batch=16,                 # 批次大小
        optimizer="SGD",          # 使用 SGD 优化器
        device="0",               # GPU 设备索引
        save=True,                # 保存 checkpoint
        # NWD 配置（透传到 v8DetectionLoss）
        nwd=True,
        nwd_c=10.0,
        nwd_alpha=1.0,
        # 实验输出
        project="runs/ssdc_uav_train",
        name="yolov12-NWD_ssdc-uav_re0_exp01",
        exist_ok=True,
    )
    print(f"E1 训练完成: {results}")


if __name__ == "__main__":
    train()
