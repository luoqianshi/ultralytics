from ultralytics import YOLO
import yaml
from pathlib import Path
import os
import shutil

class SaveLastNCheckpointsCallback:
    """
    自定义回调函数：用于保存最近 N 个 epoch 的模型权重。
    Ultralytics 默认只保存 last.pt 和 best.pt。
    此回调会在每个 epoch 结束时，将当前的 last.pt 复制为 epoch_X.pt，并维护最多 N 个历史文件。
    """
    def __init__(self, n=3):
        self.n = n
        self.saved_epochs = [] # 记录已保存的文件路径

    def on_train_epoch_end(self, trainer):
        """
        在每个训练 epoch 结束时被调用。
        """
        # 获取当前 epoch (trainer.epoch 是 0-indexed，所以加 1)
        current_epoch = trainer.epoch + 1

        # 确保权重目录存在
        weights_dir = os.path.join(trainer.save_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)

        # 定义目标文件路径
        target_path = os.path.join(weights_dir, f'epoch_{current_epoch}.pt')

        # trainer.last 是当前 last.pt 的路径
        if os.path.exists(trainer.last):
            try:
                # 复制 last.pt 到 epoch_X.pt
                shutil.copy2(trainer.last, target_path)
                self.saved_epochs.append(target_path)

                # 打印日志 (可选)
                # print(f"已保存检查点: {target_path}")

                # 如果保存数量超过 N，删除最早的一个
                if len(self.saved_epochs) > self.n:
                    to_remove = self.saved_epochs.pop(0)
                    if os.path.exists(to_remove):
                        os.remove(to_remove)
                        # print(f"已移除旧检查点: {to_remove}")
            except Exception as e:
                print(f"保存检查点时出错: {e}")

def train():
    """
    YOLO12 + FreqFusion + DyHead + Powerful-CIoU2 训练脚本（公平对比协议）
    - 数据集: SSDC-UAV
    - 训练: 300 epochs, imgsz=640, batch=16, SGD
    - 损失: Powerful-CIoU2（powerful_iou=True）
    - 随机初始化权重，从零开始训练（不加载任何预训练权重）
    - 其余超参与 train_yolov12n-PIoU2_DyHead_ssdc-uav_re0_300Epoch_mAP50.py 逐项一致，
      唯一自变量 = 模型结构（FreqFusion 直融合 neck），保证与 baseline/其它改进可比
    """
    # 1. 配置路径
    # SSDC-UAV 数据集配置（与全部对比实验一致）
    yaml_path = Path(r'D:\Data\New_Codes\Python_Codes\ultralytics\datasets\SSDC-UAV_yolo\ssdc-uav.yaml')

    print(f"使用的配置文件路径: {yaml_path}")

    # 2. 加载模型（随机初始化，from_scratch）
    # FreqFusion 直融合版：整体替代 Upsample+Concat（HR=主干跳连, LR=前向深层）
    # 如需容量对齐消融（保留 Concat 的 HRUp 版），仅换此行：
    #   yolo12n-FreqFusion_DyHead.yaml
    model = YOLO(r'D:\Data\New_Codes\Python_Codes\ultralytics\scripts\improved_yolo12\yolo12n-FreqFusionDirect_DyHead.yaml')

    # 注册自定义回调函数：保存最近 3 个 epoch 的权重
    save_callback = SaveLastNCheckpointsCallback(n=3)
    model.add_callback("on_train_epoch_end", save_callback.on_train_epoch_end)

    # 3. 开始训练
    print(f"开始使用配置文件训练: {yaml_path}")
    results = model.train(
        data=str(yaml_path),      # 数据集配置文件路径
        epochs=300,               # 与 baseline / 全部对比实验对齐
        imgsz=640,                # [对齐] 输入图像尺寸
        batch=16,                 # 批次大小 (与对比实验一致)
        project='runs/ssdc_uav_train',  # [对齐] 全部对比实验权重所在目录（测试脚本按此路径找 best.pt）
        name='yolo12n_PIoU2-FreqFusion_DyHead_ssdc_uav_exp1_re0_300Epoch_mAP50',  # 实验名称（命名沿用既有约定）
        device='0',               # 使用的 GPU 设备索引
        save=True,                # 保存 checkpoint
        optimizer='SGD',          # 使用 SGD 优化器（与对比实验一致）
        # 不加载任何预训练权重（from_scratch；与 baseline 复跑协议一致）
        # Powerful-CIoU2 配置
        powerful_iou=True,        # 启用 Powerful-CIoU2
        # 关闭确定性训练（与对比实验一致；随机种子走 ultralytics 默认 seed=0，对应命名中的 re0）
        deterministic=False,
        # 为 best_metric 设置为 mAP50
        best_metric="mAP50",
    )


if __name__ == '__main__':
    train()
