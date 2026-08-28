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

                # 如果保存数量超过 N，删除最早的一个
                if len(self.saved_epochs) > self.n:
                    to_remove = self.saved_epochs.pop(0)
                    if os.path.exists(to_remove):
                        os.remove(to_remove)
            except Exception as e:
                print(f"保存检查点时出错: {e}")

def train():
    """
    YOLOv12-FreqFusion_hrup 训练脚本
    用于在 SSDC-UAV 数据集上训练 YOLOv12-FreqFusion_hrup 模型（双输入式 FreqFusion 上采样器，
    新增从主干网络引入HR分支，引入原生的高分辨率特征图，用于补足边界位置信息，以提高难例检测能力；
    类 DySample 即插即用集成：仅替换官方 head 第 9/12 层的 nn.Upsample，保留 Concat，
    层索引与官方对齐，yolo12s.pt 预训练迁移率与 DySample 版一致）。
    训练协议逐项对齐 DySample 组，构成同位上采样算子的公平对比。
    """
    # 1. 配置路径
    # 指定用户提供的 dataset.yaml 配置文件路径
    # 请确保此路径指向正确的数据集配置文件
    # 注意：根据用户之前的输入，路径在 datasets/SSDC-UAV_yolo 下

    yaml_path = Path(r'D:\Data\New_Codes\Python_Codes\ultralytics\datasets\SSDC-UAV_yolo\ssdc-uav.yaml')

    print(f"使用的配置文件路径: {yaml_path}")

    # 2. 加载模型
    # FreqFusionUpsample（单输入即插即用适配器）替代 Upsample，head 层索引 9-21 与官方对齐，
    # 预训练迁移率与 DySample 版一致（对比 DySample 运行日志 "Transferred 697/697"）。
    model = YOLO(r'D:\Data\New_Codes\Python_Codes\ultralytics\scripts\improved_yolo12\yolo12s-FreqFusion_hrup.yaml')
    # 使用官方 Model.load 机制：内部走 load_checkpoint → BaseModel.load → intersect_dicts + strict=False
    # 自动处理 EMA 权重、键名/形状过滤、DDP overrides、迁移比例日志
    # model.load('yolo12s.pt')  # 会在控制台打印 "Transferred X/Y items from pretrained weights"

    # 注册自定义回调函数：保存最近 3 个 epoch 的权重
    save_callback = SaveLastNCheckpointsCallback(n=3)
    model.add_callback("on_train_epoch_end", save_callback.on_train_epoch_end)

    # 3. 开始训练
    # - Optimizer: SGD
    # - Epochs: 150
    # - Input Size: imgsz=640, batch=16

    print(f"开始使用配置文件训练: {yaml_path}")
    results = model.train(
        data=str(yaml_path),      # 数据集配置文件路径
        epochs=150,               # [对齐] 与 DySample 组保持一致
        imgsz=640,                # [对齐] 输入图像尺寸
        batch=16,                 # 批次大小 (显存允许的情况下尽量大，16是官方推荐)
        project='runs/ssdc_uav_train',   # 训练结果保存的项目目录
        name='yolo12s_FreqFusion_hrup_ssdc_uav_exp01',# 实验名称
        device='0',               # 使用的 GPU 设备索引
        save=True,                # 保存 checkpoint
        optimizer='SGD',          # 使用 SGD 优化器
        pretrained='yolo12s.pt',  # 预训练权重路径
        # 关闭确定性训练(防止命令行不断打印Warning信息)
        deterministic=False,
    )


if __name__ == '__main__':
    train()
