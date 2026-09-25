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

    def on_model_save(self, trainer):
        """
        在 save_model() 成功写入 last.pt 后被调用（trainer.run_callbacks("on_model_save")）。
        注意：不能挂 on_train_epoch_end——该事件在 save_model() 之前触发，
        会导致 epoch_X.pt 内容与文件名错位一个 epoch（且第 1 个 epoch 因 last.pt
        尚不存在而被静默跳过）。
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
    YOLO12n + NeckSimAM 训练脚本
    - 随机初始化模型参数
    - 数据集: SSDC-UAV
    - 训练: 300 epochs, imgsz=640, batch=16, SGD
    - 损失: 默认 CIoU+DFL（未启用 powerful_iou）
    - 为 best_metric 设置为 mAP50
    """
    # 1. 配置路径
    # 指定用户提供的 dataset.yaml 配置文件路径
    # 请确保此路径指向正确的数据集配置文件
    # 注意：根据用户之前的输入，路径在 datasets/SSDC-UAV_yolo 下

    yaml_path = Path(r'D:\Data\New_Codes\Python_Codes\ultralytics\datasets\SSDC-UAV_yolo\ssdc-uav.yaml')

    print(f"使用的配置文件路径: {yaml_path}")

    # 2. 加载模型
    model = YOLO(r'D:\Data\New_Codes\Python_Codes\ultralytics\scripts\improved_yolo12\yolo12n-NeckSimAM.yaml')

    # 注册自定义回调函数：保存最近 3 个 epoch 的权重
    # 注意：必须挂 on_model_save（save_model() 成功写盘后触发），挂 on_train_epoch_end
    # 会早于 last.pt 写入，导致 epoch_X.pt 内容错位一个 epoch
    save_callback = SaveLastNCheckpointsCallback(n=3)
    model.add_callback("on_model_save", save_callback.on_model_save)

    # 3. 开始训练

    print(f"开始使用配置文件训练: {yaml_path}")
    results = model.train(
        data=str(yaml_path),      # 数据集配置文件路径
        epochs=300,               # 与 baseline 对齐 (E0': 300ep)
        imgsz=640,                # [对齐] 输入图像尺寸
        batch=16,                 # 批次大小 (显存允许的情况下尽量大，16是官方推荐)
        project='runs/ssdc_uav_train',   # 训练结果保存的项目目录
        name='yolo12n_NeckSimAM_ssdc_uav_exp1_re0_300Epoch_mAP50',  # 实验名称
        device='0',               # 使用的 GPU 设备索引
        save=True,                # 保存 checkpoint
        optimizer='SGD',          # 使用 SGD 优化器
        # pretrained='yolo12s.pt',  # 加载 COCO 预训练权重
        # 关闭确定性训练
        deterministic=False,
        # 为 best_metric 设置为 mAP50
        best_metric="mAP50",
    )


if __name__ == '__main__':
    train()
