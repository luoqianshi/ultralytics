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

    def on_fit_epoch_end(self, trainer):
        """
        在每个训练 epoch 结束时被调用（含验证与 save_model 之后）。

        注意：必须挂在 on_fit_epoch_end 而非 on_train_epoch_end——
        后者触发时本 epoch 的 last.pt 尚未写入（save_model 在其后执行），
        会导致 epoch_N.pt 实际保存的是第 N-1 个 epoch 的权重。
        """
        # 获取当前 epoch (trainer.epoch 是 0-indexed，所以加 1)
        current_epoch = trainer.epoch + 1

        # 防重入：训练结束后的 final_eval() 会以 epoch+1 额外触发一次
        # on_fit_epoch_end，跳过超出总轮数的调用，避免多存一个 epoch 文件
        if current_epoch > trainer.epochs:
            return

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
    YOLOv12-DySample-DyHead-EMA 训练脚本
    用于在 SSDC-UAV 数据集上训练「颈部三输出接 EMA + DySample 上采样 + DyHead 检测头」的改进模型。
    结构变更（相对 yolo12-DySample_DyHead.yaml）：
      颈部层 14/17/20 之后各插入一个 EMA 注意力（层 21/22/23），DyHead 改从 EMA 输出层接收输入。
    """
    # 1. 配置路径
    yaml_path = Path(r'D:\Data\New_Codes\Python_Codes\ultralytics\datasets\SSDC-UAV_yolo\ssdc-uav.yaml')

    print(f"使用的配置文件路径: {yaml_path}")

    # 2. 加载模型
    model = YOLO(r'D:\Data\New_Codes\Python_Codes\ultralytics\scripts\improved_yolo12\yolo12s-DySample_DyHead_EMA.yaml')

    # 注册自定义回调函数：保存最近 3 个 epoch 的权重
    # 注意：挂在 on_fit_epoch_end（save_model 之后），确保复制到的是本 epoch 的 last.pt
    save_callback = SaveLastNCheckpointsCallback(n=3)
    model.add_callback("on_fit_epoch_end", save_callback.on_fit_epoch_end)

    # 3. 开始训练
    print(f"开始使用配置文件训练: {yaml_path}")
    results = model.train(
        data=str(yaml_path),      # 数据集配置文件路径
        epochs=150,               # [对齐] 与 baseline 及其他改进实验保持一致 (E0': 150ep)
        imgsz=640,                # [对齐] 输入图像尺寸
        batch=16,                 # 批次大小 (显存允许的情况下尽量大，16是官方推荐)
        project='runs/ssdc_uav_train',   # 训练结果保存的项目目录
        name='yolo12_PIoU2_DySample_DyHead_EMA_ssdc_uav_exp01',  # 实验名称
        device='0',               # 使用的 GPU 设备索引
        save=True,                # 保存 checkpoint
        optimizer='SGD',          # 使用 SGD 优化器
        pretrained=r'yolo12s.pt',
        # Powerful-IoU2 配置 (新增)
        powerful_iou=True,        # 启用 Powerful-IoU2
        # 关闭确定性训练，防止报警输出
        deterministic=False,
    )


if __name__ == '__main__':
    train()
