import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

__all__ = ["Detect_ASFF"]


class ASFFV5(nn.Module):
    def __init__(self, level, ch, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        ASFF version for YoloV5 .
        different than YoloV3
        multiplier should be 1, 0.5 which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you need change code manually.
        """
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [int(ch[2] * multiplier), int(ch[1] * multiplier),
                    int(ch[0] * multiplier)]
        # print(self.dim)

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(ch[1] * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(int(ch[0] * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                ch[2] * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(ch[2] * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(ch[0] * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(ch[1] * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(ch[2] * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(ch[1] * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(
                ch[0] * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0 = x[2]  # l
        x_level_1 = x[1]  # m
        x_level_2 = x[0]  # s
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
        #      level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print('level_0_weight_v: ', level_0_weight_v.shape)
        # print('level_1_weight_v: ', level_1_weight_v.shape)
        # print('level_2_weight_v: ', level_2_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class Detect_ASFF(Detect):
    """ASFF（Adaptive Spatial Feature Fusion）检测头。

    集成模式与 Detect_Conv2Formers 一致：继承原生 Detect（8.4.60），
    在 cv2/cv3 之前用 3 个 ASFFV5 对 P3/P4/P5 三层特征做自适应空间融合，
    其余行为（DFL 解码、stride 构建、bias_init、训练/推理 forward、导出、fuse）全部继承。
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=(), multiplier=1, rfb=False):
        """前 4 个位置参数由 parse_model 注入（与框架 Detect 签名一致）；
        multiplier/rfb 为 ASFFV5 附加参数，取默认值 1/False（yaml 中不可直接配置）。"""
        super().__init__(nc, reg_max, end2end, ch)
        self.l0_fusion = ASFFV5(level=0, ch=ch, multiplier=multiplier, rfb=rfb)  # 输出 P5 尺度, ch[2] 通道
        self.l1_fusion = ASFFV5(level=1, ch=ch, multiplier=multiplier, rfb=rfb)  # 输出 P4 尺度, ch[1] 通道
        self.l2_fusion = ASFFV5(level=2, ch=ch, multiplier=multiplier, rfb=rfb)  # 输出 P3 尺度, ch[0] 通道

    def forward(self, x):
        """x: [P3, P4, P5] 特征图（来自 yaml [[14, 17, 20]]）。"""
        p5 = self.l0_fusion(x)  # ASFFV5 level=0 -> P5 尺度
        p4 = self.l1_fusion(x)  # ASFFV5 level=1 -> P4 尺度
        p3 = self.l2_fusion(x)  # ASFFV5 level=2 -> P3 尺度
        return super().forward([p3, p4, p5])


if __name__ == "__main__":
    # Generating Sample image
    image1 = (1, 64, 32, 32)
    image2 = (1, 128, 16, 16)
    image3 = (1, 256, 8, 8)

    image1 = torch.rand(image1)
    image2 = torch.rand(image2)
    image3 = torch.rand(image3)
    image = [image1, image2, image3]
    channel = (64, 128, 256)
    # Model
    mobilenet_v1 = Detect_ASFF(nc=80, ch=channel)

    out = mobilenet_v1(image)
    print(out)
