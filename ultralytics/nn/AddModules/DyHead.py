# DyHead 检测头（源自 microsoft/DynamicHead, CVPR 2021, 适配集成到 Ultralytics YOLO12）
#
# 与 Detect_XXX 系列改进的本质区别：DyHead 取消每尺度独立分支的并行设计——
# 先用 1x1 Conv 将 P3/P4/P5 统一到 hidc 通道，再经 DyHeadBlock（跨尺度可变形卷积 +
# 尺度注意力 + 任务注意力）串行融合三层特征，最后进入以 hidc 为输入重建的 cv2/cv3。
#
# 集成方式与 Detect_ASFF 一致：继承原生 Detect，仅重写 __init__ 与 forward；
# DFL 解码、stride 构建、bias_init、训练/推理输出格式、导出、fuse 等全部继承。
# DCN 算子基于 torchvision.ops.deform_conv2d（DCNv2，支持 mask 调制），无需 mmcv。

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    """Hard Sigmoid，等价 mmcv 的 HSigmoid(bias=3.0, divisor=6.0)。"""

    def __init__(self, inplace=True, h_max=1):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class DyReLU(nn.Module):
    """任务注意力：基于输入自适应生成每通道 ReLU 的缩放/偏置参数（Dynamic ReLU）。"""

    def __init__(self, inp, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
                 init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
        super().__init__()
        self.oup = inp
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        if reduction == 4:
            squeeze = inp // reduction
        else:
            squeeze = _make_divisible(inp // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, self.oup * self.exp),
            h_sigmoid(),
        )
        if use_spatial:
            self.spa = nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
        else:
            self.spa = None

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]

            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:  # bias but not PL
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1

            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)

        elif self.exp == 1:
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3
            out = out * ys

        return out


class DyDCNv2(nn.Module):
    """DyHead 使用的 ModulatedDeformConv2d(DCNv2) + GroupNorm。

    基于 torchvision.ops.deform_conv2d 实现（传入 mask 即 DCNv2），无需 mmcv。
    卷积参数由内部 nn.Conv2d 承载（不调用其 forward），以兼容 state_dict/优化器/DDP。
    """

    def __init__(self, in_channels, out_channels, stride=1, use_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=not use_norm)
        self.norm = nn.GroupNorm(16, out_channels) if use_norm else None

    def forward(self, x, offset, mask):
        if not x.is_cuda:
            # 20260823 规避：torchvision 0.16 的 deform_conv2d CPU 内核在大通道+小空间特征
            # （如构建期 get_flops 的 128x128 输入 -> P5 为 4x4）下触发访问冲突(0xC0000005)，
            # 进程无报错直接退出。CPU 路径仅用于 stride 探测/FLOPs 统计等构建期计算，
            # 只依赖输出形状，故降级为普通卷积（形状与 DCN 完全一致）；GPU 路径保持 DCNv2 真算子。
            # 注意：CPU 推理 DyHead 时 DCN 语义退化为普通卷积，推理/训练请使用 GPU。
            x = F.conv2d(
                x, self.conv.weight, self.conv.bias,
                stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,
            )
            return self.norm(x) if self.norm is not None else x
        x = deform_conv2d(
            x.contiguous(),
            offset.contiguous(),
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            mask=mask.contiguous(),
        )
        if self.norm is not None:
            x = self.norm(x)
        return x


class DyHeadBlock(nn.Module):
    """DyHead Block：跨尺度（高/中/低层）可变形卷积 + 尺度注意力 + 任务注意力。

    HSigmoid 等价官方 act_cfg（bias=3.0, divisor=6.0）。
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
    """

    def __init__(self, in_channels, zero_init_offset=True):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        self.spatial_conv_high = DyDCNv2(in_channels, in_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, in_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, in_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1),
            nn.ReLU(inplace=True),
            h_sigmoid(),
        )
        self.task_attn_module = DyReLU(in_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.zero_init_offset:
            nn.init.zeros_(self.spatial_conv_offset.weight)
            nn.init.zeros_(self.spatial_conv_offset.bias)

    def forward(self, x):
        """x: [P3, P4, P5] 特征图 list，返回融合后同结构 list。"""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, : self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # torchvision 的 deform_conv2d 要求 offset 与卷积输出同尺寸，无法复刻 mmcv
                # "先卷积后上采样"的怪异顺序（https://github.com/microsoft/DynamicHead/issues/25），
                # 改为等价做法：先将高层特征上采样对齐到当前层网格，再做可变形卷积精修
                high_input = F.interpolate(
                    x[level + 1], size=x[level].shape[-2:], mode="bilinear", align_corners=True
                )
                high_feat = self.spatial_conv_high(high_input, offset, mask)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))

        return outs


class DyHead(Detect):
    """YOLO Detect 检测头的 DyHead 改进（DynamicHead）。

    流程：1x1 通道对齐（P3/P4/P5 -> hidc）-> DyHeadBlock × block_num 跨尺度融合 ->
    重建的 cv2/cv3（输入统一为 hidc）-> Detect 原生 DFL 解码与输出。
    其余行为（stride 构建、bias_init、训练/推理 forward、导出、fuse）全部继承 Detect。
    """

    def __init__(self, nc=80, hidc=256, block_num=2, reg_max=16, end2end=False, ch=()):
        """前 3 个参数来自 yaml args（[nc, hidc, block_num]），后 3 个由 parse_model 注入。

        对应 yaml 写法: [[14, 17, 20], 1, DyHead, [nc, 128, 1]]
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.conv = nn.ModuleList(Conv(x, hidc, 1) for x in ch)  # 1x1 通道对齐
        self.dyhead = nn.Sequential(*[DyHeadBlock(hidc) for _ in range(block_num)])
        # 重建 cv2/cv3：输入通道由 ch 改为统一 hidc（结构沿用 Detect 非 legacy 分支）
        c2, c3 = max(16, hidc // 4, self.reg_max * 4), max(hidc, min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(hidc, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for _ in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(hidc, c3, 3), Conv(c3, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for _ in ch
        )
        if end2end:  # Detect.__init__ 的深拷贝发生在 cv2/cv3 重建前，这里对重建后的头重新深拷贝
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
        x = self.dyhead(x)  # list[Tensor] -> list[Tensor]，nn.Sequential 对 list 透传
        return super().forward(x)  # 训练: dict(boxes, scores, feats)；推理: (y, preds)
