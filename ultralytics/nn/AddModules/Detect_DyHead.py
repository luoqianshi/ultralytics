# DyHead 检测头（基于框架原生 Detect 改进）：
#   - cv2 (box 回归分支) 使用 DynamicConv(timm CondConv2d) 动态卷积替换标准卷积
#   - cv3 (cls 分类分支) 使用 DWConv + Conv 结构（与原生非 legacy Detect 相同）
#   - forward / 损失 / stride 初始化 / 导出 / fuse 等逻辑完全复用原生 Detect，
#     保证与当前框架（dict 输出、(nc, reg_max, end2end, ch) 签名）兼容
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import CondConv2d

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect

__all__ = ["Detect_dyhead"]


class DynamicConv(nn.Module):
    """Dynamic Conv layer: 基于输入内容自适应路由多个专家卷积核 (CondConv)。"""

    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding="", dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                                     groups, bias, num_experts)

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        return self.cond_conv(x, routing_weights)


class Detect_dyhead(Detect):
    """DyHead 检测头：box 分支使用 DynamicConv，cls 分支使用 DWConv+Conv。

    与原生 Detect 的唯一区别是头部分支结构，其余行为（DFL 解码、stride、
    bias_init、训练/推理 forward、导出、fuse）全部继承自 Detect。
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """参数签名与当前框架 Detect 一致，由 parse_model 以位置参数传入。"""
        super().__init__(nc, reg_max, end2end, ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        # box 回归分支：DynamicConv 动态卷积（DyHead 核心改进）
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DynamicConv(x, c2),
                DynamicConv(c2, c2),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        # cls 分类分支：DWConv + Conv
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        if end2end:
            # super().__init__ 的 one2one 头按标准 cv2/cv3 深拷贝，
            # 替换分支后需基于新结构重建，避免 one2one 残留标准头
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
