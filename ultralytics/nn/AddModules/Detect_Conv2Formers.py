# Detect_Conv2Formers 检测头：box 回归分支（cv2）使用 Conv2Former 模块
# 集成模式与 Detect_PPA 一致：继承原生 Detect，仅替换 cv2 分支，
# 其余行为（DFL 解码、stride 构建、bias_init、训练/推理 forward、导出、fuse）全部继承。

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

__all__ = ["Detect_Conv2Formers", "Conv2Formers"]


class LayerNorm(nn.Module):
    """From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)，支持 channels_first。"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim),
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x


class Conv2FormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.0):
        super().__init__()
        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class Conv2Formers(nn.Module):
    """CSP 风格容器：cv1 分流 + n 个 Conv2FormerBlock + cv2 聚合（可单独作为 yaml 层使用）。"""

    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.cb = nn.ModuleList(Conv2FormerBlock(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(cb(y[-1]) for cb in self.cb)
        return self.cv2(torch.cat(y, 1))


class Detect_Conv2Formers(Detect):
    """Conv2Formers 检测头：box 分支使用 Conv2Former 模块，cls 分支沿用原生 DWConv+Conv。

    与原生 Detect 的唯一区别是 cv2 分支结构，其余行为（DFL 解码、stride、
    bias_init、训练/推理 forward、导出、fuse）全部继承自 Detect。
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """参数签名与当前框架 Detect 一致，由 parse_model 以位置参数传入。"""
        super().__init__(nc, reg_max, end2end, ch)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        # box 回归分支：Conv2Former 模块（卷积调制注意力 + MLP）
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv2Formers(x, c2),
                Conv2Formers(c2, c2),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        if end2end:  # one2one 分支同步替换为 Conv2Former 结构
            self.one2one_cv2 = deepcopy(self.cv2)
