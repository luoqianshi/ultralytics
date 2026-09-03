import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
try:
    from mmcv.cnn import ConvModule, build_norm_layer
    from mmengine.model import BaseModule
    from mmengine.model import constant_init
    from mmengine.model.weight_init import trunc_normal_init, normal_init
except ImportError as e:
    pass

# @TODO 20260903 仅导出改进模块：
# 本文件重定义了 Conv/Bottleneck/C3/C3k（与 ultralytics 内置同名但实现不同），
# 若不加 __all__，tasks.py 中 `from .AddModules import *` 会遮蔽官方模块，静默改变整个网络结构
__all__ = ["Mona", "A2C2f_Mona"]


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )
    def forward(self, x):
        AiFHG = x
        conv1_x = self.conv1_AiFHG(x)
        conv2_x = self.conv2_AiFHG(x)
        conv3_x = self.conv3_AiFHG(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + AiFHG
        AiFHG = x
        x = self.projector(x)
        return AiFHG + x

class Mona(nn.Module):
    def __init__(self,in_dim,AiFHG=4):
        super().__init__()
        self.project1_AiFHG = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2_AiFHG = nn.Linear(64, in_dim)
        self.dropout_AiFHG = nn.Dropout(p=0.1)
        self.adapter_conv_AiFHG = MonaOp(64)
        self.norm_AiFHG = nn.LayerNorm(in_dim)
        self.gamma_AiFHG = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax_AiFHG = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C, -1).transpose(-1, -2)
        AiFHG = x
        x = self.norm_AiFHG(x) * self.gamma_AiFHG + x * self.gammax_AiFHG
        project1 = self.project1_AiFHG(x)  #降维操作，减少计算量
        b, n, c = project1.shape
        h, w = H,W
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv_AiFHG(project1)  #使用多尺度卷积操作，3*3，5*5，7*7代表不同大小的卷积核
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)   #使用激活函数
        nonlinear = self.dropout_AiFHG(nonlinear)
        project2 = self.project2_AiFHG(nonlinear)  #升维操作，还原方便残差连接
        out = AiFHG + project2
        out = out.reshape(B, H, W,C).permute(0,3,1,2)
        return out

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
 
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class A2C2f_Mona(nn.Module):  
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True, den=[0.5]):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        num_heads = c_ // 32
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(
                Mona(c_) 
                for _ in range(2)
            )) if a2 else C3k(c_, c_, 2, shortcut, g) 
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + (self.gamma * self.cv2(torch.cat(y, 1)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.cv2(torch.cat(y, 1))

