import torch.nn as nn
import torch

# 本文件定义了 swish（通用激活名），若不加 __all__，tasks.py 中 `from .AddModules import *`
# 会把 swish 一并导出，未来任何新增模块若也定义 swish 即会静默互相覆盖。
__all__ = ["BiFPN"]


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class BiFPN(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = swish()
        self.epsilon = 0.0001

    def forward(self, x):
        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon) 
        weighted_feature_maps = [weights[i] * x[i] for i in range(len(x))]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)
        return result