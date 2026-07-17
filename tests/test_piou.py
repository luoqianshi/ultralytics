"""Unit tests for Powerful-IoU (piou) function."""
import torch
import pytest
from ultralytics.utils.metrics import piou, bbox_iou


def test_piou_shape():
    """输出形状应与输入一致。"""
    box1 = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32)
    box2 = torch.tensor([[12, 12, 52, 52], [30, 30, 70, 70]], dtype=torch.float32)
    result = piou(box1, box2, xywh=False, PIoU2=True)
    assert result.shape == box1.shape[:-1] + (1,) or result.shape == box1.shape


def test_piou_perfect_prediction():
    """完全重合的两个框，相似度应为 1.0（loss = 0）。"""
    box = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    result = piou(box, box, xywh=False, PIoU2=True)
    assert torch.allclose(result, torch.ones_like(result), atol=1e-5)


def test_piou_no_overlap():
    """完全不重叠的两个框，相似度应 < 1.0（loss > 0）。"""
    box1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    box2 = torch.tensor([[100, 100, 110, 110]], dtype=torch.float32)
    result = piou(box1, box2, xywh=False, PIoU2=True)
    assert result.item() < 1.0


def test_piou_gradient_flow():
    """梯度应正常回传。"""
    box1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32, requires_grad=True)
    box2 = torch.tensor([[12, 12, 52, 52]], dtype=torch.float32)
    result = piou(box1, box2, xywh=False, PIoU2=True)
    loss = (1 - result).sum()
    loss.backward()
    assert box1.grad is not None
    assert not torch.allclose(box1.grad, torch.zeros_like(box1.grad))


def test_piou_higher_iou_higher_similarity():
    """IoU 更高的框对应应返回更高的相似度（更低 loss）。"""
    gt = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    pred_good = torch.tensor([[11, 11, 51, 51]], dtype=torch.float32)
    pred_bad = torch.tensor([[20, 20, 60, 60]], dtype=torch.float32)
    sim_good = piou(pred_good, gt, xywh=False, PIoU2=True)
    sim_bad = piou(pred_bad, gt, xywh=False, PIoU2=True)
    assert sim_good.item() > sim_bad.item()


def test_piou_xywh_format():
    """支持 xywh 格式输入。"""
    box1_xywh = torch.tensor([[30, 30, 40, 40]], dtype=torch.float32)
    box2_xywh = torch.tensor([[32, 32, 40, 40]], dtype=torch.float32)
    result = piou(box1_xywh, box2_xywh, xywh=True, PIoU2=True)
    assert result.shape[0] == 1


def test_piou_piou_mode():
    """测试 PIoU (v1) 模式。"""
    box1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    box2 = torch.tensor([[12, 12, 52, 52]], dtype=torch.float32)
    result = piou(box1, box2, xywh=False, PIoU=True)
    assert result.shape[0] == 1


def test_piou_default_return():
    """当不指定 PIoU 或 PIoU2 时，应返回标准 IoU。"""
    box1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    box2 = torch.tensor([[12, 12, 52, 52]], dtype=torch.float32)
    piou_result = piou(box1, box2, xywh=False)
    iou_result = bbox_iou(box1, box2, xywh=False)
    assert torch.allclose(piou_result, iou_result, atol=1e-5)