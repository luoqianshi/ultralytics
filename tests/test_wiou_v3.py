"""Unit tests for Wise-IoU v3 (wiou_v3) function."""
import torch
import pytest
from ultralytics.utils.metrics import wiou_v3, bbox_iou


def test_wiou_v3_shape():
    """输出形状应与输入一致。"""
    box1 = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32)
    box2 = torch.tensor([[12, 12, 52, 52], [30, 30, 70, 70]], dtype=torch.float32)
    iou_mean = torch.tensor(0.5)
    result = wiou_v3(box1, box2, iou_mean=iou_mean)
    assert result.shape == box1.shape[:-1] + (1,) or result.shape == box1.shape


def test_wiou_v3_perfect_prediction():
    """完全重合的两个框，相似度应为 1.0（loss = 0）。"""
    box = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    iou_mean = torch.tensor(0.5)
    result = wiou_v3(box, box, iou_mean=iou_mean)
    assert torch.allclose(result, torch.ones_like(result), atol=1e-5)


def test_wiou_v3_no_overlap():
    """完全不重叠的两个框，相似度应 < 0（loss > 1）。"""
    box1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    box2 = torch.tensor([[100, 100, 110, 110]], dtype=torch.float32)
    iou_mean = torch.tensor(0.5)
    result = wiou_v3(box1, box2, iou_mean=iou_mean)
    assert result.item() < 0.0


def test_wiou_v3_gradient_flow():
    """梯度应通过 center_dist 和 L_IoU 回传，但不通过 r。"""
    box1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32, requires_grad=True)
    box2 = torch.tensor([[12, 12, 52, 52]], dtype=torch.float32)
    iou_mean = torch.tensor(0.5)
    result = wiou_v3(box1, box2, iou_mean=iou_mean)
    loss = (1 - result).sum()
    loss.backward()
    assert box1.grad is not None
    assert not torch.allclose(box1.grad, torch.zeros_like(box1.grad))


def test_wiou_v3_higher_iou_higher_similarity():
    """IoU 更高的框对应应返回更高的相似度（更低 loss）。"""
    gt = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    pred_good = torch.tensor([[11, 11, 51, 51]], dtype=torch.float32)
    pred_bad = torch.tensor([[20, 20, 60, 60]], dtype=torch.float32)
    iou_mean = torch.tensor(0.5)
    sim_good = wiou_v3(pred_good, gt, iou_mean=iou_mean)
    sim_bad = wiou_v3(pred_bad, gt, iou_mean=iou_mean)
    assert sim_good.item() > sim_bad.item()


def test_wiou_v3_xywh_format():
    """支持 xywh 格式输入。"""
    box1_xywh = torch.tensor([[30, 30, 40, 40]], dtype=torch.float32)
    box2_xywh = torch.tensor([[32, 32, 40, 40]], dtype=torch.float32)
    iou_mean = torch.tensor(0.5)
    result = wiou_v3(box1_xywh, box2_xywh, iou_mean=iou_mean, xywh=True)
    assert result.shape[0] == 1


def test_wiou_v3_consistent_with_bbox_iou_on_perfect():
    """完美预测时，wiou_v3 与 bbox_iou 都应返回 ~1.0。"""
    box = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    iou_mean = torch.tensor(0.5)
    wiou_sim = wiou_v3(box, box, iou_mean=iou_mean)
    iou_sim = bbox_iou(box, box, xywh=False)
    assert torch.allclose(wiou_sim, iou_sim, atol=1e-5)
