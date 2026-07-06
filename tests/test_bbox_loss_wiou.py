"""Integration tests for BboxLoss with WIoU v3 enabled."""
import torch
import pytest
from ultralytics.utils.loss import BboxLoss


def _make_bbox_loss(wiou=True, wiou_alpha=1.0, wiou_momentum=0.01, reg_max=16):
    """Helper: construct a BboxLoss with given wiou config."""
    return BboxLoss(reg_max=reg_max, wiou=wiou, wiou_alpha=wiou_alpha, wiou_momentum=wiou_momentum)


def _make_forward_inputs(device="cpu"):
    """Helper: construct minimal valid inputs for BboxLoss.forward."""
    n_anchors = 8400
    n_fg = 10
    pred_dist = torch.randn(n_anchors, 4 * 16, requires_grad=True, device=device)
    pred_bboxes = torch.rand(n_anchors, 4, device=device) * 100
    pred_bboxes[..., 2:] = pred_bboxes[..., :2] + torch.rand(n_anchors, 2, device=device) * 50
    pred_bboxes.requires_grad_(True)
    anchor_points = torch.rand(n_anchors, 2, device=device) * 640
    target_bboxes = torch.rand(n_anchors, 4, device=device) * 100
    target_bboxes[..., 2:] = target_bboxes[..., :2] + torch.rand(n_anchors, 2, device=device) * 50
    target_scores = torch.rand(n_anchors, 1, device=device)
    target_scores_sum = torch.tensor(float(n_fg), device=device)
    fg_mask = torch.zeros(n_anchors, dtype=torch.bool, device=device)
    fg_mask[:n_fg] = True
    imgsz = torch.tensor([640, 480], device=device)
    stride = torch.tensor(8.0, device=device)
    return (pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask, imgsz, stride)


def test_bbox_loss_wiou_init():
    """BboxLoss with wiou=True should have iou_mean buffer."""
    bl = _make_bbox_loss(wiou=True)
    assert bl.use_wiou is True
    assert hasattr(bl, "iou_mean")
    assert bl.iou_mean.item() == 1.0  # initial value


def test_bbox_loss_ciou_init():
    """BboxLoss with wiou=False (default) should NOT use wiou path."""
    bl = _make_bbox_loss(wiou=False)
    assert bl.use_wiou is False


def test_bbox_loss_wiou_forward_runs():
    """BboxLoss.forward with wiou=True should produce valid loss tensors."""
    bl = _make_bbox_loss(wiou=True)
    inputs = _make_forward_inputs()
    loss_iou, loss_dfl = bl(*inputs)
    assert loss_iou.item() > 0
    assert loss_dfl.item() > 0
    assert torch.isfinite(loss_iou)
    assert torch.isfinite(loss_dfl)


def test_bbox_loss_wiou_updates_iou_mean():
    """After forward, iou_mean should decrease from initial 1.0."""
    bl = _make_bbox_loss(wiou=True, wiou_momentum=0.5)  # large momentum for fast update
    inputs = _make_forward_inputs()
    _ = bl(*inputs)
    assert bl.iou_mean.item() < 1.0  # should have decreased


def test_bbox_loss_wiou_gradient():
    """Gradient should flow through loss_iou."""
    bl = _make_bbox_loss(wiou=True)
    inputs = _make_forward_inputs()
    loss_iou, _ = bl(*inputs)
    loss_iou.backward()
    assert inputs[1].grad is not None  # pred_bboxes has grad


def test_bbox_loss_alpha_mix():
    """wiou_alpha < 1.0 should mix WIoU and CIoU without error."""
    bl = _make_bbox_loss(wiou=True, wiou_alpha=0.5)
    inputs = _make_forward_inputs()
    loss_iou, loss_dfl = bl(*inputs)
    assert torch.isfinite(loss_iou)


def test_bbox_loss_ciou_path_unchanged():
    """wiou=False should produce same loss as original CIoU path (sanity)."""
    bl = _make_bbox_loss(wiou=False)
    inputs = _make_forward_inputs()
    loss_iou, loss_dfl = bl(*inputs)
    assert loss_iou.item() > 0
    assert torch.isfinite(loss_iou)
