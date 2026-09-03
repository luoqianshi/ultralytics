"""Unit tests for VarifocalNetLoss (Varifocal Loss extracted from VarifocalNet, CVPR 2021)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import VarifocalNetLoss, v8DetectionLoss


def test_varifocal_net_loss_init():
    """Default params should match the official VarifocalNet defaults."""
    vfl = VarifocalNetLoss()
    assert vfl.alpha == 0.75
    assert vfl.gamma == 2.0
    assert vfl.iou_weighted is True


def test_varifocal_net_loss_output_shape():
    """Output must be element-wise with the same shape as inputs."""
    torch.manual_seed(0)
    pred = torch.randn(2, 100, 3)
    target = torch.rand(2, 100, 3)
    out = VarifocalNetLoss()(pred, target)
    assert out.shape == pred.shape
    assert out.dtype == torch.float32


def test_varifocal_net_loss_negative_weight_formula():
    """For negatives (t=0), loss == alpha * p^gamma * BCE(p, 0)."""
    torch.manual_seed(0)
    pred = torch.randn(4, 5)
    target = torch.zeros(4, 5)
    alpha, gamma = 0.75, 2.0
    out = VarifocalNetLoss(alpha=alpha, gamma=gamma)(pred, target)
    p = pred.sigmoid()
    expected = alpha * p.pow(gamma) * F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    assert torch.allclose(out, expected, atol=1e-6)


def test_varifocal_net_loss_positive_weighted_by_target():
    """iou_weighted=True: positives weighted by target value; iou_weighted=False: weight 1."""
    torch.manual_seed(0)
    pred = torch.randn(4, 5)
    target = torch.zeros(4, 5)
    target[0, 0] = 0.8  # a positive with IACS 0.8
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    out = VarifocalNetLoss()(pred, target)
    assert torch.allclose(out[0, 0], target[0, 0] * bce[0, 0], atol=1e-6)  # weight = target

    out2 = VarifocalNetLoss(iou_weighted=False)(pred, target)
    assert torch.allclose(out2[0, 0], bce[0, 0], atol=1e-6)  # weight = 1
    assert torch.allclose(out[1:], out2[1:], atol=1e-6)  # negatives identical


def test_varifocal_net_loss_gradient_flow():
    """Gradients must flow back to pred and be finite."""
    torch.manual_seed(0)
    pred = torch.randn(3, 7, 2, requires_grad=True)
    target = torch.rand(3, 7, 2)
    loss = VarifocalNetLoss()(pred, target).sum()
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def _make_model(args):
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("yolo12.yaml", nc=3, verbose=False)
    model.args = args
    return model


def test_v8detection_loss_flag_off_backward_compat():
    """Without varifocal_loss key, v8DetectionLoss keeps the original BCE path (backward compat)."""
    criterion = v8DetectionLoss(_make_model({}))
    assert criterion.use_varifocal_loss is False
    assert isinstance(criterion.bce, nn.BCEWithLogitsLoss)
    assert not hasattr(criterion, "varifocal_loss")


def test_v8detection_loss_flag_on():
    """With varifocal_loss=True, criterion builds a VarifocalNetLoss instance with official defaults."""
    criterion = v8DetectionLoss(_make_model({"varifocal_loss": True}))
    assert criterion.use_varifocal_loss is True
    assert isinstance(criterion.varifocal_loss, VarifocalNetLoss)
    assert criterion.varifocal_loss.alpha == 0.75
    assert criterion.varifocal_loss.gamma == 2.0


def test_v8detection_loss_vfl_forward_smoke():
    """End-to-end: VFL branch through get_assigned_targets_and_loss (IACS target construction + backward)."""
    torch.manual_seed(0)
    model = _make_model(IterableSimpleNamespace(varifocal_loss=True, box=7.5, cls=0.5, dfl=1.5, overlap_mask=False))
    model.train()
    criterion = v8DetectionLoss(model)
    batch = {
        "batch_idx": torch.tensor([0, 0, 1]),
        "cls": torch.tensor([[0.0], [2.0], [1.0]]),
        "bboxes": torch.rand(3, 4) * 0.2 + 0.3,
        "img": torch.zeros(2, 3, 320, 320),
    }
    preds = model(batch["img"])
    loss, items = criterion(preds, batch)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(items).all()
    loss.sum().backward()  # gradients must flow
