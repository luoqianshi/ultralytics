# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Smoke test: DySample 改进模型通过官方路径部分加载 yolo12s.pt 预训练权重。

验证点（对应 spec 第 2 节参数映射分析）：
1. DySample 模型能从 yaml 构建 + 加载 yolo12s.pt 不报错
2. 骨干层（model.0.conv.weight）成功迁移（与预训练值一致）
3. DySample 新增参数（model.9.offset.weight）保留 normal_init(std=0.001)，未被预训练覆盖
4. Detect 头（model.21.cv2.0.0.weight）保持 nc=1 形状，未被 nc=80 预训练值污染
"""
import pytest
import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.downloads import attempt_download_asset


@pytest.mark.slow
def test_dysample_partial_load_from_yolo12s():
    """DySample 模型通过 BaseModel.load() 部分加载 yolo12s.pt 应成功迁移骨干、保留 DySample/Detect 初始化。"""
    yaml_path = "scripts/improved_yolo12/yolo12-DySample.yaml"

    # 1. 构建 DySample 模型（nc=1，对齐 SSDC-UAV）
    model = DetectionModel(yaml_path, nc=1, ch=3, verbose=False)
    sd_before = {k: v.clone() for k, v in model.state_dict().items()}

    # 2. 加载 yolo12s.pt 官方预训练权重
    weights_path = attempt_download_asset("yolo12s.pt")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # 3. 通过官方 BaseModel.load() 路径加载（走 intersect_dicts）
    model.load(ckpt, verbose=False)

    sd_after = model.state_dict()

    # 4. 骨干层 model.0.conv.weight 应被迁移（与预训练值一致）
    backbone_key = "model.0.conv.weight"
    pretrained_backbone = ckpt["model"].float().state_dict()[backbone_key]
    assert sd_after[backbone_key].equal(pretrained_backbone), \
        "骨干层 model.0.conv.weight 应被预训练权重覆盖"

    # 5. DySample 新增参数 model.9.offset.weight 应保留 normal_init（未被预训练覆盖）
    dysample_key = "model.9.offset.weight"
    assert dysample_key not in ckpt["model"].float().state_dict(), \
        "yolo12s.pt 不应包含 DySample 的 offset 参数"
    assert sd_after[dysample_key].equal(sd_before[dysample_key]), \
        "DySample offset.weight 应保留 normal_init，未被加载流程修改"

    # 6. Detect 头 model.21.cv2.0.0.weight 应保持 nc=1 形状（未被 nc=80 污染）
    detect_key = "model.21.cv2.0.0.weight"
    assert sd_after[detect_key].shape[0] == 1, \
        f"Detect 头应保持 nc=1，实际 shape={sd_after[detect_key].shape}"
    assert sd_after[detect_key].equal(sd_before[detect_key]), \
        "Detect 头应保留随机初始化，未被 nc=80 预训练值覆盖"
