import os
from pathlib import Path

import torch
from PIL import Image

from dataselector.features.feature_extractor import FeatureExtractor


def test_resnet50_get_dim():
    ext = FeatureExtractor(model_name="resnet50")
    assert ext.get_feature_dimension() == 2048


def test_dinov2_mocked_extract(monkeypatch, tmp_path):
    # Monkeypatch torch.hub.load to return a dummy DINOv2-like module
    class DummyDino(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = 384

        def forward(self, x):
            B = x.shape[0]
            return torch.randn(B, self.out_dim)

    monkeypatch.setattr(torch.hub, "load", lambda repo, name: DummyDino())

    # Create a test image
    p = tmp_path / "img.png"
    Image.new("RGB", (512, 512), color=(128, 128, 128)).save(p)

    ext = FeatureExtractor(model_name="dinov2")
    feats = ext.extract_features_batch([str(p)], data_dir=tmp_path, batch_size=1, crop_size=(512, 512))

    assert feats.shape[0] == 1
    assert feats.shape[1] == ext.get_feature_dimension()
    assert ext.get_feature_dimension() == 384
