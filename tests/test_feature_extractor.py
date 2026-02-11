import torch
from PIL import Image
from torch import nn

from dataselector.features import feature_extractor as feature_module
from dataselector.features.feature_extractor import FeatureExtractor


def test_resnet50_get_dim(monkeypatch):
    class DummyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 2048, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    monkeypatch.setattr(feature_module.models, "resnet50", lambda weights=None: DummyResNet())
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
    feats = ext.extract_features_batch(
        [str(p)], data_dir=tmp_path, batch_size=1, crop_size=(512, 512)
    )

    assert feats.shape[0] == 1
    assert feats.shape[1] == ext.get_feature_dimension()
    assert ext.get_feature_dimension() == 384


def test_dinov2_mean_pooling_uses_patchtokens(monkeypatch, tmp_path):
    class DummyDino(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward_features(self, x):
            batch = x.shape[0]
            # 4 patch tokens, 384 dim
            patch = torch.ones(batch, 4, 384)
            patch[:, 1, :] = 3.0
            return {"x_norm_patchtokens": patch}

        def forward(self, x):
            # Should not be used in this test
            return torch.zeros(x.shape[0], 384)

    monkeypatch.setattr(torch.hub, "load", lambda repo, name: DummyDino())

    p = tmp_path / "img.png"
    Image.new("RGB", (512, 512), color=(128, 128, 128)).save(p)

    ext = FeatureExtractor(model_name="dinov2", pooling="mean")
    feats = ext.extract_features_batch(
        [str(p)], data_dir=tmp_path, batch_size=1, crop_size=(512, 512)
    )

    assert feats.shape == (1, 384)
    # Mean over tokens [1,3,1,1] = 1.5
    assert float(feats[0, 0]) == 1.5


def test_dinov2_model_provenance(monkeypatch):
    class DummyDino(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 384)

    monkeypatch.setattr(torch.hub, "load", lambda repo, name: DummyDino())

    ext = FeatureExtractor(
        model_name="dinov2",
        model_variant="dinov2_vits14",
        dinov2_repo="facebookresearch/dinov2",
        dinov2_ref="main",
        pooling="cls",
    )
    prov = ext.get_model_provenance()

    assert prov["model_name"] == "dinov2"
    assert prov["repo"] == "facebookresearch/dinov2"
    assert prov["ref"] == "main"
    assert prov["model_variant"] == "dinov2_vits14"
    assert prov["pooling"] == "cls"
