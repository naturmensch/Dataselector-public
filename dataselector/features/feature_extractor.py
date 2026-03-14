"""
Feature Extraction mittels Deep Learning (ResNet50 oder DINOv2).

Dieses Modul extrahiert visuelle Features aus Kartenbildern
mittels eines vortrainierten CNN oder ViT.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm


def preprocess_historical_image(img: Image.Image) -> Image.Image:
    """
    Bereitet eine historische Karte für SOTA-Modelle vor:
    1. Graustufen (entfernt Farbrauschen).
    2. Autokontrast (verstärkt Linien).
    3. Zurück zu RGB (für Modell-Kompatibilität).
    """
    # In Graustufen wandeln
    img_gray = img.convert("L")
    # Kontrast maximieren (wichtig für verblasste Karten!)
    img_enhanced = ImageOps.autocontrast(img_gray)
    # Zurück in 3 Kanäle (R=G=B)
    return img_enhanced.convert("RGB")


class FeatureExtractor:
    """Extrahiert Deep Learning Features aus Bildern."""

    def __init__(
        self,
        model_name: str = "dinov2",  # Standardmäßig jetzt SOTA!
        device: Optional[str] = None,
        input_size: int = 224,
        default_crop_size: Tuple[int, int] = (2048, 2048),
        pooling: str = "cls",
        model_variant: str = "dinov2_vits14",
        dinov2_repo: str = "facebookresearch/dinov2",
        dinov2_ref: str = "main",
    ):
        """
        Args:
            model_name: Modellname ('resnet50' oder 'dinov2')
            device: 'cuda'|'cpu' or None (auto)
            input_size: Model-Input-Größe (z. B. 224)
            default_crop_size: Default für center-crop (w, h)
            pooling: DINOv2 Pooling-Strategie ('cls' oder 'mean')
            model_variant: DINOv2 Modell-Name (z. B. dinov2_vits14)
            dinov2_repo: DINOv2 Repository (owner/repo)
            dinov2_ref: DINOv2 Referenz (branch/tag/commit)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.input_size = int(input_size)
        self.default_crop_size = tuple(default_crop_size)
        self.pooling = str(pooling).strip().lower()
        self.model_variant = str(model_variant).strip()
        self.dinov2_repo = str(dinov2_repo).strip()
        self.dinov2_ref = str(dinov2_ref).strip()
        if self.pooling not in {"cls", "mean", "global_avg"}:
            raise ValueError(
                f"Unbekannte pooling-Strategie: {self.pooling}. Erlaubt: cls|mean|global_avg"
            )
        if self.model_name.lower() == "dinov2" and self.pooling == "global_avg":
            # Keep a deterministic fallback for callers that use framework-neutral naming.
            self.pooling = "mean"
        self._model_provenance: Dict[str, Any] = {}

        # If DINOv2 is selected and the user did not override input_size, use
        # a patch-compatible size (multiple of 14).
        if self.model_name.lower() == "dinov2" and int(input_size) == 224:
            self.input_size = 392

        self.model = self._load_model()
        self.transform = self._get_transforms(self.input_size)
        # DINOv2 braucht oft eine spezifische Normalisierung,
        # wir nutzen hier den Standard-Transform.

        # Cap torch threads for deterministic CPU runs
        try:
            torch.set_num_threads(1)
        except Exception:
            # Best-effort; do not fail if not applicable
            pass

    def _load_model(self) -> nn.Module:
        print(f"Lade Modell: {self.model_name} auf {self.device}...")

        model_key = self.model_name.lower()

        if model_key == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1])
            self._model_provenance = {
                "model_name": "resnet50",
                "weights": "ResNet50_Weights.IMAGENET1K_V1",
                "pooling": "global_avg",
                "input_size": int(self.input_size),
            }
        elif model_key == "dinov2":
            try:
                # Canonical DINOv2 entrypoint with explicit repo/ref/model variant.
                repo_spec = (
                    f"{self.dinov2_repo}:{self.dinov2_ref}"
                    if self.dinov2_ref
                    else self.dinov2_repo
                )
                model = torch.hub.load(repo_spec, self.model_variant)
                self._model_provenance = {
                    "model_name": "dinov2",
                    "repo": self.dinov2_repo,
                    "ref": self.dinov2_ref or None,
                    "repo_spec": repo_spec,
                    "model_variant": self.model_variant,
                    "pooling": self.pooling,
                    "input_size": int(self.input_size),
                }
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load 'dinov2' via torch.hub "
                    f"('{self.dinov2_repo}:{self.dinov2_ref}', '{self.model_variant}')."
                ) from exc
        else:
            raise ValueError(f"Unbekanntes Modell: {self.model_name}")

        model.eval().to(self.device)
        return model

    def _extract_dinov2_tensor_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 features using configured pooling strategy."""
        model = self.model
        output: Any

        # Prefer explicit feature dict if available for deterministic pooling behavior.
        if hasattr(model, "forward_features"):
            output = model.forward_features(batch)
        else:
            output = model(batch)

        if isinstance(output, dict):
            if self.pooling == "cls":
                if "x_norm_clstoken" in output:
                    output = output["x_norm_clstoken"]
                elif "x_prenorm" in output and output["x_prenorm"].ndim == 3:
                    output = output["x_prenorm"][:, 0, :]
                elif "x_norm_patchtokens" in output:
                    output = output["x_norm_patchtokens"].mean(dim=1)
                else:
                    output = model(batch)
            else:  # mean pooling
                if "x_norm_patchtokens" in output:
                    output = output["x_norm_patchtokens"].mean(dim=1)
                elif "x_prenorm" in output and output["x_prenorm"].ndim == 3:
                    output = output["x_prenorm"].mean(dim=1)
                elif "x_norm_clstoken" in output:
                    output = output["x_norm_clstoken"]
                else:
                    output = model(batch)

        if isinstance(output, torch.Tensor):
            if output.ndim == 3:
                if self.pooling == "cls":
                    output = output[:, 0, :]
                else:
                    output = output.mean(dim=1)
            return output.reshape(output.shape[0], -1)

        # Fallback for unexpected model outputs.
        output = model(batch)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"Unexpected DINOv2 output type: {type(output)}; expected Tensor/dict"
            )
        if output.ndim == 3:
            if self.pooling == "cls":
                output = output[:, 0, :]
            else:
                output = output.mean(dim=1)
        return output.reshape(output.shape[0], -1)

    def _get_transforms(self, input_size: int = 224) -> transforms.Compose:
        resize_size = max(256, int(input_size))
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_center_crop(
        self, image_path: Path, crop_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        crop_size = (
            tuple(crop_size) if crop_size is not None else self.default_crop_size
        )

        # Original öffnen
        raw_img = Image.open(image_path)

        # UNSERE NEUE FUNKTION ANWENDEN
        img = preprocess_historical_image(raw_img)

        width, height = img.size

        left = (width - crop_size[0]) // 2
        top = (height - crop_size[1]) // 2
        right = left + crop_size[0]
        bottom = top + crop_size[1]

        # Falls das Bild kleiner als der gewünschte Crop ist, vergrößern wir es
        if width < crop_size[0] or height < crop_size[1]:
            img = img.resize(crop_size, Image.LANCZOS)
            return img

        return img.crop((left, top, right, bottom))

    def extract_features_single(self, image: Image.Image) -> np.ndarray:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_name.lower() == "dinov2":
                features = self._extract_dinov2_tensor_features(img_tensor)
            else:
                features = self.model(img_tensor).reshape(1, -1)
        features = features.squeeze().cpu().numpy()
        return features

    def extract_features_batch(
        self,
        image_paths: List[Union[str, Path]],
        data_dir: Path,
        batch_size: int = 8,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Args:
            image_paths: Liste von Dateinamen (str/Path) relativ zu `data_dir`
            crop_size: Optional overrides default center-crop (w,h)
        """
        crop_size = (
            tuple(crop_size) if crop_size is not None else self.default_crop_size
        )
        all_features = []

        # fallback tensor shape depends on self.input_size
        fallback_tensor = torch.zeros(3, self.input_size, self.input_size)

        for i in tqdm(
            range(0, len(image_paths), batch_size), desc="SOTA Feature Extraction"
        ):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = []

            for img_name in batch_paths:
                # allow Path or string
                img_path = (
                    Path(img_name)
                    if isinstance(img_name, (str, Path))
                    else Path(img_name)
                )
                if not img_path.is_absolute():
                    img_path = data_dir / img_path

                if not img_path.exists():
                    print(f"Warnung: Bild nicht gefunden: {img_path}")
                    batch_tensors.append(fallback_tensor)
                    continue

                try:
                    # Hier wird automatisch gecroppt UND vorverarbeitet!
                    img = self.extract_center_crop(img_path, crop_size=crop_size)
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Fehler bei {img_path}: {e}")
                    batch_tensors.append(fallback_tensor)

            batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                if self.model_name.lower() == "dinov2":
                    features = self._extract_dinov2_tensor_features(batch)
                else:
                    features = self.model(batch)
                    # Ensure flat vector (ResNet50 outputs (N, 2048, 1, 1))
                    features = features.reshape(features.shape[0], -1)

            features = features.reshape(features.shape[0], -1).cpu().numpy()
            all_features.append(features)

        if not all_features:
            return np.empty((0, self.get_feature_dimension()), dtype=np.float32)
        return np.vstack(all_features)

    def get_feature_dimension(self) -> int:
        model_key = self.model_name.lower()
        if model_key == "resnet50":
            return 2048
        if model_key == "dinov2":
            return 384
        return 0

    def get_model_provenance(self) -> Dict[str, Any]:
        """Return deterministic model-loading metadata for run provenance."""
        provenance = dict(self._model_provenance)
        provenance["device"] = str(self.device)
        provenance["crop_size"] = [
            int(self.default_crop_size[0]),
            int(self.default_crop_size[1]),
        ]
        return provenance
