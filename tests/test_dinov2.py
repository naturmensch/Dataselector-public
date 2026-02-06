import numpy as np
import pytest
from PIL import Image

from dataselector.features.feature_extractor import (
    FeatureExtractor,
    preprocess_historical_image,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _require_numba():
    pytest.importorskip("numba", exc_type=ImportError)


def test_dinov2_initialization():
    """Testet, ob DINOv2 korrekt initialisiert wird."""
    try:
        # Wir nutzen CPU für den Test, um GPU-Abhängigkeit zu vermeiden
        extractor = FeatureExtractor(model_name="dinov2", device="cpu")
        assert extractor.model is not None
        # DINOv2-Small (vits14) hat 384 Dimensionen
        assert extractor.get_feature_dimension() == 384
    except Exception as e:
        pytest.fail(
            f"DINOv2 Initialisierung fehlgeschlagen (Internetverbindung prüfen?): {e}"
        )


def test_preprocessing():
    """Testet die Bildvorverarbeitung (Graustufen -> Autokontrast -> RGB)."""
    # Erstelle ein Dummy-Bild (Grau mit etwas Rauschen)
    img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    processed = preprocess_historical_image(img)

    assert processed.mode == "RGB"
    assert processed.size == (100, 100)
    # Prüfen, ob es ein PIL Image ist
    assert isinstance(processed, Image.Image)


def test_feature_extraction_flow(tmp_path):
    """Testet den kompletten Flow mit einem Dummy-Bild."""
    # Dummy Bild speichern
    img_path = tmp_path / "test_map.png"
    # Ein Bild mit Struktur erstellen, damit Features nicht null sind
    Image.new("RGB", (500, 500), color="white").save(img_path)

    try:
        extractor = FeatureExtractor(model_name="dinov2", device="cpu")
    except Exception:
        pytest.skip("DINOv2 konnte nicht geladen werden")

    # Extraktion testen
    features = extractor.extract_features_batch([str(img_path)], data_dir=tmp_path)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, 384)
