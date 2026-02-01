import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def MultiCriteriaFacilityLocation():
    pytest.importorskip("numba", exc_type=ImportError)
    import importlib

    mod = importlib.import_module("src.multi_criteria_facility_location")
    return mod.MultiCriteriaFacilityLocation


def _make_meta():
    return pd.DataFrame(
        {
            "N": [50.0, 51.0, 52.0],
            "left": [10.0, 11.0, 12.0],
            "year": [1900, 1914, 1918],
        }
    )


def test_valid_weights_do_not_raise(MultiCriteriaFacilityLocation):
    meta = _make_meta()
    # weights sum to 1.0
    m = MultiCriteriaFacilityLocation(
        n_samples=2,
        metadata=meta,
        alpha_visual=0.7,
        beta_spatial=0.15,
        gamma_temporal=0.15,
    )
    assert m.alpha == pytest.approx(0.7)
    assert m.beta == pytest.approx(0.15)
    assert m.gamma == pytest.approx(0.15)


def test_invalid_weights_raise_value_error(MultiCriteriaFacilityLocation):
    meta = _make_meta()
    with pytest.raises(ValueError) as excinfo:
        MultiCriteriaFacilityLocation(
            n_samples=2,
            metadata=meta,
            alpha_visual=0.5,
            beta_spatial=0.1,
            gamma_temporal=0.2,
        )
    assert "Gewichte müssen sich zu 1.0 addieren" in str(excinfo.value)
