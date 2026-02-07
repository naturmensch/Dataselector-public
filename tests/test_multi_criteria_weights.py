import pandas as pd
import pytest

from dataselector.selection.multi_criteria_facility_location import (
    MultiCriteriaFacilityLocation,
)


def _make_meta():
    return pd.DataFrame(
        {
            "ul_x": [9.95, 10.95, 11.95],
            "ul_y": [50.05, 51.05, 52.05],
            "lr_x": [10.05, 11.05, 12.05],
            "lr_y": [49.95, 50.95, 51.95],
            "year": [1900, 1914, 1918],
        }
    )


def test_valid_weights_do_not_raise():
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


def test_invalid_weights_raise_value_error():
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
