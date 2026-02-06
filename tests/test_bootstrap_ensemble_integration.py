from pathlib import Path

import pandas as pd
import pytest

from dataselector.workflows.bootstrap import run_bootstrap_pareto

pytestmark = pytest.mark.integration


def test_bootstrap_pareto_rejects_ensemble_mode(tmp_path: Path, capsys):
    pareto = pd.DataFrame(
        {
            "alpha": [0.6],
            "beta": [0.2],
            "gamma": [0.2],
            "min_distance_km": [28],
            "n_selected": [10],
        }
    )
    pareto_csv = tmp_path / "pareto.csv"
    out_csv = tmp_path / "bootstrap.csv"
    pareto.to_csv(pareto_csv, index=False)

    rc = run_bootstrap_pareto(
        pareto_csv=pareto_csv,
        n_boot=20,
        output_csv=out_csv,
        random_seed=1,
        uq_method="ensemble",
    )

    assert rc == 2
    output = capsys.readouterr()
    text = (output.out + output.err).lower()
    assert "not implemented" in text
    assert "ensemble" in text
    assert not out_csv.exists()
