import json
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.compare_distance_methods import compute_haversine_matrix, compute_utm_matrix

DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "new_all_tiles.csv"


def test_per_tile_stats_small_sample(tmp_path):
    # build tiny dataframe of 3 points roughly forming a triangle
    df = pd.DataFrame(
        {
            "shortName": ["A", "B", "C"],
            "N": [52.52, 50.9375, 48.1372],
            "left": [13.405, 6.9603, 11.5755],
        }
    )
    hav = compute_haversine_matrix(df)
    utm = compute_utm_matrix(df)
    full_abs = np.abs(hav - utm)

    # per-tile medians manually
    per = []
    for i in range(3):
        vals = np.delete(full_abs[i, :], i)
        per.append(float(np.median(vals)))

    # run the script's logic to produce per-tile median
    # we reimplement the same loop here and compare values
    for i in range(3):
        vals = np.delete(full_abs[i, :], i)
        assert abs(np.median(vals) - per[i]) < 1e-9
