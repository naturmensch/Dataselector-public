import json
from pathlib import Path


def create_minimal_autoscale(root: Path, n_samples: int = 40):
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    # Keep both names during migration because some workflows still read legacy filenames.
    (out / "optuna_autoscale_selected_n_samples.txt").write_text(str(n_samples))
    (out / "autoscale_selected_n_samples.txt").write_text(str(n_samples))
    best = {
        "user_attrs": {
            "alpha": 0.33,
            "beta": 0.33,
            "gamma": 0.34,
            "min_distance_km": 50,
        }
    }
    (out / "optuna_autoscale_best_latest.json").write_text(json.dumps(best))
