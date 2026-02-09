from __future__ import annotations

import os
from pathlib import Path

AUTOSCALE_SELECTION_FILES = (
    "autoscale_selected_n_samples.txt",
    "optuna_autoscale_selected_n_samples.txt",
)


def _parse_positive_int(raw: str, *, context: str) -> int:
    try:
        value = int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"{context}: could not parse integer from '{raw}'") from exc
    if value <= 0:
        raise ValueError(f"{context}: value must be > 0 (got {value})")
    return value


def resolve_selection_n_samples(
    explicit_n_samples: int | None,
    *,
    context: str,
    root: Path | None = None,
    config_path: Path | None = None,
    experiment_run_dir: Path | str | None = None,
) -> tuple[int, str]:
    """Resolve target selection size without implicit numeric fallbacks.

    Resolution order:
    1) explicit argument
    2) config selection.n_samples
    3) autoscale artifact file in experiment run dir
    4) fail fast
    """
    if explicit_n_samples is not None:
        return _parse_positive_int(
            str(explicit_n_samples), context=f"{context}: explicit_n_samples"
        ), "explicit"

    resolved_root = root if root is not None else Path.cwd()
    resolved_config = (
        config_path if config_path is not None else resolved_root / "config" / "pipeline_config.yaml"
    )

    # Config source
    cfg_value = None
    if resolved_config.exists():
        try:
            import yaml

            with open(resolved_config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg_value = cfg.get("selection", {}).get("n_samples")
        except Exception:
            cfg_value = None

    if cfg_value is not None:
        return _parse_positive_int(
            str(cfg_value), context=f"{context}: config selection.n_samples"
        ), "config"

    # Autoscale artifact source
    exp_dir = (
        Path(experiment_run_dir)
        if experiment_run_dir is not None
        else (
            Path(os.environ["EXPERIMENT_RUN_DIR"])
            if "EXPERIMENT_RUN_DIR" in os.environ
            else None
        )
    )
    if exp_dir is not None:
        for fname in AUTOSCALE_SELECTION_FILES:
            candidate = exp_dir / fname
            if not candidate.exists():
                continue
            return _parse_positive_int(
                candidate.read_text(encoding="utf-8"),
                context=f"{context}: {fname}",
            ), f"experiment_artifact:{fname}"

    raise ValueError(
        f"{context}: could not resolve selection target n_samples. "
        "Provide --n-samples (or explicit function argument), set "
        "'selection.n_samples' in config/pipeline_config.yaml, or provide "
        "autoscale artifact file "
        f"({', '.join(AUTOSCALE_SELECTION_FILES)})."
    )
