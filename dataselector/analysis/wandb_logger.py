"""Weights & Biases (wandb) integration for KDR100 experiment tracking.

Provides centralized logging interface for Optuna trials, bootstrap iterations,
and phase completion metrics.

Usage:
    from dataselector.analysis.wandb_logger import WandBLogger

    logger = WandBLogger(
        project="kdr100",
        run_name="phase1_hamburg_2026",
        tags=["phase-1", "optuna"]
    )

    logger.log_trial(trial_num=1, objective=0.85, params={...})
    logger.finish()
"""

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    """Experiment tracking with Weights & Biases.

    Gracefully handles case where wandb is not installed or not configured.
    Falls back to no-op logging if wandb unavailable.
    """

    def __init__(
        self,
        project: str = "kdr100",
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        disabled: bool = False,
    ):
        """Initialize wandb run.

        Args:
            project: wandb project name
            run_name: human-readable run identifier
            tags: list of tags (e.g., ["phase-1", "hamburg"])
            notes: markdown notes for run
            disabled: if True, disable all logging (for testing)
        """
        self.project = project
        self.run_name = run_name
        self.tags = tags or []
        self.notes = notes
        self.disabled = disabled or not WANDB_AVAILABLE
        self.run = None

        if not self.disabled:
            try:
                self.run = wandb.init(
                    project=project,
                    name=run_name,
                    tags=tags,
                    notes=notes,
                    settings=wandb.Settings(start_method="fork"),
                )
                print(f"✅ wandb initialized: {project}/{run_name}")
            except Exception as e:
                print(f"⚠️  wandb init failed: {e}; logging disabled")
                self.disabled = True

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters.

        Args:
            config: dictionary of configuration values
        """
        if self.disabled or self.run is None:
            return
        try:
            self.run.config.update(config)
        except Exception as e:
            print(f"⚠️  wandb config log failed: {e}")

    def log_trial(
        self,
        trial_num: int,
        objective: float,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log Optuna trial result.

        Args:
            trial_num: trial number (1-indexed)
            objective: objective value (loss/score)
            params: trial parameters
            metrics: additional metrics
        """
        if self.disabled or self.run is None:
            return
        try:
            data = {"trial": trial_num, "objective": objective}
            if params:
                data.update({f"param_{k}": v for k, v in params.items()})
            if metrics:
                data.update(metrics)
            self.run.log(data)
        except Exception as e:
            print(f"⚠️  wandb trial log failed: {e}")

    def log_bootstrap(
        self,
        iteration: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log bootstrap iteration results.

        Args:
            iteration: bootstrap iteration number
            metrics: stability metrics (Jaccard, coverage, etc.)
        """
        if self.disabled or self.run is None:
            return
        try:
            data = {"bootstrap_iter": iteration}
            data.update({f"boot_{k}": v for k, v in metrics.items()})
            self.run.log(data)
        except Exception as e:
            print(f"⚠️  wandb bootstrap log failed: {e}")

    def log_phase_completion(
        self,
        phase: int,
        duration_seconds: float,
        summary: Dict[str, Any],
    ) -> None:
        """Log phase completion with summary statistics.

        Args:
            phase: phase number (0-5)
            duration_seconds: time taken
            summary: summary statistics
        """
        if self.disabled or self.run is None:
            return
        try:
            data = {
                "phase": phase,
                "phase_duration_sec": duration_seconds,
            }
            data.update({f"phase{phase}_{k}": v for k, v in summary.items()})
            self.run.log(data)
        except Exception as e:
            print(f"⚠️  wandb phase log failed: {e}")

    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str = "result",
        description: Optional[str] = None,
    ) -> None:
        """Log artifact (CSV, JSON, image).

        Args:
            artifact_path: path to file
            artifact_type: type label (e.g., "result", "plot", "data")
            description: human-readable description
        """
        if self.disabled or self.run is None:
            return
        try:
            p = Path(artifact_path)
            if p.exists():
                artifact = wandb.Artifact(
                    name=p.stem,
                    type=artifact_type,
                    description=description,
                )
                artifact.add_file(str(artifact_path))
                self.run.log_artifact(artifact)
        except Exception as e:
            print(f"⚠️  wandb artifact log failed: {e}")

    def log_plot(
        self,
        plot_path: str,
        step: Optional[int] = None,
    ) -> None:
        """Log a plot image.

        Args:
            plot_path: path to image file
            step: optional step/iteration number
        """
        if self.disabled or self.run is None:
            return
        try:
            import matplotlib.image as mpimg

            img = mpimg.imread(plot_path)
            self.run.log(
                {"plot": wandb.Image(img)},
                step=step,
            )
        except Exception as e:
            print(f"⚠️  wandb plot log failed: {e}")

    def finish(self) -> None:
        """Finalize run."""
        if self.disabled or self.run is None:
            return
        try:
            self.run.finish()
            print("✅ wandb run finished")
        except Exception as e:
            print(f"⚠️  wandb finish failed: {e}")


def get_wandb_logger(
    project: str = "kdr100",
    disabled: bool = False,
) -> WandBLogger:
    """Factory function for wandb logger.

    Args:
        project: wandb project name
        disabled: if True, create no-op logger

    Returns:
        WandBLogger instance (no-op if disabled or wandb unavailable)
    """
    return WandBLogger(project=project, disabled=disabled)
