"""Professional Experiment Management System with versioning and reproducibility tracking.

Key principles:
1. Every run gets a dated, named directory with complete provenance
2. Hierarchical structure: config/ results/ logs/ artifacts/ monitor/
3. Incremental result saving during execution (not just at the end)
4. Complete reproducibility: Git commit, Python environment, parameters all tracked
5. Status tracking: manifest.json, status.log, metrics.csv

Usage:
    em = ExperimentManager(name="hamburg_optuna_n2000", description="Optuna sweep with Hamburg preselection")
    run_dir = em.initialize()  # Creates outputs/runs/20260116_T160213_hamburg_optuna_n2000/

    # Save artifacts at different stages
    em.save_config("selection", {"alpha": 0.5, "beta": 0.3, "gamma": 0.2})
    em.save_results("trials", trials_df)
    em.log("Trial 100 complete", level="info")
    em.mark_complete("optuna_stage")
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ExperimentManager:
    """Manages versioned experiment directories with full provenance tracking."""

    def __init__(
        self,
        name: str,
        description: str = "",
        base_dir: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
        capture_provenance: bool = True,
    ):
        """Initialize experiment manager.

        Args:
            name: Experiment identifier (e.g., 'hamburg_optuna_n2000')
            description: Human-readable description
            base_dir: Base output directory (default: outputs/runs/)
            metadata: Additional metadata dict (e.g., pipeline version, thesis_chapter)
            capture_provenance: When False, skip expensive provenance collection (helpful in tests)
        """
        self.name = name
        self.description = description
        self.base_dir = Path(base_dir or "outputs/runs")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped run directory
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_T%H%M%S")
        self.run_id = f"{self.timestamp}_{name}"
        self.run_dir = self.base_dir / self.run_id

        # Initialize directory structure
        self._init_directories()

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize manifest
        self.manifest = {
            "experiment": {
                "name": name,
                "description": description,
                "run_id": self.run_id,
                "timestamp_utc": self.timestamp,
                "status": "initialized",
                "stages": {},  # Track which stages completed
            },
            "provenance": {} if not capture_provenance else self._capture_provenance(),
            "metadata": metadata or {},
            "results": {},
            "artifacts": [],
        }
        self.stages = {}  # Track stages separately as well
        self._capture_provenance_enabled = bool(capture_provenance)

        self.logger.info(f"Experiment initialized: {self.run_id}")

    def _init_directories(self):
        """Create hierarchical directory structure."""
        dirs = [
            self.run_dir,
            self.run_dir / "config",  # Configurations at different stages
            self.run_dir / "results",  # CSV, metrics, best trials
            self.run_dir / "logs",  # Main execution logs
            self.run_dir / "artifacts",  # Intermediate outputs (plots, selections)
            self.run_dir / "monitor",  # Monitoring snapshots and status
            self.run_dir / "monitor" / "snapshots",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Configure hierarchical logging."""
        logger = logging.getLogger(self.run_id)
        # If logger is already configured (handlers present), reuse it to avoid duplicate handlers
        if logger.handlers:
            return logger
        logger.setLevel(logging.DEBUG)

        # Handler 1: Main log file (all levels)
        main_log = self.run_dir / "logs" / "experiment.log"
        fh_main = logging.FileHandler(main_log)
        fh_main.setLevel(logging.DEBUG)

        # Handler 2: Status log (info+ only, for monitoring)
        status_log = self.run_dir / "logs" / "status.log"
        fh_status = logging.FileHandler(status_log)
        fh_status.setLevel(logging.INFO)

        # Handler 3: Console (info+ only)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
        fmt = logging.Formatter(
            "[%(levelname)s %(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh_main.setFormatter(fmt)
        fh_status.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh_main)
        logger.addHandler(fh_status)
        logger.addHandler(ch)

        return logger

    def _capture_provenance(self) -> Dict[str, Any]:
        """Capture complete reproducibility information.

        This can be an expensive operation (subprocess calls, imports). We keep it
        separate so callers (or tests) can opt-out during initialization.
        """
        prov = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": "unknown",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "cwd": str(Path.cwd()),
        }

        # Try to resolve hostname in a safe manner
        try:
            out = subprocess.run(["hostname", "-f"], capture_output=True, text=True)
            if out and out.stdout:
                prov["hostname"] = out.stdout.strip()
        except Exception:
            pass

        # Git information (if available)
        try:
            git_dir = Path.cwd()
            if (git_dir / ".git").exists():
                prov["git"] = {
                    "commit": subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=git_dir,
                        capture_output=True,
                        text=True,
                    ).stdout.strip(),
                    "branch": subprocess.run(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        cwd=git_dir,
                        capture_output=True,
                        text=True,
                    ).stdout.strip(),
                    "dirty": bool(
                        subprocess.run(
                            ["git", "diff", "--quiet"],
                            cwd=git_dir,
                        ).returncode
                    ),
                    "diff": (
                        subprocess.run(
                            ["git", "diff", "--stat"],
                            cwd=git_dir,
                            capture_output=True,
                            text=True,
                        ).stdout.strip()
                        if subprocess.run(
                            ["git", "diff", "--quiet"],
                            cwd=git_dir,
                        ).returncode
                        else ""
                    ),
                }
        except Exception as e:
            self.logger.warning(f"Could not capture git info: {e}")

        # Python packages (optuna, pytorch, etc.)
        prov["packages"] = {}
        for pkg in ["optuna", "torch", "pandas", "numpy", "scikit-learn", "apricot"]:
            try:
                mod = __import__(pkg)
                prov["packages"][pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                prov["packages"][pkg] = "not installed"

        return prov

    def capture_provenance(self) -> Dict[str, Any]:
        """Public helper to (re-)capture provenance on demand.

        Returns the provenance dict and stores it in the manifest.
        """
        prov = self._capture_provenance()
        self.manifest["provenance"] = prov
        return prov

    @classmethod
    def from_existing(cls, run_dir: str):
        """Attach to an existing run directory and load manifest if present.

        This is useful when a top-level pipeline creates the run directory and
        sub-stage scripts need to write into that same run.
        """
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Allocate without calling __init__ (to avoid re-creating timestamps)
        em = cls.__new__(cls)
        em.run_dir = run_dir
        em.run_id = run_dir.name
        parts = em.run_id.split("_", 2)
        em.timestamp = parts[0] if parts else ""
        em.name = parts[2] if len(parts) >= 3 else em.run_id
        em.description = ""
        em.base_dir = run_dir.parent

        # Ensure directory layout exists and logging configured
        em._init_directories()
        em.logger = em._setup_logging()

        # Load existing manifest if present, otherwise create a minimal manifest
        manifest_file = em.run_dir / "manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, "r") as f:
                    em.manifest = json.load(f)
            except Exception as e:
                em.logger.warning(f"Could not load existing manifest: {e}")
                em.manifest = {
                    "experiment": {
                        "name": em.name,
                        "description": em.description,
                        "run_id": em.run_id,
                        "timestamp_utc": em.timestamp,
                        "status": "attached",
                        "stages": {},
                    },
                    "provenance": em._capture_provenance(),
                    "metadata": {},
                    "results": {},
                    "artifacts": [],
                }
        else:
            em.manifest = {
                "experiment": {
                    "name": em.name,
                    "description": em.description,
                    "run_id": em.run_id,
                    "timestamp_utc": em.timestamp,
                    "status": "attached",
                    "stages": {},
                },
                "provenance": em._capture_provenance(),
                "metadata": {},
                "results": {},
                "artifacts": [],
            }

        em.stages = em.manifest["experiment"].get("stages", {})
        em.logger.info(f"Attached to existing experiment: {em.run_id}")
        return em

    def save_config(self, stage_name: str, config_dict: Dict[str, Any]):
        """Save configuration at a specific stage.

        Args:
            stage_name: Identifier for this config (e.g., 'exploration', 'optuna', 'selection')
            config_dict: Configuration parameters
        """
        config_file = self.run_dir / "config" / f"config_{stage_name}.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

        self.manifest["artifacts"].append(
            {
                "type": "config",
                "stage": stage_name,
                "path": str(config_file.relative_to(self.run_dir)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.logger.info(f"Saved config: {stage_name} → {config_file.name}")

    def save_results(self, result_name: str, data, format: str = "auto"):
        """Save results (CSV, JSON, pickle, etc.).

        Args:
            result_name: Identifier (e.g., 'trials', 'best_trial', 'convergence')
            data: DataFrame, dict, or other serializable object
            format: 'auto', 'csv', 'json', 'pkl', 'npy'
        """
        import pandas as pd

        # Determine format
        if format == "auto":
            if isinstance(data, pd.DataFrame):
                format = "csv"
            elif isinstance(data, dict):
                format = "json"
            elif isinstance(data, (list, tuple)):
                format = "json"
            else:
                format = "pkl"

        # Save file
        result_file = self.run_dir / "results" / f"{result_name}.{format}"

        if format == "csv":
            data.to_csv(result_file, index=False)
        elif format == "json":
            with open(result_file, "w") as f:
                if isinstance(data, dict):
                    json.dump(data, f, indent=2, default=str)
                else:
                    json.dump(data, f, indent=2, default=str)
        elif format == "pkl":
            import pickle

            with open(result_file, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.manifest["artifacts"].append(
            {
                "type": "results",
                "name": result_name,
                "format": format,
                "path": str(result_file.relative_to(self.run_dir)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "shape": str(getattr(data, "shape", "N/A")),
            }
        )
        self.logger.info(
            f"Saved results: {result_name} ({format}, shape={getattr(data, 'shape', 'N/A')})"
        )

    def save_artifact(
        self, source_path: Path, artifact_name: str, category: str = "other"
    ):
        """Copy and track an artifact from anywhere.

        Args:
            source_path: Source file path
            artifact_name: Name in the experiment (e.g., 'convergence_plot')
            category: Category (e.g., 'plot', 'log', 'data')
        """
        source_path = Path(source_path)
        if not source_path.exists():
            self.logger.warning(f"Artifact source not found: {source_path}")
            return

        dest_path = self.run_dir / "artifacts" / f"{artifact_name}{source_path.suffix}"
        shutil.copy(source_path, dest_path)

        self.manifest["artifacts"].append(
            {
                "type": "artifact",
                "name": artifact_name,
                "category": category,
                "path": str(dest_path.relative_to(self.run_dir)),
                "source": str(source_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.logger.info(f"Archived artifact: {artifact_name} ← {source_path.name}")

    def log(self, message: str, level: str = "info"):
        """Log a message."""
        getattr(self.logger, level.lower())(message)

    def mark_stage_complete(
        self, stage_name: str, summary: Optional[Dict[str, Any]] = None
    ):
        """Mark a pipeline stage as complete.

        Args:
            stage_name: Stage identifier (e.g., 'exploration', 'fine_sweep', 'optuna', 'bootstrap')
            summary: Optional summary dict (e.g., n_trials, best_value, runtime)
        """
        self.stages[stage_name] = {
            "status": "complete",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary or {},
        }
        self.manifest["experiment"]["stages"][stage_name] = self.stages[stage_name]
        self.logger.info(f"✓ Stage complete: {stage_name}")
        # Persist manifest and write a monitor snapshot for live tools
        try:
            self.save_manifest()
            self._write_monitor_snapshot()
        except Exception as e:
            self.logger.warning(f"Could not write monitor snapshot: {e}")

    def _write_monitor_snapshot(self):
        """Write a lightweight monitor summary for dashboards / live watchers."""
        snapshot = {
            "run_id": self.run_id,
            "status": self.manifest["experiment"].get("status", "running"),
            "stages": self.stages,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        out = self.run_dir / "monitor" / "summary.json"
        with open(out, "w") as f:
            json.dump(snapshot, f, indent=2)
        self.logger.debug(f"Wrote monitor snapshot: {out}")

    def save_manifest(self):
        """Persist manifest to disk using atomic write (temp file + move).

        This reduces the risk of corruption if the process is terminated during a write.
        """
        manifest_file = self.run_dir / "manifest.json"
        # Do not override a terminal status (complete/failed) when flushing during shutdown
        if self.manifest["experiment"].get("status") not in ("complete", "failed"):
            self.manifest["experiment"]["status"] = "running"
        self.manifest["experiment"]["stages"] = self.stages
        tmp_file = manifest_file.with_suffix(".json.tmp")
        try:
            with open(tmp_file, "w") as f:
                json.dump(self.manifest, f, indent=2, default=str)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    # Windows may not support fsync for some file types; ignore if unsupported
                    pass
            os.replace(tmp_file, manifest_file)
        except Exception as e:
            self.logger.warning(f"Failed to write manifest atomically: {e}")
            # fallback: try a plain write
            try:
                with open(manifest_file, "w") as f:
                    json.dump(self.manifest, f, indent=2, default=str)
            except Exception as e2:
                self.logger.error(f"Second attempt to write manifest failed: {e2}")

    def start_heartbeat(self, interval_seconds: int = 300):
        """Start a background heartbeat that periodically flushes manifest and monitor snapshot.

        Args:
            interval_seconds: Interval in seconds between heartbeats (default: 300s / 5min)
        """
        if (
            getattr(self, "_heartbeat_thread", None)
            and getattr(self, "_heartbeat_thread").is_alive()
        ):
            self.logger.debug("Heartbeat already running")
            return
        self._heartbeat_stop = threading.Event()

        def _loop():
            while not self._heartbeat_stop.wait(interval_seconds):
                try:
                    self.save_manifest()
                    self._write_monitor_snapshot()
                    self.logger.debug("Heartbeat: manifest flushed")
                except Exception as e:
                    self.logger.warning(f"Heartbeat error: {e}")

        self._heartbeat_thread = threading.Thread(
            target=_loop, daemon=True, name=f"heartbeat-{self.run_id}"
        )
        self._heartbeat_thread.start()
        self.logger.info(f"Started heartbeat (every {interval_seconds}s)")

    def stop_heartbeat(self):
        """Stop the background heartbeat if running."""
        if getattr(self, "_heartbeat_stop", None):
            self._heartbeat_stop.set()
            if getattr(self, "_heartbeat_thread", None):
                self._heartbeat_thread.join(timeout=5)
            self.logger.info("Stopped heartbeat")

    def mark_complete(self, success: bool = True, summary: Optional[str] = None):
        """Mark the entire experiment as complete."""
        # Ensure heartbeat stopped and final manifest flushed
        try:
            self.stop_heartbeat()
        except Exception:
            pass
        self.manifest["experiment"]["status"] = "complete" if success else "failed"
        if summary:
            self.manifest["experiment"]["summary"] = summary
        self.manifest["experiment"]["completion_time"] = datetime.now(
            timezone.utc
        ).isoformat()
        self.save_manifest()

        status = "✓ COMPLETE" if success else "✗ FAILED"
        self.logger.info(f"{status}: {self.run_id}")

    def get_path(self, subdir: str = "") -> Path:
        """Get path to a subdirectory in the run."""
        if subdir:
            return self.run_dir / subdir
        return self.run_dir

    def summary(self) -> str:
        """Return a summary of the experiment."""
        return f"""
═══════════════════════════════════════════════════════════
Experiment Summary
═══════════════════════════════════════════════════════════
Run ID:        {self.run_id}
Path:          {self.run_dir}
Description:   {self.description}
Status:        {self.manifest["experiment"]["status"]}
Stages:        {list(self.stages.keys())}
Artifacts:     {len(self.manifest["artifacts"])}
───────────────────────────────────────────────────────────
Git Commit:    {self.manifest["provenance"].get("git", {}).get("commit", "N/A")[:8]}
Python:        {self.manifest["provenance"]["python_version"]}
Started:       {self.manifest["provenance"]["timestamp_utc"]}
═══════════════════════════════════════════════════════════
"""
