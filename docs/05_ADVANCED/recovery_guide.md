# Recovery & Monitor Resume Guide

This document explains how the monitor handles resume, integrity checks and trial reconstruction.

- Resume modes: `--restart last|<run_id>`, `--force-restart`, `--dry-run-restart`
- Optuna DB reconstruction: `scripts/xxl_full_run_monitor.py --no-reconstruct` to opt-out
- Backups: `optuna_study.db.bak_resume_<ts>` are created automatically

(Include step-by-step recovery examples.)