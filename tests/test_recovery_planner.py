from pathlib import Path

from tests._helpers.load_script import load_script

ROOT = Path(__file__).resolve().parents[1]
recovery = load_script(
    ROOT / "scripts" / "recovery.py", module_name="scripts.recovery_test"
)
RecoveryPlanner = recovery.RecoveryPlanner


def test_planner_reconstruct_then_finalize():
    # DB has more completed trials than CSV -> reconstruct then finalize
    state = {
        "csv_exists": True,
        "csv_completed": 10,
        "db_exists": True,
        "db_integrity_ok": True,
        "db_completed": 20,
        "repro_done": True,
        "final_exists": False,
        "n_samples": 20,
    }
    planner = RecoveryPlanner(configured_n=20)
    tasks = planner.plan(state)
    assert [t.name for t in tasks] == ["reconstruct", "finalize"] or [
        t.name for t in tasks
    ] == ["reconstruct", "optuna", "repro", "finalize"]


def test_planner_needs_optuna_and_repro():
    state = {
        "csv_exists": True,
        "csv_completed": 40,
        "db_exists": False,
        "db_integrity_ok": False,
        "db_completed": 0,
        "repro_done": False,
        "final_exists": False,
    }
    planner = RecoveryPlanner(configured_n=100)
    tasks = planner.plan(state)
    names = [t.name for t in tasks]
    assert "optuna" in names
    assert "repro" in names
    assert "finalize" in names


def test_planner_only_finalize_when_complete():
    state = {
        "csv_exists": True,
        "csv_completed": 100,
        "db_exists": False,
        "db_integrity_ok": False,
        "db_completed": 0,
        "repro_done": True,
        "final_exists": False,
    }
    planner = RecoveryPlanner(configured_n=100)
    tasks = planner.plan(state)
    names = [t.name for t in tasks]
    assert names == ["finalize"]
