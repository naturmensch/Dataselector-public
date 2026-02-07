from dataselector.workflows.recovery import RecoveryPlanner


def test_planner_basic_optuna_repro_finalize():
    planner = RecoveryPlanner(configured_n=10, repro_seeds=[43, 44])
    state = {
        "csv_exists": True,
        "csv_completed": 3,
        "db_exists": False,
        "repro_done": False,
        "final_exists": False,
        "n_samples": 5,
    }
    tasks = planner.plan(state)
    names = [t.name for t in tasks]
    assert names == ["optuna", "repro", "finalize"]
    # The optuna task must request the remaining trials
    opt_task = tasks[0]
    assert opt_task.params["n_trials"] == 7


def test_planner_reconstruct_when_db_no_csv():
    planner = RecoveryPlanner(configured_n=5, repro_seeds=[1])
    state = {
        "db_exists": True,
        "csv_exists": False,
        "csv_completed": 0,
        "repro_done": True,
        "final_exists": True,
    }
    tasks = planner.plan(state)
    names = [t.name for t in tasks]
    assert "reconstruct" in names


def test_planner_no_optuna_if_already_complete():
    planner = RecoveryPlanner(configured_n=3, repro_seeds=[])
    state = {
        "csv_exists": True,
        "csv_completed": 3,
        "repro_done": False,
        "final_exists": False,
    }
    tasks = planner.plan(state)
    names = [t.name for t in tasks]
    assert "optuna" not in names
    assert "repro" in names and "finalize" in names
