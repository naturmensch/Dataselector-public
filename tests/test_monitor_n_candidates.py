from pathlib import Path
import importlib.util


def _load_monitor_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "xxl_full_run_monitor.py"
    spec = importlib.util.spec_from_file_location("monitor_mod", str(module_path))
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    return monitor


def test_detect_n_candidates_from_csv(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv = data_dir / "new_all_tiles.csv"
    # header + 676 entries
    lines = ["longName\n"] + [f"tile_{i}\n" for i in range(676)]
    csv.write_text("".join(lines))

    monitor = _load_monitor_module()
    detected = monitor._detect_n_candidates(root=tmp_path)
    assert detected == 676


def test_detect_n_candidates_from_env(monkeypatch, tmp_path):
    # ensure no CSV present
    monkeypatch.delenv("DATASET_N_CANDIDATES", raising=False)
    monitor = _load_monitor_module()
    monkeypatch.setenv("DATASET_N_CANDIDATES", "42")
    detected = monitor._detect_n_candidates(root=tmp_path)
    assert detected == 42
