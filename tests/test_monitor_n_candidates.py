from dataselector.workflows.xxl_monitor import _detect_n_candidates


def test_detect_n_candidates_from_csv(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv = data_dir / "new_all_tiles.csv"

    lines = ["longName\n"] + [f"tile_{i}\n" for i in range(676)]
    csv.write_text("".join(lines), encoding="utf-8")

    detected = _detect_n_candidates(root=tmp_path)
    assert detected == 676


def test_detect_n_candidates_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("DATASET_N_CANDIDATES", raising=False)
    monkeypatch.setenv("DATASET_N_CANDIDATES", "42")

    detected = _detect_n_candidates(root=tmp_path)
    assert detected == 42
