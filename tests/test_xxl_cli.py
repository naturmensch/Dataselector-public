import json
import sys
from pathlib import Path
from tests._helpers.load_script import load_script
ROOT = Path(__file__).resolve().parents[1]
xxl = load_script(ROOT / "scripts" / "xxl_KDR146_run_thesis_complete.py", module_name="scripts.xxl_KDR146_run_thesis_complete_test")


def test_cli_finalize_phase(monkeypatch, tmp_path):
    # Make module operate in tmp_path
    monkeypatch.setattr(xxl, "ROOT", tmp_path)

    # Fake extractor: write a trivial final selection JSON
    def fake_extract(root):
        out_latest = tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json"
        out_latest.parent.mkdir(parents=True, exist_ok=True)
        out_latest.write_text(json.dumps({"run_id": "test_run", "best_value": 1.0}))
        return {"run_id": "test_run", "best_value": 1.0}

    monkeypatch.setattr(xxl, "_extract_xxl_final_statistics", fake_extract)

    monkeypatch.setattr(sys, "argv", ["xxl", "--phase", "finalize"])
    ret = xxl.main()
    assert ret == 0
    assert (tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json").exists()


def test_cli_repro_phase(monkeypatch, tmp_path):
    monkeypatch.setattr(xxl, "ROOT", tmp_path)

    calls = []

    def fake_run_cmd(cmd, retries=2, delay=5, cwd=None, fail_ok=False):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(xxl, "run_cmd_with_retry", fake_run_cmd)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xxl",
            "--phase",
            "repro",
            "--seeds",
            "101,102",
            "--n-trials",
            "10",
            "--n-candidates",
            "50",
        ],
    )
    ret = xxl.main()
    assert ret == 0
    # Expect 2 reproducibility runs
    assert len(calls) == 2
    assert any("--seed 101" in c for c in calls)
    assert any("--seed 102" in c for c in calls)
    assert all("--n-trials 10" in c and "--n-candidates 50" in c for c in calls)
