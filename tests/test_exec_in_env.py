import subprocess
import sys
import shutil
import pytest

MAMBA = shutil.which('mamba')
if MAMBA is None:
    pytest.skip('mamba not available; skipping env wrapper integration tests', allow_module_level=True)


def conda_env_exists(name: str) -> bool:
    res = subprocess.run(['mamba', 'env', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return name in res.stdout


def test_exec_in_env_runs_simple_command():
    env_name = 'dataselector'
    if not conda_env_exists(env_name):
        pytest.skip(f'Conda env {env_name} not found; skipping')

    out = subprocess.check_output(['./scripts/exec_in_env.sh', '--env', env_name, '--', 'python', '-c', "print('ok-from-env')"], text=True)
    assert 'ok-from-env' in out
