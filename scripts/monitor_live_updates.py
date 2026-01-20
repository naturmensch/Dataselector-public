#!/usr/bin/env python3
import time
from pathlib import Path
import subprocess
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'outputs' / 'XXL_FULL_RUN.log'
PRELOGS = list(ROOT.glob('outputs/pre_run_*.log'))
SEL = ROOT / 'outputs' / 'selected_sampler.json'

def tail(path, n=20):
    if not path.exists():
        return ''
    try:
        return subprocess.check_output(['tail','-n',str(n),str(path)], text=True)
    except Exception:
        try:
            return '\n'.join(path.read_text().splitlines()[-n:])
        except Exception:
            return ''

def proc_info(patterns=['compare_samplers_multi_seed','optuna_optimize']):
    try:
        ps = subprocess.check_output(['pgrep','-a','-f','|'.join(patterns)], text=True, stderr=subprocess.DEVNULL)
        lines = [l for l in ps.splitlines() if l.strip()]
    except Exception:
        # fallback to pgrep for each
        lines = []
        for p in patterns:
            try:
                out = subprocess.check_output(['pgrep','-a','-f',p], text=True, stderr=subprocess.DEVNULL)
                lines.extend([l for l in out.splitlines() if l.strip()])
            except Exception:
                pass
    info = []
    for l in lines:
        fields = l.split(None, 1)
        pid = fields[0]
        cmd = fields[1] if len(fields)>1 else ''
        try:
            stat = subprocess.check_output(['ps','-p',pid,'-o','pid,ppid,%cpu,%mem,etimes,cmd','--no-headers'], text=True)
            info.append(stat.strip())
        except Exception:
            info.append(f"{pid} {cmd}")
    return '\n'.join(info)

print('Starting live monitor (updates every 60s). To stop: kill this process or Ctrl-C.')
while True:
    ts = datetime.now().isoformat()
    print('\n' + '='*80)
    print(f'[{ts}] Live monitor snapshot')
    print('-'*80)
    print('XXL_FULL_RUN.log (tail):')
    print(tail(LOG, 20))
    print('-'*80)
    latest_pre = None
    prefiles = sorted(ROOT.glob('outputs/pre_run_*.log'))
    if prefiles:
        latest_pre = prefiles[-1]
        print(f'Pre-run log ({latest_pre.name}) tail:')
        print(tail(latest_pre, 20))
    else:
        print('No pre-run log present yet')

    print('-'*80)
    print('Active processes matching compare_samplers/optuna:')
    print(proc_info())
    print('-'*80)
    print('Selected sampler artifact:')
    if SEL.exists():
        try:
            print(SEL.read_text())
        except Exception:
            print('Could not read selected_sampler.json')
    else:
        print('NOT PRESENT')

    print('='*80)
    time.sleep(60)
