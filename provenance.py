"""Run directories and provenance sidecars.

Every script writes to results/<script>/<YYYYMMDD-HHMM>/ (never overwriting an
earlier run) and saves a meta.json next to its arrays with the checkpoint, git
state, normalization, totals source, z grid, stride, split and feature labels.

Usage:
    from provenance import make_run_dir, write_meta
    run_dir = make_run_dir(__file__)
    write_meta(run_dir, checkpoint=ckpt_path, stride=4, split='test', ...)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))


def git_state(repo=REPO):
    """Commit hash, branch and whether the working tree differs from HEAD."""
    try:
        run = lambda *a: subprocess.check_output(['git', '-C', repo, *a], text=True,
                                                 stderr=subprocess.DEVNULL).strip()
        return {'git_hash': run('rev-parse', 'HEAD'),
                'git_branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
                'git_dirty': run('status', '--porcelain', '--untracked-files=no') != ''}
    except (subprocess.CalledProcessError, OSError):
        return {'git_hash': None, 'git_branch': None, 'git_dirty': None}


def make_run_dir(script, base=None):
    """results/<script-name>/<stamp>/ ; a suffix is added if the stamp exists."""
    base = base or os.path.join(REPO, 'results')
    name = os.path.splitext(os.path.basename(script))[0]
    stamp = time.strftime('%Y%m%d-%H%M')
    run_dir, i = os.path.join(base, name, stamp), 1
    while os.path.exists(run_dir):
        run_dir, i = os.path.join(base, name, f'{stamp}_{i}'), i + 1
    os.makedirs(run_dir)
    return run_dir


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def write_meta(run_dir, **meta):
    """Write meta.json (timestamp, argv, git state, plus the given fields)."""
    meta = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'argv': sys.argv,
            'cwd': os.getcwd(), **git_state(), **meta}
    with open(os.path.join(run_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=_default)
    return meta
