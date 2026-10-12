"""Table of held-out dex metrics over gnn_inference.py runs (replaces the stride comparison of stride_tests.ipynb).

    python plot_scripts/stride_table.py [results/gnn_inference]      (default: <repo>/results/gnn_inference; prints markdown, redirect to results/stride_table.md)
    'grid' = evaluation grid; 'zeroed' = inputs set to zero at inference (ablation runs of gnn_inference.py), '-' = none
"""
import glob
import json
import os
import sys

base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'gnn_inference')
rows = []
for run in sorted(glob.glob(os.path.join(base, '*'))):
    try:
        meta = json.load(open(os.path.join(run, 'meta.json')))
        m = json.load(open(os.path.join(run, 'metrics.json')))
    except (FileNotFoundError, json.JSONDecodeError):
        continue
    ck = meta['checkpoint'].split('/checkpoints/')[-1]
    bands = m['per_band']
    rows.append((os.path.basename(run), ck, meta['stride'], meta['split'], meta.get('grid', 'index'), ','.join(meta.get('zero_features', [])) or '-',
                 m['region']['guard'], m['region']['n_columns_metric'],
                 m['all']['rms'], m['relevance_weighted_rms'], m['all']['median_abs'], m['all']['p95_abs'],
                 bands['0-0.5']['rms'] if bands.get('0-0.5') else float('nan'), bands['1-1.5']['rms'] if bands.get('1-1.5') else float('nan'),
                 bands['1.5-2.2']['rms'] if bands.get('1.5-2.2') else float('nan'), bands['>3.5']['rms'] if bands.get('>3.5') else float('nan'),
                 m.get('audit_region_358_484_rms_all', float('nan'))))
hdr = ('run', 'checkpoint', 's', 'split', 'grid', 'zeroed', 'guard', 'ncol', 'RMS', 'wRMS', 'med|e|', 'p95|e|', '0-0.5', '1-1.5', '1.5-2.2', '>3.5', 'audit[358,484)')
print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
for r in rows:
    print('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r) + ' |')
