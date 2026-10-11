# ablation_table.py: held-out metrics of the gnn_inference runs grouped by checkpoint, one row per set of zeroed inputs
# (zero_features in meta.json; '' = none), so that the effect of removing an input at inference can be read off.
#   python plot_scripts/ablation_table.py [out.md]
import json, os, sys
from glob import glob

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, 'results', 'ablation_table.md')
bands = ['z<0', '0-0.5', '0.5-1', '1-1.5', '1.5-2.2', '2.2-3.5', '>3.5']

rows = {}
for meta_path in sorted(glob(os.path.join(repo, 'results', 'gnn_inference', '*', 'meta.json'))):
    meta = json.load(open(meta_path))
    metrics_path = os.path.join(os.path.dirname(meta_path), 'metrics.json')
    if not os.path.exists(metrics_path) or meta.get('split') != 'test':
        continue
    metrics = json.load(open(metrics_path))
    name = os.path.basename(os.path.dirname(meta['checkpoint'])) + '/' + os.path.basename(meta['checkpoint'])
    key = (name, meta['stride'], ','.join(meta.get('zero_features', [])))
    rows[key] = (os.path.basename(os.path.dirname(meta_path)), metrics)       # the latest run of each combination wins

lines = ['# Input ablations at inference (held-out corner, guard 32 px, RMS log10(n_pred/n_target) in dex)', '',
         '| checkpoint | stride | zeroed inputs | all | relevance-weighted | ' + ' | '.join(bands) + ' | run |',
         '|---|---|---|---|---|' + '---|' * len(bands) + '---|']
for (name, stride, zeroed) in sorted(rows):
    run, m = rows[(name, stride, zeroed)]
    if not any(k[0] == name and k[2] for k in rows):      # only checkpoints with at least one ablation run
        continue
    lines.append(f"| {name} | {stride} | {zeroed or '-'} | {m['all']['rms']:.4f} | {m['relevance_weighted_rms']:.4f} | "
                 + ' | '.join(f"{m['per_band'][b]['rms']:.4f}" for b in bands) + f" | {run} |")
open(out_path, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\nwritten to {out_path}')
