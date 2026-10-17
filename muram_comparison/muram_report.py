"""One-page MURaM report assembled from the result directories (plan WP3 step 3.5).

    python muram_comparison/muram_report.py [output.md]

Collects: the latest 1.5D-vs-3D floor (results/bifrost_15d_floor), every muram_inference run that has
compare_populations.json (GNN vs LTE vs climatology per band), its ood_table and synthesis jsons.
"""
import sys
import glob
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, 'results')
LEVELS = ['4s', '3d3/2', '3d5/2', '4p1/2', '4p3/2', 'CaIII']
BANDS = ['z<0', '0-0.5', '0.5-1', '1-1.5', '1.5-2.2', '2.2-3.5', '>3.5']

class args:                        # usage: python muram_report.py [output.md]
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, 'muram_report.md')
L = [f'# MURaM comparison report ({time.strftime("%Y-%m-%d %H:%M")})', '',
     'All heights above each simulation\'s own column-median tau500 = 1. Populations in cm^-3; dlog = log10(model / SNAPI), the reference being the SNAPI 1.5D NLTE solution (HDU2 of the che_full FITS).',
     'Quantitative claims are meaningful only where the 1.5D-vs-3D floor (first table) is below the model error (plan D-3).', '']

floors = sorted(glob.glob(os.path.join(RES, 'bifrost_15d_floor', '*', 'floor.json')))
if floors:
    fl = json.load(open(floors[-1]))
    L += [f"## 1.5D-vs-3D reference floor on Bifrost ({fl['n_columns']} columns, {os.path.dirname(floors[-1])})", '',
          'RMS [bias] of log10(n_1.5D / n_3D) per band and level; Lightweaver 1.5D with ' +
          f"vturb {fl['setup'].get('vturb', 0)} m/s, {'PRD' if fl['setup'].get('prd') else 'CRD'}, {'H active' if fl['setup'].get('active_h') else 'Ca only'}.", '',
          '| band | ' + ' | '.join(LEVELS) + ' | all |', '|' + '---|' * (len(LEVELS) + 2)]
    for b in BANDS:
        d = fl['per_band'].get(b)
        if d:
            L.append(f'| {b} | ' + ' | '.join(f"{d[l]['rms']:.3f} [{d[l]['bias']:+.2f}]" for l in LEVELS) + f" | {d['all']['rms']:.3f} [{d['all']['bias']:+.2f}] |")
    L.append('')

for run in sorted(glob.glob(os.path.join(RES, 'muram_inference', '*'))):
    cp = os.path.join(run, 'compare_populations.json')
    if not os.path.exists(cp):
        continue
    meta = json.load(open(os.path.join(run, 'meta.json')))
    c = json.load(open(cp))
    L += [f"## {os.path.basename(run)}: {meta['checkpoint'].split('/checkpoints/')[-1]}", '',
          f"features {meta['features_labels']}, stride {meta['stride']}, {c['n_columns']} columns ({meta['decimate']}x{meta['decimate']} averaged), "
          f"tau=1 index {meta['tau1_index_median']:.2f}, totals = {meta['totals_source']}", '',
          f"Median |dlog n| [signed median] on nodes with SNAPI Ca II fraction > {c['relevance_threshold']:g}:", '',
          '| band | n | method | ' + ' | '.join(LEVELS) + ' | all | dlog total |', '|' + '---|' * (len(LEVELS) + 5)]
    for b in BANDS:
        for m in ('gnn', 'lte', 'climatology'):
            meth = c['methods'][m]
            d = meth['per_band_relevant'].get(b) if 'per_band_relevant' in meth else meth.get(b)
            if not d or not d['n_nodes']:
                continue
            if 'levels' in d:          # layout of the first version of compare_populations.py
                L.append(f"| {b} | {d['n_nodes']} | {m} | " + ' | '.join(f"{d['levels'][l]['dlog_n']['p50']:.3f} [{d['levels'][l]['dlog_n']['median_signed']:+.2f}]" for l in LEVELS)
                         + f" | {d['all_levels']['dlog_n']['p50']:.3f} | {d['dlog_total']['median_signed']:+.3f} |")
            else:
                L.append(f"| {b} | {d['n_nodes']} | {m} | " + ' | '.join(f"{d[l]['median_abs']:.3f} [{d[l]['median_signed']:+.2f}]" for l in LEVELS)
                         + f" | {d['all']['median_abs']:.3f} | {d['dlog_total_median']:+.3f} |")
    L.append('')
    ood = os.path.join(run, 'ood_table.md')
    if os.path.exists(ood):
        txt = open(ood).read().split('\n')
        band_tab = txt[txt.index('') + 1:] if '' in txt else txt[-10:]
        L += ['OOD fractions per band (MURaM nodes outside Bifrost [p0.1, p99.9] in the model input space):', ''] + [l for l in txt if l.startswith('| band') or (l.startswith('|') and any(l.startswith(f'| {b} ') for b in BANDS))] + ['']
    for sj in sorted(glob.glob(os.path.join(run, 'muram_synthesis_*.json'))):
        s = json.load(open(sj))
        if 'reference_pops_vs_HDU0' in s:
            r = s['reference_pops_vs_HDU0']
            L.append(f"Ca II K, {s['n_columns']} sample columns (Lightweaver CRD synthesis, mu = 1): SNAPI populations vs SNAPI spectra (cross-code): continuum {r['continuum_median_abs_pct']:.1f} %, "
                     f"wings {r['wing_median_abs_pct']:.1f} %, K3 median {r['centre_median_abs_pct']:.1f} % (bias {r['centre_bias_pct']:+.0f} %). Against the SNAPI-population synthesis: "
                     + ' | '.join(f"{k.split('_vs_')[0]} K3 median {v['centre_median_abs_pct']:.0f} % (bias {v['centre_bias_pct']:+.0f} %)" for k, v in s.items() if k.endswith('_vs_reference_pops')))
    L.append('')
os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out, 'w').write('\n'.join(L))
print('\n'.join(L[:40])); print(f'... written to {args.out}')
