# ensemble.py: the seed ensemble of several gnn_inference.py or muram_inference.py runs of the same region, grid and stride:
# the mean of log10 n over the members, their spread (std over the seeds, ddof 1) and, for Bifrost runs, the metrics of the
# ensemble (as gnn_inference.py: own grid and end to end at the native levels) and the coverage of the per-node error bar
# sigma^2 = a^2 + (c s)^2 + F(z, l)^2.
#   python ensemble.py <run_dir_1> <run_dir_2> [<run_dir_3> ...]
# Writes results/ensemble/<stamp>/ with the same files as a member run (the ensemble predictions under the member's file
# name, the member's targets, masks, grid and atmosphere), spread_dex.npy, metrics.json and meta.json, so that
# compare_populations.py, ood_table.py, muram_synthesis.py, synthesis.py and plot_scripts/inference_plots.py run on it
# like on a single run.
import json
import os
import shutil
import sys

import numpy as np
from scipy.interpolate import interp1d

from provenance import make_run_dir, write_meta

# Per-node error bar of the ensemble mean on Bifrost-like atmospheres, native resolution: sigma^2 = a^2 + (c s)^2 + F(z, l)^2
# with s the seed spread of log10 n and F(z, l) the interpolation floor of the evaluation grid per native height and Ca level
# (RMS over the held-out columns of log10 of the targets on the grid interpolated to the native levels minus the native
# targets). a, c fitted per height band by Gaussian maximum likelihood on the held-out corner for the three-seed ensemble on
# the dense:60 grid at stride 4 (2026-10-17): 2-sigma coverage 0.942 on the other half of the columns (0.934-0.949 per band).
ERROR_BAR_A_C = {'z<0': (0.0031, 0.909), '0-0.5': (0.0015, 0.755), '0.5-1': (0.0000, 2.448), '1-1.5': (0.0244, 2.965),
                 '1.5-2.2': (0.0518, 3.008), '2.2-3.5': (0.0773, 2.646), '>3.5': (0.0025, 4.351)}
BANDS = [(-np.inf, 0.0, 'z<0'), (0.0, 0.5, '0-0.5'), (0.5, 1.0, '0.5-1'), (1.0, 1.5, '1-1.5'), (1.5, 2.2, '1.5-2.2'), (2.2, 3.5, '2.2-3.5'), (3.5, np.inf, '>3.5')]
WINDOWS = [('8542 core', 1.15, 1.73), ('H core', 1.54, 2.75), ('K core', 1.60, 2.95), ('TR', 1.84, 3.82)]      # Bifrost FITS frame (p5-p95)
level_names = ['4s', '3d3/2', '3d5/2', '4p1/2', '4p3/2', 'CaIII']
rms = lambda v: float(np.sqrt(np.nanmean(v**2)))

members = sys.argv[1:]
assert len(members) >= 2, 'give at least two run directories'
metas = [json.load(open(os.path.join(m, 'meta.json'))) for m in members]
bifrost = 'targets' in metas[0]['files']
for key in ('stride', 'grid') + (('split', 'x0', 'x1', 'y0', 'y1', 'guard') if bifrost else ('crop', 'tau1_index_median')):
    assert all(m[key] == metas[0][key] for m in metas), (key, [m[key] for m in metas])
logs = np.stack([np.log10(np.load(os.path.join(m, m_meta['files']['predictions']))) for m, m_meta in zip(members, metas)])   # (M, nz, ny, nx, nlev)
mean_log = logs.mean(0)
spread = logs.std(0, ddof=1).astype(np.float32)
del logs

run_dir = make_run_dir(__file__)
for name, f in metas[0]['files'].items():
    if name != 'predictions':
        shutil.copy(os.path.join(members[0], f), run_dir)
for f in ('metric_mask.npy', 'test_mask.npy', 'zz_grid.npy'):
    if os.path.exists(os.path.join(members[0], f)):
        shutil.copy(os.path.join(members[0], f), run_dir)
np.save(os.path.join(run_dir, metas[0]['files']['predictions']), 10.0 ** mean_log)
np.save(os.path.join(run_dir, 'spread_dex.npy'), spread)
print(f"{len(members)} members, mean log10 n and spread (median {np.median(spread):.4f} dex) written to {run_dir}")

metrics = {}
if bifrost:
    meta0 = metas[0]
    zz_grid = np.load(os.path.join(run_dir, 'zz_grid.npy'))
    metric_mask = np.load(os.path.join(run_dir, 'metric_mask.npy'))
    targets = np.load(os.path.join(run_dir, meta0['files']['targets']))[:, metric_mask, :]
    pred = (10.0 ** mean_log)[:, metric_mask, :]
    nlev = pred.shape[-1]
    datadir = meta0['datadir'] if os.path.isabs(meta0['datadir']) else os.path.join(os.path.dirname(os.path.abspath(__file__)), meta0['datadir'])
    geometry_grid = np.load(os.path.join(os.path.dirname(datadir), 'en024048_hion', 'grid_bifrost.npz'))['z']
    pops_orig = np.memmap(os.path.join(datadir, 'AR_385_CaII_5L_pops.dat'), dtype='<f4', mode='r', shape=(len(geometry_grid), 504, 504, nlev))
    native = np.asarray(pops_orig[:, meta0['y0']:meta0['y1'], meta0['x0']:meta0['x1'], :], dtype=np.float64)[:, metric_mask, :]

    # --- own-grid metrics, as gnn_inference.py: e = log10(n_pred / n_target) per level and height band, Ca II-relevance weighted ---
    with np.errstate(divide='ignore', invalid='ignore'):
        e_dex = np.log10(pred) - np.log10(targets)
    e_dex[~np.isfinite(e_dex)] = np.nan
    z_nodes = np.broadcast_to(zz_grid[:, None, None], e_dex.shape)
    w_rel = np.clip((1.0 - targets[..., -1] / targets.sum(-1)) / 1e-2, 0.0, 1.0)[..., None]
    metrics = {'region': {k: int(meta0[k]) for k in ('x0', 'x1', 'y0', 'y1', 'guard')},
               'all': {'rms': rms(e_dex), 'median_abs': float(np.nanmedian(np.abs(e_dex))), 'p95_abs': float(np.nanpercentile(np.abs(e_dex), 95))},
               'relevance_weighted_rms': float(np.sqrt(np.nansum(w_rel * e_dex**2) / np.nansum(w_rel * np.isfinite(e_dex)))),
               'per_level': {level_names[l]: {'rms': rms(e_dex[..., l]), 'bias': float(np.nanmean(e_dex[..., l]))} for l in range(nlev)},
               'per_band': {name: {'rms': rms(e_dex[(z_nodes >= lo) & (z_nodes < hi)])} for lo, hi, name in BANDS}}
    metrics['region']['n_columns_metric'] = int(metric_mask.sum())
    # --- end-to-end at the native levels, as gnn_inference.py, plus the line-formation windows and the error bar ---
    z_eval = np.clip(geometry_grid, zz_grid[0], zz_grid[-1])
    to_native = lambda a: interp1d(zz_grid, a, axis=0, assume_sorted=True)(z_eval)
    with np.errstate(divide='ignore', invalid='ignore'):
        l_native = np.log10(native)
        e_native = to_native(np.log10(pred)) - l_native
        e_floor = to_native(np.log10(targets)) - l_native
    e_native[~np.isfinite(e_native)] = np.nan
    e_floor[~np.isfinite(e_floor)] = np.nan
    w_native = np.clip((1.0 - native[..., -1] / native.sum(-1)) / 1e-2, 0.0, 1.0)[..., None]
    band_of = lambda e, lo, hi: e[(geometry_grid >= lo) & (geometry_grid < hi)]
    metrics['native'] = {'all': rms(e_native), 'floor_all': rms(e_floor),
                         'relevance_weighted_rms': float(np.sqrt(np.nansum(w_native * e_native**2) / np.nansum(w_native * np.isfinite(e_native)))),
                         'per_band': {name: {'rms': rms(band_of(e_native, lo, hi)), 'floor_rms': rms(band_of(e_floor, lo, hi))} for lo, hi, name in BANDS},
                         'per_level': {level_names[l]: {'rms': rms(e_native[..., l])} for l in range(nlev)},
                         'windows': {name: {'rms': rms(band_of(e_native, lo, hi)), 'floor_rms': rms(band_of(e_floor, lo, hi))} for name, lo, hi in WINDOWS}}
    s = to_native(spread[:, metric_mask].astype(np.float64))
    F = np.sqrt(np.nanmean(e_floor ** 2, axis=1))                                             # (n_native, nlev)
    sigma = np.empty_like(e_native)
    for lo, hi, name in BANDS:
        sel = (geometry_grid >= lo) & (geometry_grid < hi)
        a, c = ERROR_BAR_A_C[name]
        sigma[sel] = np.sqrt(a ** 2 + (c * s[sel]) ** 2 + F[sel][:, None, :] ** 2)
    ok = np.isfinite(e_native)
    cover = lambda m: {'cover_1sigma': float(np.mean(np.abs(e_native[m & ok]) < sigma[m & ok])), 'cover_2sigma': float(np.mean(np.abs(e_native[m & ok]) < 2 * sigma[m & ok])),
                       'rms_e': rms(e_native[m]), 'median_spread': float(np.median(s[m]))}
    metrics['error_bar'] = {'definition': 'sigma^2 = a^2 + (c s)^2 + F(z, l)^2 at the native levels; a, c per band (ERROR_BAR_A_C), s = seed spread, F = floor of the grid',
                            'all': cover(np.ones(e_native.shape, bool)),
                            **{name: cover(np.broadcast_to(((geometry_grid >= lo) & (geometry_grid < hi))[:, None, None], e_native.shape)) for lo, hi, name in BANDS}}
    print(f"RMS log10(n_pred/n_target) all levels: {metrics['all']['rms']:.4f} dex | Ca II-relevance-weighted {metrics['relevance_weighted_rms']:.4f} dex")
    print(f"native resolution ({len(geometry_grid)} levels): RMS {metrics['native']['all']:.4f} dex, Ca II-weighted {metrics['native']['relevance_weighted_rms']:.4f}, "
          f"interpolation floor of the grid {metrics['native']['floor_all']:.4f}")
    print('native per band : ' + '  '.join(f"{k} {v['rms']:.4f} (floor {v['floor_rms']:.4f})" for k, v in metrics['native']['per_band'].items()))
    print('native windows  : ' + '  '.join(f"{k} {v['rms']:.4f}" for k, v in metrics['native']['windows'].items()))
    print('error bar 2-sigma coverage: ' + '  '.join(f"{k} {v['cover_2sigma']:.3f}" for k, v in metrics['error_bar'].items() if k != 'definition') + '  (Gaussian 0.954)')
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1)

# the metadata of the first member (its checkpoint stands for the config, grid and climatology that the scoring scripts read)
write_meta(run_dir, **{k: v for k, v in metas[0].items() if k not in ('timestamp', 'argv', 'cwd', 'git_hash', 'git_branch', 'git_dirty', 'device')},
           members=[os.path.abspath(m) for m in members], checkpoints=[m['checkpoint'] for m in metas],
           ensemble='mean of log10 n over the members; spread_dex.npy = std over the members (ddof 1)')
print(f"Saved to {run_dir}")
