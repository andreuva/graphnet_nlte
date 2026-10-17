"""Per-height out-of-distribution table for a MURaM inference run (plan WP3 step 3.2).  python ood_table.py [run_dir]

For every level and input component, the fraction of MURaM nodes whose normalized feature (the model's own input
space: the checkpoint's statistics, the scalars interpolated in log10, the LTE input computed on the native levels)
falls outside Bifrost's per-level [p0.1, p99.9] range; plus a per-node flag (any component outside) saved next to the
predictions as ood_flag.npy.

"""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.interpolate import interpn, interp1d

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]
from normalization import normalize_features, lte_log_fractions, expand_labels     # noqa: E402
from vertical_grids import training_grid        # noqa: E402
import muram_common as mc                                     # noqa: E402

FILE_OF = {'vel': 'AR_385_veloc.dat', 'temp': 'AR_385_temp.dat', 'n_e': 'AR_385_ne.dat',
           'rho': 'AR_385_mass.dat', 'press': 'AR_385_press.dat'}
NCOMP = {'vel': 3}
BANDS = [(-np.inf, 0.0, 'z<0'), (0.0, 0.5, '0-0.5'), (0.5, 1.0, '0.5-1'), (1.0, 1.5, '1-1.5'),
         (1.5, 2.2, '1.5-2.2'), (2.2, 3.5, '2.2-3.5'), (3.5, np.inf, '>3.5')]

# usage: python ood_table.py [results/muram_inference/<stamp>]   (default: the most recent run)
results_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(REPO, 'results', 'muram_inference', '*', 'meta.json')))[-1][:-len('meta.json')]
bifrost_step = 4            # column subsampling of Bifrost for the percentiles
plo, phi = 0.1, 99.9        # Bifrost percentile range per level and feature

meta = json.load(open(os.path.join(results_dir, 'meta.json')))
checkpoint = torch.load(meta['checkpoint'], map_location='cpu', weights_only=False)
config = checkpoint['config']
labels = meta['features_labels']
fnp = checkpoint['feature_norm_params']
grid = np.load(os.path.join(results_dir, meta['files']['grid']))
new_z_m, z_rel = grid['muram_fractional_index'], grid['z_rel_tau1_Mm']
dec = meta.get('decimate', 1)
py, px = meta['origin'] if 'origin' in meta else meta['parities'][0]
comp_names = expand_labels(labels)
# 'lte' is derived from temp and n_e on both sides (normalization.lte_log_fractions), on the native levels as in the pipeline
raw_labels = [l for l in labels if l != 'lte'] + [l for l in ('temp', 'n_e') if 'lte' in labels and l not in labels]

# ---- Bifrost percentiles per level in normalized space (subsampled columns) ----
datadir = os.path.normpath(os.path.join(REPO, config['data']['datadir'])) if not os.path.isabs(config['data']['datadir']) else config['data']['datadir']
nx, ny, nz = config['data']['nx'], config['data']['ny'], config['data']['nz_orig']
_, new_z_b, _ = training_grid(config, np.load(config['data']['grid_file'])['z'], meta.get('grid'))   # the levels of this MURaM run
pts = tuple(np.meshgrid(new_z_b, np.arange(0, ny, bifrost_step), np.arange(0, nx, bifrost_step), indexing='ij', sparse=True))
axes = (np.arange(nz), np.arange(ny), np.arange(nx))
mm = lambda l: np.memmap(os.path.join(datadir, FILE_OF[l]), dtype='<f4', mode='r', shape=(nz, ny, nx, NCOMP.get(l, 1)))
feats_b = {l: interpn(axes, mm(l), pts) if l == 'vel' else 10.0 ** interpn(axes, np.log10(np.asarray(mm(l), dtype=np.float64)), pts) for l in raw_labels}
if 'lte' in labels:        # on the native levels, then interpolated linearly in z (as gnn_train.py / gnn_inference.py)
    lte_nat = lte_log_fractions(mm('temp')[:, ::bifrost_step, ::bifrost_step], mm('n_e')[:, ::bifrost_step, ::bifrost_step], config['normalization']['log_offset'])
    feats_b['lte'] = interp1d(np.arange(nz), lte_nat, axis=0, assume_sorted=True)(new_z_b)
feats_b = [feats_b[l] for l in labels]
norm_b = np.concatenate(normalize_features(feats_b, labels, fnp)[0], axis=-1)      # (nz_new, ny/s, nx/s, ncomp)
lo = np.percentile(norm_b, plo, axis=(1, 2))                                                   # (nz_new, ncomp)
hi = np.percentile(norm_b, phi, axis=(1, 2))

# ---- MURaM in the same space ----
cube = mc.load_snap()
fields = mc.muram_si_on_levels(cube, new_z_m, raw_labels)
ny_m, nx_m = np.load(os.path.join(results_dir, meta['files']['rho'])).shape[1:]
feats_m = {l: fields[l][:, py:py + dec * ny_m:dec, px:px + dec * nx_m:dec] for l in raw_labels}     # the predicted native columns
if 'lte' in labels:        # on MURaM's native levels of the predicted columns, then interpolated in z (as muram_inference.py)
    T_nat = np.asarray(cube.Temp)[:, px:px + dec * nx_m:dec, py:py + dec * ny_m:dec].transpose(0, 2, 1)
    ne_nat = np.asarray(cube.ne)[:, px:px + dec * nx_m:dec, py:py + dec * ny_m:dec].transpose(0, 2, 1) * 1e6
    lte_nat = lte_log_fractions(T_nat, ne_nat, config['normalization']['log_offset'])
    feats_m['lte'] = interp1d(np.arange(cube.Temp.shape[0]), lte_nat, axis=0, assume_sorted=True)(new_z_m)
feats_m = [feats_m[l] for l in labels]
norm_m = np.concatenate(normalize_features(feats_m, labels, fnp)[0], axis=-1)      # (nz_new, ny_m, nx_m, ncomp)
outside = (norm_m < lo[:, None, None, :]) | (norm_m > hi[:, None, None, :])
frac = outside.mean(axis=(1, 2))                                                                    # (nz_new, ncomp)
flag = outside.any(axis=-1)
np.save(os.path.join(results_dir, 'ood_flag.npy'), flag)
np.savez(os.path.join(results_dir, 'ood_table.npz'), fraction_outside=frac, levels_z_rel_Mm=z_rel, components=np.array(comp_names),
         bifrost_lo=lo, bifrost_hi=hi, any_component_fraction=flag.mean(axis=(1, 2)))

lines = [f'# MURaM nodes outside Bifrost [p{plo}, p{phi}] in the model input space', '',
         '| level | z [Mm] | ' + ' | '.join(comp_names) + ' | any |', '|' + '---|' * (len(comp_names) + 3)]
for k in range(len(z_rel)):
    lines.append(f'| {k} | {z_rel[k]:.2f} | ' + ' | '.join(f'{100 * frac[k, c]:.0f}%' for c in range(frac.shape[1])) + f' | {100 * flag[k].mean():.0f}% |')
lines += ['', '| band | ' + ' | '.join(comp_names) + ' | any |', '|' + '---|' * (len(comp_names) + 2)]
for lo_b, hi_b, name in BANDS:
    m = (z_rel >= lo_b) & (z_rel < hi_b)
    if m.any():
        lines.append(f'| {name} | ' + ' | '.join(f'{100 * frac[m, c].mean():.0f}%' for c in range(frac.shape[1])) + f' | {100 * flag[m].mean():.0f}% |')
open(os.path.join(results_dir, 'ood_table.md'), 'w').write('\n'.join(lines))
print('\n'.join(lines[-(len(BANDS) + 3):]))
print(f'Written ood_flag.npy, ood_table.npz and ood_table.md to {results_dir}')
