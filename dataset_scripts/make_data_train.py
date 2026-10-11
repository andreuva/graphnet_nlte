"""Regenerate the Bifrost training data from the FITS files (plan WP4 step 4.1).

Reads en024048_hion/atmos/BIFROST_en024048_hion_<var>_385.fits (SI units, 496 levels), slices the vertical
indices [52:477] (425 levels, the grid of en024048_hion/grid_bifrost.npz) and writes data_train_v2/ with the
same file names and layout as data_train/ (float32 little-endian, (nz, ny, nx, ncomp)), with the magnetic
field read from the bx, by AND bz files (data_train has Bz == By because the notebook read `by` twice),
plus AR_385_nH_levels.dat (the six hydrogen level populations) and a MANIFEST.json with shapes, units,
sources and SHA-256 of every file. The Ca II populations are copied unchanged from data_train.

    python dataset_scripts/make_data_train.py [output_dir]
"""
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(REPO)
ATMOS = os.path.join(BASE, 'en024048_hion', 'atmos')
GRID = os.path.join(BASE, 'en024048_hion', 'grid_bifrost.npz')
OLD = os.path.join(BASE, 'data_train')
Z0, Z1 = 52, 477
NLEV = 6

class args:                        # usage: python make_data_train.py [output_dir]
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'data_train_v2')
os.makedirs(args.out, exist_ok=True)
sys.path.insert(0, REPO)
from provenance import git_state      # noqa: E402


def fits_path(var):
    return os.path.join(ATMOS, f'BIFROST_en024048_hion_{var}_385.fits')


def read(var, log=False):
    with fits.open(fits_path(var), memmap=True) as h:
        a = np.asarray(h[0].data[Z0:Z1], dtype=np.float32)        # (425, ny, nx), big-endian -> native float32
        unit, btype, z = h[0].header.get('BUNIT'), h[0].header.get('BTYPE'), np.asarray(h[1].data)
    return (10.0 ** a).astype(np.float32) if log else a, str(unit), str(btype), z


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


manifest = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_dir': ATMOS, 'z_slice': [Z0, Z1], 'grid_file': GRID,
            'layout': '(nz, ny, nx, ncomp) float32 little-endian, C order; nz = 425, ny = nx = 504', **git_state(REPO),
            'notes': ['Bz read from the bz file (data_train/AR_385_B.dat has Bz == By).',
                      'AR_385_CaII_5L_pops.dat copied from data_train unchanged (cm^-3, provenance: see README).',
                      'AR_385_nh.dat = 10^lgn1 (ground-level neutral hydrogen), AR_385_np.dat = 10^lgn6 (protons), both m^-3.'],
            'files': {}}
t0 = time.time()
z_fits = None
spec = {  # output file: (list of (fits var, log?), unit)
    'AR_385_temp.dat': ([('lgtg', True)], 'K'),
    'AR_385_ne.dat': ([('lgne', True)], 'm^-3'),
    'AR_385_nh.dat': ([('lgn1', True)], 'm^-3 (H ground level)'),
    'AR_385_np.dat': ([('lgn6', True)], 'm^-3 (protons)'),
    'AR_385_mass.dat': ([('lgr', True)], 'kg m^-3'),
    'AR_385_press.dat': ([('lgp', True)], 'Pa'),
    'AR_385_B.dat': ([('bx', False), ('by', False), ('bz', False)], 'T (components x, y, z)'),
    'AR_385_veloc.dat': ([('ux', False), ('uy', False), ('uz', False)], 'm s^-1 (components x, y, z; positive z = upward)'),
    'AR_385_nH_levels.dat': ([(f'lgn{i}', True) for i in range(1, 7)], 'm^-3 (H levels 1-5 and protons)'),
}
for name, (vars_, unit) in spec.items():
    comps, sources = [], []
    for var, log in vars_:
        a, bunit, btype, z = read(var, log)
        if z_fits is None:
            z_fits = z
        assert np.array_equal(z, z_fits), var
        comps.append(a)
        sources.append({'fits': os.path.basename(fits_path(var)), 'BUNIT': bunit, 'BTYPE': btype, 'log10_in_file': log})
    arr = np.stack(comps, axis=-1)
    path = os.path.join(args.out, name)
    arr.astype('<f4').tofile(path)
    manifest['files'][name] = {'shape': list(arr.shape), 'dtype': '<f4', 'unit': unit, 'sources': sources, 'sha256': sha256(path)}
    print(f'{name:26s} {arr.shape} written  ({time.time() - t0:.0f} s)')
    del arr, comps

# Ca II populations: copied unchanged
src = os.path.join(OLD, 'AR_385_CaII_5L_pops.dat')
dst = os.path.join(args.out, 'AR_385_CaII_5L_pops.dat')
shutil.copyfile(src, dst)
manifest['files']['AR_385_CaII_5L_pops.dat'] = {'shape': [Z1 - Z0, 504, 504, NLEV], 'dtype': '<f4', 'unit': 'cm^-3',
                                                'sources': [{'copied_from': src}], 'sha256': sha256(dst)}
# the grid file must be the FITS z axis sliced the same way
zg = np.load(GRID)['z']
manifest['grid_check'] = {'max_abs_diff_Mm': float(np.max(np.abs(zg - z_fits[Z0:Z1]))), 'nz': int(len(zg))}
assert manifest['grid_check']['max_abs_diff_Mm'] < 1e-5, manifest['grid_check']

# ---- Verification against data_train ----
sh = (Z1 - Z0, 504, 504)
mm = lambda d, n, c=1: np.memmap(os.path.join(d, n), dtype='<f4', mode='r', shape=sh + (c,))
checks = {}
for n, c in (('AR_385_temp.dat', 1), ('AR_385_ne.dat', 1), ('AR_385_nh.dat', 1), ('AR_385_np.dat', 1), ('AR_385_mass.dat', 1),
             ('AR_385_press.dat', 1), ('AR_385_veloc.dat', 3)):
    a, b = mm(OLD, n, c), mm(args.out, n, c)
    checks[n] = 'identical to data_train' if np.array_equal(a, b) else f'DIFFERS: max rel {float(np.max(np.abs(a - b) / (np.abs(a) + 1e-30))):.3e}'
Bo, Bn = mm(OLD, 'AR_385_B.dat', 3), mm(args.out, 'AR_385_B.dat', 3)
checks['AR_385_B.dat'] = {'bx_identical': bool(np.array_equal(Bo[..., 0], Bn[..., 0])), 'by_identical': bool(np.array_equal(Bo[..., 1], Bn[..., 1])),
                          'old_bz_equals_old_by': bool(np.array_equal(Bo[..., 1], Bo[..., 2])), 'new_bz_equals_new_by': bool(np.array_equal(Bn[..., 1], Bn[..., 2])),
                          'new_bz_rms_T': float(np.sqrt(np.mean(np.asarray(Bn[::8, ::4, ::4, 2], dtype=np.float64) ** 2)))}
rho = mm(args.out, 'AR_385_mass.dat'); pops = np.memmap(dst, dtype='<f4', mode='r', shape=sh + (NLEV,))
ratio = pops[::16, ::8, ::8].sum(-1) / rho[::16, ::8, ::8, 0]
checks['sum_pops_over_rho'] = {'median': float(np.median(ratio)), 'rel_scatter': float(np.std(ratio) / np.mean(ratio))}
T = mm(args.out, 'AR_385_temp.dat'); V = mm(args.out, 'AR_385_veloc.dat', 3)
k = int(np.argmin(np.abs(zg - 0.0614)))                           # tau500 = 1 level
hot = T[k, :, :, 0] > np.percentile(T[k, :, :, 0], 70)
checks['uz_sign'] = {'level': k, 'median_uz_in_hot_granules_m_s': float(np.median(V[k, :, :, 2][hot])), 'expect': 'positive = upflow'}
manifest['checks'] = checks
with open(os.path.join(args.out, 'MANIFEST.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
print(json.dumps(checks, indent=1))
print(f'Done in {(time.time() - t0) / 60:.1f} min: {args.out}')
