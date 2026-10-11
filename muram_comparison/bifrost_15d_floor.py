"""1.5D-vs-3D reference floor on Bifrost (plan WP3 step 3.4).

Solves the Ca II NLTE problem column by column with Lightweaver on strided Bifrost columns, on the NATIVE
425-level grid, with the setup assumed for the MURaM reference (CaII_atom active only, Bifrost n_e and n_H,
v_turb 0, CRD, 5-point quadrature), and compares with the 3D targets: log10(n_1.5D / n_3D) per level and height
band. This is the floor below which no Bifrost-trained GNN can be distinguished from the 1.5D MURaM reference.

    python muram_comparison/bifrost_15d_floor.py [stride] [n_jobs] [max_cols]
"""
import json
import multiprocessing
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

import lightweaver as lw
import lightweaver.constants as const
from lightweaver.atomic_table import DefaultAtomicAbundance
from lightweaver.rh_atoms import (Al_atom, C_atom, CaII_atom, Fe_atom, H_6_atom, He_9_atom, MgII_atom, N_atom,
                                  Na_atom, OI_ord_atom, S_atom, Si_atom)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]
from provenance import make_run_dir, write_meta     # noqa: E402
import muram_common as mc                           # noqa: E402

DATADIR = os.path.join(os.path.dirname(REPO), 'data_train')
NZ, NY, NX, NLEV = 425, 504, 504, 6
BANDS = [(-np.inf, 0.0, 'z<0'), (0.0, 0.5, '0-0.5'), (0.5, 1.0, '0.5-1'), (1.0, 1.5, '1-1.5'),
         (1.5, 2.2, '1.5-2.2'), (2.2, 3.5, '2.2-3.5'), (3.5, np.inf, '>3.5')]

# usage: python bifrost_15d_floor.py [stride] [n_jobs] [max_cols]   -- stride 11 -> 46^2 = 2116 columns
class args:                        # the setup assumed for the MURaM reference; change here if the reference turns out different
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else min(32, multiprocessing.cpu_count())
    max_cols = int(sys.argv[3]) if len(sys.argv) > 3 else None
    vturb = 0.0
    prd = False
    conserve_charge = False        # Bifrost n_e kept fixed
    active_h = False               # Ca active only

run_dir = make_run_dir(__file__)
z_native = np.load(mc.BIFROST_GRID)['z']                       # Mm, FITS frame
z_tau = z_native - mc.ZTAU51_BIFROST                            # tau500 = 1 frame
mm = lambda name, nc=1: np.memmap(os.path.join(DATADIR, name), dtype='<f4', mode='r', shape=(NZ, NY, NX, nc))
temp, vel, ne, rho, pops = mm('AR_385_temp.dat'), mm('AR_385_veloc.dat', 3), mm('AR_385_ne.dat'), mm('AR_385_mass.dat'), mm('AR_385_CaII_5L_pops.dat', NLEV)
ys = np.arange(args.stride // 2, NY, args.stride)
xs = np.arange(args.stride // 2, NX, args.stride)
cols = [(int(iy), int(ix)) for iy in ys for ix in xs]
if args.max_cols:
    cols = cols[:args.max_cols]
print(f"{len(cols)} columns, stride {args.stride}; writing to {run_dir}")

SL = slice(None, None, -1)                                      # top first for Lightweaver
zz_m = np.ascontiguousarray(z_native[SL].astype(np.float64) * 1e6)      # Lightweaver needs float64 buffers


def solve_column(iy, ix, T, vz, n_e, rh, vturb, prd, conserve, active_h):
    try:
        atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=zz_m, convertScales=False,
                                      temperature=T, vlos=vz, vturb=np.full_like(T, vturb), ne=n_e,
                                      nHTot=rh / (const.Amu * DefaultAtomicAbundance.massPerH))
        atmos.quadrature(5)
        aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(),
                                He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
        aSet.set_active(*(('H', 'Ca') if active_h else ('Ca',)))
        spect = aSet.compute_wavelength_grid()
        eqPops = aSet.compute_eq_pops(atmos)
        ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=conserve)
        lw.iterate_ctx_se(ctx, prd=prd, quiet=True)
        n = np.array(eqPops.atomicPops['Ca'].n)                 # (nlev, nz) m^-3, top first
        return iy, ix, n[:, ::-1].T * 1e-6                       # (nz, nlev) cm^-3, bottom first
    except Exception as exc:                                    # noqa: BLE001
        print(f'column ({iy},{ix}) failed: {exc}')
        return iy, ix, None


t0 = time.time()
res = Parallel(n_jobs=args.n_jobs, verbose=2, batch_size=4)(
    delayed(solve_column)(iy, ix, np.array(temp[SL, iy, ix, 0], dtype=np.float64), np.array(vel[SL, iy, ix, 2], dtype=np.float64),
                          np.array(ne[SL, iy, ix, 0], dtype=np.float64), np.array(rho[SL, iy, ix, 0], dtype=np.float64),
                          args.vturb, args.prd, args.conserve_charge, args.active_h) for iy, ix in cols)
print(f"done in {(time.time() - t0) / 60:.1f} min")
n15 = np.full((len(cols), NZ, NLEV), np.nan)
n3d = np.full((len(cols), NZ, NLEV), np.nan)
for i, (iy, ix, n) in enumerate(res):
    if n is not None:
        n15[i] = n
        n3d[i] = pops[:, iy, ix, :]
ok = np.isfinite(n15[:, 0, 0])
np.savez_compressed(os.path.join(run_dir, 'bifrost_15d_vs_3d.npz'), columns=np.array(cols), n15d_cm3=n15, n3d_cm3=n3d, z_native_Mm=z_native)
e = np.log10(n15[ok]) - np.log10(n3d[ok])                      # (ncol, nz, nlev)
zk = np.broadcast_to(z_tau[None, :, None], e.shape)
out = {'n_columns': int(ok.sum()), 'n_failed': int((~ok).sum()), 'setup': {k: v for k, v in vars(args).items() if not k.startswith('_')}, 'per_band': {}}
for lo, hi, name in BANDS:
    m = (zk >= lo) & (zk < hi)
    out['per_band'][name] = {mc.LEVEL_NAMES[l]: {'rms': float(np.sqrt(np.nanmean(e[..., l][m[..., l]]**2))),
                                                 'median_abs': float(np.nanmedian(np.abs(e[..., l][m[..., l]]))),
                                                 'bias': float(np.nanmean(e[..., l][m[..., l]]))} for l in range(NLEV)}
    out['per_band'][name]['all'] = {'rms': float(np.sqrt(np.nanmean(e[m]**2))), 'median_abs': float(np.nanmedian(np.abs(e[m]))), 'bias': float(np.nanmean(e[m]))}
with open(os.path.join(run_dir, 'floor.json'), 'w') as f:
    json.dump(out, f, indent=1)
write_meta(run_dir, **{k: v for k, v in vars(args).items() if not k.startswith('_')}, datadir=DATADIR, bands_frame='tau500=1 (z_native - ZTAU51)')
print(f"\n1.5D vs 3D on {ok.sum()} Bifrost columns: log10(n_1.5D/n_3D) RMS [bias] per band (tau500=1 frame)")
print(f"{'band':8s} " + ' '.join(f'{n:>16s}' for n in mc.LEVEL_NAMES + ['all']))
for name, d in out['per_band'].items():
    print(f"{name:8s} " + ' '.join(f"{d[n]['rms']:7.3f} [{d[n]['bias']:+6.3f}]" for n in mc.LEVEL_NAMES + ['all']))
print(f"Written to {run_dir}")
