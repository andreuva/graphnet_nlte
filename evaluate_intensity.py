"""
Acceptance test for the Si I GraphNet, measured where it matters: in the emergent profile.

The training loss (MSE on log10(b)/5) is a poor proxy for the quantity the network exists to
produce. It is dominated by transition-region and coronal points whose populations are
astrophysically negligible -- roughly 16% of all target values sit on the +-10 clip plateau at
|y| = 2, while the entire Si I 1083.0 nm line-forming region lives inside |y| <= 0.2. A
checkpoint can improve its MSE without improving a single synthesised profile.

This script answers the question a referee asks instead: *does using the predicted departure
coefficients beat just assuming LTE, and by how much?* For each column it synthesises the
1083.0 nm profile three times --

    reference : the stored, converged lightweaver departure coefficients (ground truth)
    graphnet  : the network's prediction
    lte       : b = 1 everywhere, i.e. no NLTE correction at all

-- and reports the error of `graphnet` and `lte` against `reference` at the line core, averaged
over the core window, and in equivalent width. LTE is the baseline the network has to beat; the
gap between them is the entire value of the model.

Run it on `validation` (a different Bifrost snapshot, snap530), not on `test`: databases
generated before the split fix in generate_database.py share ~80% of their test columns with
train, so a test-split number is a memorisation score.

Usage
-----
    python evaluate_intensity.py --rd ../data_1d_si_v2/ \
                                 --ck checkpoints_si_v2/20260911-231808 \
                                 --n 100 --sav checkpoints_si_v2/20260911-231808/accept/

`--ck` takes a *_best.pth file or a run directory (the most recent *_best.pth in it is used).
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
from tqdm import tqdm

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

import torch

import api


# Vacuum rest wavelength of the Si I 3s2 3p 4p 3PE (j=11) -> 3s2 3p 4s 3PO (i=8) transition,
# 1e7 / (49188.617 - 39955.053) cm-1. This is the line usually quoted by its air wavelength,
# 10827 A -- do not look for the core at 1082.79 nm, that is already the blue wing.
LINE_CORE_NM = 1083.0038

# Default synthesis grid: 2 pm sampling across +-0.6 nm of the core, the same sampling as
# api.DEFAULT_WAVE. The Doppler width of Si is 4.4 pm at the coolest temperature in the training
# set (2500 K, vturb=0) and 6.2 pm at 5000 K, so this keeps at least two samples across the core
# everywhere.
WAVE_LO, WAVE_HI, WAVE_N = 1082.4, 1083.6, 601

# Okabe-Ito, chosen for colour-vision deficiency safety: the reference is a neutral ink rather
# than a hue (it is truth, not a competing series), and blue/vermillion is a CVD-safe pair.
COLOR = {'reference': '#2B3A3C', 'graphnet': '#0072B2', 'lte': '#D55E00'}

# np.trapz was renamed in numpy 2.0; keep working on either.
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def load_split(datadir, prefix):
    """Read the feature and target arrays of one split."""
    out = {}
    for name in ('T', 'z', 'ne', 'vturb', 'vlos', 'logdeparture'):
        with open(os.path.join(datadir, f'{prefix}_{name}.pkl'), 'rb') as fh:
            out[name] = pickle.load(fh)
    return out


def build_atmosphere(T, z, ne, vturb, vlos):
    """
    One lightweaver atmosphere + Si population table, built exactly the way api.py builds it at
    inference time, so that what we measure here is what a caller of api.intensity_gnn gets.
    """
    import lightweaver as lw
    atmos, aSet, spect, eqPops = api._build_atmosphere(T, z, ne, vturb, vlos)
    eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
    return lw, atmos, spect, eqPops


def synthesise(T, z, ne, vturb, vlos, wave, log_dep=None):
    """
    Emergent intensity for one column, with the Si populations set from `log_dep` (or left at
    LTE if None). A single formal solution, no statistical-equilibrium iteration -- the same
    path as api.intensity_gnn, which reproduces the fully converged profile to ~4e-6 in this
    wavelength window.
    """
    lw, atmos, spect, eqPops = build_atmosphere(T, z, ne, vturb, vlos)
    if log_dep is not None:
        nstar = np.asarray(eqPops.atomicPops['Si'].nStar)
        eqPops.atomicPops['Si'].n = np.ascontiguousarray((10.0 ** log_dep) * nstar, dtype=np.float64)
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
    return np.asarray(ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False))


def population_sum_error(T, z, ne, vturb, vlos, log_dep):
    """
    |sum_i n_i / n_Total - 1| per depth point. The converged lightweaver solution satisfies this
    to machine precision; the network decodes all 16 levels independently and so does not, which
    puts a floor on the achievable line depth independent of profile shape.
    """
    _, _, _, eqPops = build_atmosphere(T, z, ne, vturb, vlos)
    nstar = np.asarray(eqPops.atomicPops['Si'].nStar)
    ntot = np.asarray(eqPops.atomicPops['Si'].nTotal)
    return np.abs(((10.0 ** log_dep) * nstar).sum(axis=0) / ntot - 1.0)


def profile_metrics(wave, I_ref, I_test, core_idx, continuum):
    """Error of one trial profile against the reference, at the core and integrated."""
    rel = np.abs(I_test - I_ref) / np.abs(I_ref)
    ew_ref = _trapz(1.0 - I_ref / continuum, wave)
    ew_test = _trapz(1.0 - I_test / continuum, wave)
    return {
        'core_rel': float(rel[core_idx]),
        'window_rel_mean': float(rel.mean()),
        'window_rel_max': float(rel.max()),
        'depth': float(1.0 - I_test[core_idx] / continuum),
        'ew_pm': float(ew_test * 1e3),
        'ew_rel': float(abs(ew_test - ew_ref) / abs(ew_ref)) if ew_ref != 0 else np.nan,
    }


def summarise(rows, key, field):
    vals = np.array([r[key][field] for r in rows], dtype=float)
    return vals[np.isfinite(vals)]


def make_figure(rows, example, wave, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a in ax:
        a.grid(True, lw=0.5, alpha=0.25, color='#8a9a9c')
        a.set_axisbelow(True)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)

    # --- A: an example profile. Identity of three series, so: legend, 2px lines, one y-axis.
    a = ax[0]
    # The reference runs underneath as a wide, soft band so the two trials stay readable on top
    # of it -- at a good checkpoint the GraphNet curve sits almost exactly on the reference.
    a.plot(wave, example['I_ref'] * 1e8, lw=4.5, color=COLOR['reference'], alpha=0.30,
           solid_capstyle='round', label='reference (NLTE)')
    a.plot(wave, example['I_gnn'] * 1e8, lw=1.8, color=COLOR['graphnet'], label='GraphNet')
    a.plot(wave, example['I_lte'] * 1e8, lw=1.8, color=COLOR['lte'], label='LTE', ls=(0, (5, 2)))
    a.set_xlabel('wavelength [nm]')
    a.set_ylabel(r'$I$  [$10^{-8}$ J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]')
    a.set_title(f'Si I {LINE_CORE_NM:.2f} nm, column {example["index"]}', fontsize=10, loc='left')
    a.legend(frameon=False, fontsize=9)

    # --- B, C: paired per-column comparison against the diagonal. A point below the y=x line is
    # a column where the network beat LTE; the whole question in one mark per column.
    for a, field, label in ((ax[1], 'core_rel', 'line-core'), (ax[2], 'ew_rel', 'equivalent width')):
        x = summarise(rows, 'lte', field)
        y = summarise(rows, 'graphnet', field)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        lim = [max(1e-6, min(x.min(), y.min()) * 0.5), max(x.max(), y.max()) * 2]
        a.plot(lim, lim, lw=1.0, color='#8a9a9c', zorder=1)
        a.scatter(x, y, s=26, color=COLOR['graphnet'], alpha=0.75,
                  edgecolor='white', linewidth=0.6, zorder=2)
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlim(lim)
        a.set_ylim(lim)
        a.set_xlabel(f'LTE rel. error, {label}')
        a.set_ylabel(f'GraphNet rel. error, {label}')
        won = float(np.mean(y < x)) * 100 if n else 0.0
        a.set_title(f'GraphNet better on {won:.0f}% of columns', fontsize=10, loc='left')
        a.annotate('below the line = GraphNet wins', xy=(0.04, 0.93), xycoords='axes fraction',
                   fontsize=8.5, color='#5C6E70')

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rd', '--readir', default='../data_1d_si_v2/', metavar='READIR',
                   help='directory holding the database pickles')
    p.add_argument('--dtst', '--dataset', default='validation', metavar='DTST',
                   help="split to evaluate; keep this on 'validation' (see module docstring)")
    p.add_argument('--ck', '--checkpoint', default=api.DEFAULT_CHECKPOINT, metavar='CKPT',
                   help='*_best.pth file, or a run directory to take the latest one from')
    p.add_argument('--n', default=100, type=int, metavar='N', help='number of columns to evaluate')
    p.add_argument('--seed', default=0, type=int, metavar='SEED', help='seed for column selection')
    p.add_argument('--device', default='cpu', metavar='DEV', help="torch device, e.g. 'cpu' or 'cuda:0'")
    p.add_argument('--sav', '--savedir', default=None, metavar='SAVEDIR',
                   help='where to write the pickle and figure (default: alongside the checkpoint)')
    args = p.parse_args()

    checkpoint = api._resolve_checkpoint(args.ck)
    savedir = args.sav or os.path.join(os.path.dirname(checkpoint), 'acceptance')
    os.makedirs(savedir, exist_ok=True)

    if args.dtst == 'test':
        print("WARNING: the 'test' split of any database generated before the spatial-split fix in\n"
              "         generate_database.py overlaps the training set by ~80%. Use 'validation'.\n")

    print(f'=> checkpoint {checkpoint}')
    model, hyperparams, norm_stats = api._load_model(checkpoint, args.device)
    print(f'=> reading {args.dtst} split from {args.rd}')
    data = load_split(args.rd, args.dtst)

    n_total = len(data['T'])
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_total)
    wave = np.linspace(WAVE_LO, WAVE_HI, WAVE_N)

    rows, n_rejected, example = [], 0, None
    pbar = tqdm(total=min(args.n, n_total), ncols=100, desc='Synthesising')
    for idx in order:
        if len(rows) >= args.n:
            break
        idx = int(idx)
        T, z = np.ascontiguousarray(data['T'][idx], np.float64), np.ascontiguousarray(data['z'][idx], np.float64)
        ne = np.ascontiguousarray(data['ne'][idx], np.float64)
        vturb = np.ascontiguousarray(data['vturb'][idx], np.float64)
        vlos = np.ascontiguousarray(data['vlos'][idx], np.float64)
        dep_ref = np.asarray(data['logdeparture'][idx], dtype=np.float64)
        if not np.all(np.isfinite(dep_ref)):
            continue

        # api.compute_dep_coeffs would refuse this column, so record it and keep going: the
        # quadratic extrapolation of the vturb perturbation in generate_database drives vturb
        # negative in ~9% of columns, and only vturb**2 ever reaches the physics.
        if np.any(vturb < 0):
            n_rejected += 1

        # The deployed feature construction and model, minus the input guard above. Verified
        # bit-for-bit identical to Dataset.py's graph for this node/edge configuration.
        node, edge_index, edge_attr = api._build_graph(T, z, ne, vturb, vlos, norm_stats)
        with torch.no_grad():
            out = model(node.to(args.device), edge_attr.to(args.device), edge_index.to(args.device),
                        torch.zeros((1, 1), dtype=torch.float32, device=args.device),
                        torch.zeros(node.shape[0], dtype=torch.long, device=args.device))
        # Same clamp api.compute_dep_coeffs applies, so this measures the deployed output: the
        # masked loss leaves the network unsupervised wherever n_i/n_Total < 1e-9, and 10**log_dep
        # feeds straight into the populations.
        dep_gnn = np.clip((out.cpu().numpy() * 5.0).T.astype(np.float64), -10.0, 10.0)

        try:
            I_ref = synthesise(T, z, ne, vturb, vlos, wave, dep_ref)
            I_gnn = synthesise(T, z, ne, vturb, vlos, wave, dep_gnn)
            I_lte = synthesise(T, z, ne, vturb, vlos, wave, None)
            cons = population_sum_error(T, z, ne, vturb, vlos, dep_gnn)
        except Exception as exc:                                   # noqa: BLE001
            tqdm.write(f'  column {idx}: {type(exc).__name__}: {exc} -- skipped')
            continue

        # Compare at the reference profile's own minimum rather than at the rest wavelength, so
        # that a Doppler-shifted core is still measured at its core. Continuum from the
        # reference, and reused for all three, to keep the depths on one scale.
        core_idx = int(np.argmin(I_ref))
        continuum = float(np.max(I_ref))
        if continuum <= 0:
            continue

        # The target the network was trained against is the clipped one; the reference used for
        # the synthesis above is the raw stored value (the clip moves the profile by <= 2e-7).
        dep_clipped = np.clip(dep_ref, -10.0, 10.0)
        photosphere = z < 8e5

        rows.append({
            'index': idx,
            'n_depth': len(T),
            'reference': {'core_rel': 0.0, 'window_rel_mean': 0.0, 'window_rel_max': 0.0,
                          'depth': float(1.0 - I_ref[core_idx] / continuum),
                          'ew_pm': float(_trapz(1.0 - I_ref / continuum, wave) * 1e3),
                          'ew_rel': 0.0},
            'graphnet': profile_metrics(wave, I_ref, I_gnn, core_idx, continuum),
            'lte': profile_metrics(wave, I_ref, I_lte, core_idx, continuum),
            'dep_mae_all': float(np.mean(np.abs(dep_gnn - dep_clipped))),
            'dep_mae_photosphere': float(np.mean(np.abs(dep_gnn[:, photosphere] - dep_clipped[:, photosphere]))),
            'pop_sum_err_median': float(np.median(cons[photosphere])),
            'pop_sum_err_max': float(np.max(cons[photosphere])),
        })
        if example is None:
            example = {'index': idx, 'I_ref': I_ref, 'I_gnn': I_gnn, 'I_lte': I_lte}
        pbar.update(1)
    pbar.close()

    if not rows:
        print('No columns evaluated.')
        return

    # ---------------------------------------------------------------- report
    def line(label, field, fmt='{:.4f}'):
        g, l = summarise(rows, 'graphnet', field), summarise(rows, 'lte', field)
        gain = np.median(l) / np.median(g) if np.median(g) > 0 else np.inf
        print(f'  {label:<34s} {fmt.format(np.median(l)):>12s} {fmt.format(np.median(g)):>12s} {gain:>9.1f}x')

    print(f'\n{"="*78}')
    print(f'Si I {LINE_CORE_NM:.4f} nm acceptance test  --  {len(rows)} columns of '
          f'"{args.dtst}" ({args.rd})')
    print(f'checkpoint: {os.path.basename(checkpoint)}')
    print('=' * 78)
    print(f'\n  {"metric (median over columns)":<34s} {"LTE":>12s} {"GraphNet":>12s} {"gain":>10s}')
    print(f'  {"-"*34} {"-"*12} {"-"*12} {"-"*10}')
    line('line-core rel. intensity error', 'core_rel')
    line('mean rel. error over window', 'window_rel_mean')
    line('max rel. error over window', 'window_rel_max')
    line('equivalent-width rel. error', 'ew_rel')

    for field, label in (('core_rel', 'line core'), ('ew_rel', 'equivalent width')):
        g, l = summarise(rows, 'graphnet', field), summarise(rows, 'lte', field)
        n = min(len(g), len(l))
        print(f'  GraphNet beats LTE on {100*np.mean(g[:n] < l[:n]):5.1f}% of columns  ({label})')

    d_ref = summarise(rows, 'reference', 'depth')
    d_gnn = summarise(rows, 'graphnet', 'depth')
    d_lte = summarise(rows, 'lte', 'depth')
    print(f'\n  line depth 1 - I/Ic at core:  reference {np.median(d_ref):.3f}   '
          f'GraphNet {np.median(d_gnn):.3f}   LTE {np.median(d_lte):.3f}')

    mae = np.array([r['dep_mae_photosphere'] for r in rows])
    cons = np.array([r['pop_sum_err_median'] for r in rows])
    print(f'  log10(b) MAE below 800 km:    {np.median(mae):.4f} dex')
    print(f'  |sum n_i / n_Total - 1|:      {np.median(cons):.4f} median below 800 km '
          f'(reference satisfies this to 1e-16)')
    if n_rejected:
        print(f'\n  NOTE: {n_rejected} of {len(rows)} columns carry vturb < 0 and would be rejected by '
              f'api.compute_dep_coeffs.')

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out_pkl = os.path.join(savedir, f'acceptance_{args.dtst}_{stamp}.pkl')
    with open(out_pkl, 'wb') as fh:
        pickle.dump({'rows': rows, 'wave': wave, 'example': example, 'checkpoint': checkpoint,
                     'datadir': args.rd, 'split': args.dtst, 'hyperparams': dict(hyperparams),
                     'n_vturb_negative': n_rejected}, fh)
    out_png = os.path.join(savedir, f'acceptance_{args.dtst}_{stamp}.png')
    make_figure(rows, example, wave, out_png)
    print(f'\n  wrote {out_pkl}\n  wrote {out_png}\n')


if __name__ == '__main__':
    main()
