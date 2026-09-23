"""
Quick-look plots for the prediction pickles written by test_prediction.py (Formal.test).

For every `<split>_checkpoint_*.pkl` of a run it produces, in `<run>/plots/`:

    train_loss_vs_epochs_<name>.png      training and validation loss curves, best epoch marked
    Si_checkpoint_<name>_at_<stamp>.png  grid of random columns: log10(b) of the two levels of the
                                         1083.0 nm transition (target vs GraphNet), the other 14
                                         levels in faint grey, the height range where the line
                                         core forms shaded, and T(z) as an inset
    Intensities_SiI_checkpoint_<name>_at_<stamp>.png
                                         the Si I profiles those populations give, normalised to
                                         the continuum, with a residual strip (GraphNet and LTE
                                         against the target) under each panel and a +-1e-3 I_c
                                         noise band; the panel title carries the RMS residual

and, when more than one pickle is found, a scatter of test loss against the number of
message-passing steps. For the quantitative acceptance numbers use evaluate_intensity.py; this
script is for looking at individual columns.

Usage
-----
    python plot_scripts/explore_tests.py --ck checkpoints_si_v3/20260923-171902
    python plot_scripts/explore_tests.py --ck checkpoints_si_v3/    # every run below this tree
    python plot_scripts/explore_tests.py --ck <run>/validation_checkpoint_<run>_at_<stamp>.pkl

`--ck` takes a run directory, a checkpoint tree, a *.pth file or a prediction pickle. The
database is read from the directory recorded in the pickle unless `--rd` overrides it.
"""
import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
from tqdm import tqdm

# This script lives in plot_scripts/; api.py and evaluate_intensity.py live one level up.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_MODULE_DIR)
for _d in (_REPO_DIR, _MODULE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import api
from evaluate_intensity import (LINE_CORE_NM, LOWER_LEVEL, UPPER_LEVEL, CORE_SEARCH_NM, column_arrays,
                                continuum_level, contribution_weights, load_split, weighted_quantile)

# Si I 1074-1085 nm window at 2 pm sampling (api.DEFAULT_WAVE): the whole 4s-4p multiplet, with
# the Doppler core resolved. An earlier 1100-point version of this grid under-sampled the core.
DEFAULT_WAVE = (1074.0, 1085.0, 5501)

# Photon-noise band drawn under the residuals, in units of the continuum.
NOISE = 1e-3

COLOR = {'target': '#2B3A3C', 'prediction': '#2a78d6', 'lte': '#eb6834', 'other': '#b8c2c4',
         'T': '#5C6E70', 'band': '#8a9a9c', 'formation': '#1baf7a'}


def find_prediction_pickles(ck, dtst):
    """Prediction pickles of the run(s) that `ck` points at."""
    if os.path.isfile(ck) and ck.endswith('.pkl'):
        return [os.path.abspath(ck)]
    if os.path.isfile(ck):                       # a *.pth: its run directory
        ck = os.path.dirname(ck)
    if not os.path.isdir(ck):
        raise FileNotFoundError(f'{ck} is neither a directory, a checkpoint nor a prediction pickle')
    files = sorted(glob.glob(os.path.join(ck, '**', f'{dtst}_checkpoint_*.pkl'), recursive=True))
    if not files:
        raise FileNotFoundError(f"no '{dtst}_checkpoint_*.pkl' under {ck}; run test_prediction.py first")
    return [os.path.abspath(f) for f in files]


def load_test(path):
    """One prediction pickle, with the loss curves filled in from the *.pth if they are missing."""
    with open(path, 'rb') as fh:
        test = pickle.load(fh)
    if test.get('train_loss') is None:
        pth = os.path.join(os.path.dirname(path), os.path.basename(test['checkpoint']))
        if os.path.exists(pth):
            import torch
            ckpt = torch.load(pth, map_location='cpu', weights_only=False)
            test['train_loss'] = ckpt.get('train_loss')
            test['valid_loss'] = ckpt.get('valid_loss')
    return test


def synthesise(T, z, ne, vturb, vlos, wave, log_dep_target, log_dep_pred):
    """
    Emergent intensity for the LTE, target and predicted populations of one column, each with
    the api.intensity_gnn synthesis on the same atmosphere, plus the height
    range holding 90% of the 1083.0 nm line-core contribution function of the target.
    """
    import lightweaver as lw
    atmos, aSet, spect, eqPops = api._build_atmosphere(T, z, ne, np.maximum(vturb, 0.0), vlos)
    eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
    si = eqPops.atomicPops['Si']
    nstar = np.array(si.nStar, dtype=np.float64)
    mu = float(atmos.muz[-1])
    out = {}
    for name, log_dep in (('lte', None), ('target', log_dep_target), ('prediction', log_dep_pred)):
        si.n[:] = nstar if log_dep is None else (10.0 ** log_dep) * nstar
        ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
        ctx.formal_sol_gamma_matrices()          # J for the scattering term, as api.intensity_gnn
        I, rc = ctx.compute_rays(wave, [mu], stokes=False, returnCtx=True)
        out[name] = np.asarray(I, dtype=np.float64)
        if name == 'target':
            # where the 1083.0 nm core forms: contribution function of the target populations
            lo = int(np.searchsorted(wave, LINE_CORE_NM - CORE_SEARCH_NM))
            hi = int(np.searchsorted(wave, LINE_CORE_NM + CORE_SEARCH_NM))
            core = lo + int(np.argmin(out[name][lo:hi]))
            rc.depthData.fill = True
            rc.formal_sol_gamma_matrices()
            chi = np.array(rc.depthData.chi)[core, 0, 1, :]
            eta = np.array(rc.depthData.eta)[core, 0, 1, :]
            w, _, _ = contribution_weights(chi, eta, np.asarray(atmos.z), mu)
            zz = np.asarray(atmos.z)
            out['z_formation_km'] = (weighted_quantile(zz, w, 0.05) / 1e3, weighted_quantile(zz, w, 0.95) / 1e3)
    return out


def _style(ax):
    ax.grid(True, lw=0.5, alpha=0.25, color=COLOR['band'])
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def plot_loss_curves(test, name, plotdir, plt):
    train_loss = test.get('train_loss')
    if train_loss is None:
        return None
    train_loss = np.asarray(train_loss, dtype=float)
    valid_loss = test.get('valid_loss')
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    _style(ax)
    ax.plot(epochs, train_loss, color=COLOR['prediction'], lw=1.8, label='training')
    if valid_loss is not None and len(valid_loss) == len(train_loss):
        valid_loss = np.asarray(valid_loss, dtype=float)
        ax.plot(epochs, valid_loss, color=COLOR['lte'], lw=1.8, label='validation')
        k = int(np.nanargmin(valid_loss))
        ax.scatter([epochs[k]], [valid_loss[k]], s=40, color=COLOR['lte'], zorder=3, edgecolor='white', lw=0.8)
        ax.annotate(f'best epoch {epochs[k]}: {valid_loss[k]:.2e}', xy=(epochs[k], valid_loss[k]),
                    xytext=(8, 10), textcoords='offset points', fontsize=8.5, color=COLOR['T'])
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (masked MSE on log$_{10}$(b)/5)')
    ax.set_yscale('log')
    ax.set_title(f'{name}   (test loss of this pickle: {test["loss"].mean():.2e})', fontsize=10, loc='left')
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(plotdir, f'train_loss_vs_epochs_{name}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_loss_vs_architecture(tests, names, path, plt):
    loss = np.array([t['loss'].mean() for t in tests])
    msn = np.array([t['hyperparams']['n_message_passing_steps'] for t in tests])
    latdim = np.array([t['hyperparams']['latent_size'] for t in tests])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    _style(ax)
    ax.scatter(msn, loss, s=latdim ** 2 / 8, alpha=0.5, color=COLOR['prediction'], edgecolor='white')
    for x, y, n in zip(msn, loss, names):
        ax.annotate(n.split('_checkpoint_')[-1], (x, y), fontsize=6.5, color=COLOR['T'],
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('message-passing steps')
    ax.set_ylabel('test loss (masked MSE)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('test loss vs message-passing steps; marker area ~ latent size$^2$', fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_populations(rows, results, path, plt, inset_axes):
    """Grid of log10(b) against height; the two levels of the 1083 nm transition in colour."""
    n = len(rows)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5.6 * ncol, 3.9 * nrow), squeeze=False)
    axes = axes.ravel()
    for a in axes[n:]:
        a.set_visible(False)
    for a, row, res in zip(axes, rows, results):
        idx, T, z, ne, vturb, vlos, dep_t, dep_p = row
        zk = z / 1e3
        _style(a)
        if res is not None:
            zlo, zhi = res['z_formation_km']
            a.axvspan(zlo, zhi, color=COLOR['formation'], alpha=0.12, lw=0)
        a.axhline(0.0, color=COLOR['lte'], lw=1.1, ls=(0, (5, 2)))
        others = [k for k in range(dep_t.shape[0]) if k not in (LOWER_LEVEL, UPPER_LEVEL)]
        a.plot(zk, dep_t[others].T, color=COLOR['other'], lw=0.8)
        a.plot(zk, dep_p[others].T, color=COLOR['other'], lw=0.8, ls=(0, (2, 2)))
        for lev, lw_, alpha in ((LOWER_LEVEL, 3.2, 0.35), (UPPER_LEVEL, 3.2, 0.35)):
            a.plot(zk, dep_t[lev], color=COLOR['target'], lw=lw_, alpha=alpha, solid_capstyle='round')
        a.plot(zk, dep_p[LOWER_LEVEL], color=COLOR['prediction'], lw=1.6, label='lower level (4s $^3$P$^o_2$)')
        a.plot(zk, dep_p[UPPER_LEVEL], color=COLOR['prediction'], lw=1.6, ls=(0, (4, 1.5)),
               label='upper level (4p $^3$P$_2$)')
        # focus on the range the two line levels span, not on the +-10 clip plateau of the others
        cool = T < 2e4 if np.any(T < 2e4) else np.ones_like(T, dtype=bool)
        both = np.concatenate([dep_t[[LOWER_LEVEL, UPPER_LEVEL]][:, cool].ravel(),
                               dep_p[[LOWER_LEVEL, UPPER_LEVEL]][:, cool].ravel()])
        lo, hi = np.nanmin(both), np.nanmax(both)
        pad = 0.15 * max(hi - lo, 0.5)
        a.set_ylim(max(lo - pad, -3.5), min(hi + pad, 8.5))
        a.set_xlim(zk.min(), zk.max())
        if res is not None:
            err = np.abs(dep_p[[LOWER_LEVEL, UPPER_LEVEL]] - dep_t[[LOWER_LEVEL, UPPER_LEVEL]])
            inside = (zk >= zlo) & (zk <= zhi)
            mae = float(np.mean(err[:, inside])) if inside.any() else float('nan')
            a.set_title(f'column {idx} ({len(T)} pts): |d log b| = {mae:.3f} dex where the core forms',
                        fontsize=9, loc='left')
        else:
            a.set_title(f'column {idx} ({len(T)} pts)', fontsize=9, loc='left')
        axins = inset_axes(a, width='32%', height='32%', loc='upper left', borderpad=0.8)
        axins.plot(zk, T / 1e3, color=COLOR['T'], lw=1.1)
        axins.set_ylim(2, 16)
        axins.tick_params(labelsize=6, length=2, pad=1)
        axins.set_title('T [kK]', fontsize=7, pad=2)
        axins.patch.set_alpha(0.85)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.995))
    fig.supxlabel('height [km]')
    fig.supylabel(r'$\log_{10}(n/n^*)$   ink band: target, blue: GraphNet, grey: other 14 levels, '
                  'green: 90% of the 1083.0 nm core contribution')
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.07, hspace=0.35, wspace=0.18)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_intensities(rows, results, wave, path, plt):
    """Grid of continuum-normalised profiles with a residual strip under each panel."""
    n = len(rows)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(5.6 * ncol, 4.3 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.28, wspace=0.22, left=0.05, right=0.99, top=0.94, bottom=0.05)
    first = True
    for k, (row, res) in enumerate(zip(rows, results)):
        idx = row[0]
        inner = outer[k // ncol, k % ncol].subgridspec(2, 1, height_ratios=[2.6, 1], hspace=0.06)
        a = fig.add_subplot(inner[0])
        b = fig.add_subplot(inner[1], sharex=a)
        _style(a)
        _style(b)
        if res is None:
            a.set_title(f'column {idx}: synthesis failed', fontsize=9, loc='left')
            continue
        Ic = continuum_level(res['target'])
        r_pred = (res['prediction'] - res['target']) / Ic
        r_lte = (res['lte'] - res['target']) / Ic
        a.plot(wave, res['target'] / Ic, color=COLOR['target'], lw=3.0, alpha=0.35, solid_capstyle='round',
               label='target (stored NLTE b)')
        a.plot(wave, res['prediction'] / Ic, color=COLOR['prediction'], lw=1.3, label='GraphNet')
        a.plot(wave, res['lte'] / Ic, color=COLOR['lte'], lw=1.1, ls=(0, (5, 2)), label='LTE')
        a.set_ylabel(r'$I/I_c$', fontsize=9)
        a.tick_params(labelbottom=False, labelsize=8)
        rms_p, rms_l = np.sqrt(np.mean(r_pred ** 2)), np.sqrt(np.mean(r_lte ** 2))
        a.set_title(f'column {idx}: RMS residual GraphNet {rms_p:.1e}, LTE {rms_l:.1e}', fontsize=9, loc='left')
        b.axhspan(-NOISE * 1e3, NOISE * 1e3, color=COLOR['band'], alpha=0.18, lw=0)
        b.axhline(0.0, color=COLOR['target'], lw=0.7)
        b.plot(wave, r_lte * 1e3, color=COLOR['lte'], lw=1.0, ls=(0, (5, 2)))
        b.plot(wave, r_pred * 1e3, color=COLOR['prediction'], lw=1.2)
        lim = max(3.0 * NOISE * 1e3, 1.15 * np.max(np.abs(r_pred)) * 1e3)
        b.set_ylim(-lim, lim)
        b.set_ylabel(r'$\Delta I/I_c$ [$10^{-3}$]', fontsize=8)
        b.tick_params(labelsize=8)
        if k // ncol == nrow - 1 or k + ncol >= n:
            b.set_xlabel('wavelength [nm]', fontsize=9)
        if first:
            handles, labels = a.get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, fontsize=9, loc='upper center', ncol=3,
                       bbox_to_anchor=(0.5, 0.995), title=f'residual band: $\\pm${NOISE:g} $I_c$',
                       title_fontsize=8.5)
            first = False
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ck', '--checkpoint', default=api.DEFAULT_CHECKPOINT, metavar='CKPT',
                   help='run directory, checkpoint tree, *.pth file or prediction pickle')
    p.add_argument('--dtst', '--dataset', default='validation', metavar='DTST',
                   help='split whose prediction pickles to plot')
    p.add_argument('--rd', '--readir', default=None, metavar='READIR',
                   help='database directory (default: the one recorded in each pickle)')
    p.add_argument('--n', default=25, type=int, metavar='N', help='columns per grid (a square number fits best)')
    p.add_argument('--seed', default=0, type=int, metavar='SEED', help='seed for the column selection')
    p.add_argument('--wave', default=list(DEFAULT_WAVE), type=float, nargs=3, metavar=('LO', 'HI', 'N'),
                   help='synthesis grid [nm]: start, end, number of points')
    p.add_argument('--sav', '--savedir', default=None, metavar='SAVEDIR',
                   help='where to write the figures (default: <run>/plots/ of each pickle)')
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    files = find_prediction_pickles(args.ck, args.dtst)
    tests = [load_test(f) for f in files]
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    plotdirs = [args.sav or os.path.join(os.path.dirname(f), 'plots') for f in files]
    for d in set(plotdirs):
        os.makedirs(d, exist_ok=True)

    print(f'=> {len(files)} prediction pickle(s) for split "{args.dtst}" under {args.ck}')
    for f, t in zip(files, tests):
        print(f'   {os.path.relpath(f)}: test loss {t["loss"].mean():.5f}, {len(t["target"])} columns, '
              f'checkpoint {os.path.basename(t["checkpoint"])}')

    if len(tests) > 1:
        path = os.path.join(plotdirs[0], f'loss_vs_msn_{args.dtst}.png')
        plot_loss_vs_architecture(tests, names, path, plt)
        print(f'   wrote {path}')

    wave = np.linspace(args.wave[0], args.wave[1], int(args.wave[2]))
    if not (wave[0] <= LINE_CORE_NM <= wave[-1]):
        print(f'   WARNING: the grid does not contain the 1083.0 nm core; the formation-height shading is skipped')
    rng = np.random.default_rng(args.seed)
    data_cache = {}

    for test, name, plotdir in zip(tests, names, plotdirs):
        path = plot_loss_curves(test, name, plotdir, plt)
        if path:
            print(f'   wrote {path}')

        datadir = args.rd or test['datadir']
        if datadir not in data_cache:
            print(f'=> reading {args.dtst} split from {datadir}')
            data_cache[datadir] = load_split(datadir, args.dtst)
        data = data_cache[datadir]
        n_pred = len(test['prediction'])
        if n_pred != len(data['T']):
            print(f'   WARNING: {name} holds {n_pred} columns but the split has {len(data["T"])}; '
                  f'predictions are matched by position and may not belong to these columns')

        sampler = rng.choice(min(n_pred, len(data['T'])), size=min(args.n, n_pred), replace=False)
        rows = []
        for idx in sampler:
            idx = int(idx)
            T, z, ne, vturb, vlos = column_arrays(data, idx)
            # targets and predictions are stored scaled by 1/5 (Dataset.py)
            dep_t = np.clip(np.asarray(test['target'][idx], dtype=np.float64).T * 5.0, -10.0, 10.0)
            dep_p = np.clip(np.asarray(test['prediction'][idx], dtype=np.float64).T * 5.0, -10.0, 10.0)
            rows.append((idx, T, z, ne, vturb, vlos, dep_t, dep_p))

        print(f'=> synthesising {len(rows)} columns x 3 profiles')
        results = []
        for idx, T, z, ne, vturb, vlos, dep_t, dep_p in tqdm(rows, ncols=100):
            try:
                results.append(synthesise(T, z, ne, vturb, vlos, wave, dep_t, dep_p))
            except Exception as exc:                               # noqa: BLE001
                tqdm.write(f'   column {idx}: {type(exc).__name__}: {exc} -- skipped')
                results.append(None)

        stamp = time.strftime('%Y%m%d-%H%M%S')
        path = os.path.join(plotdir, f'Si_checkpoint_{name}_at_{stamp}.png')
        plot_populations(rows, results, path, plt, inset_axes)
        print(f'   wrote {path}')
        path = os.path.join(plotdir, f'Intensities_SiI_checkpoint_{name}_at_{stamp}.png')
        plot_intensities(rows, results, wave, path, plt)
        print(f'   wrote {path}')


if __name__ == '__main__':
    main()
