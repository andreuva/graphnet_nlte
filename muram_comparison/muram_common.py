"""Shared helpers for the MURaM comparison scripts (implementation plan WP2 steps 2.3, 2.4, 2.5, 2.9).

Conventions
- `MuramCube` arrays are (z, x, y) with the z index increasing upward; `cube.vx`/`cube.Bx` are the VERTICAL
  components, `cube.vy`/`By` point along x and `cube.vz`/`Bz` along y. Everything returned here is in the
  Bifrost layout (z, y, x), vectors stacked as (x, y, z), SI units.
- Heights are measured from the column-median tau500 = 1 level (`muram_heights`). Bifrost heights in
  grid_bifrost.npz are in the FITS frame, whose <tau500> = 1 sits at ZTAU51_BIFROST; subtract it to compare.
- MURaM stores B / sqrt(4 pi) in Gauss: multiply by sqrt(4 pi) * 1e-4 for Tesla (audit M4).
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interpn

import muram as mio

MURAM_DIR = '/dat/milic/MURaM_enhanced_network/'
SNAP = 499000
ZTAU51_BIFROST = 0.0613866                     # Mm, Bifrost FITS header ZTAU51 (en024048_hion, snapshot 385)
BIFROST_GRID = '/dat/andreuva/gpu/graphnet/en024048_hion/grid_bifrost.npz'
REFERENCE_FITS = os.path.join(MURAM_DIR, 'che_full_499000_lwsynth_200.0.fits')
FITS_Z_SLICE = slice(15, 416)                  # MURaM levels stored in the reference FITS (stored top-first)
B_MURAM_TO_TESLA = np.sqrt(4.0 * np.pi) * 1e-4
LEVEL_NAMES = ['4s', '3d3/2', '3d5/2', '4p1/2', '4p3/2', 'CaIII']
# Static line-core tau = 1 and transition-region heights over 7938 Bifrost columns (plan section 2.2),
# p5-p95, in the Bifrost FITS frame (subtract ZTAU51_BIFROST for the tau500 = 1 frame).
FORMATION_BANDS_BIFROST_FRAME = [('8542 core', 1.15, 1.73), ('H core', 1.54, 2.75), ('K core', 1.60, 2.95), ('TR', 1.84, 3.82)]


def load_snap(path=MURAM_DIR, snap=SNAP):
    return mio.MuramSnap(path, snap)


def spacing_km(cube):
    """(dz, dx, dy) in km from the snapshot header (not the 24 km of the old scripts: dx = dy = 23.4375 km)."""
    dz, dx, dy = np.asarray(cube.Temp.dX) / 1e5
    return float(dz), float(dx), float(dy)


def tau1_index(tau, step=1):
    """Per-column fractional z index of the highest tau500 = 1 crossing (linear in log tau). Shape (nx//step, ny//step)."""
    lt = np.log10(np.clip(np.asarray(tau[:, ::step, ::step]), 1e-30, None))
    nz = lt.shape[0]
    k_hi = nz - 1 - np.argmax((lt >= 0)[::-1], axis=0)          # highest level with tau >= 1
    k_hi = np.clip(k_hi, 0, nz - 2)
    i, j = np.indices(k_hi.shape)
    c0, c1 = lt[k_hi, i, j], lt[k_hi + 1, i, j]
    return k_hi + c0 / (c0 - c1)


def muram_heights(cube, step=4):
    """Heights [Mm] of all MURaM levels relative to the column-median tau500 = 1 level, and that fractional index."""
    dz = spacing_km(cube)[0] / 1e3
    idx = float(np.nanmedian(tau1_index(cube.tau, step)))
    return (np.arange(cube.Temp.shape[0]) - idx) * dz, idx


def bifrost_training_grid(config, grid_file=None):
    """(zz_grid_bifrost [50, Mm, FITS frame], z_native [425], new_z_b [fractional native indices]) as in gnn_train.py."""
    grid_file = grid_file or BIFROST_GRID
    z_native = np.load(grid_file)['z']
    nz = config['data']['nz_orig']
    d = config['dataset']
    new_z_b = np.clip(np.concatenate([np.linspace(0, nz * d['logspace_fraction'], d['nz_linear'], endpoint=False),
                                      np.logspace(np.log10(nz * d['logspace_fraction']), np.log10(nz - 1), d['nz_log'])]), 0, nz - 1)
    return np.interp(new_z_b, np.arange(nz), z_native), z_native, new_z_b


def align_bifrost_levels_to_muram(zz_grid_bifrost, idx_tau1, dz_mm):
    """Fractional MURaM z indices at which the Bifrost training levels sit, both simulations aligned at tau500 = 1."""
    return idx_tau1 + (zz_grid_bifrost - ZTAU51_BIFROST) / dz_mm


def muram_si_on_levels(cube, new_z, labels=('temp', 'vel', 'b', 'n_e', 'rho', 'press')):
    """Interpolate MURaM variables linearly to fractional z indices `new_z`; SI, Bifrost layout (z, y, x)."""
    nz, nx, ny = cube.Temp.shape
    z, x, y = (np.arange(d) for d in (nz, nx, ny))
    pts = tuple(np.meshgrid(new_z, x, y, indexing='ij', sparse=True))

    def interp(var):
        return interpn((z, x, y), getattr(cube, var), pts).transpose(0, 2, 1)       # (z, x, y) -> (z, y, x)

    out = {}
    for lab in labels:
        if lab == 'temp':
            out[lab] = interp('Temp')[..., None]
        elif lab == 'vel':                                                           # cm/s -> m/s; (x, y, z) components
            out[lab] = np.stack([interp('vy'), interp('vz'), interp('vx')], axis=-1) / 100.0
        elif lab == 'b':                                                             # B/sqrt(4pi) [G] -> T
            out[lab] = np.stack([interp('By'), interp('Bz'), interp('Bx')], axis=-1) * B_MURAM_TO_TESLA
        elif lab == 'n_e':
            out[lab] = interp('ne')[..., None] * 1e6                                 # cm^-3 -> m^-3
        elif lab == 'rho':
            out[lab] = interp('rho')[..., None] * 1e3                                # g cm^-3 -> kg m^-3
        elif lab == 'press':
            out[lab] = interp('Pres')[..., None] / 10.0                              # dyn cm^-2 -> Pa
        elif lab == 'n_h':
            out[lab] = interp('H_1')[..., None] * 1e6                                # ChE ground-level H, cm^-3 -> m^-3
        elif lab == 'n_p':
            out[lab] = interp('H_6')[..., None] * 1e6                                # ChE protons
        else:
            raise ValueError(lab)
    return out


def plot_grids(grids, bands=None, spacing_panel=True, path=None, title=None, transition=None, removed=None,
               xlim=None):
    """Tick plot of vertical grids, one row each (plan WP2 step 2.9; generalizes compare_datasets.py's compare_grids).

    grids:   list of (label, z [Mm], color); the first row is drawn at the top.
    bands:   list of (label, lo, hi) shaded height ranges (formation heights); transition: vertical line [Mm].
    removed: dict label -> boolean mask of nodes drawn in grey (removed from the graph, kept in the atmosphere).
    spacing_panel: add a panel with the local spacing dz against height (log y) for every grid.
    """
    n = len(grids)
    fig, axes = plt.subplots(2 if spacing_panel else 1, 1, figsize=(13, 0.45 * n + 2.2 + (3.0 if spacing_panel else 0)),
                             sharex=True, gridspec_kw={'height_ratios': [0.45 * n + 1.2, 3.0]} if spacing_panel else None)
    ax = axes[0] if spacing_panel else axes
    handles = []
    for row, (label, zg, color) in enumerate(grids):
        zg = np.asarray(zg)
        yrow = n - 1 - row
        keep = np.ones(len(zg), dtype=bool) if not removed or label not in removed else ~np.asarray(removed[label])
        h, = ax.plot(zg[keep], np.full(keep.sum(), yrow), '|', markersize=15, color=color, label=f'{label} ({len(zg)} levels)')
        if (~keep).any():
            ax.plot(zg[~keep], np.full((~keep).sum(), yrow), '|', markersize=15, color='0.6')
        handles.append(h)
    ax.set_yticks(range(n)); ax.set_yticklabels([g[0] for g in grids][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    if bands:
        cmap = plt.get_cmap('Set2')
        for i, (label, lo, hi) in enumerate(bands):
            ax.axvspan(lo, hi, color=cmap(i), alpha=0.25, lw=0, label=f'{label} (p5-p95)')
    if transition is not None:
        ax.axvline(transition, color='k', ls='--', lw=1, label=f'dense/sparse transition {transition:.2f} Mm')
    ax.grid(True, axis='x', ls='--', alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=9, borderaxespad=0)
    if title:
        ax.set_title(title)
    if spacing_panel:
        ax2 = axes[1]
        for label, zg, color in grids:
            zg = np.sort(np.asarray(zg))
            ax2.plot(0.5 * (zg[1:] + zg[:-1]), np.diff(zg) * 1e3, '.-', ms=3, lw=1, color=color, label=label)
        if bands:
            for i, (label, lo, hi) in enumerate(bands):
                ax2.axvspan(lo, hi, color=plt.get_cmap('Set2')(i), alpha=0.25, lw=0)
        if transition is not None:
            ax2.axvline(transition, color='k', ls='--', lw=1)
        ax2.set_yscale('log'); ax2.set_ylabel(r'local spacing $\Delta z$ [km]'); ax2.grid(True, ls='--', alpha=0.5)
        ax2.set_xlabel('Height [Mm]')
    else:
        ax.set_xlabel('Height [Mm]')
    if xlim:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig
