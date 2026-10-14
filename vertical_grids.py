"""Two-zone vertical grids in physical height and their verification (plan section 2.2, D-9, D-10, WP5 rungs 5-6).

Principle (the user's design): dense sampling where the Ca II lines form, few levels in the corona where the
populations are irrelevant and error-prone; the bottom and top native levels are always kept so the column's
extent and boundary conditions never change; the atmosphere is never moved, only the sampling of it varies.

Heights are measured from tau500 = 1 (Bifrost: z - ZTAU51_BIFROST), so the same grid family applies to Bifrost
and MURaM. All functions return heights in Mm, increasing.

    python vertical_grids.py            # runs the unit test and writes results/vertical_grids/<stamp>/grids.png
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
ZTAU51_BIFROST = 0.0613866
# Formation heights measured on 7938 Bifrost columns (plan 2.2), p5-p95, tau500 = 1 frame
FORMATION_BANDS = [('8542 core', 1.15 - ZTAU51_BIFROST, 1.73 - ZTAU51_BIFROST), ('H core', 1.54 - ZTAU51_BIFROST, 2.75 - ZTAU51_BIFROST),
                   ('K core', 1.60 - ZTAU51_BIFROST, 2.95 - ZTAU51_BIFROST), ('TR', 1.84 - ZTAU51_BIFROST, 3.82 - ZTAU51_BIFROST)]
K_CORE_P95, TR_P95 = 2.95 - ZTAU51_BIFROST, 3.82 - ZTAU51_BIFROST
# Random-draw ranges of the augmentation (plan 2.2 / rung 5b-c)
DENSE_SPACING_KM = (60.0, 110.0)
TRANSITION_MM = (3.8, 4.8)
N_SPARSE = (5, 10)
JITTER_FRACTION = 0.3
REMOVE_DENSE = (0.0, 0.15)
REMOVE_SPARSE = (0.0, 0.5)
DENSE_START_MM = -0.1            # the dense zone starts just below tau500 = 1
N_SUB = 2                        # interior levels between the bottom native level and the dense-zone start


def two_zone_grid(bottom, top, spacing_km=80.0, transition=4.5, n_sparse=7, dense_start=DENSE_START_MM, n_sub=N_SUB):
    """Fixed two-zone grid (D-10): bottom, n_sub sub-photospheric levels, dense zone at `spacing_km` from
    `dense_start` to `transition`, `n_sparse` geometrically spaced levels up to and including `top`."""
    sub = np.linspace(bottom, dense_start, n_sub + 2)[:-1] if dense_start > bottom + 1e-9 else np.empty(0)   # bottom + n_sub interior
    dense = np.arange(dense_start, transition + 1e-9, spacing_km / 1e3)
    # geometric gaps from the dense spacing at the transition up to the top
    d0 = spacing_km / 1e3
    gaps = np.geomspace(d0, 1.0, n_sparse + 1)                                     # relative shape, rescaled below
    gaps = gaps * (top - dense[-1]) / gaps.sum()
    sparse = dense[-1] + np.cumsum(gaps)
    sparse[-1] = top
    return np.concatenate([sub, dense, sparse])


def training_grid(config, z_native, grid=None):
    """The vertical grid of a training run from its config: (level heights in the Bifrost FITS frame [Mm], fractional
    native indices for the interpolation of the cube, transition height [Mm, FITS frame] for the grid figure).

    [dataset] grid = 'index' (default): nz_linear levels linearly spaced in native index over the first logspace_fraction
    of the cube, then nz_log log-spaced indices up to the top (the original 30 + 20 grid).
    [dataset] grid = 'two_zone' (rung 6, D-10): physical height above tau500 = 1, identical for Bifrost and MURaM: the
    bottom native level, N_SUB sub-photospheric levels, dense_spacing_km from DENSE_START_MM to transition_Mm, then
    n_sparse geometrically spaced levels up to the top native level (two_zone_grid).
    grid (optional) overrides [dataset] grid, e.g. to evaluate a model on another grid; missing parameters take the
    defaults of the ladder (30 + 20 levels over 0.33 of the cube; 80 km, 4.5 Mm, 7 sparse levels). Inference-only forms:
    'two_zone:<km>' sets the dense spacing; 'dense[:<km>]' is the two-zone grid with its dense zone starting at the bottom
    native level (no sparse sub-photospheric levels).
    """
    d = config['dataset']
    nz = len(z_native)
    assert nz == config['data']['nz_orig'], (nz, config['data']['nz_orig'])
    kind, _, spacing_arg = (grid or d.get('grid', 'index')).partition(':')
    if kind == 'index':
        frac, n_lin, n_log = d.get('logspace_fraction', 0.33), d.get('nz_linear', 30), d.get('nz_log', 20)
        idx = np.clip(np.concatenate([np.linspace(0, nz * frac, n_lin, endpoint=False),
                                      np.logspace(np.log10(nz * frac), np.log10(nz - 1), n_log)]), 0, nz - 1)
        z = np.interp(idx, np.arange(nz), z_native)
        return z, idx, z[n_lin]
    if kind in ('two_zone', 'dense'):
        transition = d.get('transition_Mm', 4.5)
        spacing = float(spacing_arg) if spacing_arg else d.get('dense_spacing_km', 80)
        bottom = z_native[0] - ZTAU51_BIFROST
        h = two_zone_grid(bottom, z_native[-1] - ZTAU51_BIFROST, spacing, transition, d.get('n_sparse', 7),
                          dense_start=bottom if kind == 'dense' else DENSE_START_MM)
        z = h + ZTAU51_BIFROST
        return z, np.interp(z, z_native, np.arange(nz)), transition + ZTAU51_BIFROST
    raise ValueError(f"unknown [dataset] grid = '{kind}'")


def zone_of(z, transition):
    """0 = sub-photospheric, 1 = dense, 2 = sparse."""
    return np.where(z < DENSE_START_MM - 1e-9, 0, np.where(z <= transition + 1e-9, 1, 2))


def random_two_zone_grid(bottom, top, rng, snap_to=None):
    """One draw of the augmented grid (rung 5b-c). Returns (heights, params, removed_mask_on_full_grid)."""
    spacing = rng.uniform(*DENSE_SPACING_KM)
    transition = rng.uniform(*TRANSITION_MM)
    n_sparse = int(rng.integers(N_SPARSE[0], N_SPARSE[1] + 1))
    z = two_zone_grid(bottom, top, spacing, transition, n_sparse)
    # jitter of the interior levels, at most JITTER_FRACTION of the local spacing, order preserved
    local = np.minimum(np.diff(z, prepend=z[0] - (z[1] - z[0])), np.diff(z, append=z[-1] + (z[-1] - z[-2])))
    jit = rng.uniform(-JITTER_FRACTION, JITTER_FRACTION, len(z)) * local
    jit[0] = jit[-1] = 0.0
    zj = z + jit
    zj = np.maximum.accumulate(zj)                                                 # keep monotonic
    # removal: interior dense levels at a low rate (never two adjacent), interior sparse levels at a high rate
    zone = zone_of(z, transition)
    p_dense, p_sparse = rng.uniform(*REMOVE_DENSE), rng.uniform(*REMOVE_SPARSE)
    removed = np.zeros(len(z), dtype=bool)
    for i in range(1, len(z) - 1):
        if zone[i] == 1 and rng.random() < p_dense and not removed[i - 1]:
            removed[i] = True
        elif zone[i] == 2 and rng.random() < p_sparse:
            removed[i] = True
    # the formation region is never sampled coarser than twice the drawn spacing: undo removals that would
    for i in np.where(removed & (zone == 1))[0]:
        lo = i - 1
        while removed[lo]:
            lo -= 1
        hi = i + 1
        while removed[hi]:
            hi += 1
        if zj[hi] - zj[lo] > 2.0 * spacing / 1e3:
            removed[i] = False
    kept = zj[~removed]
    if snap_to is not None:                                                        # MURaM: nearest native level, no vertical interpolation
        kept = np.unique(snap_to[np.abs(snap_to[None, :] - kept[:, None]).argmin(axis=1)])
    params = {'spacing_km': spacing, 'transition_Mm': transition, 'n_sparse': n_sparse, 'p_remove_dense': p_dense, 'p_remove_sparse': p_sparse}
    return kept, params, (zj, removed)


def check_grid(z, params, bottom, top, atol=1e-6):
    """The conditions of plan 2.2 / WP5 'Verification of the grids'. Raises AssertionError with the reason."""
    assert abs(z[0] - bottom) < atol and abs(z[-1] - top) < atol, 'bottom/top native level missing'
    assert np.all(np.diff(z) > 0), 'not strictly increasing'
    tr = params['transition_Mm']
    assert DENSE_SPACING_KM[0] <= params['spacing_km'] <= DENSE_SPACING_KM[1] and TRANSITION_MM[0] <= tr <= TRANSITION_MM[1], 'parameters out of range'
    assert N_SPARSE[0] <= params['n_sparse'] <= N_SPARSE[1], 'n_sparse out of range'
    dense = z[(z >= DENSE_START_MM - 1e-9) & (z <= tr + 1e-9)]
    max_gap_km = np.diff(dense).max() * 1e3
    assert max_gap_km <= 2.0 * params['spacing_km'] + 1e-6, f'dense-zone gap {max_gap_km:.0f} km > 2x spacing {params["spacing_km"]:.0f} km'
    assert K_CORE_P95 < tr and TR_P95 < tr, 'K-core / TR p95 not inside the dense zone'
    return max_gap_km


def unit_test(bottom, top, n_draws=200, seed=0):
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_draws):
        z, params, _ = random_two_zone_grid(bottom, top, rng)
        gaps.append(check_grid(z, params, bottom, top))
    return {'n_draws': n_draws, 'max_dense_gap_km': float(max(gaps)), 'median_dense_gap_km': float(np.median(gaps))}


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(REPO, 'muram_comparison'))
    import muram_common as mc
    from provenance import make_run_dir, write_meta
    z_native = np.load(mc.BIFROST_GRID)['z'] - ZTAU51_BIFROST                   # Bifrost, tau500 = 1 frame
    bottom, top = float(z_native[0]), float(z_native[-1])
    nz = len(z_native)
    new_z = np.clip(np.concatenate([np.linspace(0, nz * 0.33, 30, endpoint=False), np.logspace(np.log10(nz * 0.33), np.log10(nz - 1), 20)]), 0, nz - 1)
    current = np.interp(new_z, np.arange(nz), z_native)
    fixed = two_zone_grid(bottom, top, 80.0, 4.5, 7)
    budget50 = two_zone_grid(bottom, top, 112.0, 4.5, 7)
    muram_native = (np.arange(800) - 47.6) * 0.02                                # MURaM levels, tau500 = 1 frame (dz = 20 km)
    muram_in_range = muram_native[(muram_native >= bottom - 0.3) & (muram_native <= top + 0.3)]
    test = unit_test(bottom, top)
    print('unit test:', test)
    rng = np.random.default_rng(1)
    rows = [('native Bifrost (425)', z_native, 'royalblue'), ('native MURaM (20 km, in range)', muram_in_range, 'darkorange'),
            ('current training grid (50)', current, 'forestgreen'), (f'fixed two-zone 80 km / 4.5 Mm ({len(fixed)})', fixed, 'crimson'),
            (f'same 50-level budget 112 km ({len(budget50)})', budget50, 'purple')]
    removed = {}
    for i in range(12):
        kept, params, (zfull, rem) = random_two_zone_grid(bottom, top, rng)
        label = f"draw {i}: {params['spacing_km']:.0f} km, tr {params['transition_Mm']:.2f}, {params['n_sparse']} sparse, -{rem.sum()} nodes ({len(kept)})"
        rows.append((label, zfull, 'k'))
        removed[label] = rem
    mu_kept, mu_params, (mu_full, mu_rem) = random_two_zone_grid(bottom, top, np.random.default_rng(7), snap_to=muram_native)
    rows.append((f'draw snapped to MURaM levels ({len(mu_kept)})', mu_kept, 'darkorange'))
    run_dir = make_run_dir(__file__)
    mc.plot_grids(rows, bands=FORMATION_BANDS, transition=4.5, removed=removed, path=os.path.join(run_dir, 'grids.png'),
                  title='Two-zone grids (tau500 = 1 frame): fixed, same-budget and random draws; removed nodes in grey', xlim=(-0.8, 13.0))
    write_meta(run_dir, unit_test=test, ranges={'dense_spacing_km': DENSE_SPACING_KM, 'transition_Mm': TRANSITION_MM, 'n_sparse': N_SPARSE,
               'jitter_fraction': JITTER_FRACTION, 'remove_dense': REMOVE_DENSE, 'remove_sparse': REMOVE_SPARSE, 'dense_start_Mm': DENSE_START_MM},
               fixed_grid_levels=len(fixed), budget50_levels=len(budget50), fixed_grid_Mm=fixed)
    print(f'fixed grid: {len(fixed)} levels; same-budget grid: {len(budget50)} levels; figure in {run_dir}/grids.png')
