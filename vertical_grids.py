"""
Vertical grids in height above tau500 = 1 [Mm] and the linear interpolation in height onto them.

Bifrost heights: grid_bifrost.npz z - ZTAU51_BIFROST; MURaM heights: muram_comparison/muram.py muram_heights. The grids
spend their levels where the Ca II lines form (a dense zone from the photosphere to ~4.5 Mm) and only a few in the
corona; the bottom and top native levels are always kept, so the atmosphere is only resampled, never moved or cut.

    python vertical_grids.py        # checks the design rules of the random training grids
"""
import numpy as np

ZTAU51_BIFROST = 0.0613866       # height of <tau500> = 1 in the Bifrost FITS frame [Mm] (FITS header ZTAU51)
DENSE_START_MM = -0.1            # the dense zone of the training grids starts just below tau500 = 1
# Ranges of the random training grids. The transition stays above the Bifrost K-core and transition-region heights
# (p95 2.9 and 3.8 Mm above tau500 = 1)
DENSE_SPACING_KM = (60.0, 110.0)
TRANSITION_MM = (3.8, 4.8)
N_SPARSE = (5, 10)
JITTER_FRACTION = 0.3            # of the smaller of the two gaps around a level
REMOVE_DENSE = (0.0, 0.15)       # range of the removal probability of a dense-zone level (one probability per grid)
REMOVE_SPARSE = (0.0, 0.5)       # the same for a coronal level


def two_zone_grid(bottom, top, spacing_km, transition, n_sparse, dense_start):
    """
    bottom, two levels up to dense_start (if it lies above the bottom), levels every spacing_km from dense_start up to
    transition, then n_sparse levels with geometrically growing gaps and top.
    """
    sub = np.linspace(bottom, dense_start, 4)[:-1] if dense_start > bottom + 1e-9 else np.empty(0)
    dense = np.arange(dense_start, transition + 1e-9, spacing_km / 1e3)
    gaps = np.geomspace(spacing_km / 1e3, 1.0, n_sparse + 1)
    sparse = dense[-1] + np.cumsum(gaps * (top - dense[-1]) / gaps.sum())
    sparse[-1] = top
    return np.concatenate([sub, dense, sparse])


def fixed_grid(name, bottom, top):
    """
    Validation and inference grids, 'family:km':
    'dense:60' = 60 km steps from the bottom level up to 4.5 Mm, then 7 coronal levels (the deployment grid, 93 levels
    on Bifrost); 'two_zone:80' = the same with two sub-photospheric levels and the dense zone from DENSE_START_MM.
    """
    family, km = name.split(':')
    dense_start = {'dense': bottom, 'two_zone': DENSE_START_MM}[family]
    return two_zone_grid(bottom, top, float(km), 4.5, 7, dense_start)


def random_grid(bottom, top):
    """
    The levels of one training sample (global numpy generator, reseeded in every DataLoader worker): a two-zone grid
    with a random dense spacing, transition and number of coronal levels; the interior levels jittered; interior levels
    removed with a probability drawn per grid (the sub-photospheric ones never), but a dense-zone level is put back when
    its kept neighbours would be more than twice the spacing apart.
    """
    spacing = np.random.uniform(*DENSE_SPACING_KM)
    transition = np.random.uniform(*TRANSITION_MM)
    z = two_zone_grid(bottom, top, spacing, transition, np.random.randint(N_SPARSE[0], N_SPARSE[1] + 1), DENSE_START_MM)
    dense = (z >= DENSE_START_MM - 1e-9) & (z <= transition + 1e-9)
    corona = z > transition + 1e-9

    gap = np.diff(z)
    jitter = np.random.uniform(-JITTER_FRACTION, JITTER_FRACTION, len(z) - 2) * np.minimum(gap[:-1], gap[1:])
    z = z + np.concatenate([[0.0], jitter, [0.0]])

    p_remove = np.where(dense, np.random.uniform(*REMOVE_DENSE), np.where(corona, np.random.uniform(*REMOVE_SPARSE), 0.0))
    keep = np.random.random(len(z)) >= p_remove
    keep[0] = keep[-1] = True
    for i in np.flatnonzero(~keep & dense):          # upwards, on the evolving mask
        lo, hi = i - 1, i + 1
        while not keep[lo]:
            lo -= 1
        while not keep[hi]:
            hi += 1
        keep[i] = z[hi] - z[lo] > 2.0 * spacing / 1e3
    return z[keep]


def interp_weights(z_from, z_to):
    """
    Linear interpolation in height from the levels z_from to the heights z_to (inside the range of z_from, no
    extrapolation): the level k0 of z_from below every height and the weight w of the level k0 + 1.
    """
    assert np.all(np.diff(z_from) > 0), 'the levels must increase'
    assert z_from[0] - 1e-6 <= np.min(z_to) and np.max(z_to) <= z_from[-1] + 1e-6, 'heights outside the range of the levels'
    fi = np.interp(z_to, z_from, np.arange(len(z_from)))
    k0 = np.minimum(fi.astype(np.int64), len(z_from) - 2)
    return k0, fi - k0


def to_levels(a, z_from, z_to, log=False):
    """a (levels z_from on axis 0) at the heights z_to, linear in height, or linear in log10(a) with log=True."""
    k0, w = interp_weights(z_from, z_to)
    w = w.reshape((-1,) + (1,) * (np.ndim(a) - 1))
    lo, hi = np.asarray(a[k0], dtype=np.float64), np.asarray(a[k0 + 1], dtype=np.float64)
    if log:
        return 10.0 ** ((1 - w) * np.log10(lo) + w * np.log10(hi))
    return (1 - w) * lo + w * hi


if __name__ == '__main__':
    z = np.load('../en024048_hion/grid_bifrost.npz')['z'].astype(np.float64) - ZTAU51_BIFROST
    n_levels = []
    for i in range(2000):
        np.random.seed(i)
        spacing = np.random.uniform(*DENSE_SPACING_KM) / 1e3          # the first two draws of random_grid
        transition = np.random.uniform(*TRANSITION_MM)
        np.random.seed(i)
        h = random_grid(z[0], z[-1])
        assert h[0] == z[0] and h[-1] == z[-1] and np.all(np.diff(h) > 0)
        assert np.diff(h[(h >= DENSE_START_MM) & (h <= transition)]).max() <= 2 * spacing + 1e-9
        n_levels.append(len(h))
    print(f'random_grid: 2000 draws keep the bottom and top levels, increase, and have dense-zone gaps <= 2x the spacing; '
          f'{min(n_levels)}-{max(n_levels)} levels (median {int(np.median(n_levels))}). '
          f'dense:60 has {len(fixed_grid("dense:60", z[0], z[-1]))} levels.')
