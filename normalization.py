"""
Normalization of the network inputs and targets (v2 representation, plan D-4 / D-5).

Inputs ('features'): log10 of every scalar (temp, n_e, rho, press, n_h, n_p) standardized by ONE global mean and standard
deviation of the whole training cube on the training grid (no per-height statistics) and clipped at +-clip sigma;
velocities divided by vel_scale (10 km/s). The magnetic field is not an input. The fitted statistics are stored in the
checkpoint and re-applied unchanged to any other atmosphere.
Derived input 'lte' (rung 8a, plan D-5 note): the log10 LTE (Saha-Boltzmann) fractions of the six levels of the Ca II model
atom computed from the atmosphere's own temp and n_e (lte_log_fractions), standardized per component; the raw n_e is then
not an input, so n_e enters the network only through the Saha dependence.

Targets ('pops'): log10(f + log_offset) with f = n_level / sum_levels(n), the representation produced by the 'log_softmax'
head of graphnet.py (fractions sum exactly to one). Errors in this space are dex. Absolute populations of ANY atmosphere
follow from the fractions as n_total [cm^-3] = CA_TOTAL_PER_RHO * rho [kg m^-3].

The per-height normalizations, the height and B inputs and the linear head of the earlier models are in the parent commit.
"""

import numpy as np

# components of the multi-component inputs, in the order of the expanded labels (velx, vely, velz; lte_4s, ... lte_III)
COMPONENTS = {'vel': ['x', 'y', 'z'], 'lte': ['_4s', '_3d32', '_3d52', '_4p12', '_4p32', '_III']}


def expand_labels(labels):
    """Expanded input labels: one per component ('vel' -> velx, vely, velz; 'lte' -> six levels; scalars unchanged)."""
    return [f'{l}{c}' for l in labels for c in COMPONENTS.get(l, [''])]


def lte_log_fractions(temp, n_e, log_offset=1e-12, chunk=1 << 20):
    """
    log10 of the LTE fractions of the six levels of the Ca II model atom (4s, 3d3/2, 3d5/2, 4p1/2, 4p3/2, Ca III; the order of
    the targets) from temp [K] and n_e [m^-3] of the atmosphere itself (lightweaver lte_pops with Debye lowering, the call of
    the LTE baseline in muram_comparison/compare_populations.py). Any array shape, with or without a trailing singleton axis;
    returns float32 (..., 6) with the fractions floored at log_offset (the sum over the six levels is one before the floor).
    """
    from lightweaver.atomic_set import lte_pops
    from lightweaver.rh_atoms import CaII_atom
    temp, n_e = np.asarray(temp), np.asarray(n_e)
    if temp.ndim > 1 and temp.shape[-1] == 1:
        temp, n_e = temp[..., 0], n_e[..., 0]
    assert temp.shape == n_e.shape, (temp.shape, n_e.shape)
    atom = CaII_atom()
    assert len(atom.levels) == 6
    out = np.empty(temp.shape + (6,), dtype=np.float32)
    flat_T, flat_ne, flat_out = temp.reshape(-1), n_e.reshape(-1), out.reshape(-1, 6)
    for i in range(0, flat_T.size, chunk):
        T = np.asarray(flat_T[i:i + chunk], dtype=np.float64)
        f = lte_pops(atom, T, np.asarray(flat_ne[i:i + chunk], dtype=np.float64), np.ones_like(T))     # (6, n) fractions
        flat_out[i:i + chunk] = np.log10(np.maximum(f.T, log_offset))
    return out

# Total Ca number density per unit mass density in the training targets: the sum over the six levels of
# data_train/AR_385_CaII_5L_pops.dat [cm^-3] divided by AR_385_mass.dat [kg m^-3] is 9.65e14 at every node
# (relative scatter 2e-7), i.e. the targets assume one fixed Ca abundance (plan WP2 step 2.2).
CA_TOTAL_PER_RHO = 9.65e14


def normalize_features(features, labels, norm_params=None, clip=5.0, vel_scale=1e4):
    """
    Args:
        features: list of arrays (nz, ny, nx[, ncomp]); labels: their names ('vel', 'temp', 'n_e', 'rho', 'press', 'n_h', 'n_p', 'lte')
        norm_params: None to fit the statistics (training), or the dictionary saved by the training run (inference)

    Returns:
        normalized: list of arrays with a trailing component axis
        norm_params: {'version': 2, 'means', 'stds', 'clip', 'vel_scale'} (means/stds of the log10 scalars, per label; lists of six for 'lte')
    """
    fit = norm_params is None
    if fit:
        norm_params = {'version': 2, 'means': [], 'stds': [], 'clip': clip, 'vel_scale': vel_scale}

    normalized = []
    for i, (feature, label) in enumerate(zip(features, labels)):
        if feature.ndim == 3:
            feature = feature[:, :, :, np.newaxis]
        if 'vel' in label:
            mean, std = 0.0, norm_params['vel_scale']
            normed = feature / std
        elif label == 'lte':               # already log10, six components: one mean and std per component, no clip (the floor bounds it)
            mean, std = (feature.mean(axis=(0, 1, 2), dtype=np.float64).tolist(), feature.std(axis=(0, 1, 2), dtype=np.float64).tolist()) \
                if fit else (norm_params['means'][i], norm_params['stds'][i])
            normed = (feature - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)   # bounded in [log10(offset), 0]: no clip
        else:
            logf = np.log10(feature)
            mean, std = (float(logf.mean()), float(logf.std())) if fit else (norm_params['means'][i], norm_params['stds'][i])
            normed = np.clip((logf - mean) / std, -norm_params['clip'], norm_params['clip'])
        if fit:
            norm_params['means'].append(mean)
            norm_params['stds'].append(std)
        normalized.append(normed)

    return normalized, norm_params


def normalize_pops(pops, log_offset=1e-12):
    """
    Args:
        pops: (nz, ny, nx, nlev) populations

    Returns:
        normalized_pops: log10(f + log_offset)
        norm_params: {'totals' (nz, ny, nx), 'means' (nz, nlev) mean fraction per height for the climatology baseline, 'log_offset'}
    """
    total_pops_all = np.sum(pops, axis=3)
    fractional_pops = pops / (total_pops_all[:, :, :, np.newaxis] + log_offset)
    normalized_pops = np.log10(fractional_pops + log_offset)

    norm_params = {
        'means': np.mean(fractional_pops, axis=(1, 2)),
        'totals': total_pops_all,
        'log_offset': log_offset
    }

    return normalized_pops, norm_params


def denormalize_pops(normalized_pops, norm_params):
    """Populations from log10 fractions: (10**normalized - log_offset) * totals, with 'totals' of the atmosphere predicted."""
    total_pops_all = norm_params['totals']
    log_offset = norm_params['log_offset']

    reconstructed_fractional_pops = np.maximum(10**normalized_pops - log_offset, 0.0)
    reconstructed_pops = reconstructed_fractional_pops * \
        (total_pops_all[:, :, :, np.newaxis] + log_offset)

    return reconstructed_pops
