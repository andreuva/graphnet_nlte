"""
Normalization of the network inputs and targets (v2 representation, plan D-4 / D-5).

Inputs ('features'): log10 of every scalar (temp, n_e, rho, press, n_h, n_p) standardized by ONE global mean and standard
deviation of the whole training cube on the training grid (no per-height statistics) and clipped at +-clip sigma;
velocities divided by vel_scale (10 km/s). The magnetic field is not an input. The fitted statistics are stored in the
checkpoint and re-applied unchanged to any other atmosphere.

Targets ('pops'): log10(f + log_offset) with f = n_level / sum_levels(n), the representation produced by the 'log_softmax'
head of graphnet.py (fractions sum exactly to one). Errors in this space are dex. Absolute populations of ANY atmosphere
follow from the fractions as n_total [cm^-3] = CA_TOTAL_PER_RHO * rho [kg m^-3].

The per-height 'log' / 'pow' normalizations of the pre-v2 models were removed in the commit after 4b1ebbf.
"""

import numpy as np

# Total Ca number density per unit mass density in the training targets: the sum over the six levels of
# data_train/AR_385_CaII_5L_pops.dat [cm^-3] divided by AR_385_mass.dat [kg m^-3] is 9.65e14 at every node
# (relative scatter 2e-7), i.e. the targets assume one fixed Ca abundance (plan WP2 step 2.2).
CA_TOTAL_PER_RHO = 9.65e14


def normalize_features(features, labels, norm_params=None, clip=5.0, vel_scale=1e4):
    """
    Args:
        features: list of arrays (nz, ny, nx[, ncomp]); labels: their names ('vel', 'temp', 'n_e', 'rho', 'press', 'n_h', 'n_p')
        norm_params: None to fit the statistics (training), or the dictionary saved by the training run (inference)

    Returns:
        normalized: list of arrays with a trailing component axis
        norm_params: {'version': 2, 'means', 'stds', 'clip', 'vel_scale'} (means/stds of the log10 scalars, per label)
    """
    fit = norm_params is None
    if fit:
        norm_params = {'version': 2, 'means': [], 'stds': [], 'clip': clip, 'vel_scale': vel_scale}

    normalized = []
    for i, (feature, label) in enumerate(zip(features, labels)):
        if feature.ndim == 3:
            feature = feature[:, :, :, np.newaxis]
        if label == 'b':
            raise ValueError("B is not an input of the v2 representation (plan D-4)")
        if 'vel' in label:
            mean, std = 0.0, norm_params['vel_scale']
            normed = feature / std
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
