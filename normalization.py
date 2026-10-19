"""
Normalization and denormalization functions for NLTE populations and physical features.

Inputs: velocities divided by a fixed 10 km/s; the other quantities (temp, rho, press) as log10, standardized by ONE
mean and standard deviation of the whole training cube and clipped at +-5 sigma; 'lte', the six log10 LTE fractions
of the Ca levels (lte_log_fractions), standardized per level. The statistics are fitted once on the training cube,
stored in the checkpoint and applied unchanged to any other atmosphere (normalize_features_with_params).
Targets: log10 of the fractions of the six levels, log10(n_level / n_total + log_offset).
"""

import numpy as np

VEL_SCALE = 1e4                 # [m/s]: velocities enter as v / 10 km/s, with no offset (Dataset.py rotates them)
CLIP = 5.0                      # standardized log10 scalars are clipped at +-CLIP

# Total Ca number density per unit mass density of the training targets: the sum over the six levels of
# AR_385_CaII_5L_pops.dat [cm^-3] divided by AR_385_mass.dat [kg m^-3] is 9.65e14 at every node (scatter 2e-7)
CA_TOTAL_PER_RHO = 9.65e14


def normalize_pops(pops, log_offset=1e-12):
    """
    Normalize populations as the log10 of the fraction of every level.

    Args:
        pops: Population array of shape (nz, ny, nx, nlev)
        log_offset: Small value to avoid log(0) (default: 1e-12)

    Returns:
        normalized_pops: log10(fractions + log_offset)
        norm_params: Dictionary with the totals (nz, ny, nx) and log_offset for denormalization
    """
    total_pops_all = np.sum(pops, axis=3)
    fractional_pops = pops / (total_pops_all[:, :, :, np.newaxis] + log_offset)
    normalized_pops = np.log10(fractional_pops + log_offset)

    norm_params = {
        'totals': total_pops_all,
        'log_offset': log_offset
    }

    return normalized_pops, norm_params

def denormalize_pops(normalized_pops, norm_params):
    """
    Denormalize populations back to original space: the fractions times the totals of the atmosphere predicted.

    Args:
        normalized_pops: Normalized population array (nz, ny, nx, nlev)
        norm_params: Dictionary with 'totals' (nz, ny, nx) and 'log_offset' keys

    Returns:
        reconstructed_pops: Populations in original space
    """
    total_pops_all = norm_params['totals']
    log_offset = norm_params['log_offset']

    reconstructed_fractional_pops = np.maximum(10**normalized_pops - log_offset, 0.0)
    reconstructed_pops = reconstructed_fractional_pops * \
        (total_pops_all[:, :, :, np.newaxis] + log_offset)

    return reconstructed_pops

def lte_log_fractions(temp, n_e, log_offset=1e-12):
    """
    log10 of the LTE (Saha-Boltzmann) fractions of the six levels of the Ca II model atom (4s, 3d3/2, 3d5/2, 4p1/2,
    4p3/2, Ca III: the order of the targets) from the atmosphere's own temperature and electron density
    (Lightweaver lte_pops).

    Args:
        temp: Temperature [K], array (nz, ny, nx, 1) like the other feature arrays (or (nz, n, 1))
        n_e: Electron density [m^-3], same shape

    Returns:
        lte: float32 array (..., 6), the fractions floored at log_offset
    """
    from lightweaver.atomic_set import lte_pops
    from lightweaver.rh_atoms import CaII_atom
    atom = CaII_atom()
    lte = np.empty(temp.shape[:-1] + (6,), dtype=np.float32)
    for k in range(temp.shape[0]):
        T = np.asarray(temp[k], dtype=np.float64).ravel()
        f = lte_pops(atom, T, np.asarray(n_e[k], dtype=np.float64).ravel(), np.ones_like(T))     # (6, n) fractions
        lte[k] = np.log10(np.maximum(f.T, log_offset)).reshape(lte.shape[1:])
    return lte

def calculate_mean_std(features, labels):
    """
    Calculate the mean and standard deviation of each feature over the whole cube.

    Velocities are not standardized (mean 0, std VEL_SCALE). 'lte' has one mean and std per level.
    The other quantities are standardized in log10.

    Args:
        features: List of feature arrays (nz, ny, nx, ncomp)
        labels: List of feature labels ('vel', 'temp', 'rho', 'press', 'lte')

    Returns:
        means: List of means (lists of six for 'lte')
        stds: List of standard deviations
    """
    means = []
    stds = []
    for feature, label in zip(features, labels):
        if 'vel' in label:
            means.append(0.0)
            stds.append(VEL_SCALE)
        elif label == 'lte':
            means.append(feature.mean(axis=(0, 1, 2), dtype=np.float64).tolist())
            stds.append(feature.std(axis=(0, 1, 2), dtype=np.float64).tolist())
        else:
            log_feature = np.log10(feature)
            means.append(float(log_feature.mean(dtype=np.float64)))
            stds.append(float(log_feature.std(dtype=np.float64)))
    return means, stds

def normalize_features(features, labels):
    """
    Fit the normalization on the training cube and normalize the features with it.

    Args:
        features: List of feature arrays
        labels: List of feature labels

    Returns:
        normalized: List of normalized feature arrays
        norm_params: Dictionary with 'means', 'stds' and 'clip', saved in the checkpoint
    """
    means, stds = calculate_mean_std(features, labels)
    norm_params = {
        'means': means,
        'stds': stds,
        'clip': CLIP
    }

    return normalize_features_with_params(features, labels, norm_params), norm_params

def normalize_features_with_params(features, labels, norm_params):
    """
    Normalize physical features using pre-computed statistics (the ones of the training run).

    Args:
        features: List of feature arrays
        labels: List of feature labels
        norm_params: Dictionary with 'means', 'stds' and 'clip'

    Returns:
        normalized: List of normalized float32 feature arrays
    """
    normalized = []
    for feature, label, mean, std in zip(features, labels, norm_params['means'], norm_params['stds']):
        if feature.ndim == 3:
            feature = feature[:, :, :, np.newaxis]

        if 'vel' in label:
            normed = feature / std
        elif label == 'lte':           # already log10 and bounded by the floor: no clip
            normed = (feature - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
        else:
            normed = np.clip((np.log10(feature) - mean) / std, -norm_params['clip'], norm_params['clip'])

        normalized.append(normed.astype(np.float32))

    return normalized
