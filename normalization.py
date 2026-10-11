"""
Normalization and denormalization functions for NLTE populations and physical features.

This module contains the normalization strategy from porta_graph_test.ipynb
to ensure consistency across training and inference scripts.
"""

import numpy as np

# Total Ca number density per unit mass density in the training targets: the sum over the six levels of
# data_train/AR_385_CaII_5L_pops.dat [cm^-3] divided by AR_385_mass.dat [kg m^-3] is 9.65e14 at every node
# (relative scatter 2e-7), i.e. the targets assume one fixed Ca abundance. Absolute populations for ANY
# atmosphere follow from predicted fractions as n_total [cm^-3] = CA_TOTAL_PER_RHO * rho [kg m^-3]
# (plan WP2 step 2.2; replaces stretching Bifrost's totals onto another simulation).
CA_TOTAL_PER_RHO = 9.65e14

# ===================================================================
# ORIGINAL 'POW' NORMALIZATION
# ===================================================================
def normalize_pops_pow(pops, factor=4., log_offset=1e-12):
    """
    Normalize populations using logarithmic transformation with height-stratified statistics.

    Args:
        pops: Population array of shape (nz, ny, nx, nlev)
        factor: Scaling factor for the normalized populations (default: 4.0)
        log_offset: Small value to avoid log(0) (default: 1e-12)

    Returns:
        normalized_pops: Normalized populations
        norm_params: Dictionary with normalization parameters for denormalization
    """
    total_pops_all = np.sum(pops, axis=-1)
    fractional_pops = pops / (total_pops_all[:, :, :, np.newaxis] + log_offset)
    normalized_pops = np.log10(1 / (fractional_pops + log_offset))**(1 / factor)

    norm_params = {
        'totals': total_pops_all,
        'factor': factor,
        'log_offset': log_offset
    }

    return normalized_pops, norm_params

def denormalize_pops_pow(normalized_pops, norm_params):
    """
    Denormalize populations back to original space.

    Args:
        normalized_pops: Normalized population array
        norm_params: Dictionary with 'totals', 'factor', and 'log_offset' keys

    Returns:
        reconstructed_pops: Populations in original space
    """
    total_pops_all = norm_params['totals']
    factor = norm_params['factor']
    log_offset = norm_params['log_offset']

    # normalized_pops = np.log10(fractional_pops + log_offset)**(1/factor)
    reconstructed_fractional_pops = np.power(10, -normalized_pops**factor) - log_offset
    reconstructed_pops = reconstructed_fractional_pops * \
        (total_pops_all[:, :, :, np.newaxis] + log_offset)

    return reconstructed_pops

# ===================================================================
# ORIGINAL 'LOG' NORMALIZATION
# ===================================================================
def normalize_pops_log(pops, factor=4., log_offset=1e-12):
    """
    Normalize populations using logarithmic transformation with height-stratified statistics.

    Args:
        pops: Population array of shape (nz, ny, nx, nlev)
        factor: Scaling factor for the normalized populations (default: 4.0)
        log_offset: Small value to avoid log(0) (default: 1e-12)

    Returns:
        normalized_pops: Normalized populations
        norm_params: Dictionary with normalization parameters for denormalization
    """
    total_pops_all = np.sum(pops, axis=3)
    # total_pops_mean = np.mean(total_pops_all, axis=(1,2))
    # fractional_pops = pops / (total_pops_mean[:, np.newaxis, np.newaxis, np.newaxis] + log_offset)
    fractional_pops = pops / (total_pops_all[:, :, :, np.newaxis] + log_offset)
    mean_populations = np.mean(fractional_pops, axis=(1, 2))

    mean_broadcast = mean_populations[:, np.newaxis, np.newaxis, :]
    normalized_pops = np.log10(fractional_pops / (mean_broadcast + log_offset)) / factor

    norm_params = {
        'means': mean_populations,
        'totals': total_pops_all,
        # 'totals': total_pops_mean,
        'factor': factor,
        'log_offset': log_offset
    }

    return normalized_pops, norm_params

def denormalize_pops_log(normalized_pops, norm_params):
    """
    Denormalize populations back to original space.

    Args:
        normalized_pops: Normalized population array
        norm_params: Dictionary with 'means', 'totals', 'factor', and 'log_offset' keys

    Returns:
        reconstructed_pops: Populations in original space
    """
    mean_populations = norm_params['means']
    total_pops_all = norm_params['totals']
    # total_pops_all = norm_params['totals'][:, ::2, ::2]
    factor = norm_params['factor']
    log_offset = norm_params['log_offset']

    temp = normalized_pops * factor
    temp = 10**temp

    mean_broadcast = mean_populations[:, np.newaxis, np.newaxis, :]
    reconstructed_fractional_pops = temp * (mean_broadcast + log_offset)

    reconstructed_pops = reconstructed_fractional_pops * \
        (total_pops_all[:, :, :, np.newaxis] + log_offset)
        # (total_pops_all[:, np.newaxis, np.newaxis, np.newaxis] + log_offset)

    return reconstructed_pops

# ===================================================================
# 'LOGF' TARGETS (plan rung 3, D-5): log10 of the level fractions, no per-height means
# ===================================================================
def normalize_pops_logf(pops, factor=1., log_offset=1e-12):
    """
    normalized = log10(f + log_offset) with f = n / sum_levels(n): the representation produced by the 'log_softmax'
    head of graphnet.py (sum of the fractions exactly 1). factor must be 1 so that dex = |factor| * |d normalized|
    keeps holding everywhere; 'means' (per-height mean fractions) are kept for the climatology baseline.
    """
    assert factor == 1.0, "normalize_pops_logf: set [normalization] factor = 1.0 for 'logf' targets"
    total_pops_all = np.sum(pops, axis=3)
    fractional_pops = pops / (total_pops_all[:, :, :, np.newaxis] + log_offset)
    normalized_pops = np.log10(fractional_pops + log_offset)

    norm_params = {
        'means': np.mean(fractional_pops, axis=(1, 2)),
        'totals': total_pops_all,
        'factor': factor,
        'log_offset': log_offset
    }

    return normalized_pops, norm_params

def denormalize_pops_logf(normalized_pops, norm_params):
    total_pops_all = norm_params['totals']
    log_offset = norm_params['log_offset']

    reconstructed_fractional_pops = np.maximum(10**normalized_pops - log_offset, 0.0)
    reconstructed_pops = reconstructed_fractional_pops * \
        (total_pops_all[:, :, :, np.newaxis] + log_offset)

    return reconstructed_pops

def calculate_mean_std(features, labels):
    """
    Calculate mean and standard deviation for each feature type.

    For velocities and magnetic fields, statistics are computed on the magnitude.
    For other quantities, statistics are computed directly on the values.

    Args:
        features: List of feature arrays
        labels: List of feature labels (e.g., 'vel', 'b', 'temp', 'n_h', 'n_e', 'n_p')

    Returns:
        means: List of mean values per height
        stds: List of standard deviations per height
    """
    means = []
    stds = []
    for feature, label in zip(features, labels):
        if 'vel' in label or 'b' in label:
            means.append(np.mean(np.linalg.norm(feature, axis=3), axis=(1, 2)))
            stds.append(np.std(np.linalg.norm(feature, axis=3), axis=(1, 2)))
        else:
            means.append(np.mean(feature, axis=(1, 2, 3)))
            stds.append(np.std(feature, axis=(1, 2, 3)))
    return means, stds

# ===================================================================
# ORIGINAL 'POW' NORMALIZATION
# ===================================================================
def normalize_features_pow(features, labels, log_offset=1e-12):
    """
    Normalize physical features using height-stratified statistics and scale to [-1, 1] range.

    - Velocities and magnetic fields: divided by std of magnitude, then scaled to [-1, 1]
    - Densities (n_*): log10(value / mean), then scaled to [-1, 1]
    - Other quantities: (value - mean) / std, then scaled to [-1, 1]

    Args:
        features: List of feature arrays
        labels: List of feature labels
        log_offset: Small value to avoid division by zero (default: 1e-12)

    Returns:
        normalized: List of normalized feature arrays (scaled to [-1, 1])
        norm_params: Dictionary with 'means', 'stds', 'scale_factors', and 'log_offset'
    """
    # Normalize features and targets as done during training

    normalized = []
    scale_factors = []
    means, stds = calculate_mean_std(features, labels)

    for feature, label, mean, std in zip(features, labels, means, stds):
        mean_broadcast = mean[:, np.newaxis, np.newaxis, np.newaxis]
        std_broadcast = std[:, np.newaxis, np.newaxis, np.newaxis]
        
        if 'vel' in label or 'b' in label:
            scale_factor = 4.0
            # Normalize by std, then apply a power-law transformation
            z = (feature - mean_broadcast) / std_broadcast
            normed = np.sign(z) * np.abs(z)**(1 / scale_factor)
        elif 'n_' in label:
            scale_factor = 10.0
            # Logarithmic transformation relative to the mean, scaled by a factor
            normed = np.log10(feature / mean_broadcast) / scale_factor
        else: # Assuming 'temp' or other quantities
            scale_factor = 1.0
             # Logarithmic transformation relative to the mean, scaled by a factor
            normed = np.log10(feature / mean_broadcast) / scale_factor

        normalized.append(normed)
        scale_factors.append(scale_factor)

    norm_params = {
        'means': means,
        'stds': stds,
        'scale_factors': scale_factors,
        'log_offset': log_offset
    }

    return normalized, norm_params

def denormalize_features_pow(normalized_features, labels, norm_params):
    """
    Denormalize features back to original space.

    Args:
        normalized_features: List of normalized feature arrays
        labels: List of feature labels
        norm_params: Dictionary with 'means', 'stds', 'scale_factors', and 'log_offset'

    Returns:
        denormalized: List of feature arrays in original space
    """
    means = norm_params['means']
    stds = norm_params['stds']
    scale_factors = norm_params['scale_factors']

    denormalized = []
    for feature, label, mean, std, scale_factor in zip(normalized_features, labels, means, stds, scale_factors):
        std_broadcast = std[:, np.newaxis, np.newaxis, np.newaxis]
        mean_broadcast = mean[:, np.newaxis, np.newaxis, np.newaxis]

        if 'vel' in label or 'b' in label:
            z_reconstructed = np.sign(feature) * (np.abs(feature)**scale_factor)
            denormalized.append(z_reconstructed * std_broadcast + mean_broadcast)
        elif 'n_' in label:
            denormalized.append(mean_broadcast * 10**(feature * scale_factor))
        else:
            denormalized.append(mean_broadcast * 10**(feature * scale_factor))

    return denormalized


def normalize_features_log(features, labels, log_offset=1e-12):
    """
    Normalize physical features using height-stratified statistics and scale to [-1, 1] range.

    - Velocities and magnetic fields: divided by std of magnitude, then scaled to [-1, 1]
    - Densities (n_*): log10(value / mean), then scaled to [-1, 1]
    - Other quantities: (value - mean) / std, then scaled to [-1, 1]

    Args:
        features: List of feature arrays
        labels: List of feature labels
        log_offset: Small value to avoid division by zero (default: 1e-12)

    Returns:
        normalized: List of normalized feature arrays (scaled to [-1, 1])
        norm_params: Dictionary with 'means', 'stds', 'scale_factors', and 'log_offset'
    """
    normalized = []
    scale_factors = []
    means, stds = calculate_mean_std(features, labels)

    for feature, label, mean, std in zip(features, labels, means, stds):
        # First apply the standard normalization
        if 'vel' in label or 'b' in label:
            normed = (feature) / (std[:, np.newaxis, np.newaxis, np.newaxis] + log_offset)
        elif 'n_' in label:
            normed = np.log10(feature / (mean[:, np.newaxis, np.newaxis, np.newaxis] + log_offset))
        else:
            normed = (feature - mean[:, np.newaxis, np.newaxis, np.newaxis]) / \
                    (std[:, np.newaxis, np.newaxis, np.newaxis] + log_offset)

        # Calculate scaling factor to map to [-1, 1]
        # Find the maximum absolute value
        max_abs_value = np.max(np.abs(normed))
        scale_factor = max_abs_value if max_abs_value > 0 else 1.0

        # Scale to [-1, 1]
        normed_scaled = normed / scale_factor

        normalized.append(normed_scaled)
        scale_factors.append(scale_factor)

    # Return normalization parameters
    norm_params = {
        'means': means,
        'stds': stds,
        'scale_factors': scale_factors,
        'log_offset': log_offset
    }

    return normalized, norm_params
 
def denormalize_features_log(normalized_features, labels, norm_params):
    """
    Denormalize features back to original space.

    Args:
        normalized_features: List of normalized feature arrays
        labels: List of feature labels
        norm_params: Dictionary with 'means', 'stds', 'scale_factors', and 'log_offset'

    Returns:
        denormalized: List of feature arrays in original space
    """
    means = norm_params['means']
    stds = norm_params['stds']
    scale_factors = norm_params['scale_factors']
    log_offset = norm_params['log_offset']

    denormalized = []
    for feature, label, mean, std, scale_factor in zip(normalized_features, labels, means, stds, scale_factors):
        # First reverse the [-1, 1] scaling
        feature_unscaled = feature * scale_factor

        # Then reverse the standard normalization
        std_broadcast = std[:, np.newaxis, np.newaxis, np.newaxis] + log_offset
        mean_broadcast = mean[:, np.newaxis, np.newaxis, np.newaxis]

        if 'vel' in label or 'b' in label:
            denormalized.append(feature_unscaled * std_broadcast)
        elif 'n_' in label:
            denormalized.append((10**feature_unscaled) * (mean_broadcast + log_offset))
        else:
            denormalized.append((feature_unscaled * std_broadcast) + mean_broadcast)

    return denormalized

# ===================================================================
# V2 FEATURES (plan rung 2, D-4): code-independent inputs
# ===================================================================
def normalize_features_v2(features, labels, norm_params=None, clip=5.0, vel_scale=1e4):
    """
    log10 of every scalar (temp, n_e, rho, press, n_h, n_p) standardized by ONE global mean/std of the whole array
    (no per-height statistics) and clipped at +-clip sigma; velocities divided by vel_scale (10 km/s); B is not
    accepted (dropped in D-4). With norm_params the statistics saved by the training run are applied (inference).

    Returns:
        normalized: List of normalized feature arrays
        norm_params: {'version': 2, 'means', 'stds', 'clip', 'vel_scale'} (means/stds of the log10 scalars)
    """
    fit = norm_params is None
    if fit:
        norm_params = {'version': 2, 'means': [], 'stds': [], 'clip': clip, 'vel_scale': vel_scale}

    normalized = []
    for i, (feature, label) in enumerate(zip(features, labels)):
        if feature.ndim == 3:
            feature = feature[:, :, :, np.newaxis]
        if label == 'b':
            raise ValueError("normalize_features_v2: B is not an input of the v2 representation (plan D-4)")
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

# ===================================================================
# UNIFIED INTERFACE
# ===================================================================
def normalize_features(features, labels, log_offset=1e-12, type='log'):
    print(f"Nomalaizing features with {type}")
    if type == 'log':
        return normalize_features_log(features, labels, log_offset)
    elif type == 'pow':
        return normalize_features_pow(features, labels, log_offset)
    elif type == 'v2':
        return normalize_features_v2(features, labels)
    else:
        raise ValueError("Normalization type not implemented. Choose 'log', 'pow' or 'v2'.")

def denormalize_features(normalized_features, labels, norm_params, type='log'):
    print(f"Denomalaizing features with {type}")
    if type == 'log':
        return denormalize_features_log(normalized_features, labels, norm_params)
    elif type == 'pow':
        return denormalize_features_pow(normalized_features, labels, norm_params)
    else:
        raise ValueError("Normalization type not implemented. Choose 'log' 'pow'.")

# ===================================================================
# PRE-COMPUTED PARAMETER NORMALIZATION (FOR PURE INFERENCE)
# ===================================================================
def normalize_features_log_with_params(features, labels, norm_params):
    """
    Normalize physical features using pre-computed log statistics.

    Args:
        features: List of feature arrays
        labels: List of feature labels
        norm_params: Dictionary with 'means', 'stds', 'scale_factors', and 'log_offset'

    Returns:
        normalized: List of normalized feature arrays
    """
    means = norm_params['means']
    stds = norm_params['stds']
    scale_factors = norm_params['scale_factors']
    log_offset = norm_params.get('log_offset', 1e-12)

    normalized = []
    for feature, label, mean, std, scale_factor in zip(features, labels, means, stds, scale_factors):
        if feature.ndim == 3:
            feature = feature[:, :, :, np.newaxis]
        std_broadcast = std[:, np.newaxis, np.newaxis, np.newaxis]
        mean_broadcast = mean[:, np.newaxis, np.newaxis, np.newaxis]

        if 'vel' in label or 'b' in label:
            normed = feature / (std_broadcast + log_offset)
        elif 'n_' in label:
            normed = np.log10(feature / (mean_broadcast + log_offset))
        else:
            normed = (feature - mean_broadcast) / (std_broadcast + log_offset)

        normed_scaled = normed / scale_factor
        normalized.append(normed_scaled)

    return normalized

def normalize_features_pow_with_params(features, labels, norm_params):
    """
    Normalize physical features using pre-computed pow statistics.

    Args:
        features: List of feature arrays
        labels: List of feature labels
        norm_params: Dictionary with 'means', 'stds', 'scale_factors', and 'log_offset'

    Returns:
        normalized: List of normalized feature arrays
    """
    means = norm_params['means']
    stds = norm_params['stds']
    scale_factors = norm_params['scale_factors']

    normalized = []
    for feature, label, mean, std, scale_factor in zip(features, labels, means, stds, scale_factors):
        if feature.ndim == 3:
            feature = feature[:, :, :, np.newaxis]
        std_broadcast = std[:, np.newaxis, np.newaxis, np.newaxis]
        mean_broadcast = mean[:, np.newaxis, np.newaxis, np.newaxis]

        if 'vel' in label or 'b' in label:
            z = (feature - mean_broadcast) / std_broadcast
            normed = np.sign(z) * np.abs(z)**(1 / scale_factor)
        elif 'n_' in label:
            normed = np.log10(feature / mean_broadcast) / scale_factor
        else:
            normed = np.log10(feature / mean_broadcast) / scale_factor

        normalized.append(normed)

    return normalized

def normalize_features_with_params(features, labels, norm_params, type='log'):
    """
    Normalize features using pre-computed statistics from training.

    Args:
        features: List of feature arrays
        labels: List of feature labels
        norm_params: Pre-computed normalization parameters dictionary
        type: Normalization type ('log' or 'pow')

    Returns:
        normalized: List of normalized feature arrays
    """
    print(f"Normalizing features with pre-computed parameters using {type}")
    if type == 'log':
        return normalize_features_log_with_params(features, labels, norm_params)
    elif type == 'pow':
        return normalize_features_pow_with_params(features, labels, norm_params)
    elif type == 'v2':
        return normalize_features_v2(features, labels, norm_params)[0]
    else:
        raise ValueError("Normalization type not implemented. Choose 'log', 'pow' or 'v2'.")

def normalize_pops(pops, factor=4., log_offset=1e-12, type='log'):
    print(f"Nomalaizing populations with {type}")
    if type == 'log':
        return normalize_pops_log(pops, factor, log_offset)
    elif type == 'pow':
        return normalize_pops_pow(pops, factor, log_offset)
    elif type == 'logf':
        return normalize_pops_logf(pops, factor, log_offset)
    else:
        raise ValueError("Normalization type not implemented. Choose 'log', 'pow' or 'logf'.")

def denormalize_pops(normalized_pops, norm_params, type='log'):
    print(f"Denomalaizing populations with {type}")
    if type == 'log':
        return denormalize_pops_log(normalized_pops, norm_params)
    elif type == 'pow':
        return denormalize_pops_pow(normalized_pops, norm_params)
    elif type == 'logf':
        return denormalize_pops_logf(normalized_pops, norm_params)
    else:
        raise ValueError("Normalization type not implemented. Choose 'log', 'pow' or 'logf'.")
