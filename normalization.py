"""
Normalization and denormalization functions for NLTE populations and physical features.

This module contains the normalization strategy from porta_graph_test.ipynb
to ensure consistency across training and inference scripts.
"""

import numpy as np

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
    fractional_pops = pops / (total_pops_all[:, :, :, np.newaxis] + log_offset)
    mean_populations = np.mean(fractional_pops, axis=(1, 2))

    mean_broadcast = mean_populations[:, np.newaxis, np.newaxis, :]
    normalized_pops = np.log10(fractional_pops / (mean_broadcast + log_offset)) / factor

    norm_params = {
        'means': mean_populations,
        'totals': total_pops_all,
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
    factor = norm_params['factor']
    log_offset = norm_params['log_offset']

    temp = normalized_pops * factor
    temp = 10**temp

    mean_broadcast = mean_populations[:, np.newaxis, np.newaxis, :]
    reconstructed_fractional_pops = temp * (mean_broadcast + log_offset)

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
# UNIFIED INTERFACE
# ===================================================================
def normalize_features(features, labels, log_offset=1e-12, type='log'):
    print(f"Nomalaizing features with {type}")
    if type == 'log':
        return normalize_features_log(features, labels, log_offset)
    elif type == 'pow':
        return normalize_features_pow(features, labels, log_offset)
    else:
        raise ValueError("Normalization type not implemented. Choose 'log' 'pow'.")

def denormalize_features(normalized_features, labels, norm_params, type='log'):
    print(f"Denomalaizing features with {type}")
    if type == 'log':
        return denormalize_features_log(normalized_features, labels, norm_params)
    elif type == 'pow':
        return denormalize_features_pow(normalized_features, labels, norm_params)
    else:
        raise ValueError("Normalization type not implemented. Choose 'log' 'pow'.")

def normalize_pops(pops, factor=4., log_offset=1e-12, type='log'):
    print(f"Nomalaizing populations with {type}")
    if type == 'log':
        return normalize_pops_log(pops, factor, log_offset)
    elif type == 'pow':
        return normalize_pops_pow(pops, factor, log_offset)
    else:
        raise ValueError("Normalization type not implemented. Choose 'log', 'pow'.")

def denormalize_pops(normalized_pops, norm_params, type='log'):
    print(f"Denomalaizing populations with {type}")
    if type == 'log':
        return denormalize_pops_log(normalized_pops, norm_params)
    elif type == 'pow':
        return denormalize_pops_pow(normalized_pops, norm_params)
    else:
        raise ValueError("Normalization type not implemented. Choose 'log', 'pow'.")
