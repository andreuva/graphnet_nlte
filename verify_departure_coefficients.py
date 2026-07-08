# verify_departure_coefficients.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from configobj import ConfigObj
from normalization import normalize_dep_coeffs, denormalize_dep_coeffs

# 1. Load config
config = ConfigObj('conf.dat', unrepr=True)
datadir = config['data']['datadir']
nx = config['data']['nx']
ny = config['data']['ny']
nz = config['data']['nz_orig']
nlev = config['data']['nlev']

# 2. Load calculated departure coefficients (beta) and physical quantities
print("Loading calculated departure coefficients and background quantities...")
beta_path = f'{datadir}/AR_385_CaII_5L_beta.npy'
if not os.path.exists(beta_path):
    raise FileNotFoundError(f"Could not find beta npy file: {beta_path}")
    
beta = np.load(beta_path)
temp = np.memmap(f'{datadir}/AR_385_temp.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
n_e = np.memmap(f'{datadir}/AR_385_ne.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
n_h = np.memmap(f'{datadir}/AR_385_nh.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))

# 3. Test Normalization and Denormalization Invertibility
print("\nTesting normalization and denormalization...")
normalized_beta, norm_params = normalize_dep_coeffs(beta)
reconstructed_beta = denormalize_dep_coeffs(normalized_beta, norm_params)

# Check bounds of normalized beta
norm_min, norm_max = np.min(normalized_beta), np.max(normalized_beta)
print(f"Normalized beta range: [{norm_min:.6f}, {norm_max:.6f}] (Should be within [-1.0, 1.0])")

# Check reconstruction error in log10 space (which matches the target space of training)
log_beta = np.log10(beta + 1e-12)
log_reconstructed = np.log10(reconstructed_beta + 1e-12)
abs_diff_log = np.abs(log_reconstructed - log_beta)
max_err_log = np.max(abs_diff_log)
mean_err_log = np.mean(abs_diff_log)
print(f"Reconstruction Max Log-Space Error: {max_err_log:.3e}")
print(f"Reconstruction Mean Log-Space Error: {mean_err_log:.3e}")

# Relative error in linear space
rel_diff = np.abs(reconstructed_beta - beta) / (beta + 1e-30)
max_rel_err = np.max(rel_diff)
mean_rel_err = np.mean(rel_diff)
print(f"Reconstruction Max Relative Error: {max_rel_err:.3e}")
print(f"Reconstruction Mean Relative Error: {mean_rel_err:.3e}")

# Assert correctness
assert max_err_log < 1e-4, f"Reconstruction log-space error too high! Max error: {max_err_log:.3e}"
print("Verification passed! Invertibility is exact within negligible float tolerance.\n")

# Create output directory for plots if it doesn't exist
plot_dir = 'plots_verification'
os.makedirs(plot_dir, exist_ok=True)

# 4. Generate Plot 1: Distributions
print("Generating distribution plot (sub-sampled for performance)...")
plt.figure(figsize=(15, 5))

# Sub-sample using reshape(-1) to create memory-efficient view slices
stride_val = 100000
beta_sub = beta.reshape(-1)[::stride_val]
norm_beta_sub = normalized_beta.reshape(-1)[::stride_val]
recon_beta_sub = reconstructed_beta.reshape(-1)[::stride_val]

# original beta distribution (in log10 space)
plt.subplot(131)
plt.hist(np.log10(beta_sub + 1e-12), bins=100, color='blue', alpha=0.7, density=True)
plt.title("Original log10(beta)")
plt.xlabel("log10(beta)")
plt.ylabel("Density")

# normalized beta distribution
plt.subplot(132)
plt.hist(norm_beta_sub, bins=100, color='green', alpha=0.7, density=True)
plt.title("Normalized beta")
plt.xlabel("Normalized value")
plt.ylabel("Density")

# reconstructed beta distribution
plt.subplot(133)
plt.hist(np.log10(recon_beta_sub + 1e-12), bins=100, color='red', alpha=0.5, density=True)


plt.title("Reconstructed log10(beta)")
plt.xlabel("log10(beta)")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig(f'{plot_dir}/beta_distributions.png', dpi=150)
plt.close()

# 5. Generate Plot 2: Horizontal 2D Slices (plane_plot = 20)
plane_plot = 20
print(f"Generating horizontal slices at depth k={plane_plot}...")
plt.figure(figsize=(12, 10))

# beta slice
plt.subplot(221)
im1 = plt.imshow(np.log10(beta[plane_plot, :, :, 0] + 1e-12), cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(im1, label='log10(beta_0)')
plt.title(f'Departure Coefficient beta_0 at (k={plane_plot})')
plt.xlabel('X Index')
plt.ylabel('Y Index')

# temp slice
plt.subplot(222)
im2 = plt.imshow(temp[plane_plot, :, :, 0], cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(im2, label='Temperature (K)')
plt.title(f'Temperature at (k={plane_plot})')
plt.xlabel('X Index')
plt.ylabel('Y Index')

# ne slice
plt.subplot(223)
im3 = plt.imshow(n_e[plane_plot, :, :, 0], cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(im3, label='Electron density (cm^-3)')
plt.title(f'Electron density at (k={plane_plot})')
plt.xlabel('X Index')
plt.ylabel('Y Index')

# nh slice
plt.subplot(224)
im4 = plt.imshow(n_h[plane_plot, :, :, 0], cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(im4, label='H density (cm^-3)')
plt.title(f'H density at (k={plane_plot})')
plt.xlabel('X Index')
plt.ylabel('Y Index')

plt.tight_layout()
plt.savefig(f'{plot_dir}/horizontal_slices.png', dpi=150)
plt.close()

# 6. Generate Plot 3: Vertical 2D Slices (Z-X slice at middle Y coordinate)
y_slice = ny // 2
print(f"Generating vertical slices at y_idx={y_slice}...")
plt.figure(figsize=(12, 10))

# beta vertical slice
plt.subplot(221)
im1 = plt.imshow(np.log10(beta[:, y_slice, :, 0] + 1e-12), cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(im1, label='log10(beta_0)')
plt.title(f'Departure Coefficient beta_0 (Z-X slice, y={y_slice})')
plt.xlabel('X Index')
plt.ylabel('Z Index (Depth)')

# temp vertical slice
plt.subplot(222)
im2 = plt.imshow(temp[:, y_slice, :, 0], cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(im2, label='Temperature (K)')
plt.title(f'Temperature (Z-X slice, y={y_slice})')
plt.xlabel('X Index')
plt.ylabel('Z Index (Depth)')

# ne vertical slice
plt.subplot(223)
im3 = plt.imshow(np.log10(n_e[:, y_slice, :, 0] + 1e-32), cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(im3, label='log10(Electron density)')
plt.title(f'log10(Electron density) (Z-X slice, y={y_slice})')
plt.xlabel('X Index')
plt.ylabel('Z Index (Depth)')

# nh vertical slice
plt.subplot(224)
im4 = plt.imshow(np.log10(n_h[:, y_slice, :, 0] + 1e-32), cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(im4, label='log10(H density)')
plt.title(f'log10(H density) (Z-X slice, y={y_slice})')
plt.xlabel('X Index')
plt.ylabel('Z Index (Depth)')

plt.tight_layout()
plt.savefig(f'{plot_dir}/vertical_slices.png', dpi=150)
plt.close()

print(f"Plots saved to {plot_dir}/ directory.")
