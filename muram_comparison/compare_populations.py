import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
import os
import sys
import muram as mio

# Configure matplotlib for premium, publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'font.family': 'sans-serif'
})

print("1. Constructing grids and setting up vertical coordinates...")

# Config parameters from training (saved in conf.dat)
nz_b = 425
logspace_fraction = 0.33
nz_linear = 30
nz_log = 20

bifrost_grid_file = '/dat/andreuva/gpu/graphnet/en024048_hion/grid_bifrost.npz'
if not os.path.exists(bifrost_grid_file):
    print(f"Error: Bifrost grid file not found at {bifrost_grid_file}")
    sys.exit(1)
bifrost_z_grid = np.load(bifrost_grid_file)["z"]

# Grid setup for Bifrost
z_b = np.arange(nz_b)
new_z_b_log = np.concatenate([
    np.linspace(0, nz_b * logspace_fraction, nz_linear, endpoint=False),
    np.logspace(np.log10(nz_b * logspace_fraction), np.log10(nz_b - 1), nz_log)
])
new_z_b = np.clip(new_z_b_log, 0, nz_b - 1)
zz_grid_bifrost = np.interp(new_z_b, z_b, bifrost_z_grid)

# Load MURaM snap to compute vertical coordinate muram_z_grid
pathsource = '/dat/milic/MURaM_enhanced_network/'
snap_number = 499000
cube = mio.MuramSnap(pathsource, snap_number)

dx, dy, dz = 24/1e3, 24/1e3, 20/1e3
nz, nx, ny = cube.Temp.shape
z = np.arange(nz)
z_geom = z * dz

# Load the full cube tau to prioritize accuracy (per user instruction)
print("Loading full cube tau to compute tau=1 layer...")
tau_mean = np.mean(cube.tau, axis=(1, 2))
tau_zero_index = np.argmin(np.abs(tau_mean - 1))
print(f"   tau=1 layer index: {tau_zero_index} (z={z_geom[tau_zero_index]:.4f} Mm)")

muram_z_grid = z_geom - z_geom[tau_zero_index]

# Align height interpolation exactly with Bifrost:
# Map zz_grid_bifrost to fractional indices in MURaM's vertical coordinate system
new_z = np.interp(zz_grid_bifrost, muram_z_grid, z)
new_z = np.clip(new_z, 0, nz - 1)
zz_grid_muram = np.interp(new_z, z, muram_z_grid)

print(f"   GNN vertical heights (Mm): min={zz_grid_muram.min():.4f}, max={zz_grid_muram.max():.4f}")

# 1.5D NLTE z grid is cropped from 15 to 416 (not including 416)
fits_heights = muram_z_grid[15:416]
print(f"   NLTE vertical heights (Mm): min={fits_heights.min():.4f}, max={fits_heights.max():.4f}")

# Clip the range of the GNN grid to remove parts where NLTE doesn't reach
valid_mask = (zz_grid_muram >= fits_heights.min()) & (zz_grid_muram <= fits_heights.max())
zz_gnn_clipped = zz_grid_muram[valid_mask]
print(f"   Clipped GNN vertical heights (Mm): min={zz_gnn_clipped.min():.4f}, max={zz_gnn_clipped.max():.4f}")
print(f"   Keeping {len(zz_gnn_clipped)} GNN heights out of 50.")

# Load GNN predictions
pred_path = 'muram_predictions_stride_4_full.npy'
if not os.path.exists(pred_path):
    # Try parent directory
    pred_path_parent = os.path.join('..', 'muram_predictions_stride_4_full.npy')
    if os.path.exists(pred_path_parent):
        pred_path = pred_path_parent
    else:
        # Try absolute path
        pred_path_abs = '/dat/andreuva/gpu/graphnet/graphnet_nlte/muram_predictions_stride_4_full.npy'
        if os.path.exists(pred_path_abs):
            pred_path = pred_path_abs

if not os.path.exists(pred_path):
    print(f"Error: GNN predictions not found at {pred_path}")
    sys.exit(1)
print(f"Loading GNN predictions from {pred_path}...")
pred = np.load(pred_path) # Shape: (50, 992, 992, 6) in m^-3
# Convert GNN predictions to CGS (cm^-3)
# pred = pred * 1e6

# Clip GNN predictions to the same heights
pred_clipped = pred[valid_mask, :, :, :] # Shape: (len(zz_gnn_clipped), 992, 992, 6)

fits_path = "/dat/milic/MURaM_enhanced_network/che_full_499000_lwsynth_200.0.fits"
if not os.path.exists(fits_path):
    print(f"Error: FITS file not found at {fits_path}.")
    sys.exit(1)

print("2. Memory-mapping the 29 GB FITS file...")
with fits.open(fits_path, memmap=True) as hdul:
    # FITS shape in Python: (1024, 1024, 6, 401) representing [x_idx, y_idx, level, z_fits_idx]
    print("   Loading cropped horizontal domain...")
    fits_cropped = hdul[2].data[16:-16, 16:-16, :, :] # Shape: (992, 992, 6, 401)
    
# Reverse the FITS vertical axis to match the increasing order of fits_heights (from photosphere to chromosphere)
fits_cropped = fits_cropped[..., ::-1]
print("   Reversed FITS vertical axis to match increasing heights from photosphere to chromosphere.")

# Automatic unit check: if fits_cropped values are in m^-3 (SI) and pred in cm^-3 (CGS),
# fits_cropped will be ~1e6 times larger. Let's scale fits_cropped if it is in SI units.
mean_fits_val = np.mean(fits_cropped[:, :, 0, 0])

print(f"   FITS populations in SI units (mean={mean_fits_val:.2e}). Converting to CGS...")
fits_cropped = fits_cropped * 1e-6

print("3. Reinterpolating NLTE populations onto clipped GNN heights grid...")
# fits_heights has shape (401,), and is strictly increasing.
# fits_cropped has shape (992, 992, 6, 401).
# We interpolate along the last axis (axis=-1) to target heights zz_gnn_clipped.
fits_interp_func = interp.interp1d(fits_heights, fits_cropped, axis=-1, bounds_error=False, fill_value="extrapolate")
fits_on_gnn = fits_interp_func(zz_gnn_clipped) # Shape: (992, 992, 6, len(zz_gnn_clipped))
print(f"   Interpolated NLTE shape: {fits_on_gnn.shape}")

print("4. Aligning coordinate axes...")
# GNN predictions has shape (len(zz_gnn_clipped), 992, 992, 6) -> (z, y, x, npops).
# We transpose to (x, y, npops, z) to match fits_on_gnn.
pred_aligned = pred_clipped.transpose(2, 1, 3, 0)
print(f"   Aligned GNN shape: {pred_aligned.shape}")

print("\n5. Computing comparison statistics at each height...")

# We will compare levels 0, 3, 5
levels_to_compare = [0, 3, 5]
levels_labels = {0: "Ca II Ground (lvl 0)", 3: "Ca II Metastable (lvl 3)", 5: "Ca III Ion (lvl 5)"}

# Clean terminal table formatting
print("\n" + "=" * 105)
print(f"{'Height (Mm)':^12} | {'Level':^25} | {'Pearson r':^12} | {'MARE (%)':^12} | {'MAE (cm^-3)':^12} | {'Mean Ratio (G/F)':^18} |")
print("=" * 105)

for h_idx, h in enumerate(zz_gnn_clipped):
    for lvl in levels_to_compare:
        g_slice = pred_aligned[:, :, lvl, h_idx]
        f_slice = fits_on_gnn[:, :, lvl, h_idx]
        
        g_flat = g_slice.flatten()
        f_flat = f_slice.flatten()
        
        eps = 1e-30
        rel_diff = np.abs(g_flat - f_flat) / (f_flat + eps)
        mare = np.mean(rel_diff) * 100
        mae = np.mean(np.abs(g_flat - f_flat))
        
        # Pearson correlation
        corr = np.corrcoef(g_flat, f_flat)[0, 1]
        
        # Mean Ratio
        ratio = g_flat / (f_flat + eps)
        mean_ratio = np.mean(ratio)
        
        print(f"{h:10.4f}   | {levels_labels[lvl]:<25} | {corr:10.4f}   | {mare:10.2f}%  | {mae:10.4e} | {mean_ratio:16.4f}   |")
    print("-" * 105)
print("=" * 105 + "\n")

print("6. Generating premium population comparison plots...")

plot_dir = "plots_comparison"
os.makedirs(plot_dir, exist_ok=True)

# Select a representative set of 5 heights distributed across zz_gnn_clipped
selected_indices = np.linspace(0, len(zz_gnn_clipped) - 1, 5, dtype=int)

for idx in selected_indices:
    h = zz_gnn_clipped[idx]
    print(f"   Plotting comparison maps at height: {h:.4f} Mm...")
    
    fig, axes = plt.subplots(len(levels_to_compare), 3, figsize=(18, 14), dpi=150)
    fig.suptitle(f"$z = {h:.4f}$ Mm\n(Left: GNN | Middle: 1.5D NLTE | Right: Difference)", y=0.98)
    
    for row_idx, lvl in enumerate(levels_to_compare):
        g_slice = pred_aligned[:, :, lvl, idx]
        f_slice = fits_on_gnn[:, :, lvl, idx]
        
        # Logarithmic normalization bounds for plotting
        vmin = max(1e-15, min(g_slice.min(), f_slice.min()))
        vmax = max(g_slice.max(), f_slice.max())
        
        # Plot GNN slice
        ax_g = axes[row_idx, 0]
        im_g = ax_g.imshow(g_slice.T, origin='lower', cmap='magma', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        ax_g.set_title(f"GNN - {levels_labels[lvl]}", fontsize=11)
        ax_g.set_xlabel("x pixel", fontsize=9)
        ax_g.set_ylabel("y pixel", fontsize=9)
        
        # Plot FITS slice
        ax_f = axes[row_idx, 1]
        im_f = ax_f.imshow(f_slice.T, origin='lower', cmap='magma', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        ax_f.set_title(f"1.5D NLTE - {levels_labels[lvl]}", fontsize=11)
        ax_f.set_xlabel("x pixel", fontsize=9)
        ax_f.set_ylabel("y pixel", fontsize=9)
        
        # Plot Difference slice: log10(GNN / 1.5D NLTE)
        ax_d = axes[row_idx, 2]
        diff_slice = np.log10(np.clip(g_slice, 1e-32, None)) - np.log10(np.clip(f_slice, 1e-32, None))
        
        # Find symmetric color scale based on 99th percentile of absolute differences
        vlim = max(0.1, np.percentile(np.abs(diff_slice), 99))
        im_d = ax_d.imshow(diff_slice.T, origin='lower', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        ax_d.set_title(f"Difference (dex) - {levels_labels[lvl]}", fontsize=11)
        ax_d.set_xlabel("x pixel", fontsize=9)
        ax_d.set_ylabel("y pixel", fontsize=9)
        
        # Add colorbar for each subplot to keep spacing perfectly consistent and symmetric
        cbar_g = fig.colorbar(im_g, ax=ax_g, fraction=0.046, pad=0.04)
        cbar_g.set_label(r"Population density [$cm^{-3}$]", fontsize=9)
        
        cbar_f = fig.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
        cbar_f.set_label(r"Population density [$cm^{-3}$]", fontsize=9)
        
        cbar_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
        cbar_d.set_label(r"$\log_{10}(\mathrm{GNN} / \mathrm{NLTE})$ [dex]", fontsize=9)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave room for suptitle
    save_plot_path = os.path.join(plot_dir, f"population_comparison_z_{h:.2f}Mm.png")
    plt.savefig(save_plot_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"      Saved map to {save_plot_path}")

print("\nDone! All population comparisons and plots completed successfully.")
