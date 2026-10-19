import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
import os
import sys
import muram as mio

# Add the repository folder to path (normalization, vertical_grids)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from normalization import CA_TOTAL_PER_RHO, lte_log_fractions
from vertical_grids import to_levels

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

# The run to compare, a muram_inference.py output folder:   python compare_populations.py <run_dir>
results_dir = sys.argv[1]
# Its grid (heights above MURaM's tau500 = 1 [Mm], aligned with Bifrost in muram_inference.py), the heights of all the
# MURaM levels and the MURaM column (y, x) of the first predicted column
muram_grid = np.load(os.path.join(results_dir, 'muram_grid.npz'))
zz_grid_muram = muram_grid['z']
muram_z_grid = muram_grid['z_native']
oy, ox = muram_grid['origin']

# Load MURaM snap (temperature, electron density and density of the LTE baseline)
pathsource = '/dat/milic/MURaM_enhanced_network/'
snap_number = 499000
cube = mio.MuramSnap(pathsource, snap_number)

print(f"   GNN vertical heights (Mm): min={zz_grid_muram.min():.4f}, max={zz_grid_muram.max():.4f}")

# 1.5D NLTE z grid is cropped from 15 to 416 (not including 416)
fits_heights = muram_z_grid[15:416]
print(f"   NLTE vertical heights (Mm): min={fits_heights.min():.4f}, max={fits_heights.max():.4f}")

# Clip the range of the GNN grid to remove parts where NLTE doesn't reach
valid_mask = (zz_grid_muram >= fits_heights.min()) & (zz_grid_muram <= fits_heights.max())
zz_gnn_clipped = zz_grid_muram[valid_mask]
print(f"   Clipped GNN vertical heights (Mm): min={zz_gnn_clipped.min():.4f}, max={zz_gnn_clipped.max():.4f}")
print(f"   Keeping {len(zz_gnn_clipped)} GNN heights out of {len(zz_grid_muram)}.")

# Load GNN predictions
pred_path = os.path.join(results_dir, f"muram_predictions_stride_{int(muram_grid['stride'])}_full.npy")
if not os.path.exists(pred_path):
    print(f"Error: GNN predictions not found at {pred_path}")
    sys.exit(1)
print(f"Loading GNN predictions from {pred_path}...")
pred = np.load(pred_path) # Shape: (n_levels, ny_p, nx_p, 6) in cm^-3

# Clip GNN predictions to the same heights
pred_clipped = pred[valid_mask, :, :, :] # Shape: (len(zz_gnn_clipped), ny_p, nx_p, 6)

fits_path = "/dat/milic/MURaM_enhanced_network/che_full_499000_lwsynth_200.0.fits"
if not os.path.exists(fits_path):
    print(f"Error: FITS file not found at {fits_path}.")
    sys.exit(1)

print("2. Memory-mapping the 29 GB FITS file...")
ny_p, nx_p = pred.shape[1:3]
with fits.open(fits_path, memmap=True) as hdul:
    # FITS shape in Python: (1024, 1024, 6, 401) representing [x_idx, y_idx, level, z_fits_idx]; FITS[a, b] = MURaM (x=a, y=b)
    print("   Loading cropped horizontal domain...")
    fits_cropped = hdul[2].data[ox:ox + nx_p, oy:oy + ny_p, :, :] # Shape: (nx_p, ny_p, 6, 401)
    
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
# fits_cropped has shape (nx_p, ny_p, 6, 401).
# We interpolate along the last axis (axis=-1) to target heights zz_gnn_clipped,
# linearly in log10 (a linear interpolation mixes chromospheric and coronal states at the transition region)
fits_interp_func = interp.interp1d(fits_heights, np.log10(np.clip(fits_cropped, 1e-30, None)), axis=-1, bounds_error=False, fill_value="extrapolate")
fits_on_gnn = 10.0 ** fits_interp_func(zz_gnn_clipped) # Shape: (nx_p, ny_p, 6, len(zz_gnn_clipped))
print(f"   Interpolated NLTE shape: {fits_on_gnn.shape}")

print("4. Aligning coordinate axes...")
# GNN predictions has shape (len(zz_gnn_clipped), ny_p, nx_p, 6) -> (z, y, x, npops).
# We transpose to (x, y, npops, z) to match fits_on_gnn.
pred_aligned = pred_clipped.transpose(2, 1, 3, 0)
print(f"   Aligned GNN shape: {pred_aligned.shape}")

print("\n5. Computing comparison statistics at each height...")

# We will compare all six levels (0 = 4s ground; 1, 2 = 3d metastable; 3, 4 = 4p; 5 = Ca III)
levels_to_compare = [0, 1, 2, 3, 4, 5]
levels_labels = {0: "Ca II 4s (lvl 0)", 1: "Ca II 3d3/2 (lvl 1)", 2: "Ca II 3d5/2 (lvl 2)", 3: "Ca II 4p1/2 (lvl 3)",
                 4: "Ca II 4p3/2 (lvl 4)", 5: "Ca III (lvl 5)"}

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

# Errors in dex per height band (Mm above tau500 = 1) on the nodes where the reference has Ca II (Ca II fraction > 1e-2),
# for the GNN and for the LTE baseline: the Saha-Boltzmann populations of MURaM's T and n_e times the total Ca of the GNN
# (CA_TOTAL_PER_RHO * rho). T, n_e and rho of the predicted columns (cube (z, x, y)) interpolated in log10 to the heights
print("   Reading T, n_e and rho of the predicted columns for the LTE baseline...")
def crop_on_grid(q):
    return to_levels(q[:, ox:ox + nx_p, oy:oy + ny_p], muram_z_grid, zz_gnn_clipped, log=True)   # (z, x, y)

temp = crop_on_grid(cube.Temp)                                 # K
n_e = crop_on_grid(cube.ne) * 1e6                              # cm^-3 -> m^-3
totals = CA_TOTAL_PER_RHO * crop_on_grid(cube.rho) * 1e3       # g cm^-3 -> kg m^-3, total Ca in cm^-3
lte = (10.0 ** lte_log_fractions(temp[..., np.newaxis], n_e[..., np.newaxis]) * totals[..., np.newaxis]).transpose(1, 2, 3, 0)   # (x, y, npops, z)

level_names = ['4s', '3d3/2', '3d5/2', '4p1/2', '4p3/2', 'CaIII']
bands = [(-np.inf, 0.0, 'z<0'), (0.0, 0.5, '0-0.5'), (0.5, 1.0, '0.5-1'), (1.0, 1.5, '1-1.5'), (1.5, 2.2, '1.5-2.2'), (2.2, 3.5, '2.2-3.5'), (3.5, np.inf, '>3.5')]
f_ref = fits_on_gnn / fits_on_gnn.sum(2, keepdims=True)
relevant = (1.0 - f_ref[:, :, 5, :]) > 1e-2                    # (x, y, z)

def dex_stats(d):           # median |dlog n|, signed median, p84 |dlog n|
    return f"{np.nanmedian(np.abs(d)):12.4f}   | {np.nanmedian(d):+12.4f}   | {np.nanpercentile(np.abs(d), 84):12.4f}   |"

print("=" * 109)
print(f"{'Method':^6} | {'Band (Mm)':^9} | {'Level':^6} | {'med |dlog n|':^14} | {'signed median':^14} | {'p84 |dlog n|':^14} | {'med dlog total':^14} | {'n nodes':^9} |")
print("=" * 109)
for name, arr in (('gnn', pred_aligned), ('lte', lte)):
    with np.errstate(divide='ignore', invalid='ignore'):
        dlog = np.log10(arr) - np.log10(fits_on_gnn)                    # (x, y, npops, z)
        dtot = np.log10(arr.sum(2)) - np.log10(fits_on_gnn.sum(2))      # (x, y, z)
    for lo, hi, band in bands:
        hsel = (zz_gnn_clipped >= lo) & (zz_gnn_clipped < hi)
        m = relevant[:, :, hsel] & np.isfinite(dtot[:, :, hsel])
        if m.sum() == 0:
            continue
        d_band = dlog[:, :, :, hsel].transpose(0, 1, 3, 2)[m]           # (n nodes, npops)
        for lvl in levels_to_compare:
            print(f"{name:^6} | {band:^9} | {level_names[lvl]:^6} | " + dex_stats(d_band[:, lvl]))
        print(f"{name:^6} | {band:^9} | {'all':^6} | " + dex_stats(d_band) + f" {np.nanmedian(dtot[:, :, hsel][m]):+12.4f}   | {m.sum():9d} |")
        print("-" * 109)
print("=" * 109 + "\n")

print("6. Generating premium population comparison plots...")

plot_dir = results_dir

# Select a representative set of 5 heights distributed across zz_gnn_clipped
selected_indices = np.linspace(0, len(zz_gnn_clipped) - 1, 5, dtype=int)

for idx in selected_indices:
    h = zz_gnn_clipped[idx]
    print(f"   Plotting comparison maps at height: {h:.4f} Mm...")
    
    fig, axes = plt.subplots(len(levels_to_compare), 3, figsize=(18, 4.7 * len(levels_to_compare)), dpi=150)
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
