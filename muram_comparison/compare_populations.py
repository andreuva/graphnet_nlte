import numpy as np
import scipy.interpolate as interp
import json
import torch
from lightweaver.atomic_set import lte_pops
from lightweaver.rh_atoms import CaII_atom
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
import os
import sys
import glob
import muram as mio
_here = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path[:0] = [_here, os.path.join(_here, '..')]
import muram_common as mc
from normalization import CA_TOTAL_PER_RHO

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

# The run to score: a muram_inference.py results directory (default: the most recent). Its grid.npz holds the heights of
# the 50 levels above MURaM's tau500 = 1 and their fractional MURaM indices (alignment done once, in the inference)
results_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(_here, '..', 'results', 'muram_inference', '*', 'meta.json')))[-1][:-len('meta.json')]
meta = json.load(open(os.path.join(results_dir, 'meta.json')))
grid = np.load(os.path.join(results_dir, meta['files']['grid']))
zz_grid_muram, new_z = grid['z_rel_tau1_Mm'], grid['muram_fractional_index']
muram_z_grid = (np.arange(mc.FITS_Z_SLICE.stop) - meta['tau1_index_median']) * meta['dz_km'] / 1e3   # native levels above tau500 = 1
oy, ox = meta['origin']                                                                              # native column of pred[:, 0, 0]

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
pred_path = os.path.join(results_dir, meta['files']['predictions'])
print(f"Loading GNN predictions from {pred_path}...")
pred = np.load(pred_path) # Shape: (50, ny, nx, 6) in cm^-3, NaN where not predicted
done = np.isfinite(pred[0, :, :, 0])

# Clip GNN predictions to the same heights
pred_clipped = pred[valid_mask, :, :, :] # Shape: (len(zz_gnn_clipped), 992, 992, 6)

fits_path = "/dat/milic/MURaM_enhanced_network/che_full_499000_lwsynth_200.0.fits"
if not os.path.exists(fits_path):
    print(f"Error: FITS file not found at {fits_path}.")
    sys.exit(1)

print("2. Memory-mapping the 29 GB FITS file...")
ny_p, nx_p = pred.shape[1:3]
with fits.open(fits_path, memmap=True) as hdul:
    # FITS shape in Python: (1024, 1024, 6, 401) representing [x_idx, y_idx, level, z_fits_idx]; FITS[a, b] = MURaM (x=a, y=b)
    print("   Loading the predicted horizontal domain...")
    fits_cropped = hdul[2].data[ox:ox + nx_p, oy:oy + ny_p, :, :] # Shape: (nx_p, ny_p, 6, 401)
    
# Reverse the FITS vertical axis to match the increasing order of fits_heights (from photosphere to chromosphere)
fits_cropped = fits_cropped[..., ::-1]
print("   Reversed FITS vertical axis to match increasing heights from photosphere to chromosphere.")

# The FITS populations are in m^-3; the GNN predictions in cm^-3 (as the training targets)
fits_cropped = fits_cropped * 1e-6

print("3. Reinterpolating NLTE populations onto clipped GNN heights grid...")
# fits_heights has shape (401,), and is strictly increasing.
# fits_cropped has shape (992, 992, 6, 401).
# We interpolate along the last axis (axis=-1) to target heights zz_gnn_clipped.
# linear in log10 (a linear interpolation mixes chromospheric and coronal states at the TR crossings)
fits_interp_func = interp.interp1d(fits_heights, np.log10(np.clip(fits_cropped, 1e-30, None)), axis=-1, bounds_error=False, fill_value="extrapolate")
fits_on_gnn = 10.0 ** fits_interp_func(zz_gnn_clipped) # Shape: (nx_p, ny_p, 6, len(zz_gnn_clipped))
print(f"   Interpolated NLTE shape: {fits_on_gnn.shape}")

print("4. Aligning coordinate axes...")
# GNN predictions has shape (len(zz_gnn_clipped), 992, 992, 6) -> (z, y, x, npops).
# We transpose to (x, y, npops, z) to match fits_on_gnn.
pred_aligned = pred_clipped.transpose(2, 1, 3, 0)
print(f"   Aligned GNN shape: {pred_aligned.shape}")

print("\n5. Computing comparison statistics at each height...")

# All six levels (0 = 4s; 1, 2 = 3d metastable; 3, 4 = 4p; 5 = Ca III), in dex, next to two baselines on the same nodes:
# LTE (Saha-Boltzmann from MURaM T and n_e with the CaII_atom partition functions) and climatology (Bifrost's mean
# fraction per training level, stored in the checkpoint). Headline: nodes where the reference Ca II fraction > 1e-2.
levels_to_compare = [0, 1, 2, 3, 4, 5]
levels_labels = {0: "Ca II 4s (lvl 0)", 1: "Ca II 3d3/2 (lvl 1)", 2: "Ca II 3d5/2 (lvl 2)", 3: "Ca II 4p1/2 (lvl 3)", 4: "Ca II 4p3/2 (lvl 4)", 5: "Ca III (lvl 5)"}
level_names = ['4s', '3d3/2', '3d5/2', '4p1/2', '4p3/2', 'CaIII']
checkpoint = torch.load(meta['checkpoint'], map_location='cpu', weights_only=False)
means = np.asarray(checkpoint['normalization_params']['means'])[valid_mask]                 # (nh, 6) Bifrost mean fractions
rho = np.load(os.path.join(results_dir, meta['files']['rho']))[valid_mask]                    # kg m^-3, (nh, ny, nx)
temp = np.load(os.path.join(results_dir, meta['files']['temp']))[valid_mask]
n_e = np.load(os.path.join(results_dir, meta['files']['ne']))[valid_mask]
totals = CA_TOTAL_PER_RHO * rho                                                              # cm^-3
clim = (means[:, None, None, :] * totals[..., None]).transpose(2, 1, 3, 0)                   # (x, y, lev, h) like pred_aligned
lte = np.empty_like(clim)
for h_idx in range(len(zz_gnn_clipped)):
    lte[..., h_idx] = (lte_pops(CaII_atom(), temp[h_idx].ravel().astype(np.float64), n_e[h_idx].ravel().astype(np.float64),
                                (totals[h_idx] * 1e6).ravel().astype(np.float64)) * 1e-6).T.reshape(ny_p, nx_p, 6).transpose(1, 0, 2)
done_x = done.T                                                                              # (x, y)
f_ref = fits_on_gnn / fits_on_gnn.sum(2, keepdims=True)
relevant = (1.0 - f_ref[:, :, 5, :]) > 1e-2
bands = [(-np.inf, 0.0, 'z<0'), (0.0, 0.5, '0-0.5'), (0.5, 1.0, '0.5-1'), (1.0, 1.5, '1-1.5'), (1.5, 2.2, '1.5-2.2'), (2.2, 3.5, '2.2-3.5'), (3.5, np.inf, '>3.5')]
stats = {'results': os.path.abspath(results_dir), 'checkpoint': meta['checkpoint'], 'reference': 'SNAPI 1.5D NLTE (HDU2 of ' + os.path.basename(fits_path) + ')', 'n_columns': int(done.sum()), 'relevance_threshold': 1e-2, 'methods': {}}
for name, arr in (('gnn', pred_aligned), ('lte', lte), ('climatology', clim)):
    with np.errstate(divide='ignore', invalid='ignore'):
        dlog = np.log10(arr) - np.log10(fits_on_gnn)                                           # (x, y, lev, h)
        dtot = np.log10(arr.sum(2)) - np.log10(fits_on_gnn.sum(2))                             # (x, y, h)
    stats['methods'][name] = {}
    for lo, hi, band in bands:
        hsel = (zz_gnn_clipped >= lo) & (zz_gnn_clipped < hi)
        m = done_x[:, :, None] & relevant[:, :, hsel] & np.isfinite(dtot[:, :, hsel])
        if m.sum() == 0:
            continue
        d = {level_names[l]: {'median_abs': float(np.nanmedian(np.abs(dlog[:, :, l, hsel][m]))), 'median_signed': float(np.nanmedian(dlog[:, :, l, hsel][m])),
                              'p84_abs': float(np.nanpercentile(np.abs(dlog[:, :, l, hsel][m]), 84))} for l in levels_to_compare}
        d['all'] = {'median_abs': float(np.nanmedian(np.abs(dlog[:, :, :, hsel][m[:, :, None, :].repeat(6, 2)])))}
        d['dlog_total_median'] = float(np.nanmedian(dtot[:, :, hsel][m]))
        d['n_nodes'] = int(m.sum())
        stats['methods'][name][band] = d
with open(os.path.join(results_dir, 'compare_populations.json'), 'w') as f:
    json.dump(stats, f, indent=1)

# Clean terminal table formatting (also written as compare_populations.md)
lines = [f"# GNN vs SNAPI (1.5D NLTE) reference on MURaM ({done.sum()} columns, {os.path.basename(meta['checkpoint'])})", "",
         "Median |dlog10 n| [signed median] per level on nodes with SNAPI Ca II fraction > 1e-2; bands in Mm above tau500 = 1.", ""]
for name in stats['methods']:
    lines += [f"## {name}", "", "| band | n | " + " | ".join(level_names) + " | all | dlog total |", "|" + "---|" * (len(level_names) + 4)]
    for band, d in stats['methods'][name].items():
        lines.append(f"| {band} | {d['n_nodes']} | " + " | ".join(f"{d[l]['median_abs']:.3f} [{d[l]['median_signed']:+.2f}]" for l in level_names)
                     + f" | {d['all']['median_abs']:.3f} | {d['dlog_total_median']:+.3f} |")
    lines.append("")
open(os.path.join(results_dir, 'compare_populations.md'), 'w').write("\n".join(lines))
print("\n".join(lines))

print("6. Generating premium population comparison plots...")

plot_dir = results_dir

# Select a representative set of 5 heights distributed across zz_gnn_clipped
selected_indices = np.linspace(0, len(zz_gnn_clipped) - 1, 5, dtype=int)

for idx in selected_indices:
    h = zz_gnn_clipped[idx]
    print(f"   Plotting comparison maps at height: {h:.4f} Mm...")
    
    fig, axes = plt.subplots(len(levels_to_compare), 3, figsize=(18, 4.5 * len(levels_to_compare)), dpi=120)
    fig.suptitle(f"$z = {h:.4f}$ Mm\n(Left: GNN | Middle: SNAPI 1.5D NLTE | Right: Difference)", y=0.98)
    
    for row_idx, lvl in enumerate(levels_to_compare):
        g_slice = np.where(done_x, pred_aligned[:, :, lvl, idx], np.nan)
        f_slice = np.where(done_x, fits_on_gnn[:, :, lvl, idx], np.nan)
        
        # Logarithmic normalization bounds for plotting
        vmin = max(1e-15, np.nanpercentile(f_slice, 1))
        vmax = np.nanpercentile(f_slice, 99)
        
        # Plot GNN slice
        ax_g = axes[row_idx, 0]
        im_g = ax_g.imshow(g_slice.T, origin='lower', cmap='magma', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        ax_g.set_title(f"GNN - {levels_labels[lvl]}", fontsize=11)
        ax_g.set_xlabel("x pixel", fontsize=9)
        ax_g.set_ylabel("y pixel", fontsize=9)
        
        # Plot FITS slice
        ax_f = axes[row_idx, 1]
        im_f = ax_f.imshow(f_slice.T, origin='lower', cmap='magma', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        ax_f.set_title(f"SNAPI 1.5D NLTE - {levels_labels[lvl]}", fontsize=11)
        ax_f.set_xlabel("x pixel", fontsize=9)
        ax_f.set_ylabel("y pixel", fontsize=9)
        
        # Plot Difference slice: log10(GNN / 1.5D NLTE)
        ax_d = axes[row_idx, 2]
        diff_slice = np.log10(np.clip(g_slice, 1e-32, None)) - np.log10(np.clip(f_slice, 1e-32, None))
        
        # Find symmetric color scale based on 99th percentile of absolute differences
        vlim = max(0.1, np.nanpercentile(np.abs(diff_slice), 99))
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
