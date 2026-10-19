#!/usr/bin/env python3
"""
muram_synthesis.py

Synthesizes the Ca II 8542 Å spectral line from both SNAPI (1.5D NLTE) and GNN-predicted
Ca II populations in the MURaM dataset.
The atmosphere (MURaM T, n_e, n_H from rho and the vertical velocity, vturb = 0) and both population sets are taken
on the GNN levels inside the height range of SNAPI; only Ca is active and the intensity is computed at mu = 1.

    python muram_synthesis.py <muram_inference output folder> [patch_size]

Author: Antigravity AI (Google DeepMind)
"""

import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from astropy.io import fits
import os
import sys
import time
import multiprocessing
from joblib import Parallel, delayed

# Import Lightweaver and its Fal/RH library
import lightweaver as lw
import lightweaver.constants as const
from lightweaver.atomic_table import DefaultAtomicAbundance
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, C_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, \
                                 Fe_atom, He_9_atom, MgII_atom, N_atom, Na_atom, S_atom

# Add current folder to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
import muram as mio
from vertical_grids import to_levels

# Configure matplotlib for premium, publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# ==============================================================================
# CONFIGURABLE SETTINGS
# ==============================================================================
# The muram_inference.py output folder: its grid (muram_grid.npz) and predictions; the outputs are written there too
results_dir = sys.argv[1]
muram_grid = np.load(os.path.join(results_dir, 'muram_grid.npz'))
pred_path = os.path.join(results_dir, f"muram_predictions_stride_{int(muram_grid['stride'])}_full.npy")
ny_p, nx_p = np.load(pred_path, mmap_mode='r').shape[1:3]

# We define a 32x32 patch (2nd argument) from the center of the predicted grid.
patch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
assert patch_size <= min(ny_p, nx_p), f'patch of {patch_size} px larger than the predicted crop ({ny_p}, {nx_p})'
py_start, py_end = (ny_p - patch_size) // 2, (ny_p - patch_size) // 2 + patch_size
px_start, px_end = (nx_p - patch_size) // 2, (nx_p - patch_size) // 2 + patch_size

# Number of CPU cores to use (safely utilizing a portion of the 192 cores)
n_jobs = 128

pathsource = '/dat/milic/MURaM_enhanced_network/'
snap_number = 499000
fits_path = "/dat/milic/MURaM_enhanced_network/che_full_499000_lwsynth_200.0.fits"

# ==============================================================================
# 1. SETUP GRIDS AND VERTICAL COORDINATES
# ==============================================================================
print("1. Constructing grids and setting up vertical coordinates...")

# Load MURaM snap (the atmosphere of the patch)
cube = mio.MuramSnap(pathsource, snap_number)

# The GNN grid (heights above MURaM's tau500 = 1 [Mm], aligned with Bifrost in muram_inference.py), the heights of all
# the MURaM levels and the MURaM column (y, x) of the first predicted column
zz_grid_muram = muram_grid['z']
muram_z_grid = muram_grid['z_native']
oy, ox = muram_grid['origin']

# 1.5D NLTE z grid is cropped from 15 to 416 (not including 416)
fits_heights = muram_z_grid[15:416]

# Clip the range of the GNN grid to remove parts where NLTE doesn't reach
valid_mask = (zz_grid_muram >= fits_heights.min()) & (zz_grid_muram <= fits_heights.max())
zz_gnn_clipped = zz_grid_muram[valid_mask]

print(f"   Clipped GNN vertical heights (Mm): min={zz_gnn_clipped.min():.4f}, max={zz_gnn_clipped.max():.4f}")
print(f"   Using {len(zz_gnn_clipped)} vertical layers for synthesis.")

# ==============================================================================
# 2. INTERPOLATE ATMOSPHERIC VARIABLES
# ==============================================================================
print(f"2. Interpolating physical variables for the selected patch [{py_start}:{py_end}, {px_start}:{px_end}]...")

# GNN and FITS horizontal domain corresponds to original coordinates shifted by the crop origin (oy, ox); the original
# MURaM cube shape is (nz, nx, ny). Interpolated to the clipped GNN heights, in log10 for the scalars
def patch_on_grid(q, log):
    return to_levels(q[:, ox + px_start:ox + px_end, oy + py_start:oy + py_end], muram_z_grid, zz_gnn_clipped, log=log).transpose(0, 2, 1)

temp_patch = patch_on_grid(cube.Temp, True)  # (z, y, x)
v_z_patch = patch_on_grid(cube.vx, False)    # (z, y, x)
ne_patch = patch_on_grid(cube.ne, True)      # (z, y, x)
rho_patch = patch_on_grid(cube.rho, True)    # (z, y, x)

# Convert units to SI
# Temperature is already in K
# Velocity: cm/s -> m/s
vlos_m_s = v_z_patch / 100.0
# Electron density: cm^-3 -> m^-3
ne_m3 = ne_patch * 1e6
# Total hydrogen density: density (g/cm^3 -> kg/m^3) over the mass per hydrogen atom
nHTot_m3 = rho_patch * 1e3 / (const.Amu * DefaultAtomicAbundance.massPerH)

# Reverse physical variable vertical axis to go from top to bottom (decreasing height)
temp_patch = temp_patch[::-1, :, :]
vlos_m_s = vlos_m_s[::-1, :, :]
ne_m3 = ne_m3[::-1, :, :]
nHTot_m3 = nHTot_m3[::-1, :, :]

# Lightweaver expects depthScale in meters, strictly decreasing
depthScale_m = zz_gnn_clipped[::-1] * 1e6

# ==============================================================================
# 3. LOAD LEVEL POPULATIONS FOR THE PATCH
# ==============================================================================
print("3. Loading populations for the patch...")

# GNN populations (cm^-3 initially, we scale to m^-3 by multiplying by 1e6)
if not os.path.exists(pred_path):
    print(f"Error: GNN predictions not found at {pred_path}")
    sys.exit(1)
print(f"   Loading GNN predictions from {pred_path}...")
pred = np.load(pred_path)
pred_patch_cgs = pred[valid_mask, py_start:py_end, px_start:px_end, :] # (z, y, x, 6)
gnn_pops_si = pred_patch_cgs * 1e6 # convert CGS to SI (m^-3)

# Reverse GNN populations along vertical coordinate to match decreasing heights
gnn_pops_si = gnn_pops_si[::-1, :, :, :]

# SNAPI (1.5D NLTE FITS) populations
print(f"   Memory-mapping FITS file at {fits_path}...")
if not os.path.exists(fits_path):
    print(f"Error: FITS file not found at {fits_path}")
    sys.exit(1)

with fits.open(fits_path, memmap=True) as hdul:
    # FITS[a, b] = MURaM (x=a, y=b)
    fits_patch = hdul[2].data[ox + px_start : ox + px_end, oy + py_start : oy + py_end, :, :] # (patch_size, patch_size, 6, 401)

# Reverse FITS vertical axis to match increasing heights
fits_patch = fits_patch[..., ::-1]

# Reinterpolate fits_patch onto zz_gnn_clipped, linearly in log10
fits_interp_func = interp.interp1d(fits_heights, np.log10(np.clip(fits_patch, 1e-30, None)), axis=-1, bounds_error=False, fill_value="extrapolate")
fits_patch_on_gnn = 10.0 ** fits_interp_func(zz_gnn_clipped) # Shape: (patch_size, patch_size, 6, len(zz_gnn_clipped))

# Transpose to (z, y, x, level) to match GNN patch shape
snapi_pops_si = fits_patch_on_gnn.transpose(3, 1, 0, 2) # (z, y, x, 6)

# Reverse SNAPI populations along vertical coordinate to match decreasing heights
snapi_pops_si = snapi_pops_si[::-1, :, :, :]

# ==============================================================================
# 4. PARALLEL SYNTHESIS
# ==============================================================================
# Setup wavelengths (wavelengths in nm, around 854.2 nm)
nwave = 1001
wave = np.linspace(853.9444, 854.9444, nwave)

def process_single_row(row_idx, patch_size, 
                       temp_patch,
                       vlos_m_s, 
                       ne_m3,
                       nHTot_m3,
                       gnn_pops_si, 
                       snapi_pops_si,
                       depthScale_m, wave):
    
    row_gnn = np.zeros((patch_size, len(wave)))
    row_snapi = np.zeros((patch_size, len(wave)))
    
    for col_pix in range(patch_size):
        # Extract 1D profiles for this column
        temp_atm = np.ascontiguousarray(temp_patch[:, col_pix, row_idx])
        vel_atm = np.ascontiguousarray(vlos_m_s[:, col_pix, row_idx])
        ne_atm = np.ascontiguousarray(ne_m3[:, col_pix, row_idx])
        nHTot_atm = np.ascontiguousarray(nHTot_m3[:, col_pix, row_idx])
        
        gnn_atm = np.ascontiguousarray(gnn_pops_si[:, col_pix, row_idx, :])
        snapi_atm = np.ascontiguousarray(snapi_pops_si[:, col_pix, row_idx, :])

        # 1. GNN Synthesis
        try:
            atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, 
                                          depthScale=np.ascontiguousarray(depthScale_m),
                                          convertScales=False,
                                          temperature=temp_atm, 
                                          ne=ne_atm,
                                          nHTot=nHTot_atm,
                                          vturb=np.zeros_like(temp_atm),
                                          vlos=vel_atm)
            atmos.quadrature(5)
            aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                                    Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
            aSet.set_active('Ca')
            spect = aSet.compute_wavelength_grid()
            eqPops = aSet.compute_eq_pops(atmos)

            eqPops.atomicPops['Ca'].n = np.ascontiguousarray(np.moveaxis(gnn_atm, 1, 0))
            ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
            row_gnn[col_pix, :] = ctx.compute_rays(wave, [1.0], stokes=False)
        except Exception as e:
            print(f"Error in pixel (GNN) ({row_idx}, {col_pix}): {e}")
            row_gnn[col_pix, :] = np.nan    # NaN = failed

        # 2. SNAPI Synthesis
        try:
            atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, 
                                          depthScale=np.ascontiguousarray(depthScale_m),
                                          convertScales=False,
                                          temperature=temp_atm, 
                                          ne=ne_atm,
                                          nHTot=nHTot_atm,
                                          vturb=np.zeros_like(temp_atm),
                                          vlos=vel_atm)
            atmos.quadrature(5)
            aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                                    Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
            aSet.set_active('Ca')
            spect = aSet.compute_wavelength_grid()
            eqPops = aSet.compute_eq_pops(atmos)

            eqPops.atomicPops['Ca'].n = np.ascontiguousarray(np.moveaxis(snapi_atm, 1, 0))
            ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
            row_snapi[col_pix, :] = ctx.compute_rays(wave, [1.0], stokes=False)
        except Exception as e:
            print(f"Error in pixel (SNAPI) ({row_idx}, {col_pix}): {e}")
            row_snapi[col_pix, :] = np.nan

    return row_idx, row_gnn, row_snapi

print(f"4. Starting parallel synthesis of patch ({patch_size}x{patch_size}) using {n_jobs} cores...")
start_time = time.time()

results = Parallel(n_jobs=n_jobs, verbose=5)(
    delayed(process_single_row)(
        row, patch_size,
        temp_patch, vlos_m_s, ne_m3, nHTot_m3,
        gnn_pops_si, snapi_pops_si,
        depthScale_m, wave
    ) for row in range(patch_size)
)

print(f"   Calculation finished in {(time.time() - start_time):.2f} seconds.")

# Reassemble outputs
Iwave_gnn = np.zeros((patch_size, patch_size, nwave))
Iwave_snapi = np.zeros((patch_size, patch_size, nwave))

for res in results:
    r_idx, r_gnn, r_snapi = res
    Iwave_gnn[r_idx, :, :] = r_gnn
    Iwave_snapi[r_idx, :, :] = r_snapi

# ==============================================================================
# 5. SAVE OUTPUT DATA
# ==============================================================================
print("5. Saving outputs...")
np.save(os.path.join(results_dir, 'muram_Iwave_gnn.npy'), Iwave_gnn)
np.save(os.path.join(results_dir, 'muram_Iwave_snapi.npy'), Iwave_snapi)
print(f"   Saved synthesized GNN spectra to: {os.path.join(results_dir, 'muram_Iwave_gnn.npy')}")
print(f"   Saved synthesized SNAPI spectra to: {os.path.join(results_dir, 'muram_Iwave_snapi.npy')}")

# ==============================================================================
# 6. PLOT COMPARATIVE RESULTS
# ==============================================================================
print("6. Analyzing and generating premium visualization plots...")

# Spatially averaged profiles (failed pixels are NaN)
mean_I_gnn = np.nanmean(Iwave_gnn, axis=(0, 1))
mean_I_snapi = np.nanmean(Iwave_snapi, axis=(0, 1))

continuum_idx = 0
line_core_idx = np.argmin(mean_I_snapi)

print(f"   Continuum wavelength: {wave[continuum_idx]:.4f} nm (Index {continuum_idx})")
print(f"   Line core wavelength: {wave[line_core_idx]:.4f} nm (Index {line_core_idx})")

# Figure 1: Spatially Averaged Spectra comparison
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(wave, mean_I_snapi, 'k-', linewidth=2, label='SNAPI (1.5D NLTE)')
plt.plot(wave, mean_I_gnn, 'r--', linewidth=1.8, label='GNN Predicted')
plt.xlabel('Wavelength [nm]', fontsize=13)
plt.ylabel('Intensity [W / (m$^2$ Hz sr)]', fontsize=13)
plt.title('Spatially Averaged Ca II 8542 Å Line Profile', fontsize=14, pad=15)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
spectra_plot_path = os.path.join(results_dir, 'muram_synthesis_spectra.png')
plt.savefig(spectra_plot_path, bbox_inches='tight')
plt.close()
print(f"   Saved average spectra plot to: {spectra_plot_path}")

# Figure 2: Side-by-side 2D Maps (Continuum and Line Core)
fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)

epsilon = 1e-32
cont_snapi = Iwave_snapi[:, :, continuum_idx]
cont_gnn = Iwave_gnn[:, :, continuum_idx]
cont_diff = (cont_gnn - cont_snapi) / (cont_snapi + epsilon) * 100 # Relative diff (%)

core_snapi = Iwave_snapi[:, :, line_core_idx]
core_gnn = Iwave_gnn[:, :, line_core_idx]
core_diff = (core_gnn - core_snapi) / (core_snapi + epsilon) * 100 # Relative diff (%)

# Continuum Row (Row 0)
cont_vmin = min(np.nanmin(cont_snapi), np.nanmin(cont_gnn))
cont_vmax = max(np.nanmax(cont_snapi), np.nanmax(cont_gnn))

im0 = axes[0, 0].imshow(cont_snapi.T, origin='lower', cmap='magma', vmin=cont_vmin, vmax=cont_vmax)
axes[0, 0].set_title('SNAPI - Continuum (853.94 nm)', fontsize=12)
axes[0, 0].set_ylabel('y-pixel', fontsize=11)
axes[0, 0].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im0, ax=axes[0, 0], shrink=0.8, label='Intensity [W/(m$^2$ Hz sr)]')

im1 = axes[0, 1].imshow(cont_gnn.T, origin='lower', cmap='magma', vmin=cont_vmin, vmax=cont_vmax)
axes[0, 1].set_title('GNN - Continuum (853.94 nm)', fontsize=12)
axes[0, 1].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im1, ax=axes[0, 1], shrink=0.8, label='Intensity [W/(m$^2$ Hz sr)]')

cont_lim = max(1.0, np.nanpercentile(np.abs(cont_diff), 99))
im2 = axes[0, 2].imshow(cont_diff.T, origin='lower', cmap='RdBu_r', vmin=-cont_lim, vmax=cont_lim)
axes[0, 2].set_title('Relative Difference (%)', fontsize=12)
axes[0, 2].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im2, ax=axes[0, 2], shrink=0.8, label='(GNN - SNAPI) / SNAPI [%]')

# Line Core Row (Row 1)
core_vmin = min(np.nanmin(core_snapi), np.nanmin(core_gnn))
core_vmax = max(np.nanmax(core_snapi), np.nanmax(core_gnn))

im3 = axes[1, 0].imshow(core_snapi.T, origin='lower', cmap='magma', vmin=core_vmin, vmax=core_vmax)
axes[1, 0].set_title(f'SNAPI - Line Core ({wave[line_core_idx]:.2f} nm)', fontsize=12)
axes[1, 0].set_ylabel('y-pixel', fontsize=11)
axes[1, 0].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im3, ax=axes[1, 0], shrink=0.8, label='Intensity [W/(m$^2$ Hz sr)]')

im4 = axes[1, 1].imshow(core_gnn.T, origin='lower', cmap='magma', vmin=core_vmin, vmax=core_vmax)
axes[1, 1].set_title(f'GNN - Line Core ({wave[line_core_idx]:.2f} nm)', fontsize=12)
axes[1, 1].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im4, ax=axes[1, 1], shrink=0.8, label='Intensity [W/(m$^2$ Hz sr)]')

core_lim = max(1.0, np.nanpercentile(np.abs(core_diff), 99))
im5 = axes[1, 2].imshow(core_diff.T, origin='lower', cmap='RdBu_r', vmin=-core_lim, vmax=core_lim)
axes[1, 2].set_title('Relative Difference (%)', fontsize=12)
axes[1, 2].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im5, ax=axes[1, 2], shrink=0.8, label='(GNN - SNAPI) / SNAPI [%]')

fig.suptitle('MURaM Synthesized Intensity Maps Comparison', fontsize=16, y=0.96)
plt.tight_layout(rect=[0, 0, 1, 0.94])
maps_plot_path = os.path.join(results_dir, 'muram_synthesis_maps.png')
plt.savefig(maps_plot_path, bbox_inches='tight')
plt.close()
print(f"   Saved comparative maps plot to: {maps_plot_path}")

print("\nSynthesis complete and premium figures generated successfully!")
