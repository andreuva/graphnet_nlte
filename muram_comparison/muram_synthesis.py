#!/usr/bin/env python3
"""
muram_synthesis.py

Synthesizes the Ca II K line (the line of the reference spectra, HDU0) on a regular sample of MURaM columns with
the SNAPI (1.5D NLTE, HDU2), GNN-predicted, LTE and Bifrost-climatology Ca II populations imposed on the
native 401-level atmosphere (MURaM n_e and n_H, v_turb = 0, mu = 1, CRD), and compares the Lightweaver synthesis of the
SNAPI populations with SNAPI's own spectra (HDU0): a cross-code check, not a reproduction. The reference wavelength axis (HDU1) is in air and is converted to vacuum.
"""

import numpy as np
import scipy.interpolate as interp
import json
import torch
import lightweaver.constants as const
from lightweaver.atomic_set import lte_pops
from lightweaver.atomic_table import DefaultAtomicAbundance
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from astropy.io import fits
import os
import sys
import time
import glob
import multiprocessing
from joblib import Parallel, delayed

# Import Lightweaver and its Fal/RH library
import lightweaver as lw
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, C_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, \
                                 Fe_atom, He_9_atom, MgII_atom, N_atom, Na_atom, S_atom

# Add current folder to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
import muram as mio
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
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# ==============================================================================
# CONFIGURABLE SETTINGS
# ==============================================================================
# The muram_inference.py run to synthesize (default: the most recent) and the sample: n_side x n_side columns on a
# regular grid over the predicted crop.   python muram_synthesis.py [results/muram_inference/<stamp>] [n_side]
results_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(current_dir, '..', 'results', 'muram_inference', '*', 'meta.json')))[-1][:-len('meta.json')]
n_side = int(sys.argv[2]) if len(sys.argv) > 2 else 8
meta = json.load(open(os.path.join(results_dir, 'meta.json')))

# Number of CPU cores to use
n_jobs = 32

pathsource = '/dat/milic/MURaM_enhanced_network/'
snap_number = 499000
fits_path = "/dat/milic/MURaM_enhanced_network/che_full_499000_lwsynth_200.0.fits"
pred_path = os.path.join(results_dir, meta['files']['predictions'])

# ==============================================================================
# 1. SETUP GRIDS AND VERTICAL COORDINATES
# ==============================================================================
print("1. Constructing grids and setting up vertical coordinates...")

# Load MURaM snap; the vertical coordinate and the alignment come from the inference run (grid.npz, meta.json)
cube = mio.MuramSnap(pathsource, snap_number)

nz, nx, ny = cube.Temp.shape
dz, dx, dy = (np.asarray(cube.Temp.dX) / 1e5) / 1e3
grid = np.load(os.path.join(results_dir, meta['files']['grid']))
zz_grid_muram = grid['z_rel_tau1_Mm']                                   # heights of the 50 GNN levels above tau500 = 1
tau_zero_index = meta['tau1_index_median']
muram_z_grid = (np.arange(nz) - tau_zero_index) * dz
print(f"   tau=1 layer index: {tau_zero_index:.2f}")

# 1.5D NLTE z grid is cropped from 15 to 416 (not including 416); the synthesis uses these native levels directly
nat = np.arange(mc.FITS_Z_SLICE.start, mc.FITS_Z_SLICE.stop)
fits_heights = muram_z_grid[nat]
print(f"   Using the {len(nat)} native levels of the reference ({fits_heights.min():.3f} to {fits_heights.max():.3f} Mm) for synthesis.")

# ==============================================================================
# 2. INTERPOLATE ATMOSPHERIC VARIABLES
# ==============================================================================
pred = np.load(pred_path)                                                  # (n_levels, ny_p, nx_p, 6) cm^-3, NaN = not predicted
done = np.isfinite(pred[0, :, :, 0])
oy, ox = meta['origin']                                                     # native column of pred[:, 0, 0]
ny_p, nx_p = pred.shape[1:3]
sample = [(j, i) for j in np.linspace(4, ny_p - 5, n_side).astype(int) for i in np.linspace(4, nx_p - 5, n_side).astype(int) if done[j, i]]
print(f"2. Reading the physical variables of {len(sample)} sample columns at native resolution...")

def native_column(j, i):
    xi, yj = ox + i, oy + j                                                 # cube is (z, x, y); prediction axes are (y, x)
    c = lambda a, f: np.ascontiguousarray(np.asarray(a[nat, xi, yj], dtype=np.float64)[::-1] * f)   # top first
    return c(cube.Temp, 1.0), c(cube.vx, 1e-2), c(cube.ne, 1e6), c(cube.rho, 1e3)    # K, m/s (positive up), m^-3, kg m^-3

columns = [native_column(j, i) for j, i in sample]

# Lightweaver expects depthScale in meters, strictly decreasing
depthScale_m = np.ascontiguousarray(fits_heights[::-1] * 1e6)

# ==============================================================================
# 3. LOAD LEVEL POPULATIONS FOR THE PATCH
# ==============================================================================
print("3. Loading populations for the patch...")

# GNN populations: log fractions interpolated from the grid levels to the native levels, times the total Ca density
# of each native column (CA_TOTAL_PER_RHO * rho); LTE (Saha-Boltzmann) and Bifrost climatology on the same levels
checkpoint = torch.load(meta['checkpoint'], map_location='cpu', weights_only=False)
means = np.asarray(checkpoint['normalization_params']['means'])          # Bifrost mean fraction per training level
# the run's levels may differ from the training grid (muram_inference.py [grid]): climatology interpolated in height (FITS frame, log)
z_train = np.asarray(checkpoint.get('meta', {}).get('z_grid_Mm', grid['z_bifrost_frame_Mm']))
means = 10 ** np.stack([np.interp(grid['z_bifrost_frame_Mm'], z_train, np.log10(means[:, l])) for l in range(means.shape[1])], axis=-1)

def pops_on_native(j, i, T_top, ne_top, rho_top):
    tot_cm3 = CA_TOTAL_PER_RHO * rho_top[::-1]                                        # bottom first, cm^-3
    f_gnn = pred[:, j, i, :] / pred[:, j, i, :].sum(-1, keepdims=True)
    lf = np.stack([np.interp(fits_heights, zz_grid_muram, np.log10(f_gnn[:, l])) for l in range(6)], axis=-1)
    gnn = np.ascontiguousarray((10 ** lf * tot_cm3[:, None])[::-1].T * 1e6)         # (6, nz) top first, m^-3
    lfc = np.stack([np.interp(fits_heights, zz_grid_muram, np.log10(means[:, l])) for l in range(6)], axis=-1)
    clim = np.ascontiguousarray((10 ** lfc * tot_cm3[:, None])[::-1].T * 1e6)
    lte = np.ascontiguousarray(lte_pops(CaII_atom(), T_top, ne_top, tot_cm3[::-1] * 1e6))
    return gnn, lte, clim

# SNAPI (1.5D NLTE FITS) populations and spectra of the sample columns; FITS[a, b] = MURaM (x=a, y=b)
print(f"   Memory-mapping FITS file at {fits_path}...")
with fits.open(fits_path, memmap=True) as hdul:
    wave_air = np.asarray(hdul[1].data, dtype=np.float64)
    snapi_pops_si = np.array([hdul[2].data[ox + i, oy + j, :, :] for j, i in sample], dtype=np.float64)   # (ncol, 6, 401) top first, m^-3
    I_hdu0 = np.array([hdul[0].data[ox + i, oy + j, 0, :] for j, i in sample], dtype=np.float64)          # (ncol, nwave)
pop_sets = [[snapi_pops_si[c], *pops_on_native(j, i, columns[c][0], columns[c][2], columns[c][3])] for c, (j, i) in enumerate(sample)]
set_names = ['snapi', 'gnn', 'lte', 'climatology']

# ==============================================================================
# 4. PARALLEL SYNTHESIS
# ==============================================================================
# Setup wavelengths: those of the reference spectra (HDU1, in AIR -> vacuum for Lightweaver), Ca II K window + continuum points
wave = lw.air_to_vac(wave_air)
nwave = len(wave)
lambda_K = 393.4777                                       # Ca II K rest wavelength of CaII_atom (nm, vacuum)

def process_single_column(col_idx, temp_atm, vel_atm, ne_atm, rho_atm, pops_list, depthScale_m, wave):
    
    out = np.full((len(pops_list), len(wave)), np.nan)      # NaN = failed
    try:
        atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, 
                                      depthScale=np.ascontiguousarray(depthScale_m),
                                      convertScales=False,
                                      temperature=temp_atm, 
                                      ne=ne_atm,
                                      nHTot=rho_atm / (const.Amu * DefaultAtomicAbundance.massPerH),
                                      vturb=np.zeros_like(temp_atm),
                                      vlos=vel_atm)
        atmos.quadrature(5)
        aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                                Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
        aSet.set_active('Ca')
        spect = aSet.compute_wavelength_grid()
        eqPops = aSet.compute_eq_pops(atmos)

        for k, pops_atm in enumerate(pops_list):             # reference, GNN, LTE, climatology populations imposed (6, nz) m^-3
            eqPops.atomicPops['Ca'].n = np.ascontiguousarray(pops_atm)
            ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
            out[k] = ctx.compute_rays(wave, [1.0], stokes=False)
    except Exception as e:
        print(f"Error in column {col_idx}: {e}")

    return col_idx, out

print(f"4. Starting parallel synthesis of {len(sample)} columns using {n_jobs} cores...")
start_time = time.time()

results = Parallel(n_jobs=n_jobs, verbose=5)(
    delayed(process_single_column)(c, *columns[c], pop_sets[c], depthScale_m, wave) for c in range(len(sample))
)

print(f"   Calculation finished in {(time.time() - start_time):.2f} seconds.")

# Reassemble outputs: (ncol, nwave) per population set
Iwave = {name: np.full((len(sample), nwave), np.nan) for name in set_names}
for c_idx, out in results:
    for k, name in enumerate(set_names):
        Iwave[name][c_idx] = out[k]
Iwave_gnn = Iwave['gnn']
Iwave_snapi = Iwave['snapi']

# ==============================================================================
# 5. SAVE OUTPUT DATA
# ==============================================================================
print("5. Saving outputs...")
np.savez_compressed(os.path.join(results_dir, 'muram_synthesis_K.npz'), wave_vac=wave, wave_air=wave_air, columns=np.array(sample), I_hdu0=I_hdu0, **Iwave)
print(f"   Saved synthesized spectra to: {os.path.join(results_dir, 'muram_synthesis_K.npz')}")

# Metrics: (a) reference populations vs HDU0 (how far a 1D CRD synthesis is from the reference's own spectra);
# (b) GNN, LTE and climatology populations vs the reference populations on the same atmosphere (population error only)
core = np.abs(wave - lambda_K) <= 0.05
kwin = (wave > 392.9) & (wave < 394.9)
wing = kwin & (np.abs(wave - lambda_K) >= 0.5)
i0 = int(np.argmin(np.abs(wave - lambda_K)))
def spec_stats(I, Iref):
    good = np.isfinite(I[:, 0]) & np.isfinite(Iref[:, 0])
    rel = (I[good] - Iref[good]) / Iref[good]
    return {'n': int(good.sum()), 'continuum_median_abs_pct': float(np.median(np.abs(rel[:, ~kwin])) * 100), 'wing_median_abs_pct': float(np.median(np.abs(rel[:, wing])) * 100),
            'core_mape_pct': float(np.mean(np.abs(rel[:, core])) * 100), 'centre_median_abs_pct': float(np.median(np.abs(rel[:, i0])) * 100), 'centre_bias_pct': float(np.median(rel[:, i0]) * 100)}
synth_metrics = {'results': os.path.abspath(results_dir), 'n_columns': len(sample), 'setup': 'native 401 levels, MURaM n_e and nHTot = rho / (amu massPerH), vturb 0, mu 1, CRD, Ca active; HDU1 air -> vacuum',
                 'reference': 'SNAPI 1.5D NLTE populations (HDU2) and spectra (HDU0)',
                 'reference_pops_vs_HDU0': spec_stats(Iwave_snapi, I_hdu0),
                 **{f'{name}_vs_reference_pops': spec_stats(Iwave[name], Iwave_snapi) for name in ('gnn', 'lte', 'climatology')}}
with open(os.path.join(results_dir, 'muram_synthesis_K.json'), 'w') as f:
    json.dump(synth_metrics, f, indent=1)
print(json.dumps(synth_metrics, indent=1))

# ==============================================================================
# 6. PLOT COMPARATIVE RESULTS
# ==============================================================================
print("6. Analyzing and generating premium visualization plots...")

# Spatially averaged profiles (over the sample columns), K window only
mean_I_gnn = np.nanmean(Iwave_gnn, axis=0)
mean_I_snapi = np.nanmean(Iwave_snapi, axis=0)
mean_I_hdu0 = np.nanmean(I_hdu0, axis=0)
mean_I_lte = np.nanmean(Iwave['lte'], axis=0)
mean_I_clim = np.nanmean(Iwave['climatology'], axis=0)

continuum_idx = int(np.argmin(np.abs(wave - 394.8)))      # last point of the K window
line_core_idx = i0

print(f"   Continuum wavelength: {wave[continuum_idx]:.4f} nm (Index {continuum_idx})")
print(f"   Line core wavelength: {wave[line_core_idx]:.4f} nm (Index {line_core_idx})")

# Figure 1: Spatially Averaged Spectra comparison
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(wave[kwin], mean_I_hdu0[kwin], color='gray', linewidth=2.5, alpha=0.6, label='SNAPI spectra (HDU0)')
plt.plot(wave[kwin], mean_I_snapi[kwin], 'k-', linewidth=2, label='SNAPI populations (1.5D NLTE), Lightweaver CRD synthesis')
plt.plot(wave[kwin], mean_I_gnn[kwin], 'r--', linewidth=1.8, label='GNN populations')
plt.plot(wave[kwin], mean_I_lte[kwin], ':', color='gray', linewidth=1.5, label='LTE populations')
plt.plot(wave[kwin], mean_I_clim[kwin], '-.', color='royalblue', linewidth=1.5, label='Bifrost climatology populations')
plt.xlabel('Wavelength [nm] (vacuum)', fontsize=13)
plt.ylabel('Intensity [W / (m$^2$ Hz sr)]', fontsize=13)
plt.title('Mean Ca II K profile over the sample columns', fontsize=14, pad=15)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
spectra_plot_path = os.path.join(results_dir, 'muram_synthesis_K_spectra.png')
plt.savefig(spectra_plot_path, bbox_inches='tight')
plt.close()
print(f"   Saved average spectra plot to: {spectra_plot_path}")

# Figure 2: Side-by-side 2D Maps (Continuum and Line Core)
fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)

epsilon = 1e-32
n_s = int(np.sqrt(len(sample)))                           # the sample is a regular n_side x n_side grid
as_map = lambda v: np.full(n_side * n_side, np.nan) if len(v) != n_side * n_side else v.reshape(n_side, n_side)
cont_snapi = as_map(Iwave_snapi[:, continuum_idx])
cont_gnn = as_map(Iwave_gnn[:, continuum_idx])
cont_diff = (cont_gnn - cont_snapi) / (cont_snapi + epsilon) * 100 # Relative diff (%)

core_snapi = as_map(Iwave_snapi[:, line_core_idx])
core_gnn = as_map(Iwave_gnn[:, line_core_idx])
core_diff = (core_gnn - core_snapi) / (core_snapi + epsilon) * 100 # Relative diff (%)

# Continuum Row (Row 0)
cont_vmin = np.nanmin([cont_snapi, cont_gnn])
cont_vmax = np.nanmax([cont_snapi, cont_gnn])

im0 = axes[0, 0].imshow(cont_snapi.T, origin='lower', cmap='magma', vmin=cont_vmin, vmax=cont_vmax)
axes[0, 0].set_title(f'SNAPI pops - wing ({wave[continuum_idx]:.2f} nm)', fontsize=12)
axes[0, 0].set_ylabel('y-pixel', fontsize=11)
axes[0, 0].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im0, ax=axes[0, 0], shrink=0.8, label='Intensity [W/(m$^2$ Hz sr)]')

im1 = axes[0, 1].imshow(cont_gnn.T, origin='lower', cmap='magma', vmin=cont_vmin, vmax=cont_vmax)
axes[0, 1].set_title(f'GNN pops - wing ({wave[continuum_idx]:.2f} nm)', fontsize=12)
axes[0, 1].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im1, ax=axes[0, 1], shrink=0.8, label='Intensity [W/(m$^2$ Hz sr)]')

cont_lim = max(1.0, np.nanpercentile(np.abs(cont_diff), 99))
im2 = axes[0, 2].imshow(cont_diff.T, origin='lower', cmap='RdBu_r', vmin=-cont_lim, vmax=cont_lim)
axes[0, 2].set_title('Relative Difference (%)', fontsize=12)
axes[0, 2].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im2, ax=axes[0, 2], shrink=0.8, label='(GNN - SNAPI) / SNAPI [%]')

# Line Core Row (Row 1)
core_vmin = np.nanmin([core_snapi, core_gnn])
core_vmax = np.nanmax([core_snapi, core_gnn])

im3 = axes[1, 0].imshow(core_snapi.T, origin='lower', cmap='magma', vmin=core_vmin, vmax=core_vmax)
axes[1, 0].set_title(f'SNAPI pops - K core ({wave[line_core_idx]:.2f} nm)', fontsize=12)
axes[1, 0].set_ylabel('y-pixel', fontsize=11)
axes[1, 0].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im3, ax=axes[1, 0], shrink=0.8, label='Intensity [W/(m$^2$ Hz sr)]')

im4 = axes[1, 1].imshow(core_gnn.T, origin='lower', cmap='magma', vmin=core_vmin, vmax=core_vmax)
axes[1, 1].set_title(f'GNN pops - K core ({wave[line_core_idx]:.2f} nm)', fontsize=12)
axes[1, 1].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im4, ax=axes[1, 1], shrink=0.8, label='Intensity [W/(m$^2$ Hz sr)]')

core_lim = max(1.0, np.nanpercentile(np.abs(core_diff), 99))
im5 = axes[1, 2].imshow(core_diff.T, origin='lower', cmap='RdBu_r', vmin=-core_lim, vmax=core_lim)
axes[1, 2].set_title('Relative Difference (%)', fontsize=12)
axes[1, 2].set_xlabel('x-pixel', fontsize=11)
fig.colorbar(im5, ax=axes[1, 2], shrink=0.8, label='(GNN - SNAPI) / SNAPI [%]')

fig.suptitle(f'Ca II K on the {n_side}x{n_side} sample columns: SNAPI vs GNN populations', fontsize=16, y=0.96)
plt.tight_layout(rect=[0, 0, 1, 0.94])
maps_plot_path = os.path.join(results_dir, 'muram_synthesis_K_maps.png')
plt.savefig(maps_plot_path, bbox_inches='tight')
plt.close()
print(f"   Saved comparative maps plot to: {maps_plot_path}")

print("\nSynthesis complete and premium figures generated successfully!")
