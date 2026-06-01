#!/usr/bin/env python3
# compare_height_stratification.py

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from astropy.io import fits
import muram as mio

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

def main():
    print("==================================================================")
    
    # Hardcode Bifrost grid dimensions directly to avoid slow PyTorch and CUDA driver load times
    nx_b, ny_b, nz_b = 504, 504, 425

    print(f"Bifrost parameters: nz_b={nz_b}, ny_b={ny_b}, nx_b={nx_b}")

    # ===================================================================
    # 1. LOAD AND PROCESS BIFROST DATASET (ORIGINAL COORDINATES)
    # ===================================================================
    print("\n--- Loading Bifrost Original Dataset ---")
    datadir_bifrost = '/dat/andreuva/gpu/graphnet/data_train'
    bifrost_grid_file = '/dat/andreuva/gpu/graphnet/en024048_hion/grid_bifrost.npz'
    
    if not os.path.exists(bifrost_grid_file):
        raise FileNotFoundError(f"Bifrost grid file not found at {bifrost_grid_file}")
    
    bifrost_z_grid = np.load(bifrost_grid_file)["z"] # Shape (nz_b,)
    
    # Memory mapped arrays (original grid, no interpolation)
    print("Memory-mapping original Bifrost data cubes...")
    temp_b = np.memmap(f'{datadir_bifrost}/AR_385_temp.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 1))
    b_xyz_b = np.memmap(f'{datadir_bifrost}/AR_385_B.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 3))
    vel_b = np.memmap(f'{datadir_bifrost}/AR_385_veloc.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 3))
    ne_b = np.memmap(f'{datadir_bifrost}/AR_385_ne.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 1))
    rho_b = np.memmap(f'{datadir_bifrost}/AR_385_mass.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 1))
    press_b = np.memmap(f'{datadir_bifrost}/AR_385_press.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 1))

    # Calculating Bifrost horizontal averages (mean along x and y, which are axes 1 and 2)
    # Keep in mind: Bifrost layout is (nz_b, ny_b, nx_b, C). Average axis=(1, 2)
    print("Computing Bifrost mean height stratifications...")
    mean_T_bifrost = np.mean(temp_b[..., 0], axis=(1, 2))
    mean_ne_bifrost = np.mean(ne_b[..., 0], axis=(1, 2))
    mean_rho_bifrost = np.mean(rho_b[..., 0], axis=(1, 2))
    mean_press_bifrost = np.mean(press_b[..., 0], axis=(1, 2))

    # Magnetic fields:
    # Component 0: Bx, Component 1: By, Component 2: Bz (vertical)
    # Calculate $|B| = \sqrt{Bx^2 + By^2 + Bz^2}$ point-wise, then take mean
    print("   Averaging Bifrost B components and magnitude...")
    b_mag_b = np.sqrt(b_xyz_b[..., 0]**2 + b_xyz_b[..., 1]**2 + b_xyz_b[..., 2]**2)
    mean_B_bifrost = np.mean(b_mag_b, axis=(1, 2))
    mean_Bx_bifrost = np.mean(b_xyz_b[..., 0], axis=(1, 2))
    mean_By_bifrost = np.mean(b_xyz_b[..., 1], axis=(1, 2))
    mean_Bz_bifrost = np.mean(b_xyz_b[..., 2], axis=(1, 2))
    
    # Velocities:
    # Component 0: Vx, Component 1: Vy, Component 2: Vz (vertical)
    print("   Averaging Bifrost velocity components and magnitude...")
    vel_mag_b = np.sqrt(vel_b[..., 0]**2 + vel_b[..., 1]**2 + vel_b[..., 2]**2)
    mean_V_bifrost = np.mean(vel_mag_b, axis=(1, 2))
    mean_Vx_bifrost = np.mean(vel_b[..., 0], axis=(1, 2))
    mean_Vy_bifrost = np.mean(vel_b[..., 1], axis=(1, 2))
    mean_Vz_bifrost = np.mean(vel_b[..., 2], axis=(1, 2))

    # ===================================================================
    # 2. LOAD AND PROCESS MURAM DATASET (ORIGINAL COORDINATES)
    # ===================================================================
    print("\n--- Loading MURaM Original Dataset ---")
    pathsource = '/dat/milic/MURaM_enhanced_network/'
    snap_number = 499000
    
    cube = mio.MuramSnap(pathsource, snap_number)
    nz, nx, ny = cube.Temp.shape
    print(f"MURaM original shape: nz={nz}, nx={nx}, ny={ny}")

    # Build height grid: dz = 20 km = 0.02 Mm
    dx, dy, dz = 24/1e3, 24/1e3, 20/1e3
    z_geom = np.arange(nz) * dz
    
    print("   Loading full tau cube to locate tau=1 layer...")
    tau_mean = np.mean(cube.tau, axis=(1, 2))
    tau_zero_index = np.argmin(np.abs(tau_mean - 1))
    muram_z_grid = z_geom - z_geom[tau_zero_index]
    print(f"   MURaM tau=1 index: {tau_zero_index} (z={z_geom[tau_zero_index]:.4f} Mm)")
    print(f"   MURaM grid range: {muram_z_grid.min():.4f} to {muram_z_grid.max():.4f} Mm")

    # Keep in mind: MURaM shape is (nz, nx, ny).
    # Axis 0 is the vertical coordinate. Axis 1 is x, Axis 2 is y. We average over axis=(1, 2)
    print("Computing MURaM mean height stratifications (CGS originally)...")
    mean_T_muram = np.mean(cube.Temp, axis=(1, 2)) # Already in Kelvin
    mean_ne_muram = np.mean(cube.ne, axis=(1, 2))  # CGS (cm^-3)
    mean_rho_muram = np.mean(cube.rho, axis=(1, 2)) # CGS (g/cm^3)
    mean_press_muram = np.mean(cube.Pres, axis=(1, 2)) # CGS (dyn/cm^2)

    # Align vector components due to transposed axis representations:
    # Physical vertical component is index 0 of MURaM cube (which is 'x' component in the package: Bx, vx)
    # Physical horizontal x is index 1 (which is 'y' component: By, vy)
    # Physical horizontal y is index 2 (which is 'z' component: Bz, vz)
    print("   Handling vector axis alignment and magnitudes...")
    b_mag_m = np.sqrt(cube.Bx**2 + cube.By**2 + cube.Bz**2)
    mean_B_muram = np.mean(b_mag_m, axis=(1, 2))      # CGS (Gauss)
    mean_Bz_muram = np.mean(cube.Bx, axis=(1, 2))      # Physical vertical Bz (Gauss)
    mean_Bx_muram = np.mean(cube.By, axis=(1, 2))      # Physical horizontal Bx (Gauss)
    mean_By_muram = np.mean(cube.Bz, axis=(1, 2))      # Physical horizontal By (Gauss)

    vel_mag_m = np.sqrt(cube.vx**2 + cube.vy**2 + cube.vz**2)
    mean_V_muram = np.mean(vel_mag_m, axis=(1, 2))     # CGS (cm/s)
    mean_Vz_muram = np.mean(cube.vx, axis=(1, 2))      # Physical vertical Vz (cm/s)
    mean_Vx_muram = np.mean(cube.vy, axis=(1, 2))      # Physical horizontal Vx (cm/s)
    mean_Vy_muram = np.mean(cube.vz, axis=(1, 2))      # Physical horizontal Vy (cm/s)

    # ---- Unit Conversions (CGS -> SI) ----
    print("   Converting MURaM averages to SI units...")
    # Velocity: cm/s -> m/s
    mean_Vx_muram_si = mean_Vx_muram / 100.0
    mean_Vy_muram_si = mean_Vy_muram / 100.0
    mean_Vz_muram_si = mean_Vz_muram / 100.0
    mean_V_muram_si  = mean_V_muram / 100.0

    # Magnetic field: Gauss -> Tesla (factor of (4*pi)/10000.0 used in repo)
    scale_B = (4.0 * np.pi) / 10000.0
    mean_Bx_muram_si = mean_Bx_muram * scale_B
    mean_By_muram_si = mean_By_muram * scale_B
    mean_Bz_muram_si = mean_Bz_muram * scale_B
    mean_B_muram_si  = mean_B_muram * scale_B

    # Electron density: cm^-3 -> m^-3
    mean_ne_muram_si = mean_ne_muram * 1e6

    # Mass density: g/cm^3 -> kg/m^3
    mean_rho_muram_si = mean_rho_muram * 1000.0

    # Gas pressure: dyn/cm^2 -> Pa
    mean_press_muram_si = mean_press_muram / 10.0

    # ===================================================================
    # 3. PLOT DIAGNOSTIC STRATIFICATION
    # ===================================================================
    print("\n--- Generating Plots ---")
    
    # Elegant, harmonized color palette:
    # Royal blue for Bifrost, Crimson for MURaM
    color_b = '#1f77b4' # Royal blue
    color_m = '#d62728' # Crimson
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 20), dpi=150)
    fig.suptitle("Solar Simulation Physical Quantity Stratification Comparison\n(Averages over horizontal x-y planes, plotted vs. respective physical grids)", fontsize=16, y=0.98)
    
    # 1. Temperature T(z)
    ax = axes[0, 0]
    ax.plot(bifrost_z_grid, mean_T_bifrost, label='Bifrost', color=color_b, linewidth=2)
    ax.plot(muram_z_grid, mean_T_muram, label='MURaM', color=color_m, linewidth=2)
    ax.set_title("Temperature $T(z)$")
    ax.set_ylabel("T [K]")
    ax.set_xlabel("Height $z$ [Mm]")
    ax.grid(True)
    ax.legend()
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7, label='$\\tau=1$')

    # 2. Gas Pressure P(z) [Log Scale]
    ax = axes[0, 1]
    ax.plot(bifrost_z_grid, mean_press_bifrost, label='Bifrost', color=color_b, linewidth=2)
    ax.plot(muram_z_grid, mean_press_muram_si, label='MURaM', color=color_m, linewidth=2)
    ax.set_title("Gas Pressure $P(z)$ [Log Scale]")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Height $z$ [Mm]")
    ax.set_yscale('log')
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)

    # 3. Mass Density rho(z) [Log Scale]
    ax = axes[1, 0]
    ax.plot(bifrost_z_grid, mean_rho_bifrost, label='Bifrost', color=color_b, linewidth=2)
    ax.plot(muram_z_grid, mean_rho_muram_si, label='MURaM', color=color_m, linewidth=2)
    ax.set_title("Mass Density $\\rho(z)$ [Log Scale]")
    ax.set_ylabel("Mass Density [kg/m$^3$]")
    ax.set_xlabel("Height $z$ [Mm]")
    ax.set_yscale('log')
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)

    # 4. Electron Density ne(z) [Log Scale]
    ax = axes[1, 1]
    ax.plot(bifrost_z_grid, mean_ne_bifrost, label='Bifrost', color=color_b, linewidth=2)
    ax.plot(muram_z_grid, mean_ne_muram_si, label='MURaM', color=color_m, linewidth=2)
    ax.set_title("Electron Density $n_e(z)$ [Log Scale]")
    ax.set_ylabel("Electron Density [m$^{-3}$]")
    ax.set_xlabel("Height $z$ [Mm]")
    ax.set_yscale('log')
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)

    # 5. Total Magnetic Field |B|(z)
    ax = axes[2, 0]
    ax.plot(bifrost_z_grid, mean_B_bifrost, label='Bifrost $|B|$', color=color_b, linewidth=2)
    ax.plot(muram_z_grid, mean_B_muram_si, label='MURaM $|B|$', color=color_m, linewidth=2)
    ax.plot(bifrost_z_grid, np.abs(mean_Bz_bifrost), label='Bifrost $|B_z|$', color=color_b, linestyle=':', alpha=0.8)
    ax.plot(muram_z_grid, np.abs(mean_Bz_muram_si), label='MURaM $|B_z|$', color=color_m, linestyle=':', alpha=0.8)
    ax.set_title("Magnetic Field Strength")
    ax.set_ylabel("B [Tesla]")
    ax.set_xlabel("Height $z$ [Mm]")
    ax.grid(True)
    ax.legend()
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)

    # 6. Mean Vertical Velocity Vz(z)
    ax = axes[2, 1]
    ax.plot(bifrost_z_grid, mean_Vz_bifrost, label='Bifrost $V_z$', color=color_b, linewidth=2)
    ax.plot(muram_z_grid, mean_Vz_muram_si, label='MURaM $V_z$', color=color_m, linewidth=2)
    ax.set_title("Vertical Velocity $V_z(z)$ (Averaged)")
    ax.set_ylabel("V$_z$ [m/s] (positive is upward/downward dependent)")
    ax.set_xlabel("Height $z$ [Mm]")
    ax.grid(True)
    ax.legend()
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)

    # 7. Horizontal Magnetic Field B_horiz(z)
    ax = axes[3, 0]
    b_horiz_b = np.sqrt(b_xyz_b[..., 0]**2 + b_xyz_b[..., 1]**2)
    mean_Bhoriz_bifrost = np.mean(b_horiz_b, axis=(1, 2))
    
    b_horiz_m = np.sqrt(cube.By**2 + cube.Bz**2)
    mean_Bhoriz_muram_si = np.mean(b_horiz_m, axis=(1, 2)) * scale_B
    
    ax.plot(bifrost_z_grid, mean_Bhoriz_bifrost, label='Bifrost $B_{horiz}$', color=color_b, linewidth=2)
    ax.plot(muram_z_grid, mean_Bhoriz_muram_si, label='MURaM $B_{horiz}$', color=color_m, linewidth=2)
    ax.set_title("Horizontal Magnetic Field Strength $B_{horiz}(z)$")
    ax.set_ylabel("B$_{horiz}$ [Tesla]")
    ax.set_xlabel("Height $z$ [Mm]")
    ax.grid(True)
    ax.legend()
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)

    # 8. Total Velocity Magnitude |V|(z)
    ax = axes[3, 1]
    ax.plot(bifrost_z_grid, mean_V_bifrost, label='Bifrost $|V|$', color=color_b, linewidth=2)
    ax.plot(muram_z_grid, mean_V_muram_si, label='MURaM $|V|$', color=color_m, linewidth=2)
    ax.set_title("Total Velocity Magnitude $|V|(z)$")
    ax.set_ylabel("V [m/s]")
    ax.set_xlabel("Height $z$ [Mm]")
    ax.grid(True)
    ax.legend()
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_save_path = 'compare_height_stratification.png'
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"\nSuccessfully generated and saved comparison plot to: {plot_save_path}")
    print("==================================================================")

    # ===================================================================
    # 4. POPULATION STRATIFICATION
    # ===================================================================
    plot_population_stratification(
        bifrost_z_grid=bifrost_z_grid,
        muram_z_grid=muram_z_grid,
        nx_b=nx_b, ny_b=ny_b, nz_b=nz_b,
        datadir_bifrost=datadir_bifrost,
    )


def plot_population_stratification(bifrost_z_grid, muram_z_grid,
                                   nx_b, ny_b, nz_b, datadir_bifrost):
    """
    Plot the mean height stratification of Ca II level populations for:
      - Bifrost original NLTE populations (training data)
      - GNN predicted populations (on the interpolated MURaM grid)
      - MURaM 1.5D NLTE populations (from FITS file)

    Each dataset lives on its own native height grid, so no cross-interpolation
    is performed — they are all overplotted together for a direct visual
    comparison of the mean horizontal-average profiles.
    """
    print("\n" + "="*68)
    print("SECTION 4: Level Population Height Stratification")
    print("="*68)

    nlev = 6  # Ca II 5-level + ion (6 populations)
    level_labels = [
        "Ca II Ground (lvl 0)",
        "Ca II 1st excited (lvl 1)",
        "Ca II 2nd excited (lvl 2)",
        "Ca II Metastable (lvl 3)",
        "Ca II 4th excited (lvl 4)",
        "Ca III Ion (lvl 5)",
    ]

    # ------------------------------------------------------------------
    # A. BIFROST populations  shape: (nz_b, ny_b, nx_b, nlev)
    #    Native grid: bifrost_z_grid  (nz_b,)
    # ------------------------------------------------------------------
    print("\n--- Loading Bifrost populations ---")
    pops_b_path = f'{datadir_bifrost}/AR_385_CaII_5L_pops.dat'
    if not os.path.exists(pops_b_path):
        print(f"  ERROR: Bifrost populations file not found: {pops_b_path}")
        sys.exit(1)
    pops_b = np.memmap(pops_b_path, dtype='<f4', mode='r',
                       shape=(nz_b, ny_b, nx_b, nlev))
    # Mean over horizontal axes (1, 2) -> shape (nz_b, nlev)
    print("  Computing Bifrost mean population profiles...")
    mean_pops_bifrost = np.mean(pops_b, axis=(1, 2))  # (nz_b, nlev)
    print(f"  Bifrost mean pops shape: {mean_pops_bifrost.shape}")

    # ------------------------------------------------------------------
    # B. MURaM 1.5D NLTE populations  (FITS file)
    #    FITS shape in Python: (1024, 1024, nlev, 401) = (x, y, level, z)
    #    Native grid: muram_z_grid[15:416]  (401,)
    # ------------------------------------------------------------------
    print("\n--- Loading MURaM 1.5D NLTE populations from FITS ---")
    fits_path = "/dat/milic/MURaM_enhanced_network/che_full_499000_lwsynth_200.0.fits"
    if not os.path.exists(fits_path):
        print(f"  ERROR: FITS file not found: {fits_path}")
        sys.exit(1)

    fits_heights = muram_z_grid[15:416]  # shape (401,)

    print("  Memory-mapping FITS file and cropping horizontal domain...")
    with fits.open(fits_path, memmap=True) as hdul:
        # hdul[2].data shape: (1024, 1024, nlev, 401)
        fits_cropped = hdul[2].data[16:-16, 16:-16, :, :]  # (992, 992, nlev, 401)

    # Reverse vertical axis so height increases from photosphere to chromosphere
    fits_cropped = fits_cropped[..., ::-1]  # still (992, 992, nlev, 401)

    # Unit conversion: FITS is in m^-3 (SI); convert to cm^-3 (CGS) to match Bifrost
    print("  Converting FITS populations from SI to CGS (m^-3 -> cm^-3)...")
    fits_cropped = fits_cropped * 1e-6  # (992, 992, nlev, 401)

    # Mean over horizontal axes (0, 1) -> shape (nlev, 401)
    print("  Computing MURaM mean population profiles...")
    mean_pops_muram = np.mean(fits_cropped, axis=(0, 1))  # (nlev, 401)
    print(f"  MURaM mean pops shape: {mean_pops_muram.shape}")

    # ------------------------------------------------------------------
    # C. GNN predicted populations  shape: (50, 992, 992, nlev)
    #    Native grid: zz_grid_muram  (50,) -- subset clipped to FITS range
    # ------------------------------------------------------------------
    print("\n--- Loading GNN predictions ---")
    # Re-build the interpolated GNN height grid (same logic as compare_populations.py)
    nz_b_orig = nz_b
    logspace_fraction = 0.33
    nz_linear = 30
    nz_log = 20

    z_b_idx = np.arange(nz_b_orig)
    new_z_b_log = np.concatenate([
        np.linspace(0, nz_b_orig * logspace_fraction, nz_linear, endpoint=False),
        np.logspace(np.log10(nz_b_orig * logspace_fraction),
                    np.log10(nz_b_orig - 1), nz_log)
    ])
    new_z_b = np.clip(new_z_b_log, 0, nz_b_orig - 1)
    zz_grid_bifrost_interp = np.interp(new_z_b, z_b_idx, bifrost_z_grid)

    # Map to MURaM height coordinate
    nz_m = len(muram_z_grid)
    z_m_idx = np.arange(nz_m)
    new_z_m = np.interp(zz_grid_bifrost_interp, muram_z_grid, z_m_idx)
    new_z_m = np.clip(new_z_m, 0, nz_m - 1)
    zz_grid_muram_interp = np.interp(new_z_m, z_m_idx, muram_z_grid)  # (50,)

    # Clip to the FITS vertical range
    valid_mask = ((zz_grid_muram_interp >= fits_heights.min()) &
                  (zz_grid_muram_interp <= fits_heights.max()))
    zz_gnn_clipped = zz_grid_muram_interp[valid_mask]  # subset of 50 heights

    pred_path_candidates = [
        'muram_predictions_stride_4_full.npy',
        '../muram_predictions_stride_4_full.npy',
        '/dat/andreuva/gpu/graphnet/graphnet_nlte/muram_predictions_stride_4_full.npy',
    ]
    pred_path = None
    for p in pred_path_candidates:
        if os.path.exists(p):
            pred_path = p
            break
    if pred_path is None:
        print("  ERROR: GNN predictions file not found.")
        sys.exit(1)

    print(f"  Loading GNN predictions from: {pred_path}")
    pred = np.load(pred_path)   # shape (50, 992, 992, nlev)  in cm^-3

    # Clip to valid heights
    pred_cgs_clipped = pred[valid_mask, :, :, :]  # (n_valid, 992, 992, nlev)

    # Mean over horizontal axes (1, 2) -> shape (n_valid, nlev)
    print("  Computing GNN mean population profiles...")
    mean_pops_gnn = np.mean(pred_cgs_clipped, axis=(1, 2))  # (n_valid, nlev)
    print(f"  GNN mean pops shape: {mean_pops_gnn.shape}")

    # ------------------------------------------------------------------
    # D. PLOT
    # ------------------------------------------------------------------
    print("\n--- Generating population stratification plots ---")

    color_b   = '#1f77b4'   # Royal blue  -- Bifrost
    color_g   = '#2ca02c'   # Forest green -- GNN
    color_m   = '#d62728'   # Crimson      -- MURaM NLTE

    ncols = 3
    nrows = 2  # 6 levels -> 2x3 grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), dpi=150)
    fig.suptitle(
        "Mean Level Population Height Stratification\n"
        "Ca\u202fII 5-level + ion (horizontal averages over x-y planes)",
        fontsize=16, y=0.99
    )

    for lvl in range(nlev):
        row, col = divmod(lvl, ncols)
        ax = axes[row, col]

        # --- Bifrost (full original grid, all 425 heights) ---
        ax.plot(bifrost_z_grid, mean_pops_bifrost[:, lvl],
                label='Bifrost', color=color_b, linewidth=2.0, zorder=3)

        # --- MURaM 1.5D NLTE (401 heights in the FITS range) ---
        ax.plot(fits_heights, mean_pops_muram[lvl, :],
                label='MURaM NLTE', color=color_m, linewidth=2.0, zorder=2)

        # --- GNN predicted (50 heights, clipped to FITS range) ---
        ax.plot(zz_gnn_clipped, mean_pops_gnn[:, lvl],
                label='GNN predicted', color=color_g,
                linewidth=2.0, linestyle='--', zorder=4)

        ax.set_yscale('log')
        ax.set_xlabel("Height $z$ [Mm]")
        ax.set_ylabel("Population density [cm$^{-3}$]")
        ax.set_title(level_labels[lvl])
        ax.axvline(0, color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
        ax.grid(True, which='both', alpha=0.25, linestyle='--')
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pop_plot_path = 'compare_population_stratification.png'
    plt.savefig(pop_plot_path, dpi=300)
    plt.close()
    print(f"\nSuccessfully saved population stratification plot to: {pop_plot_path}")
    print("="*68 + "\n")


if __name__ == "__main__":
    main()
