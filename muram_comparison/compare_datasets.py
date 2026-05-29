# compare_datasets.py

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interpn
import muram as mio
from normalization import normalize_features_with_params

# Use GPU 2 if available
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load Checkpoint ----
checkpoint_path = '/dat/andreuva/gpu/graphnet/graphnet_nlte/checkpoints/multistride_cpudtst_4x4_s4_m8_b8_r6_d025_press/2026.03.01-02:16:27_best.pth'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint['config']
feature_norm_params = checkpoint['feature_norm_params']
normalization_type = config['normalization'].get('type', 'log')
print("Loaded checkpoint and parameters successfully.")

# ---- Load Bifrost Geometry & Height Grid First ----
print("\n--- Loading Bifrost Geometry & Grid ---")
datadir_bifrost = '/dat/andreuva/gpu/graphnet/data_train'
bifrost_grid_file = '/dat/andreuva/gpu/graphnet/en024048_hion/grid_bifrost.npz'
nx_b, ny_b, nz_b = config['data']['nx'], config['data']['ny'], config['data']['nz_orig']
logspace_fraction = config['dataset']['logspace_fraction']
nz_linear = config['dataset']['nz_linear']
nz_log = config['dataset']['nz_log']

bifrost_z_grid = np.load(bifrost_grid_file)["z"]

# Grid setup for Bifrost
z_b, y_b, x_b = (np.arange(d) for d in (nz_b, ny_b, nx_b))
new_z_b_log = np.concatenate([
    np.linspace(0, nz_b * logspace_fraction, nz_linear, endpoint=False),
    np.logspace(np.log10(nz_b * logspace_fraction), np.log10(nz_b - 1), nz_log)
])
new_z_b = np.clip(new_z_b_log, 0, nz_b - 1)
zz_grid_bifrost = np.interp(new_z_b, z_b, bifrost_z_grid)

new_y_b, new_x_b = (np.linspace(0, d - 1, new_d) for d, new_d in zip((ny_b, nx_b), (ny_b, nx_b)))
new_zv_b, new_yv_b, new_xv_b = np.meshgrid(new_z_b, new_y_b, new_x_b, indexing='ij', sparse=True)
new_points_bifrost = (new_zv_b, new_yv_b, new_xv_b)

# ===================================================================
# 1. LOAD AND PREPARE MURAM DATASET
# ===================================================================
print("\n--- Loading MURaM Dataset ---")
pathsource = '/dat/milic/MURaM_enhanced_network/'
snap_number = 499000
cube = mio.MuramSnap(pathsource, snap_number)

# Grid parameters
dx, dy, dz = 24/1e3, 24/1e3, 20/1e3
nz, nx, ny = cube.Temp.shape
z, x, y = (np.arange(d) for d in (nz, nx, ny))
z_geom, x_geom, y_geom = z*dz, x*dx, y*dy

tau_mean = np.mean(cube.tau, axis=(1,2))
tau_zero_index = np.argmin(np.abs(tau_mean - 1))
muram_z_grid = z_geom - z_geom[tau_zero_index]

# Align height interpolation exactly with Bifrost:
# Map zz_grid_bifrost to fractional indices in MURaM's vertical coordinate system
new_z = np.interp(zz_grid_bifrost, muram_z_grid, z)
new_z = np.clip(new_z, 0, nz - 1)

new_y, new_x = (np.linspace(0, d - 1, new_d) for d, new_d in zip((ny, nx), (ny, nx)))
new_zv, new_yv, new_xv = np.meshgrid(new_z, new_y, new_x, indexing='ij', sparse=True)

zz_grid_muram = np.interp(new_z, z, muram_z_grid)
new_points_muram = (new_zv, new_xv, new_yv)

# --- PLOT COMPARISON OF GRIDS ---
plt.figure(figsize=(12, 4))
plt.plot(muram_z_grid, np.ones_like(muram_z_grid) * 3, '|', markersize=15, label='Original MURaM Grid', color='darkorange')
plt.plot(bifrost_z_grid, np.ones_like(bifrost_z_grid) * 2, '|', markersize=15, label='Original Bifrost Grid', color='royalblue')
plt.plot(zz_grid_muram, np.ones_like(zz_grid_muram) * 1, '|', markersize=15, label='Interpolated MURaM Grid (from Bifrost)', color='crimson')
plt.plot(zz_grid_bifrost, np.ones_like(zz_grid_bifrost) * 0, '|', markersize=15, label='Interpolated Bifrost Grid', color='forestgreen')
plt.yticks([0, 1, 2, 3], ['Interp. Bifrost', 'Interp. MURaM', 'Original Bifrost', 'Original MURaM'])
plt.xlabel('Height (Mm)', fontsize=12)
plt.ylim(-0.5, 3.5)
plt.legend(loc='best')
plt.title('Comparison of Original and Interpolated Geometry Grids', fontsize=14)
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
grid_plot_path = 'compare_grids.png'
plt.savefig(grid_plot_path, dpi=300)
plt.close()
print(f"Saved grid comparison plot to {grid_plot_path}")

print("Interpolating MURaM features...")
temp_muram = interpn((z, x, y), cube.Temp, new_points_muram)
b_z_m = interpn((z, x, y), cube.Bx, new_points_muram)
b_x_m = interpn((z, x, y), cube.By, new_points_muram)
b_y_m = interpn((z, x, y), cube.Bz, new_points_muram)
v_z_m = interpn((z, x, y), cube.vx, new_points_muram)
v_x_m = interpn((z, x, y), cube.vy, new_points_muram)
v_y_m = interpn((z, x, y), cube.vz, new_points_muram)
rho_muram = interpn((z, x, y), cube.rho, new_points_muram)
ne_muram = interpn((z, x, y), cube.ne, new_points_muram)
press_muram = interpn((z, x, y), cube.Pres, new_points_muram)

# Stack variables to Bifrost layout
b_xyz_muram = np.stack([b_x_m, b_y_m, b_z_m], axis=-1)
vel_muram = np.stack([v_x_m, v_y_m, v_z_m], axis=-1)

# Convert MURaM features to SI units to match Bifrost units before normalization
vel_muram_si = vel_muram / 100.0
b_xyz_muram_si = b_xyz_muram * (4.0 * np.pi) / 10000.0
temp_muram_si = temp_muram
ne_muram_si = ne_muram * 1e6
rho_muram_si = rho_muram * 1000.0
press_muram_si = press_muram / 10.0

features_labels = ['vel', 'b', 'temp', 'n_e', 'rho', 'press']
features_data_muram = [vel_muram_si, b_xyz_muram_si, temp_muram_si, ne_muram_si, rho_muram_si, press_muram_si]

norm_features_muram = normalize_features_with_params(
    features_data_muram, features_labels, feature_norm_params, normalization_type
)

# ===================================================================
# 2. LOAD AND PREPARE BIFROST DATASET
# ===================================================================
print("\n--- Loading Bifrost Dataset ---")
datadir_bifrost = '/dat/andreuva/gpu/graphnet/data_train'
bifrost_grid_file = '/dat/andreuva/gpu/graphnet/en024048_hion/grid_bifrost.npz'
nx_b, ny_b, nz_b = config['data']['nx'], config['data']['ny'], config['data']['nz_orig']

bifrost_z_grid = np.load(bifrost_grid_file)["z"]

# Memory mapped arrays
pops_b = np.memmap(f'{datadir_bifrost}/AR_385_CaII_5L_pops.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, config['data']['nlev']))
b_xyz_b = np.memmap(f'{datadir_bifrost}/AR_385_B.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 3))
temp_b = np.memmap(f'{datadir_bifrost}/AR_385_temp.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 1))
vel_b = np.memmap(f'{datadir_bifrost}/AR_385_veloc.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 3))
ne_b = np.memmap(f'{datadir_bifrost}/AR_385_ne.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 1))
rho_b = np.memmap(f'{datadir_bifrost}/AR_385_mass.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 1))
press_b = np.memmap(f'{datadir_bifrost}/AR_385_press.dat', dtype='<f4', mode='r', shape=(nz_b, ny_b, nx_b, 1))

# Grid setup
z_b, y_b, x_b = (np.arange(d) for d in (nz_b, ny_b, nx_b))
new_z_b_log = np.concatenate([
    np.linspace(0, nz_b * logspace_fraction, nz_linear, endpoint=False),
    np.logspace(np.log10(nz_b * logspace_fraction), np.log10(nz_b - 1), nz_log)
])
new_z_b = np.clip(new_z_b_log, 0, nz_b - 1)
new_y_b, new_x_b = (np.linspace(0, d - 1, new_d) for d, new_d in zip((ny_b, nx_b), (ny_b, nx_b)))
new_zv_b, new_yv_b, new_xv_b = np.meshgrid(new_z_b, new_y_b, new_x_b, indexing='ij', sparse=True)

zz_grid_bifrost = np.interp(new_z_b, z_b, bifrost_z_grid)
new_points_bifrost = (new_zv_b, new_yv_b, new_xv_b)

print("Interpolating Bifrost features...")
temp_bifrost = interpn((z_b, y_b, x_b), temp_b, new_points_bifrost)
b_xyz_bifrost = interpn((z_b, y_b, x_b), b_xyz_b, new_points_bifrost)
vel_bifrost = interpn((z_b, y_b, x_b), vel_b, new_points_bifrost)
ne_bifrost = interpn((z_b, y_b, x_b), ne_b, new_points_bifrost)
rho_bifrost = interpn((z_b, y_b, x_b), rho_b, new_points_bifrost)
press_bifrost = interpn((z_b, y_b, x_b), press_b, new_points_bifrost)

features_data_bifrost = [vel_bifrost, b_xyz_bifrost, temp_bifrost, ne_bifrost, rho_bifrost, press_bifrost]

norm_features_bifrost = normalize_features_with_params(
    features_data_bifrost, features_labels, feature_norm_params, normalization_type
)

# ===================================================================
# 3. PRINT RANGES AND PLOT DISTRIBUTIONS (PHYSICAL QUANTITIES)
# ===================================================================
print("\n" + "="*50)
print("PHYSICAL QUANTITY RANGES AND STATISTICS (BEFORE NORMALIZATION)")
print("="*50)

# Expand feature labels to individual components (all converted to SI)
labels_expanded = [
    'Vx (velocity_x) [m/s]', 'Vy (velocity_y) [m/s]', 'Vz (velocity_z) [m/s]',
    'Bx (magnetic_x) [Tesla]', 'By (magnetic_y) [Tesla]', 'Bz (magnetic_z) [Tesla]',
    'T (temperature) [Kelvin]', 'ne (electron density) [m^-3]',
    'rho (mass density) [kg/m^3]', 'press (gas pressure) [Pascal]'
]

# Extract unnormalized variables (already converted to SI in Section 1)
vel_b, b_b, temp_b, ne_b, rho_b, press_b = features_data_bifrost
vel_m_si, b_m_si, temp_m_si, ne_m_si, rho_m_si, press_m_si = features_data_muram

# Pack converted MURaM features
features_data_muram_si = features_data_muram

# Standardize shapes to ensure they are at least 4D
if vel_b.ndim == 3: vel_b = vel_b[..., np.newaxis]
if b_b.ndim == 3: b_b = b_b[..., np.newaxis]
if temp_b.ndim == 3: temp_b = temp_b[..., np.newaxis]
if ne_b.ndim == 3: ne_b = ne_b[..., np.newaxis]
if rho_b.ndim == 3: rho_b = rho_b[..., np.newaxis]
if press_b.ndim == 3: press_b = press_b[..., np.newaxis]

if vel_m_si.ndim == 3: vel_m_si = vel_m_si[..., np.newaxis]
if b_m_si.ndim == 3: b_m_si = b_m_si[..., np.newaxis]
if temp_m_si.ndim == 3: temp_m_si = temp_m_si[..., np.newaxis]
if ne_m_si.ndim == 3: ne_m_si = ne_m_si[..., np.newaxis]
if rho_m_si.ndim == 3: rho_m_si = rho_m_si[..., np.newaxis]
if press_m_si.ndim == 3: press_m_si = press_m_si[..., np.newaxis]

# Print stats in requested format
flat_bifrost = []
flat_muram = []
for idx, (b_feat, m_feat) in enumerate(zip(features_data_bifrost, features_data_muram_si)):
    if b_feat.ndim == 3: b_feat = b_feat[..., np.newaxis]
    if m_feat.ndim == 3: m_feat = m_feat[..., np.newaxis]
    if b_feat.shape[-1] == 3:
        for c in range(3):
            flat_bifrost.append(b_feat[..., c].ravel())
            flat_muram.append(m_feat[..., c].ravel())
    else:
        flat_bifrost.append(b_feat[..., 0].ravel())
        flat_muram.append(m_feat[..., 0].ravel())

for name, b_arr, m_arr in zip(labels_expanded, flat_bifrost, flat_muram):
    b_min, b_25, b_50, b_75, b_max = np.min(b_arr), np.percentile(b_arr, 25), np.percentile(b_arr, 50), np.percentile(b_arr, 75), np.max(b_arr)
    m_min, m_25, m_50, m_75, m_max = np.min(m_arr), np.percentile(m_arr, 25), np.percentile(m_arr, 50), np.percentile(m_arr, 75), np.max(m_arr)
    print(f"\nFeature: {name}")
    print(f"  Bifrost -> Min: {b_min:1.4e} | 25%: {b_25:1.4e} | 50% (Med): {b_50:1.4e} | 75%: {b_75:1.4e} | Max: {b_max:1.4e}")
    print(f"  MURaM   -> Min: {m_min:1.4e} | 25%: {m_25:1.4e} | 50% (Med): {m_50:1.4e} | 75%: {m_75:1.4e} | Max: {m_max:1.4e}")

# Create 6x2 grid of plots for separate dataset distributions
fig, axes = plt.subplots(6, 2, figsize=(14, 24))
colors_components = ['crimson', 'forestgreen', 'royalblue']

# --- ROW 0: Velocity Components (Vx, Vy, Vz) ---
# Bifrost
ax = axes[0, 0]
p5, p95 = np.percentile(vel_b, 1), np.percentile(vel_b, 99)
if p5 == p95: p5, p95 = vel_b.min(), vel_b.max()
bins = np.linspace(p5, p95, 100)
for c in range(3):
    ax.hist(vel_b[..., c].ravel(), bins=bins, range=(p5, p95), alpha=0.4, color=colors_components[c], label=f'V{chr(120+c)}', density=True)
ax.set_xlim(p5, p95)
ax.set_title("Bifrost (Training) Velocity Components", fontsize=12)
ax.set_xlabel("Velocity (m/s)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8)

# MURaM
ax = axes[0, 1]
p5, p95 = np.percentile(vel_m_si, 1), np.percentile(vel_m_si, 99)
if p5 == p95: p5, p95 = vel_m_si.min(), vel_m_si.max()
bins = np.linspace(p5, p95, 100)
for c in range(3):
    ax.hist(vel_m_si[..., c].ravel(), bins=bins, range=(p5, p95), alpha=0.4, color=colors_components[c], label=f'V{chr(120+c)}', density=True)
ax.set_xlim(p5, p95)
ax.set_title("MURaM (Synthesis) Velocity Components", fontsize=12)
ax.set_xlabel("Velocity (m/s)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8)

# --- ROW 1: Magnetic Field Components (Bx, By, Bz) ---
# Bifrost
ax = axes[1, 0]
p5, p95 = np.percentile(b_b, 5), np.percentile(b_b, 95)
if p5 == p95: p5, p95 = b_b.min(), b_b.max()
bins = np.linspace(p5, p95, 100)
for c in range(3):
    ax.hist(b_b[..., c].ravel(), bins=bins, range=(p5, p95), alpha=0.4, color=colors_components[c], label=f'B{chr(120+c)}', density=True)
ax.set_xlim(p5, p95)
ax.set_title("Bifrost (Training) Magnetic Field Components", fontsize=12)
ax.set_xlabel("B (Tesla)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8)

# MURaM
ax = axes[1, 1]
p5, p95 = np.percentile(b_m_si, 5), np.percentile(b_m_si, 95)
if p5 == p95: p5, p95 = b_m_si.min(), b_m_si.max()
bins = np.linspace(p5, p95, 100)
for c in range(3):
    ax.hist(b_m_si[..., c].ravel(), bins=bins, range=(p5, p95), alpha=0.4, color=colors_components[c], label=f'B{chr(120+c)}', density=True)
ax.set_xlim(p5, p95)
ax.set_title("MURaM (Synthesis) Magnetic Field Components", fontsize=12)
ax.set_xlabel("B (Tesla)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8)

# --- ROW 2: Temperature (T) ---
# Bifrost
ax = axes[2, 0]
t_b_flat = temp_b.ravel()
t_b_plot = t_b_flat[t_b_flat > 0]
if len(t_b_plot) == 0: t_b_plot = np.clip(t_b_flat, 1e-30, None)
p5, p95 = np.percentile(t_b_plot, 5), np.percentile(t_b_plot, 95)
if p5 == p95: p5, p95 = t_b_plot.min(), t_b_plot.max()
bins = np.logspace(np.log10(p5), np.log10(p95), 100)
ax.hist(t_b_plot, bins=bins, range=(p5, p95), alpha=0.6, color='purple', density=True)
ax.set_xscale('log')
ax.set_xlim(p5, p95)
ax.set_title("Bifrost (Training) Temperature", fontsize=12)
ax.set_xlabel("T (Kelvin)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# MURaM
ax = axes[2, 1]
t_m_flat = temp_m_si.ravel()
t_m_plot = t_m_flat[t_m_flat > 0]
if len(t_m_plot) == 0: t_m_plot = np.clip(t_m_flat, 1e-30, None)
p5, p95 = np.percentile(t_m_plot, 5), np.percentile(t_m_plot, 95)
if p5 == p95: p5, p95 = t_m_plot.min(), t_m_plot.max()
bins = np.logspace(np.log10(p5), np.log10(p95), 100)
ax.hist(t_m_plot, bins=bins, range=(p5, p95), alpha=0.6, color='purple', density=True)
ax.set_xscale('log')
ax.set_xlim(p5, p95)
ax.set_title("MURaM (Synthesis) Temperature", fontsize=12)
ax.set_xlabel("T (Kelvin)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# --- ROW 3: Electron Density (ne) ---
# Bifrost
ax = axes[3, 0]
ne_b_flat = ne_b.ravel()
ne_b_plot = ne_b_flat[ne_b_flat > 0]
if len(ne_b_plot) == 0: ne_b_plot = np.clip(ne_b_flat, 1e-30, None)
p5, p95 = np.percentile(ne_b_plot, 5), np.percentile(ne_b_plot, 95)
if p5 == p95: p5, p95 = ne_b_plot.min(), ne_b_plot.max()
bins = np.logspace(np.log10(p5), np.log10(p95), 100)
ax.hist(ne_b_plot, bins=bins, range=(p5, p95), alpha=0.6, color='teal', density=True)
ax.set_xscale('log')
ax.set_xlim(p5, p95)
ax.set_title("Bifrost (Training) Electron Density", fontsize=12)
ax.set_xlabel("ne (m^-3)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# MURaM
ax = axes[3, 1]
ne_m_flat = ne_m_si.ravel()
ne_m_plot = ne_m_flat[ne_m_flat > 0]
if len(ne_m_plot) == 0: ne_m_plot = np.clip(ne_m_flat, 1e-30, None)
p5, p95 = np.percentile(ne_m_plot, 5), np.percentile(ne_m_plot, 95)
if p5 == p95: p5, p95 = ne_m_plot.min(), ne_m_plot.max()
bins = np.logspace(np.log10(p5), np.log10(p95), 100)
ax.hist(ne_m_plot, bins=bins, range=(p5, p95), alpha=0.6, color='teal', density=True)
ax.set_xscale('log')
ax.set_xlim(p5, p95)
ax.set_title("MURaM (Synthesis) Electron Density", fontsize=12)
ax.set_xlabel("ne (m^-3)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# --- ROW 4: Mass Density (rho) ---
# Bifrost
ax = axes[4, 0]
rho_b_flat = rho_b.ravel()
rho_b_plot = rho_b_flat[rho_b_flat > 0]
if len(rho_b_plot) == 0: rho_b_plot = np.clip(rho_b_flat, 1e-30, None)
p5, p95 = np.percentile(rho_b_plot, 5), np.percentile(rho_b_plot, 95)
if p5 == p95: p5, p95 = rho_b_plot.min(), rho_b_plot.max()
bins = np.logspace(np.log10(p5), np.log10(p95), 100)
ax.hist(rho_b_plot, bins=bins, range=(p5, p95), alpha=0.6, color='brown', density=True)
ax.set_xscale('log')
ax.set_xlim(p5, p95)
ax.set_title("Bifrost (Training) Mass Density", fontsize=12)
ax.set_xlabel("rho (kg/m^3)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# MURaM
ax = axes[4, 1]
rho_m_flat = rho_m_si.ravel()
rho_m_plot = rho_m_flat[rho_m_flat > 0]
if len(rho_m_plot) == 0: rho_m_plot = np.clip(rho_m_flat, 1e-30, None)
p5, p95 = np.percentile(rho_m_plot, 5), np.percentile(rho_m_plot, 95)
if p5 == p95: p5, p95 = rho_m_plot.min(), rho_m_plot.max()
bins = np.logspace(np.log10(p5), np.log10(p95), 100)
ax.hist(rho_m_plot, bins=bins, range=(p5, p95), alpha=0.6, color='brown', density=True)
ax.set_xscale('log')
ax.set_xlim(p5, p95)
ax.set_title("MURaM (Synthesis) Mass Density", fontsize=12)
ax.set_xlabel("rho (kg/m^3)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# --- ROW 5: Gas Pressure (press) ---
# Bifrost
ax = axes[5, 0]
press_b_flat = press_b.ravel()
press_b_plot = press_b_flat[press_b_flat > 0]
if len(press_b_plot) == 0: press_b_plot = np.clip(press_b_flat, 1e-30, None)
p5, p95 = np.percentile(press_b_plot, 5), np.percentile(press_b_plot, 95)
if p5 == p95: p5, p95 = press_b_plot.min(), press_b_plot.max()
bins = np.logspace(np.log10(p5), np.log10(p95), 100)
ax.hist(press_b_plot, bins=bins, range=(p5, p95), alpha=0.6, color='olive', density=True)
ax.set_xscale('log')
ax.set_xlim(p5, p95)
ax.set_title("Bifrost (Training) Gas Pressure", fontsize=12)
ax.set_xlabel("pressure (Pa)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# MURaM
ax = axes[5, 1]
press_m_flat = press_m_si.ravel()
press_m_plot = press_m_flat[press_m_flat > 0]
if len(press_m_plot) == 0: press_m_plot = np.clip(press_m_flat, 1e-30, None)
p5, p95 = np.percentile(press_m_plot, 5), np.percentile(press_m_plot, 95)
if p5 == p95: p5, p95 = press_m_plot.min(), press_m_plot.max()
bins = np.logspace(np.log10(p5), np.log10(p95), 100)
ax.hist(press_m_plot, bins=bins, range=(p5, p95), alpha=0.6, color='olive', density=True)
ax.set_xscale('log')
ax.set_xlim(p5, p95)
ax.set_title("MURaM (Synthesis) Gas Pressure", fontsize=12)
ax.set_xlabel("pressure (Pa)", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
dist_plot_path = 'compare_features_physical_distribution.png'
plt.savefig(dist_plot_path, dpi=300)
plt.close()
print(f"\nSaved physical feature distribution comparison plot to {dist_plot_path}")

# ===================================================================
# 4. PLOT BIFROST & MURAM INITIAL QUANTITIES AT z=0 (tau=1)
# ===================================================================
print("\nGenerating side-by-side comparison of initial physical quantities at z=0 (tau=1)...")
tau_zero_bifrost_idx = np.argmin(np.abs(zz_grid_bifrost - 0))
print(f"Closest Z-index to z=0 (tau=1): {tau_zero_bifrost_idx} (z={zz_grid_bifrost[tau_zero_bifrost_idx]:1.4f} Mm)")

initial_quantities_bifrost = {
    'T': temp_bifrost[tau_zero_bifrost_idx, :, :, 0],
    'Bx': b_xyz_bifrost[tau_zero_bifrost_idx, :, :, 0],
    'By': b_xyz_bifrost[tau_zero_bifrost_idx, :, :, 1],
    'Bz': b_xyz_bifrost[tau_zero_bifrost_idx, :, :, 2],
    'Vx': vel_bifrost[tau_zero_bifrost_idx, :, :, 0],
    'Vy': vel_bifrost[tau_zero_bifrost_idx, :, :, 1],
    'Vz': vel_bifrost[tau_zero_bifrost_idx, :, :, 2]
}

initial_quantities_muram = {
    'T': temp_muram_si[tau_zero_bifrost_idx, :, :],
    'Bx': b_xyz_muram_si[tau_zero_bifrost_idx, :, :, 0],
    'By': b_xyz_muram_si[tau_zero_bifrost_idx, :, :, 1],
    'Bz': b_xyz_muram_si[tau_zero_bifrost_idx, :, :, 2],
    'Vx': vel_muram_si[tau_zero_bifrost_idx, :, :, 0],
    'Vy': vel_muram_si[tau_zero_bifrost_idx, :, :, 1],
    'Vz': vel_muram_si[tau_zero_bifrost_idx, :, :, 2]
}

# Create a 7x2 grid of subplots (Column 0: Bifrost, Column 1: MURaM)
fig_quant, axes_quant = plt.subplots(7, 2, figsize=(14, 30))

unit_labels = {
    'T': 'T (Kelvin)',
    'Bx': 'Bx (Tesla)',
    'By': 'By (Tesla)',
    'Bz': 'Bz (Tesla)',
    'Vx': 'Vx (m/s)',
    'Vy': 'Vy (m/s)',
    'Vz': 'Vz (m/s)'
}

for idx, name in enumerate(['T', 'Bx', 'By', 'Bz', 'Vx', 'Vy', 'Vz']):
    data_b = initial_quantities_bifrost[name]
    data_m = initial_quantities_muram[name]
    
    # Choose colormap based on variable
    if name == 'T':
        cmap = 'hot'
        # Shared temperature limits
        vmin = min(np.percentile(data_b, 1), np.percentile(data_m, 1))
        vmax = max(np.percentile(data_b, 99), np.percentile(data_m, 99))
    elif name.startswith('B'):
        cmap = 'coolwarm'
        # Symmetric limits for B: saturate more to \pm 0.05 Tesla to see weak fields
        vmin, vmax = -0.05, 0.05
    else: # Velocity
        cmap = 'seismic'
        # Symmetric limits for V
        vmax = max(np.percentile(np.abs(data_b), 99), np.percentile(np.abs(data_m), 99))
        vmin = -vmax
        
    # Plot Bifrost (Column 0)
    ax_b = axes_quant[idx, 0]
    im_b = ax_b.imshow(data_b, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax_b.set_title(f"Bifrost {name} at z=0 Mm", fontsize=12)
    ax_b.set_xlabel('x-pixel', fontsize=9)
    ax_b.set_ylabel('y-pixel', fontsize=9)
    fig_quant.colorbar(im_b, ax=ax_b, shrink=0.8, label=unit_labels[name])
    
    # Plot MURaM (Column 1)
    ax_m = axes_quant[idx, 1]
    im_m = ax_m.imshow(data_m, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax_m.set_title(f"MURaM {name} at z=0 Mm", fontsize=12)
    ax_m.set_xlabel('x-pixel', fontsize=9)
    ax_m.set_ylabel('y-pixel', fontsize=9)
    fig_quant.colorbar(im_m, ax=ax_m, shrink=0.8, label=unit_labels[name])

plt.tight_layout()
save_path_quant = 'compare_initial_quantities_tau1.png'
plt.savefig(save_path_quant, dpi=300)
plt.close()
print(f"Saved initial quantities side-by-side plot to {save_path_quant}")

# ===================================================================
# 5. PLOT BIFROST & MURAM TEMPERATURE COMPARISON AT DIFFERENT HEIGHTS
# ===================================================================
print("\nGenerating temperature comparison at different heights (2x10 plot)...")
# Select 10 height indices evenly spaced across the grid
nz_indices = np.linspace(0, temp_muram_si.shape[0] - 1, 10, dtype=int)

fig_temp, axes_temp = plt.subplots(2, 10, figsize=(24, 6))

for col_idx, z_idx in enumerate(nz_indices):
    height_val = zz_grid_bifrost[z_idx]
    data_b = temp_bifrost[z_idx, :, :, 0]
    data_m = temp_muram_si[z_idx, :, :]
    
    # Temperature colormap & shared limits for this height using 1st and 99th percentiles
    cmap = 'hot'
    vmin = min(np.percentile(data_b, 1), np.percentile(data_m, 1))
    vmax = max(np.percentile(data_b, 99), np.percentile(data_m, 99))
    
    # Plot Bifrost (Row 0)
    ax_b = axes_temp[0, col_idx]
    im_b = ax_b.imshow(data_b, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax_b.set_title(f"Bifrost\nz={height_val:.2f} Mm", fontsize=10)
    ax_b.set_xlabel('x-pixel', fontsize=8)
    ax_b.set_ylabel('y-pixel', fontsize=8)
    fig_temp.colorbar(im_b, ax=ax_b, shrink=0.8)
    
    # Plot MURaM (Row 1)
    ax_m = axes_temp[1, col_idx]
    im_m = ax_m.imshow(data_m, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax_m.set_title(f"MURaM\nz={height_val:.2f} Mm", fontsize=10)
    ax_m.set_xlabel('x-pixel', fontsize=8)
    ax_m.set_ylabel('y-pixel', fontsize=8)
    fig_temp.colorbar(im_m, ax=ax_m, shrink=0.8)

plt.tight_layout()
save_path_temp = 'compare_temperatures_heights.png'
plt.savefig(save_path_temp, dpi=300)
plt.close()
print(f"Saved temperature comparison at different heights plot to {save_path_temp}")

# ===================================================================
# 6. PLOT DISTRIBUTIONS OF NORMALIZED QUANTITIES
# ===================================================================
print("\nGenerating normalized feature distribution comparison plot...")

# Extract normalized arrays (norm_features_* are lists in the same order as features_labels)
# features_labels = ['vel', 'b', 'temp', 'n_e', 'rho', 'press']
norm_vel_b, norm_b_b, norm_temp_b, norm_ne_b, norm_rho_b, norm_press_b = norm_features_bifrost
norm_vel_m, norm_b_m, norm_temp_m, norm_ne_m, norm_rho_m, norm_press_m = norm_features_muram

# Ensure 4D
def ensure_4d(arr):
    return arr[..., np.newaxis] if arr.ndim == 3 else arr

norm_vel_b = ensure_4d(norm_vel_b)
norm_b_b   = ensure_4d(norm_b_b)
norm_temp_b= ensure_4d(norm_temp_b)
norm_ne_b  = ensure_4d(norm_ne_b)
norm_rho_b = ensure_4d(norm_rho_b)
norm_press_b = ensure_4d(norm_press_b)

norm_vel_m = ensure_4d(norm_vel_m)
norm_b_m   = ensure_4d(norm_b_m)
norm_temp_m= ensure_4d(norm_temp_m)
norm_ne_m  = ensure_4d(norm_ne_m)
norm_rho_m = ensure_4d(norm_rho_m)
norm_press_m = ensure_4d(norm_press_m)

fig_norm, axes_norm = plt.subplots(6, 2, figsize=(14, 24))
colors_components = ['crimson', 'forestgreen', 'royalblue']

# Helper: histogram with 1-99 percentile range
def plot_hist_norm(ax, arr, label, color, alpha=0.4, linscale=True):
    flat = arr.ravel()
    p1, p99 = np.percentile(flat, 1), np.percentile(flat, 99)
    if p1 == p99:
        p1, p99 = flat.min(), flat.max()
    bins = np.linspace(p1, p99, 100)
    ax.hist(flat, bins=bins, range=(p1, p99), alpha=alpha, color=color, label=label, density=True)
    ax.set_xlim(p1, p99)

# --- ROW 0: Normalized Velocity ---
ax = axes_norm[0, 0]
for c in range(3):
    plot_hist_norm(ax, norm_vel_b[..., c], f'V{chr(120+c)}', colors_components[c])
p1b, p99b = np.percentile(norm_vel_b, 1), np.percentile(norm_vel_b, 99)
ax.set_xlim(p1b, p99b)
ax.set_title("Bifrost (Training) Normalized Velocity", fontsize=12)
ax.set_xlabel("Normalized velocity", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8)

ax = axes_norm[0, 1]
for c in range(3):
    plot_hist_norm(ax, norm_vel_m[..., c], f'V{chr(120+c)}', colors_components[c])
p1m, p99m = np.percentile(norm_vel_m, 1), np.percentile(norm_vel_m, 99)
ax.set_xlim(p1m, p99m)
ax.set_title("MURaM (Synthesis) Normalized Velocity", fontsize=12)
ax.set_xlabel("Normalized velocity", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8)

# --- ROW 1: Normalized Magnetic Field ---
ax = axes_norm[1, 0]
for c in range(3):
    plot_hist_norm(ax, norm_b_b[..., c], f'B{chr(120+c)}', colors_components[c])
p1b, p99b = np.percentile(norm_b_b, 1), np.percentile(norm_b_b, 99)
ax.set_xlim(p1b, p99b)
ax.set_title("Bifrost (Training) Normalized Magnetic Field", fontsize=12)
ax.set_xlabel("Normalized B", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8)

ax = axes_norm[1, 1]
for c in range(3):
    plot_hist_norm(ax, norm_b_m[..., c], f'B{chr(120+c)}', colors_components[c])
p1m, p99m = np.percentile(norm_b_m, 1), np.percentile(norm_b_m, 99)
ax.set_xlim(p1m, p99m)
ax.set_title("MURaM (Synthesis) Normalized Magnetic Field", fontsize=12)
ax.set_xlabel("Normalized B", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8)

# --- ROW 2: Normalized Temperature ---
ax = axes_norm[2, 0]
plot_hist_norm(ax, norm_temp_b[..., 0], 'T', 'purple', alpha=0.6)
ax.set_title("Bifrost (Training) Normalized Temperature", fontsize=12)
ax.set_xlabel("Normalized T", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

ax = axes_norm[2, 1]
plot_hist_norm(ax, norm_temp_m[..., 0], 'T', 'purple', alpha=0.6)
ax.set_title("MURaM (Synthesis) Normalized Temperature", fontsize=12)
ax.set_xlabel("Normalized T", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# --- ROW 3: Normalized Electron Density ---
ax = axes_norm[3, 0]
plot_hist_norm(ax, norm_ne_b[..., 0], 'ne', 'teal', alpha=0.6)
ax.set_title("Bifrost (Training) Normalized Electron Density", fontsize=12)
ax.set_xlabel("Normalized ne", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

ax = axes_norm[3, 1]
plot_hist_norm(ax, norm_ne_m[..., 0], 'ne', 'teal', alpha=0.6)
ax.set_title("MURaM (Synthesis) Normalized Electron Density", fontsize=12)
ax.set_xlabel("Normalized ne", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# --- ROW 4: Normalized Mass Density ---
ax = axes_norm[4, 0]
plot_hist_norm(ax, norm_rho_b[..., 0], 'rho', 'saddlebrown', alpha=0.6)
ax.set_title("Bifrost (Training) Normalized Mass Density", fontsize=12)
ax.set_xlabel("Normalized rho", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

ax = axes_norm[4, 1]
plot_hist_norm(ax, norm_rho_m[..., 0], 'rho', 'saddlebrown', alpha=0.6)
ax.set_title("MURaM (Synthesis) Normalized Mass Density", fontsize=12)
ax.set_xlabel("Normalized rho", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# --- ROW 5: Normalized Gas Pressure ---
ax = axes_norm[5, 0]
plot_hist_norm(ax, norm_press_b[..., 0], 'press', 'olive', alpha=0.6)
ax.set_title("Bifrost (Training) Normalized Gas Pressure", fontsize=12)
ax.set_xlabel("Normalized press", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

ax = axes_norm[5, 1]
plot_hist_norm(ax, norm_press_m[..., 0], 'press', 'olive', alpha=0.6)
ax.set_title("MURaM (Synthesis) Normalized Gas Pressure", fontsize=12)
ax.set_xlabel("Normalized press", fontsize=10)
ax.set_ylabel("Probability Density", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
norm_dist_plot_path = 'compare_features_normalized_distribution.png'
plt.savefig(norm_dist_plot_path, dpi=300)
plt.close()
print(f"\nSaved normalized feature distribution comparison plot to {norm_dist_plot_path}")

