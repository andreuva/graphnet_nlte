# %%
import numpy as np
import torch, os
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interpn
import matplotlib.ticker as ticker
import multiprocessing

from graphnet import EncodeProcessDecode
from Dataset import EfficientDataset
from normalization import denormalize_pops, normalize_features, normalize_pops

# %%
import matplotlib.colors as colors
# Configure matplotlib for paper-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# %%
# Check if CUDA is available and set the device
# Set device
gpu = 0
cuda_available = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# %%
# The ONLY parameter you need to set manually is the path to the checkpoint.
checkpoint_path = '/dat/andreuva/gpu/graphnet/graphnet_nlte/checkpoints/multistride_cpudtst_4x4_s4_m8_b8_r6_d025_press/2026.03.01-02:16:27_best.pth'

# Load the entire checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Extract the configuration object used for training
config = checkpoint['config']
datadir = config['data']['datadir']
nx, ny, nz = config['data']['nx'], config['data']['ny'], config['data']['nz_orig']
nlev = config['data']['nlev']

nz_linear = config['dataset']['nz_linear']
nz_log = config['dataset']['nz_log']
new_nz = nz_linear + nz_log
logspace_fraction = config['dataset']['logspace_fraction']
log_offset = config['normalization']['log_offset']
normalization_type = config['normalization'].get('type', 'log')

print(f"Successfully loaded configuration from checkpoint {checkpoint_path}.")

# %%
# ---- memory–mapped arrays ----
pops_orig = np.memmap(f'{datadir}/AR_385_CaII_5L_pops.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, nlev))
b_xyz_orig = np.memmap(f'{datadir}/AR_385_B.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 3))
temp_orig = np.memmap(f'{datadir}/AR_385_temp.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
vel_orig = np.memmap(f'{datadir}/AR_385_veloc.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 3))
n_e_orig = np.memmap(f'{datadir}/AR_385_ne.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
# n_h_orig = np.memmap(f'{datadir}/AR_385_nh.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
# n_p_orig = np.memmap(f'{datadir}/AR_385_np.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
rho_orig = np.memmap(f'{datadir}/AR_385_mass.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
press_orig = np.memmap(f'{datadir}/AR_385_press.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))

# %%
print(f"Original data shape: {pops_orig.shape}")
print("Creating new Z grid")

z, y, x = (np.arange(d) for d in (nz, ny, nx))
new_z_log = np.concatenate([
    np.linspace(0, nz * logspace_fraction, nz_linear, endpoint=False),
    np.logspace(np.log10(nz * logspace_fraction), np.log10(nz - 1), nz_log)
])
new_z = np.clip(new_z_log, 0, nz - 1)
new_y, new_x = (np.linspace(0, d - 1, new_d) for d, new_d in zip((ny, nx), (ny, nx)))
new_zv, new_yv, new_xv = np.meshgrid(new_z, new_y, new_x, indexing='ij', sparse=True)

# %%
geometry_file = config["data"]["grid_file"]
geometry_grid = np.load(geometry_file)["z"]
zz_grid = np.interp(new_z, z, geometry_grid)

# %%
print(f"Loaded geometry grid from {geometry_file} with shape {zz_grid.shape}.")
plt.figure(figsize=(20, 2))
plt.plot(zz_grid, zz_grid * 0, '|', markersize=20, label='Interpolated grid')
plt.plot(geometry_grid, geometry_grid * 0, '|', label='Original grid')
plt.xlabel('Height (Mm)')
plt.legend()
plt.title('Comparison of Original and Interpolated Geometry Grids')
# plt.show()
plt.close()

# %%
print(f"Interpolating data to the new grid ({new_nz}, {ny}, {nx})...")
new_points = (new_zv, new_yv, new_xv)
pops = interpn((z, y, x), pops_orig, new_points)
temp = interpn((z, y, x), temp_orig, new_points)
b_xyz = interpn((z, y, x), b_xyz_orig, new_points)
vel = interpn((z, y, x), vel_orig, new_points)
n_e = interpn((z, y, x), n_e_orig, new_points)
# n_h = interpn((z, y, x), n_h_orig, new_points)
# n_p = interpn((z, y, x), n_p_orig, new_points)
rho = interpn((z, y, x), rho_orig, new_points)
press = interpn((z, y, x), press_orig, new_points)
print("Interpolation complete.")
print(f"New shape: {pops.shape}")

# %%
# ---- Normalization ----
# IMPORTANT: Use the normalization parameters saved from the training run
feature_norm_params = checkpoint['feature_norm_params']
pop_norm_params = checkpoint['normalization_params']

features_labels = ['vx', 'vy', 'vz', 'bx', 'by', 'bz', 'temp', 'ne',
                #    'n_h', 'n_p']
                   'rho', 'press']
features_labels_simple = ['vel', 'b', 'temp', 'n_e',
                        #   'n_h', 'n_p']
                          'rho', 'press']
features_data = [vel, b_xyz, temp, n_e,
                #  n_h, n_p]
                 rho, press]

# We need to re-run the normalization to get the features for the dataset
# Note: For a true inference-only script, you would apply the saved means/stds,
# but since we're using the full dataset here, re-calculating is equivalent.
normalized_features, _ = normalize_features(features_data, features_labels_simple, log_offset, normalization_type)
pops_normalized, _ = normalize_pops(pops, factor=config['normalization']['factor'], log_offset=log_offset, type=normalization_type)

# ---- Create Test Dataset ----
# All parameters are now from the loaded config object
dataset_params = {
    'list_X': normalized_features,
    'list_Y': [pops_normalized],
    'radius_neighbors': config['dataset']['radius_neighbors'],
    'xdim': config['dataset']['x_range_graph'],
    'ydim': config['dataset']['y_range_graph'],
    'fully_connected': config['dataset']['fully_connected'],
    'pos_file': config['data']['grid_file'],
    'seed': config['system']['seed'],
    'train_ratio': config['dataset']['train_ratio'],
    'nz_linear': config['dataset']['nz_linear'],
    'nz_log': config['dataset']['nz_log'],
    'logspace_fraction': config['dataset']['logspace_fraction'],
    'epoch_size_fraction': 1.0,  # Use full dataset for testing
    'max_stride': 5, # config["dataset"].get("max_stride", 2),
    'random_stride': False, #config["dataset"].get("random_stride", False),
    'split': 'full'
}

datast_test = EfficientDataset(**dataset_params)
# loader_test = DataLoader(datast_test, batch_size=config['training']['batch_size']*6, shuffle=False)

num_workers = min(12, multiprocessing.cpu_count())
loader_test = DataLoader(
    datast_test, 
    batch_size=config['training']['batch_size']*6,
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=cuda_available,
    persistent_workers=True
)

# %%
plt.figure(0,(16,12), 150)
for i in range(datast_test.features.shape[1]):
    plt.subplot(3,4,i+1)
    plt.title(features_labels[i])
    plt.hist(datast_test.features[:,i],1000, density=True)
    plt.yscale('log')
plt.tight_layout()
# plt.show()
plt.close()

# %%
print("Feature normalization:")
print("="*60)
for i, label in enumerate(features_labels):
    feature_data = datast_test.features[:, i]
    print(f"{label:>6}: min={feature_data.min():>7.3f}, max={feature_data.max():>7.3f}")

print(f"\nTarget normalization:")
print("="*60)
for i in range(datast_test.targets.shape[1]):
    target_data = datast_test.targets[:, i]
    print(f"Ca II Level {i+1}: min={target_data.min():>7.3f}, max={target_data.max():>7.3f}")

# %%
plt.figure(0,(16,12), 150)
for i in range(datast_test.targets.shape[1]):
    plt.subplot(2,3,i+1)
    plt.title(f'Level population of Ca II_{i}')
    plt.hist(datast_test.targets[:,i],250, density=True)
    plt.yscale('log')
plt.tight_layout()
# plt.show()
plt.close()

# %%
# ---- Instantiate Model ----
model_params = config['model']
model = EncodeProcessDecode(**model_params).to(device)
model.load_state_dict(checkpoint['state_dict'])
print("Model loaded successfully from checkpoint.")
print(f"Loaded model from epoch {checkpoint['epoch']} with best validation loss: {checkpoint['best_loss']:.6f}")

# %%
all_predictions, all_targets = [], []
model.eval()

print("Running inference on the test set...")
with torch.no_grad():
    for data in tqdm(loader_test):
        node, edge_attr, edge_index = data.x.to(device), data.edge_attr.to(device), data.edge_index.to(device)
        u, batch, target = data.u.to(device), data.batch.to(device), data.y.to(device)
        out = model(node, edge_attr, edge_index, u, batch)
        all_predictions.append(out.cpu().numpy())
        all_targets.append(target.cpu().numpy())

predictions_flat = np.concatenate(all_predictions, axis=0)
targets_flat = np.concatenate(all_targets, axis=0)
print("Inference complete.")

# %%
print("Reconstructing spatial data cubes with height information...")
xdim = config['dataset']['x_range_graph']
ydim = config['dataset']['y_range_graph']
# Get the size of the infered spatial grid
nx_size = datast_test.x1 - datast_test.x0
ny_size = datast_test.y1 - datast_test.y0

nodes_per_sample = new_nz * (2 * xdim + 1) * (2 * ydim + 1)

# Create empty arrays for spatial data
predictions_spatial = np.zeros((new_nz, ny_size, nx_size, nlev))
targets_spatial = np.zeros((new_nz, ny_size, nx_size, nlev))

# Create lists to store flattened data for central columns
predictions_norm_central_flat = []
targets_norm_central_flat = []
k_indices_central_flat = [] # To store the vertical index 'k'

start_idx = 0
num_test_samples = len(datast_test)

for i in tqdm(range(num_test_samples), desc="Reconstructing"):
    ix, iy = datast_test.sample_centers[i]
    
    y_range = np.arange(iy - ydim, iy + ydim + 1)
    x_range = np.arange(ix - xdim, ix + xdim + 1)
    k_range = np.arange(new_nz)
    kv, yv, xv = np.meshgrid(k_range, y_range, x_range, indexing='ij')
    
    node_pos_indices = np.stack([kv.ravel(), yv.ravel(), xv.ravel()], axis=1)
    central_mask = (node_pos_indices[:, 1] == iy) & (node_pos_indices[:, 2] == ix)
    
    end_idx = start_idx + nodes_per_sample
    preds_subvolume = predictions_flat[start_idx:end_idx]
    targets_subvolume = targets_flat[start_idx:end_idx]

    pred_column = preds_subvolume[central_mask]
    target_column = targets_subvolume[central_mask]
    k_column = node_pos_indices[central_mask, 0] # Get the 'k' indices
    
    predictions_norm_central_flat.append(pred_column)
    targets_norm_central_flat.append(target_column)
    k_indices_central_flat.append(k_column)

    if pred_column.shape[0] == new_nz:
        predictions_spatial[:, iy - datast_test.y0, ix - datast_test.x0, :] = pred_column
        targets_spatial[:, iy - datast_test.y0, ix - datast_test.x0, :] = target_column
        
    start_idx = end_idx

# Concatenate lists into flat numpy arrays for analysis
predictions_norm_flat = np.concatenate(predictions_norm_central_flat, axis=0)
targets_norm_flat = np.concatenate(targets_norm_central_flat, axis=0)
k_indices_flat = np.concatenate(k_indices_central_flat, axis=0)

# %%
predictions_spatial.shape
pop_norm_params['totals'] = pop_norm_params['totals'][:, datast_test.x0:datast_test.x1, datast_test.y0:datast_test.y1]

# %%
# --- Denormalized Analysis ---
print("Denormalizing populations...")
predictions_denorm = denormalize_pops(predictions_spatial, pop_norm_params)
targets_denorm = denormalize_pops(targets_spatial, pop_norm_params)

# %%
test_mask = np.zeros((ny_size, nx_size), dtype=bool)
for i in range(num_test_samples):
    ix, iy = datast_test.sample_centers[i]
    test_mask[iy - datast_test.y0, ix - datast_test.x0] = True

predictions_denorm_flat = predictions_denorm[:, test_mask, :].reshape(-1, nlev)
targets_denorm_flat = targets_denorm[:, test_mask, :].reshape(-1, nlev)

residuals_denorm = (predictions_denorm_flat - targets_denorm_flat) / (targets_denorm_flat + 1e-12)
mae_denorm = np.mean(np.abs(residuals_denorm), axis=0)
rmse_denorm = np.sqrt(np.mean(residuals_denorm**2, axis=0))

# --- Normalized Analysis ---
print("\nCalculating metrics on normalized values...")
residuals_normalized = (predictions_norm_flat - targets_norm_flat) / (targets_norm_flat + 1e-12)
mae_normalized = np.mean(np.abs(residuals_normalized), axis=0)
rmse_normalized = np.sqrt(np.mean(residuals_normalized**2, axis=0))

print(f"\nMAE on original populations: {mae_denorm}")
print(f"RMSE on original populations: {rmse_denorm}")

print(f"\nMAE on normalized values: {mae_normalized}")
print(f"RMSE on normalized values: {rmse_normalized}")

# %%
print(f"predictions_denorm.shape: {predictions_denorm.shape}, targets_denorm.shape: {targets_denorm.shape}")

# %%
# Save both populations for further synthesis
np.save(f'{config['training']['savedir']}predictions_patch_stride_{dataset_params["max_stride"]}_{dataset_params["split"]}.npy', predictions_denorm)
np.save(f'{config['training']['savedir']}targets_patch_stride_{dataset_params["max_stride"]}_{dataset_params["split"]}.npy', targets_denorm)

# %%
# 1. Configuration
z_slices = [6, 20, 35, 45]   
n_levels = pops.shape[-1]    # 6 levels
cols = len(z_slices)
rows = n_levels + 1          # +1 for the Target row at the top

# 2. Setup Figure
# constrained_layout handles the space for top/bottom colorbars automatically
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4*rows), 
                         constrained_layout=True, 
                         gridspec_kw={'wspace': 0, 'hspace': 0})

# 3. Plotting Loop
for r in range(rows): 
    for col_idx, z_idx in enumerate(z_slices): 
        ax = axes[r, col_idx]
        
        # --- ROW 0: TARGET POPULATION (Reference) ---
        if r == 0:
            # We plot Level 0 (or a representative level) to show the physical structure
            # You can change the index [..., 0] if you want a different reference level
            targ_panel = targets_denorm[z_idx, :, :, 0]
            
            # Plot Heatmap (Log Scale)
            im_targ = ax.imshow(np.log10(targ_panel), cmap='magma', origin='lower')
            
            # Titles & Labels
            # ax.set_title(f"Z-idx: {z_idx}", fontsize=14, pad=45) # Pad for colorbar
            if col_idx == 0:
                ax.set_ylabel(r"$\rho_{Ca_{II}}\ (i=0)$", fontsize=18, fontweight='bold')
            
            # Top Colorbar
            # location='top' places it above the plot
            cbar = fig.colorbar(im_targ, ax=ax, location='top', fraction=0.05, pad=0.02)
            cbar.set_label(r'$log_{10}(\ \rho ($' + f'z={zz_grid[z_idx]:1.1f} Mm))', fontsize=16)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.locator = ticker.MaxNLocator(nbins=5)
            cbar.update_ticks()

        # --- ROWS 1..N: ERROR MAPS ---
        else:
            lvl = r - 1 # Adjust index since row 0 is used
            
            # Get data
            pred_panel = predictions_denorm[z_idx, :, :, lvl]
            targ_panel = targets_denorm[z_idx, :, :, lvl]
            
            # Calculate Error (dex)
            diff = np.log10(pred_panel) - np.log10(targ_panel)

            # Dynamic Range (based on full column stats for this Z)
            diff_vol_at_z = np.log10(predictions_denorm[z_idx, :, :, :]) - np.log10(targets_denorm[z_idx, :, :, :])
            vmax = np.percentile(np.abs(diff_vol_at_z), 98)
            vmin = -vmax
            
            # Plot Difference
            im_err = ax.imshow(diff, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            
            # No Contours (as requested)
            
            # Labels
            if col_idx == 0:
                ax.set_ylabel(r"$Ca_{II}$" + f" (i={lvl})", fontsize=18)
                
            # Bottom Colorbar (Only for the very last row)
            if r == rows - 1:
                cbar = fig.colorbar(im_err, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
                cbar.set_label('Error (dex)', fontsize=16)
                cbar.locator = ticker.MaxNLocator(nbins=4)
                cbar.update_ticks()

        # Common Formatting
        ax.set_xticks([])
        ax.set_yticks([])

save_path_dist = os.path.join(config['training']['savedir'], f'inference_maps_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

# %%
print("\nGenerating plots...")
import matplotlib.colors as colors

# ---- Plot 1: 2D Histograms (Predicted vs. Actual) on Original Scale ----
fig1, axes1 = plt.subplots(3, 2, figsize=(10, 15), dpi=100)
axes1 = axes1.flatten()
for i in range(nlev):
    ax = axes1[i]
    valid_mask = (targets_denorm_flat[:, i] > 0) & (predictions_denorm_flat[:, i] > 0)
    x = targets_denorm_flat[valid_mask, i]
    y = predictions_denorm_flat[valid_mask, i]

    if len(x) > 0:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        bins_x = np.logspace(np.log10(xmin), np.log10(xmax), 100)
        bins_y = np.logspace(np.log10(ymin), np.log10(ymax), 100)
        
        hist = ax.hist2d(x, y, bins=[bins_x, bins_y], norm=colors.LogNorm(1e0,1e4))
        
        lims = [min(xmin, ymin), max(xmax, ymax)]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=1, label='y=x (Perfect Fit)')
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_title(r'$Ca_{II}$'+f'(i={i+1})')
    if i==nlev-1:
        ax.set_title(r'$Ca_{III}$')
    if i in [4,5]:
        ax.set_xlabel('Real populations')
    if i in [0,2,4]:
        ax.set_ylabel('Predicted Populations')
    # ax.grid(True, linestyle='--', alpha=0.6)
    if i in [0]:
        ax.legend(loc='upper left')

# fig1.suptitle('Predicted vs. Actual Departure Coefficients (Original Scale)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path_dist = os.path.join(config['training']['savedir'], f'inference_scatter_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

# %%
# ---- Plot 2: 2D Histograms (Predicted vs. Actual) on Normalized Scale ----
fig2, axes2 = plt.subplots(3, 2, figsize=(10, 15), dpi=100)
axes2 = axes2.flatten()
for i in range(predictions_norm_flat.shape[1]):
    ax = axes2[i]
    num_bins = 100
    
    ax.hist2d(targets_norm_flat[:, i], predictions_norm_flat[:, i], bins=num_bins, norm=colors.LogNorm(1e0,1e4))
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=1)
    
    ax.set_title(f'Ca II Level {i+1}')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.legend(loc='upper left')

fig2.suptitle('Predicted vs. Actual Values for Each Target Feature (Normalized)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path_dist = os.path.join(config['training']['savedir'], f'inference_scatter_normalized_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

# %%
# ---- Plot 3: Comprehensive Analysis ----
fig3, axes3 = plt.subplots(nlev, 3, figsize=(15, 22), dpi=120)
fig3.suptitle('Comprehensive Model Performance Analysis for Ca II Levels', fontsize=20)

for i in range(nlev):
    # --- Row 1: Residuals on Original Scale ---
    ax1 = axes3[i, 0]
    data = residuals_denorm[:, i] * 100
    lower, upper = -50, 50 # np.percentile(data, [3, 97])
    ax1.hist(data, bins=100, range=(lower, upper), density=True, alpha=0.75)
    ax1.axvline(0, color='r', linestyle='--', label='Perfect Fit')
    ax1.set_title(f'Level {i+1} Residuals (Original)')
    ax1.set_xlabel('Relative Error (%)')
    # ax1.set_yscale('log')
    if i == 0: ax1.set_ylabel('Density')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.text(0.05, 0.95, f'MAE: {mae_denorm[i]:.3f}\nRMSE: {rmse_denorm[i]:.3f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.7))

    # --- Row 2: Residuals on Normalized Scale ---
    ax2 = axes3[i, 1]
    data = residuals_normalized[:, i] * 100
    lower, upper = -50, 50 # np.percentile(data, [3, 97])
    ax2.set_xlim(lower, upper)
    ax2.hist(data, bins=100, range=(lower, upper), density=True, alpha=0.75, color='orange')
    ax2.axvline(0, color='r', linestyle='--')
    ax2.set_title(f'Level {i+1} Residuals (Normalized)')
    ax2.set_xlabel('Relative Error (%)')
    # ax2.set_yscale('log')
    if i == 0: ax2.set_ylabel('Density')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.text(0.05, 0.95, f'MAE: {mae_normalized[i]:.3f}\nRMSE: {rmse_normalized[i]:.3f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    # --- Row 3: Distribution of Departure Coefficients (Original Scale) ---
    ax3 = axes3[i, 2]
    valid_mask = (targets_denorm_flat[:, i] > 0) & (predictions_denorm_flat[:, i] > 0)
    x_act = targets_denorm_flat[valid_mask, i]
    x_pred = predictions_denorm_flat[valid_mask, i]

    if len(x_act) > 0:
      combined_data = np.concatenate((x_act, x_pred))
      bins = np.logspace(np.log10(combined_data.min()), np.log10(combined_data.max()), 75)
      ax3.hist(x_act, bins=bins, alpha=0.7, label='Actual')
      ax3.hist(x_pred, bins=bins, alpha=0.7, label='Predicted')
      ax3.set_xscale('log')

    ax3.set_title(f'Level {i+1} Coefficient Distribution')
    ax3.set_xlabel('Population')
    if i == 0: ax3.set_ylabel('Count')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_path_dist = os.path.join(config['training']['savedir'], f'inference_errors_histograms_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

# %%
print("Generating height-color-coded scatter plots...")

# To perform the height-based analysis, we need the z-index for each flattened point
num_pixels_in_test = test_mask.sum()
z_indices_flat = np.repeat(np.arange(new_nz), num_pixels_in_test)
z_grid_flat = np.repeat(zz_grid, num_pixels_in_test)

fig_scatter, axes_scatter = plt.subplots(3, 2, figsize=(12, 15), dpi=100, gridspec_kw={'hspace': 0.1, 'wspace': 0.15})
axes_scatter = axes_scatter.flatten()

for i in range(nlev):
    ax = axes_scatter[i]

    # Filter out zero/negative values for log scale
    valid_mask = (targets_denorm_flat[:, i] > 0) & (predictions_denorm_flat[:, i] > 0)
    x = targets_denorm_flat[valid_mask, i]
    y = predictions_denorm_flat[valid_mask, i]
    z_colors = z_grid_flat[valid_mask]

    if len(x) > 0:
        # Use a scatter plot with color mapped to z-index
        scatter = ax.scatter(x, y, c=z_colors, cmap='viridis', s=0.75, alpha=0.1, rasterized=True)
        last_scatter = scatter

        # np.corrcoef returns a matrix; we want the off-diagonal element [0, 1]
        pearson_r = np.corrcoef(x, y)[0, 1]
        r_squared = pearson_r ** 2
        
        # Add R^2 text to the plot (bottom right corner to avoid overlap)
        # transform=ax.transAxes ensures the coordinates (0.95, 0.05) are relative to the box, not data
        ax.text(0.95, 0.05, f'$R^2 = {r_squared:.5f}$', 
                transform=ax.transAxes, 
                fontsize=14, 
                horizontalalignment='right', 
                verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)) # Optional background for readability
        
        # Add the y=x line for reference
        lims = [np.min([x, y]), np.max([x, y])]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=1, label='y=x (Perfect Fit)')
        
        ax.set_xscale('log')
        ax.set_yscale('log')

    # ax.set_title(r'$Ca_{II}$'+f'(i={i+1})')
    if i in [4, 5]:
        ax.set_xlabel('Actual')
    if i in [0, 2, 4]:
        ax.set_ylabel('Predicted', fontsize=14)
    # ax.grid(True, linestyle='--', alpha=0.6)
    if i in [0]:
        ax.legend(loc='upper left', fontsize=12)

# --- Add Global Colorbar ---
if last_scatter is not None:
    # ax argument takes the list of axes to steal space from
    cbar = fig_scatter.colorbar(last_scatter, ax=axes_scatter.ravel().tolist(), pad=0.02, aspect=40, shrink=0.6)
    cbar.set_label('Height [Mm]', fontsize=14)
    cbar.solids.set(alpha=1)

save_path_dist = os.path.join(config['training']['savedir'], f'inference_scatter_height_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

# %%
print("Generating height-color-coded comprehensive analysis plots...")

# Setup Figure
fig_comp, axes_comp = plt.subplots(nlev, 3, figsize=(12, 15), dpi=100, sharex='col', 
                                   gridspec_kw={'hspace': 0.15, 'wspace': 0.1}) 

# Define height groups for color-coding
height_groups = {
    f'Bottom (z < 2) Mm': (z_indices_flat < 25),
    f'Middle (2 < z < 4) Mm': (z_indices_flat >= 25) & (z_indices_flat < 37),
    f'Top (z >= 4) Mm': (z_indices_flat >= 37)
}
colors = ['blue', 'green', 'red']

for i in range(nlev):
    # --- Row 1: Residuals on Original Scale ---
    ax1 = axes_comp[i, 0]
    all_residuals = residuals_denorm[:, i] * 100
    lower, upper = -50, 50 
    
    for color, (label, mask) in zip(colors, height_groups.items()):
        data = all_residuals[mask]
        ax1.hist(data, bins=100, range=(lower, upper), density=True, alpha=0.5, label=label, color=color)

    ax1.axvline(0, color='k', linestyle='--')
    if i == 0: ax1.set_title(f'Residuals (Original)', fontsize=14)
    if i == nlev-1: ax1.set_xlabel('Relative Error (%)', fontsize=14)
    
    # Labeling
    ax1.set_ylabel(r'$Ca_{II}$'+f' (i={i+1})', fontsize=14)
    if i == nlev - 1:
        ax1.set_ylabel(r'$Ca_{III}$', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    # if i == 0: leg1 = ax1.legend(framealpha=0.5, facecolor='white', loc='lower left', fontsize=10)

    # --- Row 2: Residuals on Normalized Scale ---
    ax2 = axes_comp[i, 1]
    all_norm_residuals = residuals_normalized[:, i] * 100
    lower, upper = -50, 50 

    for color, (label, mask) in zip(colors, height_groups.items()):
        data = all_norm_residuals[mask]
        ax2.hist(data, bins=100, range=(lower, upper), density=True, alpha=0.5, label=label, color=color)

    ax2.axvline(0, color='k', linestyle='--')
    if i == 0: ax2.set_title(f'Residuals (Normalized)', fontsize=14)
    if i == nlev-1: ax2.set_xlabel('Relative Error (%)', fontsize=14)
    
    ax2.grid(True, linestyle='--', alpha=0.5)
    if i == 0: leg2 = ax2.legend(framealpha=0.5, facecolor='white', loc='lower left', fontsize=10)
    
    # --- COMPACT FORMATTING FOR COLUMN 2 ---
    # Force scientific notation for small numbers (0.01 -> 1e-2)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    # Limit number of ticks to keep it compact
    ax2.locator_params(axis='y', nbins=5)
    # ---------------------------------------

    # --- Row 3: Distribution of Departure Coefficients ---
    ax3 = axes_comp[i, 2]
    valid_mask = (targets_denorm_flat[:, i] > 0)
    
    if np.any(valid_mask):
        bins = np.logspace(np.log10(targets_denorm_flat[valid_mask, i].min()), 
                           np.log10(targets_denorm_flat[:, i].max()), 75)
        
        for color, (label, mask) in zip(colors, height_groups.items()):
            ax3.hist(targets_denorm_flat[valid_mask & mask, i], bins=bins, 
                     alpha=0.6, label=f'Actual ({label})', 
                     histtype='step', linewidth=2, color=color)
            ax3.hist(predictions_denorm_flat[valid_mask & mask, i], bins=bins,  label=f'Predicted ({label})',
                     alpha=0.6, histtype='stepfilled', hatch='//', edgecolor=color, facecolor='none')

    if i == 0: ax3.set_title(f'Population Distribution', fontsize=14)
    if i == nlev-1: ax3.set_xlabel('Population', fontsize=14)
    
    ax3.set_xscale('log')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # --- COMPACT FORMATTING FOR COLUMN 3 ---
    # Force scientific notation for large numbers (10000 -> 1e4)
    # Note: If yscale was 'log', this is handled differently, but for linear y-scale:
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    # Limit number of ticks to keep it compact
    ax3.locator_params(axis='y', nbins=5)
    # ---------------------------------------

plt.tight_layout()
save_path_dist = os.path.join(config['training']['savedir'], f'inference_errors_hist_height_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

