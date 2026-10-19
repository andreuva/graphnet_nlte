# %%
import numpy as np
import torch, os, sys
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import multiprocessing

from graphnet import EncodeProcessDecode
from Dataset import EfficientDataset
from normalization import denormalize_pops, normalize_features_with_params, normalize_pops, lte_log_fractions
from vertical_grids import fixed_grid, to_levels, ZTAU51_BIFROST

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
# Set device: CPU unless a device is given as 4th argument (e.g. 'cuda:0')
device = torch.device(sys.argv[4] if len(sys.argv) > 4 else 'cpu')
cuda_available = device.type == 'cuda'
print(f"Using device: {device}\n")

# %%
# The parameters, all positional:
#   python gnn_inference.py <checkpoint> [stride] [test|full] [device] [grid]
# stride: hops of the graphs in pixels (default: the training max_stride); 'test' = the held-out corner of the training
# run (default), 'full' = every column whose stencil fits in the cube; grid: the vertical grid (vertical_grids.fixed_grid)
checkpoint_path = sys.argv[1]

# Load the entire checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Extract the configuration object used for training
config = checkpoint['config']
datadir = config['data']['datadir']
nx, ny, nz = config['data']['nx'], config['data']['ny'], config['data']['nz_orig']
nlev = config['data']['nlev']

train_stride = config['dataset']['max_stride']
stride = int(sys.argv[2]) if len(sys.argv) > 2 else train_stride
split = sys.argv[3] if len(sys.argv) > 3 else 'test'
grid_name = sys.argv[5] if len(sys.argv) > 5 else 'dense:60'
log_offset = config['normalization']['log_offset']
guard = config['training']['valid_guard']      # px next to the training region excluded from the metrics

print(f"Successfully loaded configuration from checkpoint {checkpoint_path}.")

# %%
# ---- memory–mapped arrays ----
pops_orig = np.memmap(f'{datadir}/AR_385_CaII_5L_pops.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, nlev))
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

# Heights above tau500 = 1 [Mm] of the native levels and of the evaluation grid. The data stay on the native levels:
# the dataset interpolates every node linearly in height between the two native levels around it
geometry_file = config["data"]["grid_file"]
z_native = np.load(geometry_file)["z"].astype(np.float64) - ZTAU51_BIFROST
zz_grid = fixed_grid(grid_name, z_native[0], z_native[-1])
new_nz = len(zz_grid)

# %%
print(f"Loaded geometry grid from {geometry_file} with shape {zz_grid.shape}.")
plt.figure(figsize=(20, 2))
plt.plot(zz_grid, zz_grid * 0, '|', markersize=20, label='Interpolated grid')
plt.plot(z_native, z_native * 0, '|', label='Original grid')
plt.xlabel('Height above tau500 = 1 (Mm)')
plt.legend()
plt.title('Comparison of Original and Interpolated Geometry Grids')
# plt.show()
plt.close()

# %%
# ---- Normalization ----
# IMPORTANT: Use the normalization parameters saved from the training run
feature_norm_params = checkpoint['feature_norm_params']

features_labels = ['vx', 'vy', 'vz', 'temp', 'rho', 'press',
                   'lte_4s', 'lte_3d32', 'lte_3d52', 'lte_4p12', 'lte_4p32', 'lte_CaIII']
features_labels_simple = ['vel', 'temp', 'rho', 'press', 'lte']
assert features_labels_simple == list(checkpoint['features_labels']), checkpoint['features_labels']

# LTE fractions of the Ca levels (an input) from the temperature and the electron density, on the native levels
lte_orig = lte_log_fractions(temp_orig, n_e_orig, log_offset)
features_data = [vel_orig, temp_orig, rho_orig, press_orig, lte_orig]

# The native cube normalized with the saved statistics of the training run (no re-fit)
normalized_features = normalize_features_with_params(features_data, features_labels_simple, feature_norm_params)
pops_normalized, pop_norm_params = normalize_pops(pops_orig, log_offset=log_offset)
del features_data, lte_orig

# ---- Create Test Dataset ----
# All parameters are now from the loaded config object
dataset_params = {
    'list_X': normalized_features,
    'list_Y': [pops_normalized],
    'z_native': z_native,
    'pixel_Mm': config['data']['pixel_Mm'],
    'xdim': config['dataset']['x_range_graph'],
    'ydim': config['dataset']['y_range_graph'],
    'z_levels': zz_grid,
    'train_ratio': config['dataset']['train_ratio'],
    'max_stride': stride,
    'split_stride': train_stride,  # the held-out corner of the training run, whatever the inference stride
    'stencil_radius': config['dataset']['stencil_radius'],
    'edge_radius_Mm': config['dataset']['edge_radius_Mm'],
    'central_reach_Mm': config['dataset']['central_reach_Mm'],
    'log_offset': log_offset,
    'split': split
}

datast_test = EfficientDataset(**dataset_params)
# loader_test = DataLoader(datast_test, batch_size=config['training']['batch_size']*6, shuffle=False)

num_workers = min(12, multiprocessing.cpu_count())
# graphs per batch scaled with the edges per graph (48 graphs of the dense:60 grid, ~139k edges each; finer grids have more)
edges_per_graph = datast_test.template['edge_index'].shape[1]
inference_batch = max(1, min(config['training']['batch_size'] * 6, int(config['training']['batch_size'] * 6 * 140000 / edges_per_graph)))
print(f"Inference batch: {inference_batch} graphs of {edges_per_graph} edges")
loader_test = DataLoader(
    datast_test, 
    batch_size=inference_batch,
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
        # keep the central column of every graph only (its new_nz nodes, level by level)
        central_mask = data.central_mask.to(device)
        all_predictions.append(out[central_mask].cpu().numpy())
        all_targets.append(target[central_mask].cpu().numpy())

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

# only the central column of every sample was kept, in the order of sample_centers
nodes_per_sample = new_nz

# Create empty arrays for spatial data
predictions_spatial = np.zeros((new_nz, ny_size, nx_size, nlev))
targets_spatial = np.zeros((new_nz, ny_size, nx_size, nlev))

start_idx = 0
num_test_samples = len(datast_test)

for i in tqdm(range(num_test_samples), desc="Reconstructing"):
    ix, iy = datast_test.sample_centers[i]
    
    end_idx = start_idx + nodes_per_sample
    pred_column = predictions_flat[start_idx:end_idx]
    target_column = targets_flat[start_idx:end_idx]

    predictions_spatial[:, iy - datast_test.y0, ix - datast_test.x0, :] = pred_column
    targets_spatial[:, iy - datast_test.y0, ix - datast_test.x0, :] = target_column
        
    start_idx = end_idx

# Flat numpy arrays of the central columns for analysis
predictions_norm_flat = predictions_flat
targets_norm_flat = targets_flat

# %%
predictions_spatial.shape
# Totals of the predicted region: the native totals interpolated in log10 to the evaluation grid (axes z, y, x)
pop_norm_params['totals'] = to_levels(pop_norm_params['totals'][:, datast_test.y0:datast_test.y1, datast_test.x0:datast_test.x1],
                                      z_native, zz_grid, log=True)

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

# Metric columns: the predicted columns at least `guard` px away from the training region (test split) and inside the
# region of the training stride (x, y < nx - xdim * train_stride), so that the test split scores the same columns at
# every stride up to the training stride
metric_mask = test_mask.copy()
if split == 'test':
    metric_mask[:guard, :] = False
    metric_mask[:, :guard] = False
metric_mask[ny - ydim * train_stride - datast_test.y0:, :] = False
metric_mask[:, nx - xdim * train_stride - datast_test.x0:] = False
print(f"\nMetric columns: {metric_mask.sum()} of the {test_mask.sum()} predicted (stride {stride}, split stride {train_stride}, guard {guard if split == 'test' else 0} px)")

predictions_denorm_flat = predictions_denorm[:, metric_mask, :].reshape(-1, nlev)
targets_denorm_flat = targets_denorm[:, metric_mask, :].reshape(-1, nlev)

residuals_denorm = (predictions_denorm_flat - targets_denorm_flat) / (targets_denorm_flat + 1e-12)
mae_denorm = np.mean(np.abs(residuals_denorm), axis=0)
rmse_denorm = np.sqrt(np.mean(residuals_denorm**2, axis=0))

# --- Errors in dex, e = log10(n_pred / n_target): all nodes, weighted by the Ca II relevance (nodes with a Ca II
# fraction below 1e-2 weigh less), per level and per height band [Mm above tau500 = 1] ---
print("\nCalculating metrics in dex...")
with np.errstate(divide='ignore', invalid='ignore'):
    e_dex = np.log10(predictions_denorm_flat) - np.log10(targets_denorm_flat)
e_dex[~np.isfinite(e_dex)] = np.nan
num_pixels_in_test = metric_mask.sum()
z_grid_flat = np.repeat(zz_grid, num_pixels_in_test)          # the flat arrays are level-major
bands = [(-np.inf, 0.0, 'z<0'), (0.0, 0.5, '0-0.5'), (0.5, 1.0, '0.5-1'), (1.0, 1.5, '1-1.5'), (1.5, 2.2, '1.5-2.2'), (2.2, 3.5, '2.2-3.5'), (3.5, np.inf, '>3.5')]
level_names = ['4s', '3d3/2', '3d5/2', '4p1/2', '4p3/2', 'CaIII']
rms = lambda e: np.sqrt(np.nanmean(e**2))
weighted_rms = lambda e, w: np.sqrt(np.nansum(w * e**2) / np.nansum(w * np.isfinite(e)))
w_rel = np.clip((1.0 - targets_denorm_flat[:, -1] / targets_denorm_flat.sum(1)) / 1e-2, 0.0, 1.0)[:, None]
print(f"RMS log10(n_pred/n_target): {rms(e_dex):.4f} dex, Ca II-weighted {weighted_rms(e_dex, w_rel):.4f} dex")
print('per level: ' + '  '.join(f"{name} {rms(e_dex[:, l]):.4f} (bias {np.nanmean(e_dex[:, l]):+.4f})" for l, name in enumerate(level_names)))
print('per band : ' + '  '.join(f"{name} {rms(e_dex[(z_grid_flat >= lo) & (z_grid_flat < hi)]):.4f}" for lo, hi, name in bands))

# --- End-to-end error at the native levels: the log10 of the predictions (and of the targets on the grid, the
# interpolation floor of the grid: what a perfect model on this grid would still miss) interpolated linearly in height
# to the native levels and compared with the native populations. It does not depend on the grid ---
native = np.asarray(pops_orig[:, datast_test.y0:datast_test.y1, datast_test.x0:datast_test.x1][:, metric_mask], dtype=np.float64)
with np.errstate(divide='ignore', invalid='ignore'):
    e_native = to_levels(np.log10(predictions_denorm[:, metric_mask]), zz_grid, z_native) - np.log10(native)
    e_floor = to_levels(np.log10(targets_denorm[:, metric_mask]), zz_grid, z_native) - np.log10(native)
e_native[~np.isfinite(e_native)] = np.nan
e_floor[~np.isfinite(e_floor)] = np.nan
w_native = np.clip((1.0 - native[..., -1] / native.sum(-1)) / 1e-2, 0.0, 1.0)[..., None]
print(f"native levels ({nz}): RMS {rms(e_native):.4f} dex, Ca II-weighted {weighted_rms(e_native, w_native):.4f} dex, "
      f"interpolation floor of the grid {rms(e_floor):.4f} dex")
print('native per band : ' + '  '.join(f"{name} {rms(e_native[(z_native >= lo) & (z_native < hi)]):.4f} "
                                       f"(floor {rms(e_floor[(z_native >= lo) & (z_native < hi)]):.4f})" for lo, hi, name in bands))
print('native per level: ' + '  '.join(f"{name} {rms(e_native[..., l]):.4f}" for l, name in enumerate(level_names)))

print(f"\nMAE on original populations: {mae_denorm}")
print(f"RMSE on original populations: {rmse_denorm}")

# %%
print(f"predictions_denorm.shape: {predictions_denorm.shape}, targets_denorm.shape: {targets_denorm.shape}")

# %%
# Save both populations for further synthesis, with the region, the metric columns and the grid (inference.npz), in a
# folder next to the checkpoint
run_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), f"bifrost_stride{stride}_{split}_{grid_name.replace(':', '')}")
os.makedirs(run_dir, exist_ok=True)
np.save(os.path.join(run_dir, f'predictions_patch_stride_{dataset_params["max_stride"]}_{dataset_params["split"]}.npy'), predictions_denorm)
np.save(os.path.join(run_dir, f'targets_patch_stride_{dataset_params["max_stride"]}_{dataset_params["split"]}.npy'), targets_denorm)
np.savez(os.path.join(run_dir, 'inference.npz'), checkpoint=os.path.abspath(checkpoint_path), stride=stride, split=split,
         x0=datast_test.x0, x1=datast_test.x1, y0=datast_test.y0, y1=datast_test.y1, metric_mask=metric_mask, z_levels=zz_grid)
print(f"Saved to {run_dir}")

# %%
# 1. Configuration
z_slices = [int(np.argmin(np.abs(zz_grid - h))) for h in (0.0, 1.25, 3.1, 6.6)]   # map heights [Mm above tau500 = 1]
n_levels = nlev    # 6 levels
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

save_path_dist = os.path.join(run_dir, f'inference_maps_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
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
save_path_dist = os.path.join(run_dir, f'inference_scatter_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
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
save_path_dist = os.path.join(run_dir, f'inference_scatter_normalized_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
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

    # --- Row 2: Error in dex ---
    ax2 = axes3[i, 1]
    data = e_dex[:, i]
    lower, upper = -0.5, 0.5
    ax2.set_xlim(lower, upper)
    ax2.hist(data[np.isfinite(data)], bins=100, range=(lower, upper), density=True, alpha=0.75, color='orange')
    ax2.axvline(0, color='r', linestyle='--')
    ax2.set_title(f'Level {i+1} Error (dex)')
    ax2.set_xlabel('log10(pred / target)')
    # ax2.set_yscale('log')
    if i == 0: ax2.set_ylabel('Density')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.text(0.05, 0.95, f'MAE: {np.nanmean(np.abs(data)):.3f}\nRMSE: {rms(data):.3f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    # --- Row 3: Distribution of Populations (Original Scale) ---
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

    ax3.set_title(f'Level {i+1} Population Distribution')
    ax3.set_xlabel('Population')
    if i == 0: ax3.set_ylabel('Count')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_path_dist = os.path.join(run_dir, f'inference_errors_histograms_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

# %%
print("Generating height-color-coded scatter plots...")

# The height-based analysis uses the height of each flattened point (z_grid_flat, from the metrics cell)

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

        # Coefficient of determination of log10(pred) against log10(target) (a linear R^2 is dominated by the
        # deepest, densest nodes)
        pos = (x > 0) & (y > 0)
        log_x, log_y = np.log10(x[pos]), np.log10(y[pos])
        r_squared = 1.0 - np.sum((log_y - log_x) ** 2) / np.sum((log_x - log_x.mean()) ** 2)
        
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

save_path_dist = os.path.join(run_dir, f'inference_scatter_height_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
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
    f'Bottom (z < 2) Mm': (z_grid_flat < 2.0),
    f'Middle (2 < z < 4) Mm': (z_grid_flat >= 2.0) & (z_grid_flat < 4.0),
    f'Top (z >= 4) Mm': (z_grid_flat >= 4.0)
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

    # --- Row 2: Error in dex ---
    ax2 = axes_comp[i, 1]
    all_dex = e_dex[:, i]
    lower, upper = -0.5, 0.5 

    for color, (label, mask) in zip(colors, height_groups.items()):
        data = all_dex[mask]
        ax2.hist(data[np.isfinite(data)], bins=100, range=(lower, upper), density=True, alpha=0.5, label=label, color=color)

    ax2.axvline(0, color='k', linestyle='--')
    if i == 0: ax2.set_title(f'Error (dex)', fontsize=14)
    if i == nlev-1: ax2.set_xlabel('log10(pred / target)', fontsize=14)
    
    ax2.grid(True, linestyle='--', alpha=0.5)
    if i == 0: leg2 = ax2.legend(framealpha=0.5, facecolor='white', loc='lower left', fontsize=10)
    
    # --- COMPACT FORMATTING FOR COLUMN 2 ---
    # Force scientific notation for small numbers (0.01 -> 1e-2)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    # Limit number of ticks to keep it compact
    ax2.locator_params(axis='y', nbins=5)
    # ---------------------------------------

    # --- Row 3: Distribution of Populations ---
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
save_path_dist = os.path.join(run_dir, f'inference_errors_hist_height_s{dataset_params["max_stride"]}_{dataset_params["split"]}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

