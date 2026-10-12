# %%
import numpy as np
import torch, os, sys, json
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
from vertical_grids import training_grid
from provenance import make_run_dir, write_meta

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
# CPU unless a device is given as 4th argument (e.g. 'cuda:0' with CUDA_VISIBLE_DEVICES=<a GPU nobody else uses>)
device_name = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
cuda_available = device_name.startswith('cuda')
device = torch.device(device_name)
print(f"Using device: {device}\n")

# %%
# The parameters to set manually: checkpoint, inference stride (None = the training max_stride) and split; all can be
# given positionally:  python gnn_inference.py <checkpoint> <stride> <split> <device> [zero_features] [grid]
checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else '/dat/andreuva/gpu/graphnet/graphnet_nlte/checkpoints/v2_rung4_rotation_seed777/2026.10.08-15:09:03_best.pth'
stride = int(sys.argv[2]) if len(sys.argv) > 2 else None
split = sys.argv[3] if len(sys.argv) > 3 else 'test'          # held-out columns only (audit E4); 'full' and 'train' still possible
# ablation: expanded input labels set to zero after normalization, e.g. 'velx,vely' (what an inversion cannot provide); '' = none
zero_features = [s for s in (sys.argv[5] if len(sys.argv) > 5 else '').split(',') if s]
grid_arg = sys.argv[6] if len(sys.argv) > 6 else ''            # evaluation grid ('index' | 'two_zone'); '' = the run's own grid
guard = 32                                                     # px next to the training region excluded from the metrics (E4)

# Load the entire checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Extract the configuration object used for training
config = checkpoint['config']
datadir = config['data']['datadir']
nx, ny, nz = config['data']['nx'], config['data']['ny'], config['data']['nz_orig']
nlev = config['data']['nlev']

log_offset = config['normalization']['log_offset']
train_stride = config['dataset'].get('max_stride', 1)
stride = stride or train_stride
interp_log = config['dataset']['interp_log']
features_labels_simple = list(checkpoint['meta']['features_labels'])      # the input quantities of the run
print(f"Inputs {features_labels_simple}, grid {config['dataset'].get('grid', 'index')}, log interpolation {interp_log}")

print(f"Successfully loaded configuration from checkpoint {checkpoint_path}.")

# %%
# ---- memory–mapped arrays ----
FILE_OF = {'vel': ('AR_385_veloc.dat', 3), 'temp': ('AR_385_temp.dat', 1), 'n_e': ('AR_385_ne.dat', 1), 'rho': ('AR_385_mass.dat', 1),
           'press': ('AR_385_press.dat', 1), 'n_h': ('AR_385_nh.dat', 1), 'n_p': ('AR_385_np.dat', 1)}
pops_orig = np.memmap(f'{datadir}/AR_385_CaII_5L_pops.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, nlev))
raw = {l: np.memmap(f'{datadir}/{FILE_OF[l][0]}', dtype='<f4', mode='r', shape=(nz, ny, nx, FILE_OF[l][1])) for l in features_labels_simple}
for arr in [pops_orig] + list(raw.values()):
    assert os.path.getsize(arr.filename) == arr.size * 4, f'{arr.filename}: file size does not match the declared shape'

# %%
print(f"Original data shape: {pops_orig.shape}")
print("Creating new Z grid")

z, y, x = (np.arange(d) for d in (nz, ny, nx))
geometry_grid = np.load(config["data"]["grid_file"])["z"]              # native level heights [Mm]
grid_name = grid_arg or config['dataset'].get('grid', 'index')
zz_grid, new_z, _ = training_grid(config, geometry_grid, grid_name)    # the training grid of the run, or the grid asked for
new_nz = len(new_z)
new_y, new_x = (np.linspace(0, d - 1, new_d) for d, new_d in zip((ny, nx), (ny, nx)))
new_zv, new_yv, new_xv = np.meshgrid(new_z, new_y, new_x, indexing='ij', sparse=True)

# %%
# %%
print(f"Training grid: {zz_grid.shape} levels.")
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
def to_grid(a, log=False):           # as in gnn_train.py: log10 for the scalars and populations when the run used interp_log
    if log:
        return 10.0 ** interpn((z, y, x), np.log10(np.asarray(a, dtype=np.float64)), new_points)
    return interpn((z, y, x), a, new_points)
pops = to_grid(pops_orig, interp_log)
features_data = [to_grid(raw[l], interp_log and l != 'vel') for l in features_labels_simple]
print("Interpolation complete.")
print(f"New shape: {pops.shape}")

# %%
# ---- Normalization ----
# IMPORTANT: Use the normalization parameters saved from the training run
feature_norm_params = checkpoint['feature_norm_params']
pop_norm_params = checkpoint['normalization_params']

features_labels = [f'{l}{c}' for l in features_labels_simple for c in (['x', 'y', 'z'] if l == 'vel' else [''])]

normalized_features = normalize_features(features_data, features_labels_simple, feature_norm_params)[0]   # the saved statistics of the training run
for zf in zero_features:               # ablation (zero is zero: the velocities are not centred)
    assert zf in features_labels, (zf, features_labels)
    l = next(l for l in features_labels_simple if zf.startswith(l))
    normalized_features[features_labels_simple.index(l)][..., {'x': 0, 'y': 1, 'z': 2}.get(zf[len(l):], 0)] = 0.0
    print(f"Ablation: input '{zf}' set to zero")
pops_normalized, _ = normalize_pops(pops, log_offset)

# ---- Create Test Dataset ----
# All parameters are now from the loaded config object
pixel_Mm = float(np.median(np.diff(np.load(config['data']['grid_file'])['x'])))     # horizontal pixel size [Mm]
dataset_params = {
    'list_X': normalized_features,
    'list_Y': [pops_normalized],
    'xdim': config['dataset']['x_range_graph'],
    'ydim': config['dataset']['y_range_graph'],
    'z_levels': zz_grid,
    'pixel_Mm': pixel_Mm,
    'seed': config['system']['seed'],
    'train_ratio': config['dataset']['train_ratio'],
    'epoch_size_fraction': 1.0,  # Use full dataset for testing
    'max_stride': stride,
    'random_stride': False,
    'split_stride': train_stride,     # the test region is the one of the training run, whatever the inference stride (E4/E5)
    'edge_rule': config['dataset'].get('edge_rule', 'index'),
    'radius_neighbors': config['dataset'].get('radius_neighbors', 6.01),
    'stencil_radius': config['dataset'].get('stencil_radius', 4.01),
    'edge_reach_Mm': config['dataset'].get('edge_reach_Mm', 0.5),
    'central_reach_Mm': config['dataset'].get('central_reach_Mm'),
    'split': split
}

datast_test = EfficientDataset(**dataset_params)
run_dir = make_run_dir(__file__)                                # results/gnn_inference/<stamp>/ (never overwrites)
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
pop_norm_params['totals'] = pop_norm_params['totals'][:, datast_test.y0:datast_test.y1, datast_test.x0:datast_test.x1]

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

# metric columns: the held-out columns at least `guard` px away from the training region (E4)
metric_mask = test_mask.copy()
if split == 'test':
    metric_mask[:guard, :] = False
    metric_mask[:, :guard] = False
predictions_denorm_flat = predictions_denorm[:, metric_mask, :].reshape(-1, nlev)
targets_denorm_flat = targets_denorm[:, metric_mask, :].reshape(-1, nlev)

residuals_denorm = (predictions_denorm_flat - targets_denorm_flat) / (targets_denorm_flat + 1e-12)
mae_denorm = np.mean(np.abs(residuals_denorm), axis=0)
rmse_denorm = np.sqrt(np.mean(residuals_denorm**2, axis=0))

# --- Error in dex: e = log10(n_pred / n_target), per level and height band (plan D-2); the normalized relative
# residual of the old version is meaningless (targets near 0) and is dropped ---
print("\nCalculating metrics in dex...")
with np.errstate(divide='ignore', invalid='ignore'):
    e_dex = np.log10(predictions_denorm_flat) - np.log10(targets_denorm_flat)
e_dex[~np.isfinite(e_dex)] = np.nan
num_pixels_in_test = metric_mask.sum()
z_grid_flat = np.repeat(zz_grid, num_pixels_in_test)                  # arrays are level-major: (nz * n_columns, nlev)
bands = [(-np.inf, 0.0, 'z<0'), (0.0, 0.5, '0-0.5'), (0.5, 1.0, '0.5-1'), (1.0, 1.5, '1-1.5'), (1.5, 2.2, '1.5-2.2'), (2.2, 3.5, '2.2-3.5'), (3.5, np.inf, '>3.5')]
level_names = ['4s', '3d3/2', '3d5/2', '4p1/2', '4p3/2', 'CaIII']
rms = lambda v: float(np.sqrt(np.nanmean(v**2)))
f_caII = 1.0 - targets_denorm_flat[:, -1] / targets_denorm_flat.sum(1)
w_rel = np.clip(f_caII / 1e-2, 0.0, 1.0)[:, None]                     # Ca II relevance weight (plan D-6)
metrics = {'region': {'x0': int(datast_test.x0), 'x1': int(datast_test.x1), 'y0': int(datast_test.y0), 'y1': int(datast_test.y1),
                      'guard': guard, 'n_columns_test': int(test_mask.sum()), 'n_columns_metric': int(metric_mask.sum())},
           'all': {'rms': rms(e_dex), 'median_abs': float(np.nanmedian(np.abs(e_dex))), 'p95_abs': float(np.nanpercentile(np.abs(e_dex), 95))},
           'relevance_weighted_rms': float(np.sqrt(np.nansum(w_rel * e_dex**2) / np.nansum(w_rel * np.isfinite(e_dex)))),
           'per_level': {level_names[l]: {'rms': rms(e_dex[:, l]), 'bias': float(np.nanmean(e_dex[:, l]))} for l in range(nlev)},
           'per_band': {name: {'rms': rms(e_dex[(z_grid_flat >= lo) & (z_grid_flat < hi)])} for lo, hi, name in bands}}
# the audit's number (T1/E5): all levels on the absolute columns [358, 484)^2, no guard
a0, a1, b0, b1 = 358 - datast_test.x0, 484 - datast_test.x0, 358 - datast_test.y0, 484 - datast_test.y0
if a0 >= 0 and b0 >= 0 and a1 <= nx_size and b1 <= ny_size:
    with np.errstate(divide='ignore', invalid='ignore'):
        e_audit = np.log10(predictions_denorm[:, b0:b1, a0:a1, :]) - np.log10(targets_denorm[:, b0:b1, a0:a1, :])
    metrics['audit_region_358_484_rms_all'] = rms(e_audit[np.isfinite(e_audit)])
print(f"\nHeld-out region x,y in [{datast_test.x0},{datast_test.x1}) minus {guard} px guard: {metric_mask.sum()} columns")
print(f"RMS log10(n_pred/n_target) all levels: {metrics['all']['rms']:.4f} dex | Ca II-relevance-weighted {metrics['relevance_weighted_rms']:.4f} dex")
print('per level: ' + '  '.join(f"{k} {v['rms']:.4f}" for k, v in metrics['per_level'].items()))
print('per band : ' + '  '.join(f"{k} {v['rms']:.4f}" for k, v in metrics['per_band'].items()))
if 'audit_region_358_484_rms_all' in metrics:
    print(f"audit-compatible RMS on [358,484)^2, no guard: {metrics['audit_region_358_484_rms_all']:.4f} dex")
print(f"\nMAE on original populations: {mae_denorm}")
print(f"RMSE on original populations: {rmse_denorm}")

# %%
print(f"predictions_denorm.shape: {predictions_denorm.shape}, targets_denorm.shape: {targets_denorm.shape}")

# %%
# Save both populations for further synthesis
tag = f'stride_{dataset_params["max_stride"]}_{dataset_params["split"]}'
np.save(os.path.join(run_dir, f'predictions_patch_{tag}.npy'), predictions_denorm)
np.save(os.path.join(run_dir, f'targets_patch_{tag}.npy'), targets_denorm)
np.save(os.path.join(run_dir, 'metric_mask.npy'), metric_mask)
np.save(os.path.join(run_dir, 'test_mask.npy'), test_mask)
np.save(os.path.join(run_dir, 'zz_grid.npy'), zz_grid)
with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=1)
write_meta(run_dir, checkpoint=os.path.abspath(checkpoint_path), checkpoint_epoch=int(checkpoint['epoch']), stride=stride,
           train_stride=train_stride, split=split, split_stride=train_stride, guard=guard, x0=int(datast_test.x0), x1=int(datast_test.x1),
           y0=int(datast_test.y0), y1=int(datast_test.y1), nx_size=int(nx_size), ny_size=int(ny_size),
           totals_source='checkpoint normalization_params[totals] (Bifrost sum of levels), sliced [:, y0:y1, x0:x1]',
           features_labels=features_labels_simple, features_labels_expanded=features_labels, datadir=datadir,
           grid=grid_name, train_grid=config['dataset'].get('grid', 'index'), interp_log=interp_log, zero_features=zero_features,
           z_grid_Mm=zz_grid, new_z_indices=new_z, device=device_name,
           files={'predictions': f'predictions_patch_{tag}.npy', 'targets': f'targets_patch_{tag}.npy'})
print(f"Saved arrays, metrics.json and meta.json to {run_dir}")

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

save_path_dist = os.path.join(run_dir, f'inference_maps_{tag}.png')
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
save_path_dist = os.path.join(run_dir, f'inference_scatter_{tag}.png')
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
save_path_dist = os.path.join(run_dir, f'inference_scatter_normalized_{tag}.png')
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
    ax2.set_title(f'Level {i+1} error (dex)')
    ax2.set_xlabel('log10(pred / target)')
    # ax2.set_yscale('log')
    if i == 0: ax2.set_ylabel('Density')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.text(0.05, 0.95, f'RMS: {rms(data):.4f} dex\nmedian |e|: {np.nanmedian(np.abs(data)):.4f}', 
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
save_path_dist = os.path.join(run_dir, f'inference_errors_histograms_{tag}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

# %%
print("Generating height-color-coded scatter plots...")

# To perform the height-based analysis, we need the z-index for each flattened point (level-major flattening)
z_indices_flat = np.repeat(np.arange(new_nz), num_pixels_in_test)

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

        # R^2 in log space (a linear R^2 is dominated by the deepest, densest nodes)
        lt, lp = np.log10(x), np.log10(y)
        r_squared = 1.0 - np.sum((lp - lt)**2) / np.sum((lt - lt.mean())**2)
        
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

save_path_dist = os.path.join(run_dir, f'inference_scatter_height_{tag}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

# %%
print("Generating height-color-coded comprehensive analysis plots...")

# Setup Figure
fig_comp, axes_comp = plt.subplots(nlev, 3, figsize=(12, 15), dpi=100, sharex='col', 
                                   gridspec_kw={'hspace': 0.15, 'wspace': 0.1}) 

# Define height groups for color-coding
height_groups = {                      # boundaries taken from the grid heights (E6: indices 25/37 are 1.78/3.66 Mm)
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
save_path_dist = os.path.join(run_dir, f'inference_errors_hist_height_{tag}.png')
plt.savefig(save_path_dist, dpi=300)
# plt.show()
plt.close()

