# %%
import muram as mio  

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
from normalization import denormalize_pops, normalize_features_with_params

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
gpu = 1
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
nlev = config['data']['nlev']

log_offset = config['normalization']['log_offset']
normalization_type = config['normalization'].get('type', 'log')

print(f"Successfully loaded configuration from checkpoint {checkpoint_path}.")

# %%
# For the SSD data with ch and corona - you can use this path:
pathsource = '/dat/milic/MURaM_enhanced_network/'

# All the other quantities are self-explanatory, except for B units and orientation of the axis. See below after loading the data.
path3D = pathsource
# There is only one snapshow and its number is 499000
snap_number = 499000


# %%
# If you used the first one, it's FULL snaps - so use MuramSnap
cube = mio.MuramSnap(pathsource, snap_number)
# But for the other one, they are so called SUBSNAPS - so you use MuramSubSnap
#cube = mio.MuramSubSnap(path3D, 0)
cube.available

# %%
cube.Temp.shape
# This means NZ, NX, NY - so the axis are transposed 
# you need to transpose them back as T = cube.Temp.transpose(1,2,0)

# BUT the same way these are transposed, meaning of vector components is also tranposed:
# So Bx in the cube is Bz in the real life, etc.
# And same for V_x

# %%
# The easiest way to look for log tau = 0 layer is: 
tau_mean = np.mean(cube.tau, axis=(1,2))
tau_zero_index = np.argmin(np.abs(tau_mean - 1))
print(f"Index of log tau = 0 layer: {tau_zero_index}")

# %%
# construct the grid knowing that this is enhanced network simulation from Przybylski 2022 (or something like that), and what you need to have for the use is 
# dx = dy = 24 km 
# dz = 20 km
dx, dy, dz = 24/1e3, 24/1e3, 20/1e3

nz, nx, ny  = cube.Temp.shape
z, x, y = (np.arange(d) for d in (nz, nx, ny))
z_geom, x_geom, y_geom = z*dz, x*dx, y*dy
muram_z_grid = z_geom-z_geom[tau_zero_index]

# %%
datadir_bifrost = config['data']['datadir']
nx_bifrost, ny_bifrost, nz_bifrost = config['data']['nx'], config['data']['ny'], config['data']['nz_orig']

nz_linear = config['dataset']['nz_linear']      # number of linear points in the grid
nz_log = config['dataset']['nz_log']            # number of log points in the grid
new_nz = nz_linear + nz_log
logspace_fraction = config['dataset']['logspace_fraction']  # fraction of the grid in logspace

# %%
print(f"Original data shape: {cube.rho.shape}")
print("Creating new Z grid aligned with Bifrost")

bifrost_geometry_file = config["data"]["grid_file"]
bifrost_z_grid = np.load(bifrost_geometry_file)["z"]

# Compute zz_grid_bifrost
z_b = np.arange(nz_bifrost) # Wait, is nz_bifrost or nz_b used? In config it is nz_bifrost
new_z_b_log = np.concatenate([
    np.linspace(0, nz_bifrost * logspace_fraction, nz_linear, endpoint=False),
    np.logspace(np.log10(nz_bifrost * logspace_fraction), np.log10(nz_bifrost - 1), nz_log)
])
new_z_b = np.clip(new_z_b_log, 0, nz_bifrost - 1)
zz_grid_bifrost = np.interp(new_z_b, z_b, bifrost_z_grid)

# Map zz_grid_bifrost to fractional indices in MURaM's vertical coordinate system
new_z = np.interp(zz_grid_bifrost, muram_z_grid, z)
new_z = np.clip(new_z, 0, nz - 1)

new_y, new_x = (np.linspace(0, d - 1, new_d) for d, new_d in zip((ny, nx), (ny, nx)))
new_zv, new_yv, new_xv = np.meshgrid(new_z, new_y, new_x, indexing='ij', sparse=True)

# %%
zz_grid = np.interp(new_z, z, muram_z_grid)
zz_grid.shape

# %%
# Save the grid as required by GNN positions loader
np.savez_compressed(f'muram_grid.npz', x=x_geom, y=y_geom, z=zz_grid)

# %%
print(f"Interpolating data to the new grid ({new_nz}, {ny}, {nx})...")
new_points = (new_zv, new_xv, new_yv)

temp = interpn((z, x, y), cube.Temp, new_points)
b_z = interpn((z, x, y), cube.Bx, new_points)
b_x = interpn((z, x, y), cube.By, new_points)
b_y = interpn((z, x, y), cube.Bz, new_points)
v_z = interpn((z, x, y), cube.vx , new_points)
v_x = interpn((z, x, y), cube.vy , new_points)
v_y = interpn((z, x, y), cube.vz , new_points)
rho = interpn((z, x, y), cube.rho, new_points)
n_e = interpn((z, x, y), cube.ne, new_points)
press = interpn((z, x, y), cube.Pres, new_points)
print("Interpolation complete.")

# %%
b_xyz = np.stack([b_x, b_y, b_z], axis=-1)
vel = np.stack([v_x, v_y, v_z], axis=-1)
print(f"B field shape stacked: {b_xyz.shape}")
print(f"Velocity shape stacked: {vel.shape}")

# Convert MURaM features to SI units to match Bifrost units before normalization
vel_si = vel / 100.0
b_xyz_si = b_xyz * (4.0 * np.pi) / 10000.0
temp_si = temp
n_e_si = n_e * 1e6
rho_si = rho * 1000.0
press_si = press / 10.0

# %%
# ---- Normalization ----
# IMPORTANT: Use the normalization parameters saved from the training run
feature_norm_params = checkpoint['feature_norm_params']
pop_norm_params = checkpoint['normalization_params']

features_labels_simple = ['vel', 'b', 'temp', 'n_e', 'rho', 'press']
features_data = [vel_si, b_xyz_si, temp_si, n_e_si, rho_si, press_si]

# Normalize features using the pre-computed training statistics to match the model's domain
normalized_features = normalize_features_with_params(
    features_data, features_labels_simple, feature_norm_params, normalization_type
)

# %%
# ---- Create Test Dataset ----
# All parameters are now from the loaded config object
dataset_params = {
    'list_X': normalized_features,
    'list_Y': [np.zeros((new_nz, nx, ny, 1))],
    'radius_neighbors': config['dataset']['radius_neighbors'],
    'xdim': config['dataset']['x_range_graph'],
    'ydim': config['dataset']['y_range_graph'],
    'fully_connected': config['dataset']['fully_connected'],
    'pos_file': 'muram_grid.npz' ,
    'seed': config['system']['seed'],
    'train_ratio': config['dataset']['train_ratio'],
    'nz_linear': config['dataset']['nz_linear'],
    'nz_log': config['dataset']['nz_log'],
    'logspace_fraction': config['dataset']['logspace_fraction'],
    'epoch_size_fraction': 1.0,  # Use full dataset for testing
    'max_stride': 4, # config["dataset"].get("max_stride", 2),
    'random_stride': False, #config["dataset"].get("random_stride", False),
    'split': 'full'
}

datast_test = EfficientDataset(**dataset_params)

# %%
# ---- Setup DataLoader ----
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
# ---- Instantiate Model ----
model_params = config['model']
model = EncodeProcessDecode(**model_params).to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
print("Model loaded successfully from checkpoint.")

# %%
# ---- Run Inference ----
all_predictions = []
print("Running GNN inference on the MURaM atmosphere...")
with torch.no_grad():
    for data in tqdm(loader_test):
        node, edge_attr, edge_index = data.x.to(device), data.edge_attr.to(device), data.edge_index.to(device)
        u, batch = data.u.to(device), data.batch.to(device)
        out = model(node, edge_attr, edge_index, u, batch)
        all_predictions.append(out.cpu().numpy())

predictions_flat = np.concatenate(all_predictions, axis=0)
print("GNN inference complete.")

# %%
# ---- Reconstruct Spatial Data Cubes ----
print("Reconstructing 3D spatial data cubes...")
xdim = config['dataset']['x_range_graph']
ydim = config['dataset']['y_range_graph']
nx_size = datast_test.x1 - datast_test.x0
ny_size = datast_test.y1 - datast_test.y0

nodes_per_sample = new_nz * (2 * xdim + 1) * (2 * ydim + 1)
predictions_spatial = np.zeros((new_nz, ny_size, nx_size, nlev))

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

    pred_column = preds_subvolume[central_mask]
    
    if pred_column.shape[0] == new_nz:
        predictions_spatial[:, iy - datast_test.y0, ix - datast_test.x0, :] = pred_column
        
    start_idx = end_idx

# %%
# ---- Denormalization ----
print("Denormalizing predicted populations...")
# Interpolate pop_norm_params['totals'] to match predictions_spatial shape
nz_t, ny_t, nx_t = pop_norm_params['totals'].shape
z_t = np.linspace(0, 1, nz_t)
y_t = np.linspace(0, 1, ny_t)
x_t = np.linspace(0, 1, nx_t)

z_new = np.linspace(0, 1, new_nz)
y_new = np.linspace(0, 1, ny_size)
x_new = np.linspace(0, 1, nx_size)

zv_new, yv_new, xv_new = np.meshgrid(z_new, y_new, x_new, indexing='ij', sparse=True)
totals_interpolated = interpn((z_t, y_t, x_t), pop_norm_params['totals'], (zv_new, yv_new, xv_new))
pop_norm_params['totals'] = totals_interpolated

predictions_denorm = denormalize_pops(predictions_spatial, pop_norm_params, normalization_type)

# %%
# ---- Save Outputs ----
savedir = '.' # config['training']['savedir']
if not os.path.exists(savedir):
    os.makedirs(savedir)

save_pred_path = os.path.join(savedir, f'muram_predictions_stride_{dataset_params["max_stride"]}_{dataset_params["split"]}.npy')
np.save(save_pred_path, predictions_denorm)
print(f"Saved synthesized absolute populations to {save_pred_path}")

# %%
# ---- Plot 1: Reconstructed Populations Slices ----
print("Generating reconstructed populations plots...")
z_slices = [new_nz // 6, new_nz // 3, new_nz // 2, 5 * new_nz // 6]
cols = len(z_slices)
rows = nlev

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True)

for lvl in range(nlev):
    for col_idx, z_idx in enumerate(z_slices):
        ax = axes[lvl, col_idx] if rows > 1 else axes[col_idx]
        pred_panel = predictions_denorm[z_idx, :, :, lvl]
        
        # Plot Heatmap (Log Scale)
        im = ax.imshow(np.log10(pred_panel + 1e-20), cmap='magma', origin='lower')
        
        # Labels and formatting
        if col_idx == 0:
            if lvl == nlev - 1:
                ax.set_ylabel(r"$Ca_{III}$", fontsize=18, fontweight='bold')
            else:
                ax.set_ylabel(r"$Ca_{II}\ (i=" + str(lvl) + r")$", fontsize=18, fontweight='bold')
        
        if lvl == 0:
            ax.set_title(f"z-idx: {z_idx} (z={zz_grid[z_idx]:1.2f} Mm)", fontsize=16)
            
        if lvl == nlev - 1:
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
            cbar.set_label(r'$log_{10}(\rho)$', fontsize=14)
            cbar.locator = ticker.MaxNLocator(nbins=4)
            cbar.update_ticks()
            
        ax.set_xticks([])
        ax.set_yticks([])

save_path_pops = os.path.join(savedir, f'muram_reconstructed_populations.png')
plt.savefig(save_path_pops, dpi=300)
plt.close()
print(f"Saved reconstructed populations plot to {save_path_pops}")