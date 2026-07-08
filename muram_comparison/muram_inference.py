# %%
import sys, os
# Add parent directory and current directory to sys.path to find graphnet/Dataset modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))

import muram as mio  
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interpn
import matplotlib.ticker as ticker
import multiprocessing
from joblib import Parallel, delayed

# Import Lightweaver modules for LTE population calculation
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom, C_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, \
                                 Fe_atom, He_9_atom, MgII_atom, N_atom, Na_atom, S_atom

from graphnet import EncodeProcessDecode
from Dataset import EfficientDataset
from normalization import denormalize_pops, normalize_features_with_params

def compute_lte_pops_muram(temp, press, zz_grid, coords=None, n_jobs=64):
    nz, ny, nx = temp.shape
    depthScale_m = zz_grid[::-1] * 1e6
    
    if coords is None:
        # Full grid fallback
        y_indices = np.arange(ny)
        chunk_size = max(1, ny // n_jobs)
        y_chunks = [y_indices[i:i + chunk_size] for i in range(0, ny, chunk_size)]
        
        def process_y_chunk(chunk):
            aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                                    Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
            aSet.set_active('H', 'Ca')
            lte_pops_chunk = np.zeros((nz, len(chunk), nx, 6))
            for i, y_idx in enumerate(chunk):
                for x_idx in range(nx):
                    temp_atm = temp[:, y_idx, x_idx][::-1]
                    press_atm = press[:, y_idx, x_idx][::-1] / 10.0
                    atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, 
                                                  depthScale=depthScale_m,
                                                  convertScales=False,
                                                  temperature=temp_atm, 
                                                  Pgas=press_atm,
                                                  vturb=1e3 * np.ones_like(temp_atm),
                                                  vlos=np.zeros_like(temp_atm))
                    eqPops = aSet.compute_eq_pops(atmos)
                    lte_pops_chunk[:, i, x_idx, :] = eqPops.atomicPops['Ca'].n.T[::-1, :]
            return chunk, lte_pops_chunk

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_y_chunk)(chunk) for chunk in y_chunks
        )
        
        lte_pops = np.zeros((nz, ny, nx, 6))
        for chunk, lte_pops_chunk in results:
            lte_pops[:, chunk, :, :] = lte_pops_chunk
        return lte_pops
    else:
        # Specific coordinates
        y_coords, x_coords = coords
        num_coords = len(y_coords)
        if num_coords == 0:
            return np.zeros((nz, ny, nx, 6))
            
        chunk_size = max(1, num_coords // n_jobs)
        coord_chunks = [slice(i, min(i + chunk_size, num_coords)) for i in range(0, num_coords, chunk_size)]
        
        def process_coord_chunk(coord_slice):
            aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                                    Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
            aSet.set_active('H', 'Ca')
            chunk_y = y_coords[coord_slice]
            chunk_x = x_coords[coord_slice]
            num_chunk = len(chunk_y)
            lte_pops_chunk = np.zeros((nz, num_chunk, 6))
            for idx in range(num_chunk):
                y_idx = chunk_y[idx]
                x_idx = chunk_x[idx]
                temp_atm = temp[:, y_idx, x_idx][::-1]
                press_atm = press[:, y_idx, x_idx][::-1] / 10.0
                atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, 
                                              depthScale=depthScale_m,
                                              convertScales=False,
                                              temperature=temp_atm, 
                                              Pgas=press_atm,
                                              vturb=1e3 * np.ones_like(temp_atm),
                                              vlos=np.zeros_like(temp_atm))
                eqPops = aSet.compute_eq_pops(atmos)
                lte_pops_chunk[:, idx, :] = eqPops.atomicPops['Ca'].n.T[::-1, :]
            return coord_slice, lte_pops_chunk

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_coord_chunk)(c_slice) for c_slice in coord_chunks
        )
        
        lte_pops = np.zeros((nz, ny, nx, 6))
        for coord_slice, lte_pops_chunk in results:
            chunk_y = y_coords[coord_slice]
            chunk_x = x_coords[coord_slice]
            for idx in range(len(chunk_y)):
                lte_pops[:, chunk_y[idx], chunk_x[idx], :] = lte_pops_chunk[:, idx, :]
        return lte_pops

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
gpu = 2
cuda_available = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# %%
# The ONLY parameter you need to set manually is the path to the checkpoint.
checkpoint_path = '/dat/andreuva/gpu/graphnet/graphnet_nlte/checkpoints/beta_train/2026.06.18-09:54:12_best.pth'

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
feature_norm_type = 'log' if normalization_type == 'beta' else normalization_type
normalized_features = normalize_features_with_params(
    features_data, features_labels_simple, feature_norm_params, feature_norm_type
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

# Downsample sample centers based on stride for faster inference
stride = dataset_params.get("max_stride", 4)
datast_test.sample_centers = [
    (x, y) for (x, y) in datast_test.sample_centers 
    if (x - datast_test.x0) % stride == 0 and (y - datast_test.y0) % stride == 0
]
print(f"Downsampled sample centers using stride {stride}. Remaining samples: {len(datast_test.sample_centers)}")

# %%
# ---- Setup DataLoader ----
num_workers = min(12, multiprocessing.cpu_count())
loader_test = DataLoader(
    datast_test, 
    batch_size=128,
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

# Interpolate GNN predictions to fill the dense spatial grid
print("Interpolating GNN predictions to fill the dense spatial grid...")
import torch.nn.functional as F

# Extract the subgrid of non-zero predictions
subgrid = predictions_spatial[:, ::stride, ::stride, :]

# Interpolate to the full dense grid using PyTorch bilinear interpolation on CPU
subgrid_tensor = torch.from_numpy(subgrid).permute(0, 3, 1, 2)
dense_tensor = F.interpolate(subgrid_tensor, size=(ny_size, nx_size), mode='bilinear', align_corners=True)
predictions_spatial = dense_tensor.permute(0, 2, 3, 1).numpy()
print("Interpolation complete. Output shape:", predictions_spatial.shape)

# %%
# ---- Denormalization ----
print("Denormalizing predicted populations...")

is_beta = (normalization_type == 'beta')
if not is_beta:
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
else:
    # Pop 'pops_lte' to avoid shape mismatch during denormalization
    pop_norm_params.pop('pops_lte', None)

predictions_denorm = denormalize_pops(predictions_spatial, pop_norm_params, normalization_type)

if is_beta:
    print("Reconstructing physical populations from departure coefficients (beta)...")
    y_start, y_end = datast_test.y0, datast_test.y1
    x_start, x_end = datast_test.x0, datast_test.x1
    temp_cropped = temp[:, y_start:y_end, x_start:x_end]
    press_cropped = press[:, y_start:y_end, x_start:x_end]
    
    # Identify non-zero columns in sparse predictions before interpolation to optimize EoS calculations
    # Wait, y_indices and x_indices should be aligned with the stride grid
    non_zero_mask = np.zeros((ny_size, nx_size), dtype=bool)
    non_zero_mask[::stride, ::stride] = True
    y_indices, x_indices = np.where(non_zero_mask)
    num_coords = len(y_indices)
    
    print(f"Found {num_coords} stride columns out of {ny_size * nx_size} total columns (sparse ratio: {num_coords / (ny_size * nx_size) * 100:.2f}%).")
    
    # Compute LTE populations in parallel only at predicted coordinates
    n_jobs_to_use = min(48, multiprocessing.cpu_count())
    print(f"Running parallel EoS solver to compute LTE populations for {num_coords} active pixels (using {n_jobs_to_use} jobs)...")
    
    pops_lte_si = compute_lte_pops_muram(temp_cropped, press_cropped, zz_grid, 
                                         coords=(y_indices, x_indices), 
                                         n_jobs=n_jobs_to_use)
                                         
    # Interpolate LTE populations to fill the dense spatial grid
    print("Interpolating LTE populations to fill the dense spatial grid...")
    subgrid_lte = pops_lte_si[:, ::stride, ::stride, :]
    subgrid_lte_tensor = torch.from_numpy(subgrid_lte).permute(0, 3, 1, 2)
    dense_lte_tensor = F.interpolate(subgrid_lte_tensor, size=(ny_size, nx_size), mode='bilinear', align_corners=True)
    pops_lte_si = dense_lte_tensor.permute(0, 2, 3, 1).numpy()
    
    # Convert from SI (m^-3) to CGS (cm^-3) to match direct populations
    pops_lte_cgs = pops_lte_si * 1e-6
    # Clip departure coefficients (beta) to a physically realistic range to avoid numerical outliers
    predictions_denorm = np.clip(predictions_denorm, 1e-4, 1e4)
    
    # Apply double-sided temperature-dependent transition to LTE (beta=1.0)
    t_threshold_low_start = 5000.0
    t_threshold_low_end = 8000.0
    t_threshold_high_start = 12000.0
    t_threshold_high_end = 20000.0
    
    temp_expanded = temp_cropped[..., np.newaxis] # Shape (nz, ny, nx, 1)
    
    # Compute weight for low-temperature: 0.0 at 5000 K, 1.0 at 8000 K
    w_low = (temp_expanded - t_threshold_low_start) / (t_threshold_low_end - t_threshold_low_start)
    w_low = np.clip(w_low, 0.0, 1.0)
    
    # Compute weight for high-temperature: 1.0 at 12000 K, 0.0 at 20000 K
    w_high = (temp_expanded - t_threshold_high_start) / (t_threshold_high_end - t_threshold_high_start)
    w_high = 1.0 - np.clip(w_high, 0.0, 1.0)
    
    # Combined weight
    w = w_low * w_high
    
    # Smoothly transition beta to 1.0 outside the chromospheric range
    predictions_denorm = w * predictions_denorm + (1.0 - w) * 1.0
    
    predictions_denorm = predictions_denorm * pops_lte_cgs
    
    # Enforce conservation of total Ca population to prevent non-physical outliers
    sum_lte = np.sum(pops_lte_cgs, axis=-1, keepdims=True)
    sum_gnn = np.sum(predictions_denorm, axis=-1, keepdims=True)
    predictions_denorm = predictions_denorm * (sum_lte / (sum_gnn + 1e-30))
    
    print("Physical populations reconstruction complete.")

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