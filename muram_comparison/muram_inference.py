# %%
import muram as mio  

import numpy as np
import torch, os, sys
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import multiprocessing

# Add the repository folder to path (graphnet, Dataset, normalization, vertical_grids)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from graphnet import EncodeProcessDecode
from Dataset import EfficientDataset
from normalization import CA_TOTAL_PER_RHO, denormalize_pops, normalize_features_with_params, lte_log_fractions
from vertical_grids import fixed_grid, interp_weights, to_levels, ZTAU51_BIFROST

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
# Set device: the 4th argument (e.g. cuda:0, with CUDA_VISIBLE_DEVICES set to a free GPU), cpu by default
device = torch.device(sys.argv[4] if len(sys.argv) > 4 else 'cpu')
cuda_available = device.type == 'cuda'
print(f"Using device: {device}\n")

# %%
# The parameters, all positional (run from the repository folder, the paths of the config are relative to it):
#   python muram_comparison/muram_inference.py <checkpoint> [stride] [crop: full | y0,y1,x0,x1] [device] [grid]
# stride: hops of the stencil in MURaM pixels (default: the training footprint, see below); crop: the MURaM columns to
# predict; grid: the vertical grid (default dense:60)
checkpoint_path = sys.argv[1]
crop = sys.argv[3] if len(sys.argv) > 3 else 'full'
grid_name = sys.argv[5] if len(sys.argv) > 5 else 'dense:60'

# Load the entire checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Extract the configuration object used for training
config = checkpoint['config']
nlev = config['data']['nlev']

log_offset = config['normalization']['log_offset']

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
# tau500 = 1 is the median over the columns of the per-column crossing (fractional index ~47.6; the argmin of the mean
# tau gives 49, 28 km too high). muram_z_grid: heights [Mm] of all the MURaM levels above it
muram_z_grid, tau_zero_index = mio.muram_heights(cube)
print(f"Index of log tau = 0 layer: {tau_zero_index:.2f}")

# %%
# the grid spacing of the enhanced network simulation from the header (dx = dy = 23.4375 km, dz = 20 km), in Mm
dz, dx, dy = (np.asarray(cube.Temp.dX) / 1e5) / 1e3
assert dx == dy, (dx, dy)

nz, nx, ny  = cube.Temp.shape

# Hops of the stencil in MURaM pixels (2nd argument). Default: the footprint of the training stencil, max_stride Bifrost
# pixels (8 MURaM pixels, ~190 km)
stride = int(sys.argv[2]) if len(sys.argv) > 2 else round(config['dataset']['max_stride'] * config['data']['pixel_Mm'] / dx)

# %%
datadir_bifrost = config['data']['datadir']

# %%
print(f"Original data shape: {cube.rho.shape}")
print("Creating new Z grid aligned with Bifrost")

bifrost_geometry_file = config["data"]["grid_file"]
bifrost_z_grid = np.load(bifrost_geometry_file)["z"].astype(np.float64) - ZTAU51_BIFROST     # above tau500 = 1 [Mm]

# The grid on the height range of the training atmosphere: the same heights above tau500 = 1 as on Bifrost
zz_grid = fixed_grid(grid_name, bifrost_z_grid[0], bifrost_z_grid[-1])
new_nz = len(zz_grid)
assert muram_z_grid[0] <= zz_grid[0] and zz_grid[-1] <= muram_z_grid[-1], 'MURaM does not cover the heights of the training atmosphere'

# Every node is interpolated between the two native MURaM levels around it (Dataset.py): only those levels are read
k0, _ = interp_weights(muram_z_grid, zz_grid)
keep = np.unique(np.concatenate([k0, k0 + 1]))
print(f"Grid {grid_name}: {new_nz} levels from {zz_grid[0]:.3f} to {zz_grid[-1]:.3f} Mm, between {len(keep)} of the {nz} MURaM levels")

# %%
# The columns to predict ('full' or 'y0,y1,x0,x1'), padded with the periodic neighbours of the domain so that every
# column of the crop is the centre of a whole stencil
y0c, y1c, x0c, x1c = (0, ny, 0, nx) if crop == 'full' else [int(v) for v in crop.split(',')]
pad = config['dataset']['x_range_graph'] * stride
yidx = np.arange(y0c - pad, y1c + pad) % ny
xidx = np.arange(x0c - pad, x1c + pad) % nx
print(f"Predicting columns y in [{y0c},{y1c}), x in [{x0c},{x1c}) at stride {stride} ({stride * dx * 1e3:.0f} km hops), {pad} px periodic padding")

# %%
print(f"Reading the native levels of the padded crop ({len(keep)}, {len(yidx)}, {len(xidx)})...")

def native(q):                  # the kept levels of the padded crop, MURaM (z, x, y) -> (z, y, x) as Bifrost
    return np.asarray(q)[np.ix_(keep, xidx, yidx)].astype(np.float32).transpose(0, 2, 1)

temp = native(cube.Temp)
v_z = native(cube.vx)
v_x = native(cube.vy)
v_y = native(cube.vz)
rho = native(cube.rho)
n_e = native(cube.ne)
press = native(cube.Pres)
print("Reading complete.")

# %%
vel = np.stack([v_x, v_y, v_z], axis=-1)
print(f"Velocity shape stacked: {vel.shape}")

# Convert MURaM features to SI units to match Bifrost units before normalization
vel_si = vel / 100.0
temp_si = temp
n_e_si = n_e * 1e6
rho_si = rho * 1000.0
press_si = press / 10.0

# %%
# ---- Normalization ----
# IMPORTANT: Use the normalization parameters saved from the training run
feature_norm_params = checkpoint['feature_norm_params']

features_labels_simple = ['vel', 'temp', 'rho', 'press', 'lte']
assert features_labels_simple == list(checkpoint['features_labels']), checkpoint['features_labels']
# The LTE fractions of the Ca levels from MURaM's own temperature and electron density, on its native levels
lte_si = lte_log_fractions(temp_si[..., np.newaxis], n_e_si[..., np.newaxis], log_offset)
features_data = [vel_si, temp_si, rho_si, press_si, lte_si]

# Normalize features using the pre-computed training statistics to match the model's domain
normalized_features = normalize_features_with_params(
    features_data, features_labels_simple, feature_norm_params
)

# %%
# ---- Create Test Dataset ----
# All parameters are now from the loaded config object
dataset_params = {
    'list_X': normalized_features,
    'list_Y': None,
    'z_native': muram_z_grid[keep],
    'pixel_Mm': dx,
    'xdim': config['dataset']['x_range_graph'],
    'ydim': config['dataset']['y_range_graph'],
    'z_levels': zz_grid,
    'train_ratio': config['dataset']['train_ratio'],
    'max_stride': stride,
    'stencil_radius': config['dataset']['stencil_radius'],
    'edge_radius_Mm': config['dataset']['edge_radius_Mm'],
    'central_reach_Mm': config['dataset']['central_reach_Mm'],
    'split': 'full'
}

datast_test = EfficientDataset(**dataset_params)
# the centres are exactly the columns of the crop
assert (datast_test.x0, datast_test.x1, datast_test.y0, datast_test.y1) == (pad, pad + x1c - x0c, pad, pad + y1c - y0c)

# %%
# ---- Setup DataLoader ----
num_workers = min(12, multiprocessing.cpu_count())
# graphs per batch scaled with the edges per graph (48 graphs of ~139k edges on dense:60), so that denser grids fit in memory
edges_per_graph = datast_test.template['edge_index'].shape[1]
inference_batch = max(1, min(config['training']['batch_size'] * 6, int(config['training']['batch_size'] * 6 * 140000 / edges_per_graph)))
loader_test = DataLoader(
    datast_test, 
    batch_size=inference_batch,
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
        # only the central column of every graph: (graphs, levels, nlev), in the order of sample_centers
        all_predictions.append(out[data.central_mask.to(device)].view(-1, new_nz, nlev).cpu().numpy())

predictions_central = np.concatenate(all_predictions, axis=0)
print("GNN inference complete.")

# %%
# ---- Reconstruct Spatial Data Cubes ----
print("Reconstructing 3D spatial data cubes...")
nx_size = datast_test.x1 - datast_test.x0
ny_size = datast_test.y1 - datast_test.y0

predictions_spatial = np.zeros((new_nz, ny_size, nx_size, nlev))

num_test_samples = len(datast_test)

for i in tqdm(range(num_test_samples), desc="Reconstructing"):
    ix, iy = datast_test.sample_centers[i]
    predictions_spatial[:, iy - datast_test.y0, ix - datast_test.x0, :] = predictions_central[i]

# %%
# ---- Denormalization ----
print("Denormalizing predicted populations...")
# Total Ca density of the MURaM columns of the crop: CA_TOTAL_PER_RHO [cm^-3 per kg m^-3] times rho, interpolated in
# log10 to the grid levels
rho_crop = rho_si[:, pad:pad + y1c - y0c, pad:pad + x1c - x0c]
pop_norm_params = {'totals': CA_TOTAL_PER_RHO * to_levels(rho_crop, muram_z_grid[keep], zz_grid, log=True),
                   'log_offset': log_offset}

predictions_denorm = denormalize_pops(predictions_spatial, pop_norm_params)

# %%
# ---- Save Outputs ----
# a folder next to the checkpoint, one per stride, grid and crop
savedir = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), f"muram_stride{stride}_{grid_name.replace(':', '')}_{crop.replace(',', '-')}")
if not os.path.exists(savedir):
    os.makedirs(savedir)

# populations [cm^-3] (z, y, x, level) of the crop; y = MURaM cube axis 2, x = MURaM cube axis 1, from (y0c, x0c)
save_pred_path = os.path.join(savedir, f'muram_predictions_stride_{dataset_params["max_stride"]}_{dataset_params["split"]}.npy')
np.save(save_pred_path, predictions_denorm)
print(f"Saved synthesized absolute populations to {save_pred_path}")

# The grid (heights above MURaM's tau500 = 1 [Mm]), the heights of all the MURaM levels and the origin of the crop
np.savez(os.path.join(savedir, 'muram_grid.npz'), z=zz_grid, z_native=muram_z_grid, origin=np.array([y0c, x0c]),
         stride=stride, checkpoint=os.path.abspath(checkpoint_path))

# %%
# ---- Plot 1: Reconstructed Populations Slices ----
print("Generating reconstructed populations plots...")
z_slices = [np.argmin(np.abs(zz_grid - h)) for h in (0.2, 0.9, 1.8, 4.8)]     # heights above tau500 = 1 [Mm]
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