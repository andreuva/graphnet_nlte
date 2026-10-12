# %%
import muram as mio  

import numpy as np
import torch, os, sys
_here = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path[:0] = [_here, os.path.join(_here, '..')]      # this folder and the repo root (graphnet, Dataset, normalization, provenance)
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interpn
import matplotlib.ticker as ticker
import multiprocessing

from graphnet import EncodeProcessDecode
from Dataset import EfficientDataset
from normalization import CA_TOTAL_PER_RHO, denormalize_pops, normalize_features
from vertical_grids import training_grid
from provenance import make_run_dir, write_meta
import muram_common as mc

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
# CPU unless a device is given as 5th argument (e.g. 'cuda:0' with CUDA_VISIBLE_DEVICES=<a GPU nobody else uses>) (M7)
device_name = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
cuda_available = device_name.startswith('cuda')
device = torch.device(device_name)
print(f"Using device: {device}\n")

# %%
# The parameters to set manually: checkpoint, inference stride (None = training max_stride), crop of native columns to
# predict ('full' or 'y0,y1,x0,x1'); all can be given positionally:
#   python muram_inference.py <checkpoint> <stride> <crop> <device> [grid]
checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else '/dat/andreuva/gpu/graphnet/graphnet_nlte/checkpoints/v2_rung4_rotation_seed777/2026.10.08-15:09:03_best.pth'
stride = int(sys.argv[2]) if len(sys.argv) > 2 else None
crop = sys.argv[3] if len(sys.argv) > 3 else 'full'
grid_arg = sys.argv[5] if len(sys.argv) > 5 else ''            # evaluation grid ('index' | 'two_zone'); '' = the run's own grid

# Load the entire checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Extract the configuration object used for training
config = checkpoint['config']
nlev = config['data']['nlev']

log_offset = config['normalization']['log_offset']
features_labels_simple = list(checkpoint['meta']['features_labels'])      # the input quantities of the run

print(f"Successfully loaded configuration from checkpoint {checkpoint_path}.")
print(f"Inputs {features_labels_simple}, grid {config['dataset'].get('grid', 'index')}")

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
# tau500 = 1 from the per-column crossing (median fractional index 47.6; the argmin of the mean tau gives 49, 28 km too high: M5)
muram_z_grid, tau_zero_index = mc.muram_heights(cube)      # heights [Mm] relative to tau500 = 1, fractional index
print(f"Index of log tau = 0 layer: {tau_zero_index:.2f}")

# %%
# grid spacing from the header (dx = dy = 23.4375 km, dz = 20 km; M6). Native resolution: the data are never averaged,
# the multi-stride training is what makes the network adapt to the horizontal resolution.
dz, dx, dy = (np.asarray(cube.Temp.dX) / 1e5) / 1e3

nz, nx, ny  = cube.Temp.shape
z, x, y = (np.arange(d) for d in (nz, nx, ny))
z_geom, x_geom, y_geom = z*dz, x*dx, y*dy

# %%
datadir_bifrost = config['data']['datadir']
nx_bifrost, ny_bifrost, nz_bifrost = config['data']['nx'], config['data']['ny'], config['data']['nz_orig']

# %%
print(f"Original data shape: {cube.rho.shape}")
print("Creating new Z grid aligned with Bifrost")

bifrost_z_grid = np.load(config["data"]["grid_file"])["z"]
grid_name = grid_arg or config['dataset'].get('grid', 'index')
zz_grid_bifrost, new_z_b, z_transition_b = training_grid(config, bifrost_z_grid, grid_name)    # the levels in the Bifrost FITS frame
new_nz = len(new_z_b)

# Map zz_grid_bifrost to fractional indices in MURaM's vertical coordinate system, both simulations aligned at tau500 = 1
# (the Bifrost grid has <tau500> = 1 at ZTAU51 = 0.0614 Mm, M5)
new_z = np.interp(zz_grid_bifrost - mc.ZTAU51_BIFROST, muram_z_grid, z)
new_z = np.clip(new_z, 0, nz - 1)

new_y, new_x = (np.linspace(0, d - 1, new_d) for d, new_d in zip((ny, nx), (ny, nx)))
new_zv, new_yv, new_xv = np.meshgrid(new_z, new_y, new_x, indexing='ij', sparse=True)

# %%
# Crop of native columns to predict, padded with the periodic neighbours of the full domain so that the border is
# predicted too (M7); the stride sets the stencil footprint
stride = stride or config['dataset'].get('max_stride', 1)
xdim_g = config['dataset']['x_range_graph']
y0c, y1c, x0c, x1c = (0, ny, 0, nx) if crop == 'full' else [int(v) for v in crop.split(',')]
pad = xdim_g * stride
yidx = np.arange(y0c - pad, y1c + pad) % ny
xidx = np.arange(x0c - pad, x1c + pad) % nx
print(f"Predicting columns y in [{y0c},{y1c}), x in [{x0c},{x1c}) at stride {stride} ({stride * dx * 1e3:.0f} km hops), {pad} px periodic padding")

savedir = make_run_dir(os.path.join(_here, 'muram_inference.py'))

# %%
print(f"Interpolating data to the new grid ({new_nz}, {ny}, {nx})...")
new_points = (new_zv, new_xv, new_yv)

temp = interpn((z, x, y), cube.Temp, new_points)
v_z = interpn((z, x, y), cube.vx , new_points)
v_x = interpn((z, x, y), cube.vy , new_points)
v_y = interpn((z, x, y), cube.vz , new_points)
rho = interpn((z, x, y), cube.rho, new_points)
n_e = interpn((z, x, y), cube.ne, new_points)
press = interpn((z, x, y), cube.Pres, new_points)
print("Interpolation complete.")

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
pop_norm_params = checkpoint['normalization_params']

with_h = 'n_h' in features_labels_simple    # the ChE hydrogen files: H_1 = ground-level neutral H, H_6 = protons, cm^-3 (= Bifrost's n_h, n_p in m^-3)
n_h_si = interpn((z, x, y), cube.H_1, new_points) * 1e6 if with_h else None
n_p_si = interpn((z, x, y), cube.H_6, new_points) * 1e6 if with_h else None
features_data = [{'vel': vel_si, 'temp': temp_si, 'n_e': n_e_si, 'rho': rho_si, 'press': press_si, 'n_h': n_h_si, 'n_p': n_p_si}[l]
                 for l in features_labels_simple]
features_data = [f[:, yidx][:, :, xidx] for f in features_data]          # the padded crop (arrays are (z, y, x))
rho_crop = rho_si[:, y0c:y1c, x0c:x1c]

# Normalize features using the pre-computed training statistics to match the model's domain
normalized_features = normalize_features(features_data, features_labels_simple, feature_norm_params)[0]

# %%
# ---- Create Test Dataset ----
# All parameters are now from the loaded config object
assert abs(dx - dy) < 1e-9, (dx, dy)
dataset_params = {
    'list_X': normalized_features,
    'list_Y': [np.zeros(features_data[0].shape[:3] + (1,))],
    'xdim': config['dataset']['x_range_graph'],
    'ydim': config['dataset']['y_range_graph'],
    'z_levels': zz_grid_bifrost,      # the training levels (only height differences enter the edge features)
    'pixel_Mm': dx,
    'seed': config['system']['seed'],
    'train_ratio': config['dataset']['train_ratio'],
    'epoch_size_fraction': 1.0,  # Use full dataset for testing
    'max_stride': stride,
    'random_stride': False, #config["dataset"].get("random_stride", False),
    'edge_rule': config['dataset'].get('edge_rule', 'index'),
    'radius_neighbors': config['dataset'].get('radius_neighbors', 6.01),
    'stencil_radius': config['dataset'].get('stencil_radius', 4.01),
    'edge_reach_Mm': config['dataset'].get('edge_reach_Mm', 0.5),
    'central_reach_Mm': config['dataset'].get('central_reach_Mm'),
    'split': 'full'
}

datast_test = EfficientDataset(**dataset_params)
assert (datast_test.x0, datast_test.x1, datast_test.y0, datast_test.y1) == (pad, x1c - x0c + pad, pad, y1c - y0c + pad)

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
# Total Ca density of the MURaM columns themselves (sum of the training targets is 9.65e14 * rho exactly; M2)
pop_norm_params = dict(pop_norm_params)
pop_norm_params['totals'] = CA_TOTAL_PER_RHO * rho_crop                 # (nz, ny, nx): the interpolated arrays have no component axis

predictions_denorm = denormalize_pops(predictions_spatial, pop_norm_params)

# %%
# ---- Save Outputs ----
save_pred_name = f'muram_predictions_stride_{dataset_params["max_stride"]}_{dataset_params["split"]}.npy'
np.save(os.path.join(savedir, save_pred_name), predictions_denorm)
print(f"Saved synthesized absolute populations to {os.path.join(savedir, save_pred_name)}")
zz_grid = np.interp(new_z, z, muram_z_grid)                 # heights of the 50 levels above MURaM's tau500 = 1
np.save(os.path.join(savedir, 'rho_si_avg.npy'), rho_crop)
np.save(os.path.join(savedir, 'temp_avg.npy'), temp_si[:, y0c:y1c, x0c:x1c])
np.save(os.path.join(savedir, 'ne_si_avg.npy'), n_e_si[:, y0c:y1c, x0c:x1c])
np.savez(os.path.join(savedir, 'grid.npz'), z_rel_tau1_Mm=zz_grid, z_bifrost_frame_Mm=zz_grid_bifrost, muram_fractional_index=new_z,
         x_Mm=np.arange(x0c, x1c) * dx, y_Mm=np.arange(y0c, y1c) * dy)
write_meta(savedir, checkpoint=os.path.abspath(checkpoint_path), checkpoint_epoch=int(checkpoint['epoch']), stride=stride,
           train_stride=config['dataset'].get('max_stride', 1), features_labels=features_labels_simple, grid=grid_name, train_grid=config['dataset'].get('grid', 'index'),
           muram_dir=pathsource, snapshot=snap_number, dz_km=dz * 1e3, dx_km=dx * 1e3, resolution='native (no averaging)', decimate=1,
           origin=[y0c, x0c], crop=[y0c, y1c, x0c, x1c], hop_km=stride * dx * 1e3, pad=pad, tau1_index_median=tau_zero_index,
           ztau51_bifrost_Mm=mc.ZTAU51_BIFROST,
           totals_source=f'CA_TOTAL_PER_RHO={CA_TOTAL_PER_RHO:.3e} * rho_MURaM[kg m^-3] at the native columns',
           layout='(z, y, x, level) of the crop; y = MURaM cube axis 2, x = MURaM cube axis 1 (native column = origin + i); cm^-3',
           device=device_name, files={'predictions': save_pred_name, 'grid': 'grid.npz', 'rho': 'rho_si_avg.npy', 'temp': 'temp_avg.npy', 'ne': 'ne_si_avg.npy'})
# grids.png: the vertical grids aligned at tau500 = 1 (plan WP2 step 2.9)
mc.plot_grids([('native Bifrost (tau=1 frame)', bifrost_z_grid - mc.ZTAU51_BIFROST, 'royalblue'), ('native MURaM (tau=1 frame)', muram_z_grid, 'darkorange'),
               ('training grid on Bifrost', zz_grid_bifrost - mc.ZTAU51_BIFROST, 'forestgreen'), ('training levels sampled in MURaM', zz_grid, 'crimson')],
              bands=[(l, lo - mc.ZTAU51_BIFROST, hi - mc.ZTAU51_BIFROST) for l, lo, hi in mc.FORMATION_BANDS_BIFROST_FRAME],
              transition=z_transition_b - mc.ZTAU51_BIFROST, path=os.path.join(savedir, 'grids.png'),
              title='Vertical grids aligned at tau500 = 1', xlim=(-1.0, 13.0))

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