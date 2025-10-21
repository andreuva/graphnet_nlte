# gnn_train.py

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from graphnet import *
from configobj import ConfigObj
from Dataset import *
from normalization import normalize_pops, denormalize_pops, normalize_features, denormalize_features
import matplotlib.pyplot as plt
import time
from datetime import datetime
from scipy.interpolate import interpn
import os

# Use unrepr=True to automatically convert strings to int, float, lists, etc.
config = ConfigObj('conf.dat', unrepr=True)

gpu = config['system']['gpu']
cuda_available = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu}" if cuda_available else "cpu")
print(f"CUDA available: {cuda_available}")
print(f"Using device: {device}\n")

lr = config['training']['lr']
batch_size = config['training']['batch_size']
n_epochs = config['training']['n_epochs']
savedir = config['training']['savedir']
smooth = config['training']['smooth']
time_format = "%Y.%m.%d-%H:%M:%S"

datadir = config['data']['datadir']
grid_file = config['data']['grid_file']
nx = config['data']['nx']
ny = config['data']['ny']
nz = config['data']['nz_orig']
nlev = config['data']['nlev']

nz_linear = config['dataset']['nz_linear']
nz_log = config['dataset']['nz_log']
new_nz = nz_linear + nz_log
logspace_fraction = config['dataset']['logspace_fraction']
log_offset = config['normalization']['log_offset']
normalization_type = config['normalization'].get('type', 'log')

# ---- memory–mapped arrays ----
pops = np.memmap(f'{datadir}/AR_385_CaII_5L_pops.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, nlev))
b_xyz = np.memmap(f'{datadir}/AR_385_B.dat',dtype='<f4',mode='r',shape=(nz, ny, nx, 3))
temp = np.memmap(f'{datadir}/AR_385_temp.dat',dtype='<f4',mode='r',shape=(nz, ny, nx, 1))
vel = np.memmap(f'{datadir}/AR_385_veloc.dat',dtype='<f4',mode='r',shape=(nz, ny, nx, 3))
n_e = np.memmap(f'{datadir}/AR_385_ne.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
n_p = np.memmap(f'{datadir}/AR_385_np.dat', dtype='<f4', mode='r',shape=(nz, ny, nx, 1))
n_h = np.memmap(f'{datadir}/AR_385_nh.dat',dtype='<f4', mode='r', shape=(nz, ny, nx, 1))

print('Populations shape:\t', pops.shape)
print('Temperature shape:\t', temp.shape)
print('Mag, field shape:\t', b_xyz.shape)
print('Velocity shape:\t\t', vel.shape)
print('N_elec shape:\t\t', n_e.shape)
print('N_nh shape:\t\t', n_h.shape)
print('N_p shape:\t\t', n_p.shape)
print('#'*60+'\n')

# ---- Interpolate data to the new grid ----
z, y, x = (np.arange(d) for d in (nz, ny, nx))
new_z_log = np.concatenate([
    np.linspace(0, nz * logspace_fraction, nz_linear, endpoint=False),
    np.logspace(np.log10(nz * logspace_fraction), np.log10(nz - 1), nz_log)
])
new_z = np.clip(new_z_log, 0, nz - 1)
new_y, new_x = (np.linspace(0, d - 1, new_d) for d, new_d in zip((ny, nx), (ny, nx)))
new_zv, new_yv, new_xv = np.meshgrid(new_z, new_y, new_x, indexing='ij', sparse=True)

print(f"Interpolating data to the new grid ({new_nz}, {ny}, {nx})...")
new_points = (new_zv, new_yv, new_xv)
pops = interpn((z, y, x), pops, new_points)
temp = interpn((z, y, x), temp, new_points)
b_xyz = interpn((z, y, x), b_xyz, new_points)
vel = interpn((z, y, x), vel, new_points)
n_e = interpn((z, y, x), n_e, new_points)
n_h = interpn((z, y, x), n_h, new_points)
n_p = interpn((z, y, x), n_p, new_points)

print("Interpolation complete.")
print('\n'+'#'*60)
print('Populations shape INTERPOLATED:\t', pops.shape)
print('Temperature shape INTERPOLATED:\t', temp.shape)
print('Mag, field shape INTERPOLATED:\t', b_xyz.shape)
print('Velocity shape INTERPOLATED:\t', vel.shape)
print('N_elec shape INTERPOLATED:\t', n_e.shape)
print('N_nh shape INTERPOLATED:\t', n_h.shape)
print('N_p shape INTERPOLATED:\t\t', n_p.shape)
print('#'*60+'\n')

model_params = config['model']
model = EncodeProcessDecode(**model_params).to(device)
print('N. total trainable parameters : {0}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# ---- Normalization ----
features_labels = ['vel', 'b', 'temp', 'n_h', 'n_e', 'n_p']
features_data = [vel, b_xyz, temp, n_h, n_e, n_p]

normalized_features, feature_norm_params = normalize_features(features_data, features_labels, log_offset, type=normalization_type)
pops_normalized, pop_norm_params = normalize_pops(pops, factor=config['normalization']['factor'], log_offset=log_offset, type=normalization_type)

features_list = normalized_features
targets_list = [pops_normalized]
features_labels_expanded = ['vx', 'vy', 'vz', 'bx', 'by', 'bz', 'temp', 'n_h', 'n_e', 'n_p', 'z_pos']

# ---- Create Datasets ----
dataset_params = {
    'list_X': features_list,
    'list_Y': targets_list,
    'radius_neighbors': config['dataset']['radius_neighbors'],
    'xdim': config['dataset']['x_range_graph'],
    'ydim': config['dataset']['y_range_graph'],
    'fully_connected': config['dataset']['fully_connected'],
    'pos_file': grid_file,
    'seed': config['system']['seed'],
    'train_ratio': config['dataset']['train_ratio'],
    'nz_linear': nz_linear,
    'nz_log': nz_log,
    'logspace_fraction': logspace_fraction,
    'epoch_size_fraction': config['training'].get('epoch_size_fraction', 0.1)
}

datast_train = EfficientDataset(**dataset_params, split='train')
datast_test = EfficientDataset(**dataset_params, split='test')

loader_train = DataLoader(datast_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(datast_test, batch_size=batch_size, shuffle=False)

# Get a single sample graph
sample_graph = datast_train[0].to(device)

# Now, provide the input as a tuple of tensors
batch_tensor = torch.zeros(sample_graph.num_nodes, dtype=torch.long).to(device)

print('\n'+'#'*60)
print("Model device:", next(model.parameters()).device)
print("sample_graph.x device:", sample_graph.x.device)
print("sample_graph.edge_attr device:", sample_graph.edge_attr.device)
print("sample_graph.edge_index device:", sample_graph.edge_index.device)
print("sample_graph.u device:", sample_graph.u.device)
print("batch_tensor device:", batch_tensor.device)
print('#'*60+'\n')

if not os.path.exists(savedir):
    os.makedirs(savedir)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=config['training']['milestones'],
    gamma=config['training']['gamma']
)

loss_fn = nn.MSELoss()
train_loss, valid_loss, lrs = [], [], []
best_loss = float('inf')

for epoch in range(1, n_epochs + 1):
    filename = time.strftime(time_format)
    model.train()
    print("\n" + "#" * 80)
    print(f"Epoch {epoch}/{n_epochs}\nt = {filename}\nLR = {scheduler.get_last_lr()}")

    loss_avg = 0.0
    for data in tqdm(loader_train, desc="Training"):
        node, edge_attr, edge_index = data.x.to(device), data.edge_attr.to(device), data.edge_index.to(device)
        u, batch, target = data.u.to(device), data.batch.to(device), data.y.to(device)
        central_mask = data.central_mask.to(device)

        optimizer.zero_grad()
        out = model(node, edge_attr, edge_index, u, batch)
        loss = loss_fn(out.squeeze()[central_mask], target.squeeze()[central_mask])
        loss.backward()
        optimizer.step()

        loss_avg = smooth * loss.item() + (1.0 - smooth) * loss_avg if loss_avg != 0.0 else loss.item()

    train_loss.append(loss_avg)

    # ------------------- VALIDATION -------------------
    model.eval()
    loss_avg = 0.0
    with torch.no_grad():
        for data in tqdm(loader_test, desc="Validating"):
            node, edge_attr, edge_index = data.x.to(device), data.edge_attr.to(device), data.edge_index.to(device)
            u, batch, target = data.u.to(device), data.batch.to(device), data.y.to(device)
            central_mask = data.central_mask.to(device)
            out = model(node, edge_attr, edge_index, u, batch)
            loss = loss_fn(out.squeeze()[central_mask], target.squeeze()[central_mask])
            loss_avg = smooth * loss.item() + (1.0 - smooth) * loss_avg if loss_avg != 0.0 else loss.item()

    valid_loss.append(loss_avg)
    print(f"Epoch {epoch} finished with validation loss: {loss_avg:.6f}")

    if valid_loss[-1] < best_loss:
        best_loss = valid_loss[-1]

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': config,  # <--- SAVE THE ENTIRE CONFIGURATION
            'feature_norm_params': feature_norm_params,
            'normalization_params': pop_norm_params,
        }

        print("Saving best model...")
        torch.save(checkpoint, os.path.join(savedir, f'{filename}_best.pth'))
    
    lrs.append(scheduler.get_last_lr())
    scheduler.step()

    # %%
    plt.figure(0, (10,15), dpi=100)
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='Validation loss')
    plt.xlabel('Itteration')
    plt.legend()
    plt.savefig(savedir + 'loss.pdf')
    plt.close()

    # %%

    plt.figure(0, (10,15), dpi=100)
    plt.plot(lrs, label='Learning rate')
    plt.xlabel('Itteration')
    plt.legend()
    plt.savefig(savedir + 'lr.pdf')
    plt.close()
