# %%
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

# %%
gpu = 0

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Set device to GPU 0 if available, otherwise CPU
device = torch.device(f"cuda:{gpu}" if cuda_available else "cpu")

# Print device info
print(f"CUDA available: {cuda_available}")
print(f"Using device: {device}")
# %%
lr = 1e-3
batch_size = 16
n_epochs = 350
savedir = 'checkpoints/physical_z/'
smooth = 0.05

time_format = "%Y.%m.%d-%H:%M:%S"

# %%
#  LOAD THE DATACUBES OF THE GRID FROM C PORTA CODE

datadir = '../data_porta'
grid_file = '../en024048_hion/grid_bifrost.npz'
# ---- grid dimensions taken from the C code ----
nx = ny = 504
nz = 476 - 52 + 1          # 425
nz_linear = 30
nz_log = 25
new_nz = nz_linear + nz_log # Interpolated z dimension = 55
logspace_fraction = 0.4   # Fraction of points in z to be log-spaced
nlev = 6                   # caii[0] … caii[5]
radius_neighbors = 4.01  # in grid points, for the graph construction
x_range_graph = 2
y_range_graph = 2
interp_nz = new_nz  # Use the linear+log sampling
log_offset = 1e-12  # Offset to avoid log(0) in populations normalization
# %%


# ---- memory–mapped array: reads only the chunks you touch ----
pops = np.memmap(f'{datadir}/AR_385_CaII_5L_pops.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, nlev))
b_xyz = np.memmap(f'{datadir}/AR_385_B.dat',dtype='<f4',mode='r',shape=(nz, ny, nx, 3))
temp = np.memmap(f'{datadir}/AR_385_temp.dat',dtype='<f4',mode='r',shape=(nz, ny, nx, 1))
vel = np.memmap(f'{datadir}/AR_385_veloc.dat',dtype='<f4',mode='r',shape=(nz, ny, nx, 3))
n_e = np.memmap(f'{datadir}/AR_385_ne.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
n_p = np.memmap(f'{datadir}/AR_385_np.dat', dtype='<f4', mode='r',shape=(nz, ny, nx, 1))
n_h = np.memmap(f'{datadir}/AR_385_nh.dat',dtype='<f4', mode='r', shape=(nz, ny, nx, 1))

# %%
print('Populations shape:\t', pops.shape)
print('Temperature shape:\t', temp.shape)
print('Mag, field shape:\t', b_xyz.shape)
print('Velocity shape:\t\t', vel.shape)
print('N_elec shape:\t\t', n_e.shape)
print('N_nh shape:\t\t', n_h.shape)
print('N_p shape:\t\t', n_p.shape)

# Define a new grid with linear + logarithmic sampling in z
z, y, x = (np.arange(d) for d in (nz, ny, nx))

# Create hybrid z-grid: linear for first portion, then logarithmic
new_z_log = np.concatenate([
    np.linspace(0, nz*logspace_fraction, nz_linear, endpoint=False),
    np.logspace(np.log10(nz*logspace_fraction), np.log10(nz-1), nz_log)
])
new_z = np.clip(new_z_log, 0, nz - 1)  # Ensure we stay within bounds
new_y, new_x = (np.linspace(0, d-1, new_d) for d, new_d in zip((ny, nx), (ny, nx)))
new_zv, new_yv, new_xv = np.meshgrid(new_z, new_y, new_x, indexing='ij', sparse=True)

print(f"Interpolating data to the new grid ({new_nz}, {ny}, {nx})...")
print(f"Z range: [{new_z.min():.2f}, {new_z.max():.2f}] (should be [0, {nz-1}])")

# Interpolate data onto the new grid
new_points = (new_zv, new_yv, new_xv)
pops_interp = interpn((z, y, x), pops, new_points)
temp_interp = interpn((z, y, x), temp, new_points)
b_xyz_interp = interpn((z, y, x), b_xyz, new_points)
vel_interp = interpn((z, y, x), vel, new_points)
n_e_interp = interpn((z, y, x), n_e, new_points)
n_h_interp = interpn((z, y, x), n_h, new_points)
n_p_interp = interpn((z, y, x), n_p, new_points)

print('\n'+'#'*60)
print('Populations shape INTERPOLATED:\t', pops_interp.shape)
print('Temperature shape INTERPOLATED:\t', temp_interp.shape)
print('Mag, field shape INTERPOLATED:\t', b_xyz_interp.shape)
print('Velocity shape INTERPOLATED:\t', vel_interp.shape)
print('N_elec shape INTERPOLATED:\t', n_e_interp.shape)
print('N_nh shape INTERPOLATED:\t', n_h_interp.shape)
print('N_p shape INTERPOLATED:\t\t', n_p_interp.shape)

# %%
pops = pops_interp
temp = temp_interp
b_xyz = b_xyz_interp
vel = vel_interp
n_e = n_e_interp
n_h = n_h_interp
n_p = n_p_interp

# %%
# Read the configuration file
config_file = 'conf.dat'
with open(config_file, 'r') as f:
    tmp = f.readlines()
    f.close()

    # Parse configuration file and transform to integers
    hyperparameters = ConfigObj(tmp)

for k, q in hyperparameters.items():
    hyperparameters[k] = int(q)

# Instantiate the model with the hyperparameters
model = EncodeProcessDecode(**hyperparameters).to(device)
# Print the number of trainable parameters
print('N. total trainable parameters : {0}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# %%

# Apply normalization using functions from normalization.py
features_labels = ['vel', 'b', 'temp', 'n_h', 'n_e', 'n_p']
features_data = [vel, b_xyz, temp, n_h, n_e, n_p]

# Normalize features (returns norm_params dict with means, stds, scale_factors, log_offset)
normalized_features, feature_norm_params = normalize_features(features_data, features_labels, log_offset)
vel_norm, b_norm, temp_norm, n_h_norm, n_e_norm, n_p_norm = normalized_features

# Create the normalized features list
features_list = [vel_norm, b_norm, temp_norm, n_h_norm, n_e_norm, n_p_norm]

features_labels_expanded = ['vx', 'vy', 'vz', 'bx', 'by', 'bz', 'temp', 'n_h', 'n_e', 'n_p', 'z_pos']

# Normalize populations (returns norm_params dict with means, totals, factor, log_offset)
pops_normalized, normalization_params = normalize_pops(pops, factor=4., log_offset=log_offset)

targets_list = [pops_normalized]

datast_train = EfficientDataset(features_list,
                                targets_list,
                                radius_neighbors=radius_neighbors,
                                xdim=x_range_graph, ydim=y_range_graph,
                                pos_file=grid_file,
                                split='train'
                                )
datast_test = EfficientDataset(features_list,
                               targets_list,
                               radius_neighbors=radius_neighbors,
                               xdim=x_range_graph, ydim=y_range_graph,
                               pos_file=grid_file,
                               split='test'
                              )

datast_prms = {'radius_neighbors': radius_neighbors,
               'pos_file': grid_file,
               'nx': nx,
               'ny': ny,
               'nz': new_nz,
              }

# Get a single sample graph
sample_graph = datast_train[0].to(device)

# Now, provide the input as a tuple of tensors
batch_tensor = torch.zeros(sample_graph.num_nodes, dtype=torch.long).to(device)

# %%
sample_graph

# %%
torch.unique(sample_graph.edge_attr)

# %%
print("Model device:", next(model.parameters()).device)
print("sample_graph.x device:", sample_graph.x.device)
print("sample_graph.edge_attr device:", sample_graph.edge_attr.device)
print("sample_graph.edge_index device:", sample_graph.edge_index.device)
print("sample_graph.u device:", sample_graph.u.device)
print("batch_tensor device:", batch_tensor.device)

# %%
print(model)

# %%
model = model.to(device)
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
total_size = param_size + buffer_size

print(f"Model size: {total_size / 1024 ** 2:.2f} MB")

# %%

# if the savedir folder does not exist, create it
if os.path.exists(savedir) == False:
    os.makedirs(savedir)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Cosine annealing learning rate scheduler. This will reduce the learning rate with a cosing law
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ 30, 60, 100, 150, 200, 250, 275], gamma=0.5)

# %%
loader_train = DataLoader( datast_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader( datast_test, batch_size=batch_size, shuffle=True)

# Loss function
loss_fn = nn.MSELoss()

# Now start the training
train_loss = []
valid_loss = []
lr = []
best_loss = float('inf')

# print(torch.cuda.memory_summary())

# %%
for epoch in range(1, n_epochs + 1):

    # filename = str(epoch) #time.strftime("%Y%m%d-%H%M%S")
    filename = time.strftime(time_format)

    # Compute training and validation steps
    ################### TRAINING ###################
    # Put the model in training mode
    model.train()
    print("\n"+"#"*80)
    # print(f"Epoch {epoch}/{n_epochs}\nt = {filename}")
    print(f"Epoch {epoch}/{n_epochs}\nt = {filename}\nLR = {scheduler.get_last_lr()}")
    # t = tqdm(loader_train)
    loss_avg = 0.0

    # for batch_idx, (data) in enumerate(t):
    for batch_idx, (data) in enumerate(loader_train):

        # Extract the node, edges, indices, target, global and batch information from the Data class

        # Move them to the GPU
        node, edge_attr, edge_index = data.x.to(device), data.edge_attr.to(device), data.edge_index.to(device)
        u, batch, target = data.u.to(device), data.batch.to(device), data.y.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Evaluate Graphnet
        out = model(node, edge_attr, edge_index, u, batch)

        # Compute loss
        loss = loss_fn(out.squeeze(), target.squeeze())

        # Compute backpropagation
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Compute smoothed loss
        if (batch_idx == 0):
            loss_avg = loss.item()
        else:
            loss_avg = smooth * loss.item() + (1.0 - smooth) * loss_avg

        # free gpu memory
        # torch.cuda.empty_cache()

    train_loss.append(loss_avg)

    ################### VALIDATION ###################
    # Do a validation of the model and return the loss

    model.eval()
    loss_avg = 0
    # t = tqdm(loader_test)

    mid_time = time.strftime(time_format)
    print(f"Epoch consumed time = {datetime.strptime(mid_time, time_format) - datetime.strptime(filename, time_format)})")

    print("Starting the Validation of the epoch:")
    with torch.no_grad():
        # for batch_idx, (data) in enumerate(t):
        for batch_idx, (data) in enumerate(loader_test):

            node, edge_attr, edge_index = data.x.to(device), data.edge_attr.to(device), data.edge_index.to(device)
            u, batch, target = data.u.to(device), data.batch.to(device), data.y.to(device)

            out = model(node, edge_attr, edge_index, u, batch)

            loss = loss_fn(out.squeeze(), target.squeeze())

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = smooth * loss.item() + (1.0 - smooth) * loss_avg

            # t.set_postfix(loss=loss_avg)
        print()

    valid_loss.append(loss_avg)

    finish_time = time.strftime(time_format)
    print(f"full epoch finished {epoch}/{n_epochs} in {datetime.strptime(finish_time, time_format) - datetime.strptime(filename, time_format)} time with loss {loss_avg}")

    # If the validation loss improves, save the model as best
    if (valid_loss[-1] < best_loss):
        best_loss = valid_loss[-1]

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'best_loss': best_loss,
            'hyperparameters': hyperparameters,
            'optimizer': optimizer.state_dict(),
            'lr': scheduler.get_last_lr(),
            'dataset_params': datast_prms,
            'features_labels': features_labels_expanded,
            'nlev': nlev,
            'feature_norm_params': feature_norm_params,
            'normalization_params': normalization_params,
        }

        print("Saving best model...")
        torch.save(checkpoint, savedir + filename + '_best.pth')
    lr.append(scheduler.get_last_lr())
    # Update the learning rate
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
plt.plot(lr, label='Learning rate')
plt.xlabel('Itteration')
plt.legend()
plt.savefig(savedir + 'lr.pdf')
plt.close()
