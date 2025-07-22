# %%
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from graphnet import *
from configobj import ConfigObj
from Dataset import *
import matplotlib.pyplot as plt
import time
from datetime import datetime

# %%
gpu = 3

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Set device to GPU 0 if available, otherwise CPU
device = torch.device(f"cuda:{gpu}" if cuda_available else "cpu")

# Print device info
print(f"CUDA available: {cuda_available}")
print(f"Using device: {device}")

# %%
#  LOAD THE DATACUBES OF THE GRID FROM C PORTA CODE

datadir = '../data_porta'
# ---- grid dimensions taken from the C code ----
nx = ny = 504
nz = 476 - 52 + 1          # 425
nlev = 6                   # caii[0] … caii[5]

# ---- memory–mapped array: reads only the chunks you touch ----
pops = np.memmap(
    f'{datadir}/AR_385_CaII_5L_pops.dat',
    dtype='<f4',           # little-endian 32-bit float
    mode='r',
    shape=(nz, ny, nx, nlev)   # (k, j, i, level)==(z, y, x, L)
)

# ---- memory–mapped array: reads only the chunks you touch ----
b_xyz = np.memmap(
    f'{datadir}/AR_385_B.dat',
    dtype='<f4',           # little-endian 32-bit float
    mode='r',
    shape=(nz, ny, nx, 3)   # (k, j, i, B_i)==(z, y, x, B)
)

# ---- memory–mapped array: reads only the chunks you touch ----
temp = np.memmap(
    f'{datadir}/AR_385_temp.dat',
    dtype='<f4',           # little-endian 32-bit float
    mode='r',
    shape=(nz, ny, nx, 1)   # (k, j, i, 1)==(z, y, x, 1)
)

# ---- memory–mapped array: reads only the chunks you touch ----
vel = np.memmap(
    f'{datadir}/AR_385_veloc.dat',
    dtype='<f4',           # little-endian 32-bit float
    mode='r',
    shape=(nz, ny, nx, 3)   # (k, j, i, 3)==(z, y, x, 3)
)

# ---- memory–mapped array: reads only the chunks you touch ----
n_e = np.memmap(
    f'{datadir}/AR_385_ne.dat',
    dtype='<f4',           # little-endian 32-bit float
    mode='r',
    shape=(nz, ny, nx, 1)   # (k, j, i, 1)==(z, y, x, 1)
)

# ---- memory–mapped array: reads only the chunks you touch ----
n_p = np.memmap(
    f'{datadir}/AR_385_np.dat',
    dtype='<f4',           # little-endian 32-bit float
    mode='r',
    shape=(nz, ny, nx, 1)   # (k, j, i, 1)==(z, y, x, 1)
)

# ---- memory–mapped array: reads only the chunks you touch ----
n_h = np.memmap(
    f'{datadir}/AR_385_nh.dat',
    dtype='<f4',           # little-endian 32-bit float
    mode='r',
    shape=(nz, ny, nx, 1)   # (k, j, i, 1)==(z, y, x, 1)
)

# ---- memory–mapped array: reads only the chunks you touch ----
# geom = np.memmap(
#     f'{datadir}/AR_385_GEOMETRY.dat',
#     dtype='<f4',           # little-endian 32-bit float
#     mode='r',
#     shape=(nz, ny, nx, 1)   # (k, j, i, 1)==(z, y, x, 1)
# )

# %%
print('Populations shape:\t', pops.shape)
print('Temperature shape:\t', temp.shape)
print('Mag, field shape:\t', b_xyz.shape)
print('Velocity shape:\t\t', vel.shape)
print('N_elec shape:\t\t', n_e.shape)
print('N_nh shape:\t\t', n_h.shape)
print('N_p shape:\t\t', n_p.shape)

from scipy.interpolate import interpn

# Define a new, higher-resolution grid
z, y, x = (np.arange(d) for d in (nz, ny, nx))

new_nz, new_ny, new_nx = 128, nx, ny
new_z, new_y, new_x = (np.linspace(0, d-1, new_d) for d, new_d in zip((nz,ny,nx), (new_nz,new_ny,new_nx)))
new_zv, new_yv, new_xv = np.meshgrid(new_z, new_y, new_x, indexing='ij', sparse=True)

# print('\n')
# print(f'original indices: {z}')
# print(f'new indices: {new_z}')

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
datast_train = EfficientDataset([vel/vel.mean(), b_xyz/b_xyz.mean(), temp/temp.mean(), np.log10(n_h/n_h.mean()), np.log10(n_e/n_e.mean()), np.log10(n_p/n_p.mean())],
                                [pops/pops.sum(axis=-1, keepdims=True)],
                                radius_neighbors=1.77)
datast_test = EfficientDataset([vel/vel.mean(), b_xyz/b_xyz.mean(), temp/temp.mean(), np.log10(n_h/n_h.mean()), np.log10(n_e/n_e.mean()), np.log10(n_p/n_p.mean())],
                                [pops/pops.sum(axis=-1, keepdims=True)],
                                radius_neighbors=1.77, split='test')
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
loader_train = DataLoader( datast_train, batch_size=10, shuffle=True)
loader_test = DataLoader( datast_test, batch_size=10, shuffle=True)

# %%
lr = 1e-3
n_epochs = 50
savedir = 'checkpoints/'
smooth = 0.05

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Cosine annealing learning rate scheduler. This will reduce the learning rate with a cosing law
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

# Loss function
loss_fn = nn.MSELoss()

# Now start the training
train_loss = []
valid_loss = []
lr = []
best_loss = float('inf')

time_format = "%Y.%m.%d-%H:%M:%S"

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

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

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
            'lr': scheduler.get_last_lr()
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


