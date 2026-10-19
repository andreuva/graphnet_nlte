# gnn_train.py
# python gnn_train.py [conf.dat] [seed]

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from graphnet import *
from configobj import ConfigObj
from Dataset import *
from normalization import normalize_pops, normalize_features, lte_log_fractions
from vertical_grids import fixed_grid, ZTAU51_BIFROST
import matplotlib.pyplot as plt
import os, sys, time, shutil
from datetime import datetime
import multiprocessing

# Use unrepr=True to automatically convert strings to int, float, lists, etc.
conf_path = sys.argv[1] if len(sys.argv) > 1 else 'conf.dat'
config = ConfigObj(conf_path, unrepr=True)
if len(sys.argv) > 2:
    config['system']['seed'] = int(sys.argv[2])
seed = config['system']['seed']

gpu = config['system']['gpu']
cuda_available = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu}" if cuda_available else "cpu")
print(f"CUDA available: {cuda_available}")
print(f"Using device: {device}\n")
torch.manual_seed(seed)
np.random.seed(seed)
if cuda_available:
    torch.cuda.set_per_process_memory_fraction(config['system']['gpu_memory_fraction'], device)

lr = config['training']['lr']
batch_size = config['training']['batch_size']
n_epochs = config['training']['n_epochs']
savedir = f"{config['training']['savedir'].rstrip('/')}_seed{seed}/"
smooth = config['training']['smooth']
valid_guard = config['training']['valid_guard']
weight_decay = config['training']['weight_decay']
warmup_epochs = config['training']['warmup_epochs']
grad_clip = config['training']['grad_clip']
ema_decay = config['training']['ema_decay']
relevance_threshold = config['training']['relevance_threshold']
coronal_weight = config['training']['coronal_weight']
time_format = "%Y.%m.%d-%H:%M:%S"

datadir = config['data']['datadir']
grid_file = config['data']['grid_file']
nx = config['data']['nx']
ny = config['data']['ny']
nz = config['data']['nz_orig']
nlev = config['data']['nlev']
pixel_Mm = config['data']['pixel_Mm']

log_offset = config['normalization']['log_offset']

# ---- memory–mapped arrays ----
pops = np.memmap(f'{datadir}/AR_385_CaII_5L_pops.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, nlev))
temp = np.memmap(f'{datadir}/AR_385_temp.dat',dtype='<f4',mode='r',shape=(nz, ny, nx, 1))
vel = np.memmap(f'{datadir}/AR_385_veloc.dat',dtype='<f4',mode='r',shape=(nz, ny, nx, 3))
n_e = np.memmap(f'{datadir}/AR_385_ne.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
# n_p = np.memmap(f'{datadir}/AR_385_np.dat', dtype='<f4', mode='r',shape=(nz, ny, nx, 1))
# n_h = np.memmap(f'{datadir}/AR_385_nh.dat',dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
rho = np.memmap(f'{datadir}/AR_385_mass.dat', dtype='<f4', mode='r',shape=(nz, ny, nx, 1))
press = np.memmap(f'{datadir}/AR_385_press.dat',dtype='<f4', mode='r', shape=(nz, ny, nx, 1))

print('Populations shape:\t', pops.shape)
print('Temperature shape:\t', temp.shape)
print('Velocity shape:\t\t', vel.shape)
print('N_elec shape:\t\t', n_e.shape)
# print('N_nh shape:\t\t', n_h.shape)
# print('N_p shape:\t\t', n_p.shape)
print('Rho shape:\t\t', rho.shape)
print('Press shape:\t\t', press.shape)
print('#'*60+'\n')

# ---- Vertical grids ----
# The cube stays on its native levels: the training graphs draw their own random levels (Dataset.py), the validation
# graphs use the fixed grid of the run. Heights above tau500 = 1 [Mm]
z_native = np.load(grid_file)['z'].astype(np.float64) - ZTAU51_BIFROST
z_valid = fixed_grid(config['dataset']['grid'], z_native[0], z_native[-1])
print(f"Native grid: {nz} levels from {z_native[0]:.3f} to {z_native[-1]:.3f} Mm above tau500 = 1")
print(f"Validation grid {config['dataset']['grid']}: {len(z_valid)} levels")

# ---- LTE fractions of the Ca levels (an input) from the temperature and the electron density ----
t0 = time.time()
lte = lte_log_fractions(temp, n_e, log_offset)
print(f"LTE fractions computed in {time.time() - t0:.0f} s, shape {lte.shape}")
print('#'*60+'\n')

model_params = config['model']
model = EncodeProcessDecode(**model_params).to(device)
assert model.head_log_offset == log_offset, 'the head offset must be the log_offset of the targets'
print('N. total trainable parameters : {0}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# ---- Normalization ----
# 'vel' must stay the first input: Dataset.py rotates the inputs 0 and 1 (vx, vy)
features_labels = ['vel', 'temp', 'rho', 'press', 'lte']
features_data = [vel, temp, rho, press, lte]

normalized_features, feature_norm_params = normalize_features(features_data, features_labels)
pops_normalized, pop_norm_params = normalize_pops(pops, log_offset=log_offset)

features_list = normalized_features
targets_list = [pops_normalized]
features_labels_expanded = ['vx', 'vy', 'vz', 'temp', 'rho', 'press',
                            'lte_4s', 'lte_3d32', 'lte_3d52', 'lte_4p12', 'lte_4p32', 'lte_CaIII']

# ---- Create Datasets ----
dataset_params = {
    'list_X': features_list,
    'list_Y': targets_list,
    'z_native': z_native,
    'pixel_Mm': pixel_Mm,
    'xdim': config['dataset']['x_range_graph'],
    'ydim': config['dataset']['y_range_graph'],
    'train_ratio': config['dataset']['train_ratio'],
    'max_stride': config['dataset']['max_stride'],
    'stencil_radius': config['dataset']['stencil_radius'],
    'edge_radius_Mm': config['dataset']['edge_radius_Mm'],
    'central_reach_Mm': config['dataset']['central_reach_Mm'],
    'log_offset': log_offset,
}

datast_train = EfficientDataset(**dataset_params, random_geometry=True, split='train')
datast_test = EfficientDataset(**dataset_params, z_levels=z_valid, split='test')
del dataset_params, features_data, features_list, targets_list, normalized_features, pops_normalized, pop_norm_params, lte

# Validate only the held-out columns at least valid_guard pixels away from the training region
datast_test.sample_centers = [(x, y) for x, y in datast_test.sample_centers
                              if x >= datast_test.x0 + valid_guard and y >= datast_test.y0 + valid_guard]
print(f"Validation: {len(datast_test.sample_centers)} held-out columns, x, y >= {datast_test.x0 + valid_guard} (guard {valid_guard} px)")

# loader_train = DataLoader(datast_train, batch_size=batch_size, shuffle=True)
# loader_test = DataLoader(datast_test, batch_size=batch_size, shuffle=False)

# A fresh random subset of epoch_size_fraction of the training columns every epoch. The workers are forked and read the
# cube of the datasets without copying it
num_workers = min(12, multiprocessing.cpu_count())
sampler = torch.utils.data.RandomSampler(datast_train, num_samples=int(len(datast_train) * config['training']['epoch_size_fraction']))
loader_train = DataLoader(
    datast_train, 
    batch_size=batch_size, 
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=cuda_available, 
    persistent_workers=True
)
loader_test = DataLoader(
    datast_test, 
    batch_size=batch_size * 6, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=cuda_available,
    persistent_workers=True
)

print('\n'+'#'*60)
print("Model device:", next(model.parameters()).device)
print('#'*60+'\n')

# ---- Initialize Optimizer, Scheduler, and Loss ----
# AdamW with weight decay on the weight matrices only
decay = [p for p in model.parameters() if p.ndim > 1]
no_decay = [p for p in model.parameters() if p.ndim <= 1]
optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}],
                              lr=lr, betas=(0.9, 0.95), eps=1e-6)
# Stepped once per batch: linear warmup over warmup_epochs, then cosine decay to zero
n_steps = n_epochs * len(loader_train)
warmup_steps = max(1, int(warmup_epochs * len(loader_train)))
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_steps - warmup_steps))], milestones=[warmup_steps])

def loss_fn(out, target):
    """
    MSE on the log10 fractions, every node weighted by its Ca II fraction (levels 0-4): w = clip(f_CaII /
    relevance_threshold, 0, 1), never below coronal_weight, so that the corona, without Ca II, does not dominate.
    """
    w = torch.clamp((10.0 ** target[:, :5]).sum(dim=1) / relevance_threshold, 0.0, 1.0).clamp_min(coronal_weight)
    return (w[:, None] * (out - target) ** 2).sum() / (w.sum() * out.shape[1])

# Exponential moving average of the weights, the decay ramped up as (1 + n) / (10 + n) over the first updates
ema_updates = 0
def ema_avg_fn(averaged_params, current_params, num_averaged):
    decay_n = min(ema_decay, (1.0 + ema_updates) / (10.0 + ema_updates))
    torch._foreach_lerp_(averaged_params, current_params, 1.0 - decay_n)
ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=ema_avg_fn)

def validate(net):
    """Mean loss over the held-out batches and RMS error [dex] over all their central nodes."""
    net.eval()
    loss_sum, sq_sum, n_values = 0.0, 0.0, 0
    with torch.no_grad():
        for data in tqdm(loader_test, desc="Validating"):
            node, edge_attr, edge_index = data.x.to(device), data.edge_attr.to(device), data.edge_index.to(device)
            u, batch, target = data.u.to(device), data.batch.to(device), data.y.to(device)
            central_mask = data.central_mask.to(device)
            out = net(node, edge_attr, edge_index, u, batch)
            out, target = out.squeeze()[central_mask], target.squeeze()[central_mask]
            loss_sum += loss_fn(out, target).item()
            sq_sum += ((out - target) ** 2).sum().item()
            n_values += out.numel()
    return loss_sum / len(loader_test), np.sqrt(sq_sum / n_values)

# ---- Initialize Training State Variables ----
best_loss = float('inf')
best_dex = float('inf')
train_loss, valid_loss, valid_dex, valid_dex_ema, lrs = [], [], [], [], []

os.makedirs(savedir, exist_ok=True)
print(f"Saving runned code, checkpoint, and config to {savedir}")
# save a copy of the current code and config for reference
shutil.copy(f'{__file__}', os.path.join(savedir, f'gnn_train_{time.strftime(time_format)}.py'))
shutil.copy(conf_path, os.path.join(savedir, f'conf_{time.strftime(time_format)}.dat'))
shutil.copy('Dataset.py', os.path.join(savedir, f'Dataset_{time.strftime(time_format)}.py'))
shutil.copy('normalization.py', os.path.join(savedir, f'normalization_{time.strftime(time_format)}.py'))
shutil.copy('graphnet.py', os.path.join(savedir, f'graphnet_{time.strftime(time_format)}.py'))
shutil.copy('vertical_grids.py', os.path.join(savedir, f'vertical_grids_{time.strftime(time_format)}.py'))

# ---- Main Training Loop ----
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update_parameters(model)
        ema_updates += 1

        loss_avg = smooth * loss.item() + (1.0 - smooth) * loss_avg if loss_avg != 0.0 else loss.item()

    train_loss.append(loss_avg)

    # ------------------- VALIDATION (raw and averaged weights) -------------------
    val_loss_avg, val_dex = validate(model)
    _, val_dex_ema = validate(ema.module)
    valid_loss.append(val_loss_avg)
    valid_dex.append(val_dex)
    valid_dex_ema.append(val_dex_ema)
    print(f"Epoch {epoch} finished with validation loss: {val_loss_avg:.6f}, held-out RMS {val_dex:.5f} dex "
          f"(EMA weights {val_dex_ema:.5f} dex)")

    best_loss = min(best_loss, val_loss_avg)
    if min(val_dex, val_dex_ema) < best_dex:
        use_ema = val_dex_ema < val_dex
        best_dex = min(val_dex, val_dex_ema)

        checkpoint = {
            'epoch': epoch,
            'state_dict': (ema.module if use_ema else model).state_dict(),
            'weights': 'ema' if use_ema else 'raw',
            'best_loss': best_loss,
            'best_dex': best_dex,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': config,
            'features_labels': features_labels,
            'feature_norm_params': feature_norm_params,
            'train_loss_history': train_loss,
            'valid_loss_history': valid_loss,
            'valid_dex_history': valid_dex,
            'valid_dex_ema_history': valid_dex_ema,
        }

        print(f"New best held-out RMS: {best_dex:.5f} dex ({checkpoint['weights']} weights). Saving model...")
        torch.save(checkpoint, os.path.join(savedir, f'{filename}_best.pth'))
    
    lrs.append(scheduler.get_last_lr())

    # ---- Plotting ----
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.plot(valid_dex, label='Held-out RMS [dex]')
    plt.plot(valid_dex_ema, label='Held-out RMS, EMA weights [dex]')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(savedir, 'loss.pdf'))
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(lrs, label='Learning Rate')
    # plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(savedir, 'lr.pdf'))
    plt.close()