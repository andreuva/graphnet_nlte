# gnn_train.py

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from graphnet import *
from configobj import ConfigObj
from Dataset import *
from normalization import normalize_pops, normalize_features, lte_log_fractions, expand_labels
from vertical_grids import training_grid
import matplotlib.pyplot as plt
import os, time, shutil, sys, csv, hashlib
from datetime import datetime
from scipy.interpolate import interpn
from glob import glob
import multiprocessing
from provenance import git_state

# Use unrepr=True to automatically convert strings to int, float, lists, etc.
conf_path = sys.argv[1] if len(sys.argv) > 1 else 'conf.dat'          # python gnn_train.py [conf.dat] [seed]
config = ConfigObj(conf_path, unrepr=True)
if len(sys.argv) > 2:                                                   # the ensemble members differ only by the seed
    config['system']['seed'] = int(sys.argv[2])

gpu = config['system']['gpu']
cuda_available = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu}" if cuda_available else "cpu")
torch.manual_seed(config['system']['seed'])
np.random.seed(config['system']['seed'])
if cuda_available and config['system'].get('gpu_memory_fraction'):
    # cap the caching allocator so that inference jobs can share the GPU (with graphs of varying size it otherwise grows to the whole card)
    torch.cuda.set_per_process_memory_fraction(float(config['system']['gpu_memory_fraction']), device)
print(f"CUDA available: {cuda_available}")
print(f"Using device: {device}\n")

lr = config['training']['lr']
batch_size = config['training']['batch_size']
n_epochs = config['training']['n_epochs']
savedir = f"{config['training']['savedir'].rstrip('/')}_seed{config['system']['seed']}/"     # one run directory per seed
smooth = config['training']['smooth']
time_format = "%Y.%m.%d-%H:%M:%S"
valid_guard = config['training'].get('valid_guard', 32)          # D-2: held-out columns this far from the training region
valid_subset = config['training'].get('valid_subset', 1.0)
# ---- recipe (code pattern from master's Formal.py): AdamW, warmup + cosine per batch, clipping, EMA, fresh subset per epoch ----
weight_decay = config['training']['weight_decay']
adam_betas = tuple(config['training']['adam_betas'])
adam_eps = config['training']['adam_eps']
warmup_epochs = config['training']['warmup_epochs']
warmup_start_factor = config['training'].get('warmup_start_factor', 0.01)
grad_clip = config['training']['grad_clip']                             # 0 = off
ema_decay = config['training']['ema_decay']                             # 0 = off; 0.9998 with the (1+n)/(10+n) ramp
features_labels = list(config['data']['features'])                      # input quantities: 'vel', 'temp', 'rho', 'press', 'lte'

datadir = config['data']['datadir']
grid_file = config['data']['grid_file']
nx = config['data']['nx']
ny = config['data']['ny']
nz = config['data']['nz_orig']
nlev = config['data']['nlev']

log_offset = config['normalization']['log_offset']
augment_rotation = config['dataset']['augment_rotation']               # one random rotation about z per training sample (the only augmentation)
relevance_threshold = config['training'].get('relevance_threshold', 1e-2)      # loss weight w = clip(f_CaII / threshold, 0, 1) per node ...
coronal_weight = config['training'].get('coronal_weight', 0.1)                  # ... floored at this value (coronal nodes down-weighted, not dropped)

# ---- Code snapshot and git state at start-up, before any data is loaded (WP4.2) ----
start_stamp = time.strftime(time_format)
git = git_state(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(savedir):
    os.makedirs(savedir)
print(f"Saving runned code, checkpoint, and config to {savedir}")
shutil.copy(f'{__file__}', os.path.join(savedir, f'gnn_train_{start_stamp}.py'))
shutil.copy(conf_path, os.path.join(savedir, f'conf_{start_stamp}.dat'))
shutil.copy('Dataset.py', os.path.join(savedir, f'Dataset_{start_stamp}.py'))
shutil.copy('normalization.py', os.path.join(savedir, f'normalization_{start_stamp}.py'))
shutil.copy('graphnet.py', os.path.join(savedir, f'graphnet_{start_stamp}.py'))
shutil.copy('vertical_grids.py', os.path.join(savedir, f'vertical_grids_{start_stamp}.py'))

# ---- memory–mapped arrays ----
FILE_OF = {'vel': ('AR_385_veloc.dat', 3), 'temp': ('AR_385_temp.dat', 1), 'n_e': ('AR_385_ne.dat', 1), 'rho': ('AR_385_mass.dat', 1),
           'press': ('AR_385_press.dat', 1)}
pops_native = np.memmap(f'{datadir}/AR_385_CaII_5L_pops.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, nlev))
# 'lte' is derived from temp and n_e (normalization.lte_log_fractions), which are loaded even when they are not inputs
raw_labels = [l for l in features_labels if l != 'lte'] + [l for l in ('temp', 'n_e') if 'lte' in features_labels and l not in features_labels]
raw = {l: np.memmap(f'{datadir}/{FILE_OF[l][0]}', dtype='<f4', mode='r', shape=(nz, ny, nx, FILE_OF[l][1])) for l in raw_labels}
for arr in [pops_native] + list(raw.values()):
    assert os.path.getsize(arr.filename) == arr.size * 4, f'{arr.filename}: file size does not match the declared shape (WP4.2)'
manifest_path = os.path.join(datadir, 'MANIFEST.json')
data_manifest_sha256 = hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest() if os.path.exists(manifest_path) else None

print('Populations shape:\t', pops_native.shape)
for l, a in raw.items():
    print(f'{l} shape:\t\t', a.shape)
print('#'*60+'\n')

# ---- Interpolate data to the new grid ----
z, y, x = (np.arange(d) for d in (nz, ny, nx))
z_native = np.load(grid_file)['z']                                      # native level heights [Mm]
zz_levels, new_z, z_transition = training_grid(config, z_native)       # validation grid: heights, fractional native indices
new_nz = len(new_z)
new_y, new_x = (np.linspace(0, d - 1, new_d) for d, new_d in zip((ny, nx), (ny, nx)))
new_zv, new_yv, new_xv = np.meshgrid(new_z, new_y, new_x, indexing='ij', sparse=True)

print(f"Interpolating data to the validation grid ({new_nz}, {ny}, {nx})...")
new_points = (new_zv, new_yv, new_xv)
def to_grid(a, log=False):           # log10 for the scalars and populations (velocities and the log10 LTE fractions stay linear)
    if log:
        return 10.0 ** interpn((z, y, x), np.log10(np.asarray(a, dtype=np.float64)), new_points)
    return interpn((z, y, x), a, new_points)
if 'lte' in features_labels:
    t0 = time.time(); print('Computing the log10 LTE Ca fractions of the native cube from temp and n_e...')
    raw['lte'] = lte_log_fractions(raw['temp'], raw['n_e'], log_offset)       # (nz, ny, nx, 6), interpolated linearly like any log quantity
    print(f'... done in {time.time() - t0:.0f} s, shape {raw["lte"].shape}')
pops = to_grid(pops_native, True)
features_data = [to_grid(raw[l], l not in ('vel', 'lte')) for l in features_labels]

print("Interpolation complete.")
print('\n'+'#'*60)
print('Populations shape INTERPOLATED:\t', pops.shape)
for l, a in zip(features_labels, features_data):
    print(f'{l} shape INTERPOLATED:\t', a.shape)
print('#'*60+'\n')

model_params = config['model']
model = EncodeProcessDecode(**model_params).to(device)
assert model_params['head'] == 'log_softmax' and model_params.get('head_log_offset', 1e-12) == log_offset, \
    "the targets are log10(f + log_offset): the model needs head = 'log_softmax' with the same head_log_offset"
assert model_params['edge_input_size'] == 3, model_params['edge_input_size']
print('N. total trainable parameters : {0}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# ---- Normalization: the training data is the native cube, its statistics are stored and applied to the validation grid too ----
native_normalized, feature_norm_params = normalize_features([raw[l] for l in features_labels], features_labels)
normalized_features = normalize_features(features_data, features_labels, feature_norm_params)[0]
native_X = np.concatenate(native_normalized, axis=-1).astype(np.float32)
del native_normalized
native_Y = normalize_pops(np.asarray(pops_native), log_offset)[0].astype(np.float32)
pops_normalized, pop_norm_params = normalize_pops(pops, log_offset)

features_list = normalized_features
targets_list = [pops_normalized]
features_labels_expanded = expand_labels(features_labels)
assert len(features_labels_expanded) == model_params['node_input_size'], (features_labels_expanded, model_params['node_input_size'])
rot_pairs = [(i, i + 1) for i, l in enumerate(features_labels_expanded) if l == 'velx']   # (x, y) columns rotated as vectors (D-7)

# ---- Create Datasets ----
pixel_Mm = float(np.median(np.diff(np.load(grid_file)['x'])))          # horizontal pixel size [Mm] (uniform, dx = dy)
assert np.allclose(np.diff(np.load(grid_file)['x']), pixel_Mm) and np.allclose(np.diff(np.load(grid_file)['y']), pixel_Mm)
dataset_params = {
    'list_X': features_list,
    'list_Y': targets_list,
    'xdim': config['dataset']['x_range_graph'],
    'ydim': config['dataset']['y_range_graph'],
    'z_levels': zz_levels,
    'pixel_Mm': pixel_Mm,
    'seed': config['system']['seed'],
    'train_ratio': config['dataset']['train_ratio'],
    'epoch_size_fraction': config['training'].get('epoch_size_fraction', 0.1),
    "max_stride": config["dataset"]["max_stride"],
    'stencil_radius': config['dataset']['stencil_radius'],
    'edge_reach_Mm': config['dataset']['edge_reach_Mm'],
    'central_reach_Mm': config['dataset'].get('central_reach_Mm'),
}

# training graphs: a random vertical grid, hops and rotation per sample, drawn from the native cube
margin = RandomGeometryDataset.stencil_margin(dataset_params['xdim'], dataset_params['ydim'], dataset_params['max_stride'])
train_centres, _, _ = split_columns(nx, ny, dataset_params['xdim'], dataset_params['ydim'], margin, margin, dataset_params['max_stride'],
                                    dataset_params['train_ratio'], 'train', dataset_params['seed'])
datast_train = RandomGeometryDataset(native_X, native_Y, z_native, pixel_Mm, train_centres, dataset_params['xdim'], dataset_params['ydim'],
                                     stencil_radius=dataset_params['stencil_radius'], edge_reach_Mm=dataset_params['edge_reach_Mm'],
                                     central_reach_Mm=dataset_params['central_reach_Mm'], max_stride=dataset_params['max_stride'],
                                     augment_rotation=augment_rotation, rot_pairs=rot_pairs,
                                     epoch_size_fraction=dataset_params['epoch_size_fraction'], log_offset=log_offset)
del native_X, native_Y
# D-2 validation: the held-out corner at the training stride, outside the guard band, all its columns
datast_test = EfficientDataset(**{**dataset_params, 'epoch_size_fraction': 1.0}, split='test')
datast_test.sample_centers = [(ix, iy) for ix, iy in datast_test.sample_centers
                              if ix >= datast_test.x0 + valid_guard and iy >= datast_test.y0 + valid_guard]
if valid_subset < 1.0:
    datast_test.sample_centers = datast_test.sample_centers[:max(1, int(len(datast_test.sample_centers) * valid_subset))]
print(f"Validation: {len(datast_test.sample_centers)} held-out columns at stride {datast_test.max_stride}, x, y >= {datast_test.x0 + valid_guard} (guard {valid_guard} px)")

# loader_train = DataLoader(datast_train, batch_size=batch_size, shuffle=True)
# loader_test = DataLoader(datast_test, batch_size=batch_size, shuffle=False)

num_workers = min(config['training'].get('num_workers', 12), multiprocessing.cpu_count())
# a fresh random subset of the training columns every epoch
n_per_epoch = int(len(datast_train.sample_centers) * datast_train.epoch_size_fraction)
datast_train.epoch_size_fraction = 1.0
sampler = torch.utils.data.RandomSampler(datast_train, replacement=False, num_samples=n_per_epoch)
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
# AdamW with decoupled weight decay on the weight matrices only (master's Formal.py pattern)
decay, no_decay = [], []
for p_ in model.parameters():
    if p_.requires_grad:
        (decay if p_.ndim > 1 else no_decay).append(p_)
optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}],
                              lr=lr, betas=adam_betas, eps=adam_eps)
# stepped once per batch: linear warmup over warmup_epochs, then cosine annealing to zero
n_steps = n_epochs * len(loader_train)
warmup_steps = max(1, min(int(warmup_epochs * len(loader_train)), n_steps // 2))
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, n_steps - warmup_steps))], milestones=[warmup_steps])
def relevance_weights(target):     # D-6: w = clip(f_CaII / threshold, 0, 1) per node, floored at coronal_weight; levels 0-4 are Ca II
    f_caii = (10.0 ** target[:, :5]).sum(dim=1)
    return torch.clamp(f_caii / relevance_threshold, 0.0, 1.0).clamp_min(coronal_weight)
def compute_loss(o, t):            # weighted MSE on log10 f; o, t: (n_central_nodes, nlev)
    w = relevance_weights(t)
    return (w[:, None] * (o - t) ** 2).sum() / (w.sum() * o.shape[1])
ema, ema_updates = None, 0
if ema_decay > 0:                  # EMA of the weights with the (1+n)/(10+n) ramp, validated next to the raw weights
    def ema_avg_fn(averaged, current, num_averaged):
        d = min(ema_decay, (1.0 + ema_updates) / (10.0 + ema_updates))
        torch._foreach_lerp_(averaged, current, 1.0 - d)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=ema_avg_fn)
print(f"Recipe: AdamW (wd {weight_decay}), warmup {warmup_epochs} epoch + cosine, grad clip {grad_clip}, EMA {ema_decay}, fresh subset every epoch")
print(f"Representation: inputs {features_labels_expanded}, targets log10(f + {log_offset}), edges (dz, d_h, |dr|), "
      f"validation grid {config['dataset']['grid']} ({new_nz} levels), random training geometry, rotation augmentation {augment_rotation}")

# ---- Initialize Training State Variables ----
best_loss = float('inf')
best_dex = float('inf')
train_loss, valid_loss, valid_dex, lrs = [], [], [], []
meta = {'conf_path': os.path.abspath(conf_path), 'features_labels': features_labels, 'features_labels_expanded': features_labels_expanded,
        'derived_inputs': {'lte': 'log10 LTE fractions of the six Ca II levels from temp [K] and n_e [m^-3] of the atmosphere itself '
                                  '(lightweaver lte_pops, Debye on), computed on the native cube (normalization.lte_log_fractions)'} if 'lte' in features_labels else {},
        'loss': f'MSE on log10 f with node weights max(clip(f_CaII / {relevance_threshold}, 0, 1), {coronal_weight})',
        'target_transform': 'log10(f + log_offset); errors are dex', 'augment_rotation': augment_rotation, 'rot_pairs': rot_pairs,
        'grid': config['dataset']['grid'], 'edge_rule': 'physical', 'interp_log': True,
        'train_geometry': 'random', 'central_reach_Mm': config['dataset'].get('central_reach_Mm'), 'datadir': datadir, 'data_manifest_sha256': data_manifest_sha256,
        'grid_file': grid_file, 'z_grid_Mm': zz_levels.tolist(), 'new_z_indices': new_z.tolist(),
        'validation': {'split': 'test', 'stride': datast_test.max_stride, 'guard': valid_guard, 'n_columns': len(datast_test.sample_centers),
                       'metric': 'RMS dex over central nodes, plain mean (_best.pth = the epoch with the lowest value, raw or EMA weights)'},
        'recipe': {'optimizer': 'adamw', 'weight_decay': weight_decay, 'adam_betas': adam_betas, 'adam_eps': adam_eps, 'schedule': 'warmup_cosine',
                   'warmup_epochs': warmup_epochs, 'grad_clip': grad_clip, 'ema_decay': ema_decay, 'fresh_subset': True},
        'git': git, 'start_time': start_stamp, 'seed': config['system']['seed']}

csv_path = os.path.join(savedir, 'training_log.csv')          # one row per epoch
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'time', 'lr', 'train_loss_ema', 'valid_loss', 'valid_dex', 'valid_dex_ema', 'best_dex', 'best_weights', 'epoch_seconds'])

# grids.png: the vertical grid in use against the native grid and the Ca II formation heights (plan WP2 step 2.9)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'muram_comparison'))
    import muram_common as mc
    grid_rows = [('native Bifrost', z_native, 'royalblue'), ('validation grid', zz_levels, 'forestgreen')]
    from vertical_grids import random_two_zone_grid, ZTAU51_BIFROST      # what the training samples look like: 8 draws of the random two-zone grid
    _rng = np.random.default_rng(0)
    for _i in range(8):
        _h, _p, _ = random_two_zone_grid(z_native[0] - ZTAU51_BIFROST, z_native[-1] - ZTAU51_BIFROST, _rng)
        grid_rows.append((f"draw {_i}: {_p['spacing_km']:.0f} km, tr {_p['transition_Mm']:.2f} ({len(_h)})", _h + ZTAU51_BIFROST, 'k'))
    mc.plot_grids(grid_rows,
                  bands=[(l, lo, hi) for l, lo, hi in mc.FORMATION_BANDS_BIFROST_FRAME], transition=z_transition,
                  path=os.path.join(savedir, 'grids.png'), title='Vertical grids (Bifrost FITS frame)', xlim=(-0.8, 13.0))
except Exception as exc:                 # noqa: BLE001  (a check figure is never a reason to stop training)
    print(f'grids.png not written: {exc}')

# ---- Main Training Loop ----
for epoch in range(1, n_epochs + 1):
    t_epoch = time.time()
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
        loss = compute_loss(out.squeeze()[central_mask], target.squeeze()[central_mask])
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update_parameters(model)
            ema_updates += 1

        loss_avg = smooth * loss.item() + (1.0 - smooth) * loss_avg if loss_avg != 0.0 else loss.item()

    train_loss.append(loss_avg)

    # ------------------- VALIDATION (D-2): plain mean over all held-out graphs, in loss and in dex -------------------
    def validate(net):   # (weighted loss, RMS dex over the central nodes of the held-out columns)
        net.eval()
        loss_sum, sq_sum, n_batches, n_nodes = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for data in tqdm(loader_test, desc="Validating"):
                node, edge_attr, edge_index = data.x.to(device), data.edge_attr.to(device), data.edge_index.to(device)
                u, batch, target = data.u.to(device), data.batch.to(device), data.y.to(device)
                central_mask = data.central_mask.to(device)
                out = net(node, edge_attr, edge_index, u, batch)
                o, t = out.squeeze()[central_mask], target.squeeze()[central_mask]
                loss_sum += compute_loss(o, t).item(); n_batches += 1
                sq = (o - t) ** 2
                sq_sum += sq.sum().item(); n_nodes += o.numel()
        return loss_sum / max(n_batches, 1), np.sqrt(sq_sum / max(n_nodes, 1))

    val_loss_avg, val_dex = validate(model)
    _, val_dex_ema = validate(ema.module) if ema is not None else (None, None)
    valid_loss.append(val_loss_avg)
    valid_dex.append(val_dex)
    epoch_seconds = time.time() - t_epoch
    print(f"Epoch {epoch} finished with validation loss: {val_loss_avg:.6f} | held-out RMS {val_dex:.5f} dex"
          + (f" | EMA weights {val_dex_ema:.5f} dex" if val_dex_ema is not None else "") + f" | {epoch_seconds / 60:.1f} min")

    use_ema = val_dex_ema is not None and val_dex_ema < val_dex
    candidate = val_dex_ema if use_ema else val_dex
    best_loss = min(best_loss, valid_loss[-1])
    checkpoint = {
        'epoch': epoch,
        'state_dict': (ema.module if use_ema else model).state_dict(),
        'weights': 'ema' if use_ema else 'raw',
        'best_loss': best_loss,
        'best_dex': min(best_dex, candidate),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': config,
        'feature_norm_params': feature_norm_params,
        'normalization_params': pop_norm_params,
        'train_loss_history': train_loss,
        'valid_loss_history': valid_loss,
        'valid_dex_history': valid_dex,
        'meta': meta,
    }
    if candidate < best_dex:
        best_dex = candidate
        print(f"New best held-out dex: {best_dex:.5f} ({checkpoint['weights']} weights). Saving model...")
        torch.save(checkpoint, os.path.join(savedir, f'{filename}_best.pth'))
    # last.pth is the resume point: raw weights plus optimizer, scheduler and EMA state
    checkpoint['state_dict'], checkpoint['weights'] = model.state_dict(), 'raw'
    if ema is not None:
        checkpoint['ema'] = ema.state_dict()
        checkpoint['ema_updates'] = ema_updates
    torch.save(checkpoint, os.path.join(savedir, 'last.pth'))
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([epoch, filename, scheduler.get_last_lr()[0], loss_avg, valid_loss[-1], valid_dex[-1], val_dex_ema, best_dex,
                                'ema' if use_ema else 'raw', round(epoch_seconds, 1)])
    
    lrs.append(scheduler.get_last_lr())

    # ---- Plotting ----
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.plot(valid_dex, label='Held-out RMS [dex]')
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