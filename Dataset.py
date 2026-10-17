# DATASET CLASS DEFINITION

import os
import torch
import numpy as np
from torch_geometric.data import Data
from vertical_grids import random_two_zone_grid, ZTAU51_BIFROST


def split_columns(nx, ny, xdim, ydim, margin_x, margin_y, split_stride, train_ratio, split, seed):
    """Sample centres of a split: the columns at least margin_x / margin_y pixels from the border, shuffled with `seed`,
    restricted to the region of the split. The train/test boundary depends on split_stride (the training stride), so that
    inference at another stride or a training run with other hops uses the same held-out corner.
    Returns (centres, (x0, x1, y0, y1), number of valid columns)."""
    valid_ix = np.arange(margin_x, nx - margin_x)
    valid_iy = np.arange(margin_y, ny - margin_y)
    all_indices = [(x, y) for x in valid_ix for y in valid_iy]

    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(all_indices)

    ss = split_stride
    x_threshold = int((nx - (xdim * ss)) * train_ratio) + (xdim * ss)
    y_threshold = int((ny - (ydim * ss)) * train_ratio) + (ydim * ss)

    if split == 'test':
        x0, y0 = x_threshold, y_threshold
        x1, y1 = nx - xdim * ss, ny - ydim * ss
        centres = [(x, y) for x, y in all_indices if x0 <= x < x1 and y0 <= y < y1]
    elif split == 'train':
        x0, y0 = xdim * ss, ydim * ss
        x1, y1 = x_threshold, y_threshold
        centres = [(x, y) for x, y in all_indices if x < x_threshold or y < y_threshold]
    elif split == 'full':
        x0, y0 = xdim * ss, ydim * ss
        x1, y1 = nx - xdim * ss, ny - ydim * ss
        centres = [(x, y) for x, y in all_indices if x0 <= x < x1 and y0 <= y < y1]
    else:
        raise ValueError("split must be 'train', 'test' or 'full'")
    return centres, (x0, x1, y0, y1), len(all_indices)


def physical_edges(z_nodes, is_central, lev_nodes, z_central, central_idx, edge_reach, central_reach):
    """Star-topology edges of the physical rule: node n -> central node c when |z_n - z_c| <= edge_reach (central_reach if n
    is itself central), or n is the central node adjacent in level; no self loops; plus c -> n for every non-central n.
    The callers pass only the nodes of the stencil disc. Returns an int64 array (2, E)."""
    dz = np.abs(z_nodes[:, None] - z_central[None, :])                                       # (N, n_central)
    reach = np.where(is_central[:, None], central_reach, edge_reach)
    adjacent = is_central[:, None] & (np.abs(lev_nodes[:, None] - lev_nodes[central_idx][None, :]) == 1)
    ok = (dz <= reach) | adjacent
    ok[central_idx, np.arange(len(central_idx))] = False                                       # no self loops
    src, dst = np.nonzero(ok)                                                                 # n -> c
    e1 = np.stack([src, central_idx[dst]])
    nc = ~is_central[src]                                                                     # c -> n, n not central
    e2 = np.stack([central_idx[dst[nc]], src[nc]])
    return np.concatenate([e1, e2], axis=1).astype(np.int64)


class EfficientDataset(torch.utils.data.Dataset):
    def __init__(self, list_X: list, list_Y: list, xdim: int, ydim: int, z_levels, pixel_Mm: float,
                 split: str = 'train', train_ratio: float = 0.75, epoch_size_fraction: float = 0.2,
                 max_stride: int = 1,
                 split_stride: int = None,
                 stencil_radius: float = 4.01, edge_reach_Mm: float = 0.5,
                 central_reach_Mm: float = None,
                 seed: int = 777):
        """
        Graphs on a fixed grid at a fixed hop (validation and inference): one graph per central column, the
        (2 xdim + 1) x (2 ydim + 1) stencil of columns around it, hop = max_stride pixels, all nz levels of every column,
        star topology (every edge touches the central column). list_X / list_Y: lists of (nz, ny, nx, ncomp) arrays on
        the grid; z_levels: heights of the nz levels [Mm]; pixel_Mm: horizontal pixel size [Mm]. Node features are the
        concatenated list_X columns; edge features (dz, d_h, |dr|) in Mm (dz = source minus target).

        Edges (physical rule): the column lies within stencil_radius stencil units of the centre and the height
        difference is <= edge_reach_Mm (adjacent levels of the central column are always linked), so the receptive field
        in km is the same for every stride and grid; the columns outside the disc stay as isolated nodes.
        central_reach_Mm (default edge_reach_Mm) is the reach between two nodes of the central column, so a long vertical
        context costs only n_levels^2 edges.
        """
        super(EfficientDataset, self).__init__()
        print(f'Dataset.py: Initializing {split} dataset...')
        self.epoch_size_fraction = epoch_size_fraction
        self.stencil_radius = stencil_radius
        self.edge_reach = edge_reach_Mm
        self.central_reach = edge_reach_Mm if central_reach_Mm is None else central_reach_Mm
        self.xdim = xdim
        self.ydim = ydim
        self.max_stride = max_stride

        # Convert everything directly to PyTorch tensors to avoid runtime conversions
        self.features = torch.tensor(np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_X], axis=1), dtype=torch.float32)
        self.targets = torch.tensor(np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_Y], axis=1), dtype=torch.float32)

        self.nz, self.ny, self.nx = list_X[0].shape[:-1]
        self.zz = np.asarray(z_levels, dtype=np.float64)
        assert len(self.zz) == self.nz, (len(self.zz), self.nz)
        self.pixel = float(pixel_Mm)

        # Physical coordinates (x, y, z) [Mm] of every node, in the (z, y, x) flattening of the cubes
        zgrid, ygrid, xgrid = np.meshgrid(self.zz, np.arange(self.ny) * self.pixel, np.arange(self.nx) * self.pixel, indexing='ij')
        self.grid_pos = torch.tensor(np.stack([xgrid.ravel(), ygrid.ravel(), zgrid.ravel()], axis=1), dtype=torch.float32)

        # split into train and test. The region boundaries depend on the stride; split_stride (default:
        # max_stride) lets inference at another stride reuse the region of the training run.
        self.split_stride = split_stride or self.max_stride
        self.sample_centers, (self.x0, self.x1, self.y0, self.y1), n_valid = split_columns(
            self.nx, self.ny, xdim, ydim, xdim * self.max_stride, ydim * self.max_stride, self.split_stride, train_ratio, split, seed)

        # PRE-ALLOCATIONS to save CPU cycles
        self.static_u = torch.zeros((1, 1), dtype=torch.float32)
        self._precompute_graph_templates()

        print(f'Dataset.py:  {split.capitalize()} dataset created. Total samples: {len(self.sample_centers)}')
        print(f'Dataset.py:  Features shape: {self.features.shape}, Targets shape: {self.targets.shape}')
        print(f'Dataset.py:  Region: x ∈ [{self.x0}, {self.x1}), y ∈ [{self.y0}, {self.y1})')
        print(f'Dataset.py:  Split ratio: {len(self.sample_centers) / n_valid * 100:.2f}% of all valid samples')
        print(f'Dataset.py:  Stride: {self.max_stride}, split stride: {self.split_stride}')
        print(f'Dataset.py:  Edges: stencil radius {self.stencil_radius}, vertical reach {self.edge_reach} Mm, central column {self.central_reach} Mm, '
              f'edges per graph: {self.templates[(self.max_stride, self.max_stride)]["edge_index"].shape[1]}')

    def _precompute_graph_templates(self):
        """
        Calculates the edge connections and flat index offsets of the stencil ONCE (one template: the fixed hop).
        """
        self.templates = {}
        k_range = np.arange(self.nz)

        for xs in (self.max_stride,):
            for ys in (self.max_stride,):
                x_offs = np.arange(-self.xdim, self.xdim + 1) * xs
                y_offs = np.arange(-self.ydim, self.ydim + 1) * ys

                kv, yv, xv = np.meshgrid(k_range, y_offs, x_offs, indexing='ij')

                # Precompute the relative flatten offset for 1D slicing
                relative_flat = kv.ravel() * self.ny * self.nx + yv.ravel() * self.nx + xv.ravel()

                central_mask = (yv.ravel() == 0) & (xv.ravel() == 0)

                # columns within stencil_radius stencil units; levels within edge_reach Mm (central_reach between central
                # nodes), or adjacent in the central column; the columns outside the disc stay as isolated nodes
                kk = kv.ravel()
                disc = np.where(np.sqrt((xv.ravel() / xs) ** 2 + (yv.ravel() / ys) ** 2) <= self.stencil_radius)[0]
                central_idx = np.searchsorted(disc, np.where(central_mask)[0])
                e = physical_edges(self.zz[kk[disc]], central_mask[disc], kk[disc], self.zz[kk[disc][central_idx]], central_idx,
                                   self.edge_reach, self.central_reach)
                edge_index = torch.tensor(disc[e], dtype=torch.long)

                self.templates[(xs, ys)] = {
                    'relative_flat': torch.tensor(relative_flat, dtype=torch.long),
                    'edge_index': edge_index,
                    'central_mask': torch.tensor(central_mask, dtype=torch.bool),
                }

    def __len__(self):
        return int(len(self.sample_centers)*self.epoch_size_fraction)

    def __getitem__(self, index):
        ix, iy = self.sample_centers[index]

        # Fetch the precomputed topology template
        template = self.templates[(self.max_stride, self.max_stride)]

        # Shift the precomputed flattened indices to the target absolute center
        center_flat = iy * self.nx + ix
        flat_indices = template['relative_flat'] + center_flat

        # Direct tensor indexing (avoids data copies and from_numpy conversions)
        node_features = self.features[flat_indices]
        node_targets = self.targets[flat_indices]

        sub_physical_pos = self.grid_pos[flat_indices]

        edge_index = template['edge_index']
        row, col = edge_index
        edge_vectors = sub_physical_pos[row] - sub_physical_pos[col]
        # signed dz (source minus target), horizontal distance, |dr|; all in Mm
        edge_attr = torch.stack([edge_vectors[:, 2], edge_vectors[:, :2].norm(dim=1), edge_vectors.norm(dim=1)], dim=1)

        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_targets
        )

        graph_data.central_mask = template['central_mask']
        graph_data.u = self.static_u

        return graph_data


class RandomGeometryDataset(torch.utils.data.Dataset):
    def __init__(self, native_X, native_Y, z_native, pixel_Mm: float, sample_centers, xdim: int, ydim: int,
                 stencil_radius: float = 4.01, edge_reach_Mm: float = 0.5, central_reach_Mm: float = None,
                 max_stride: int = 4,
                 augment_rotation: bool = False, rot_pairs=((0, 1),), epoch_size_fraction: float = 0.2, log_offset: float = 1e-12):
        """
        Training graphs with a random sampling of the atmosphere per sample (plan rung 5, D-9). The cube stays at its native
        levels (native_X: normalized features, native_Y: log10(f + log_offset), both (nz, ny, nx, ncomp) on z_native [Mm,
        FITS frame]); per sample:
        - a two-zone vertical grid in height above tau500 = 1 (vertical_grids.random_two_zone_grid: dense spacing,
          transition height, number of sparse levels, jitter and interior-level removal; bottom and top native levels kept);
        - the hops of the stencil: integer strides 1..max_stride per axis;
        - with augment_rotation one rotation by 0/90/180/270 deg about z: (vx, vy) rotated, hops swapped for 90/270 deg.
        Features and targets are interpolated linearly in height (log quantities and velocities); the target fractions are
        renormalized to sum to one. Nodes: the columns of the stencil disc (i^2 + j^2 <= stencil_radius^2) at the drawn
        levels; edges: the physical rule (physical_edges), features (dz, d_h, |dr|) in Mm as in EfficientDataset.
        The central column is always a native column, so its targets are exact native targets interpolated in height.
        """
        super().__init__()
        self.features = torch.tensor(native_X.reshape(-1, native_X.shape[-1]), dtype=torch.float32)
        self.targets = torch.tensor(native_Y.reshape(-1, native_Y.shape[-1]), dtype=torch.float32)
        self.nz, self.ny, self.nx = native_X.shape[:3]
        self.z_native = np.asarray(z_native, dtype=np.float64)
        assert len(self.z_native) == self.nz and np.all(np.diff(self.z_native) > 0)
        self.h_bottom, self.h_top = self.z_native[0] - ZTAU51_BIFROST, self.z_native[-1] - ZTAU51_BIFROST
        self.pixel = float(pixel_Mm)
        self.sample_centers = list(sample_centers)
        self.max_stride = max_stride
        self.edge_reach = edge_reach_Mm
        self.central_reach = edge_reach_Mm if central_reach_Mm is None else central_reach_Mm
        self.augment_rotation = augment_rotation
        self.rot_pairs = [tuple(p) for p in rot_pairs]
        self.epoch_size_fraction = epoch_size_fraction
        self.log_offset = log_offset
        jj, ii = np.meshgrid(np.arange(-ydim, ydim + 1), np.arange(-xdim, xdim + 1), indexing='ij')
        disc = ii ** 2 + jj ** 2 <= stencil_radius ** 2
        self.cols_i, self.cols_j = ii[disc], jj[disc]                                         # stencil units
        self.centre_col = int(np.where((self.cols_i == 0) & (self.cols_j == 0))[0][0])
        self.margin_px = self.stencil_margin(xdim, ydim, max_stride)
        reach_px = self.margin_px
        for ix, iy in self.sample_centers:                                                    # every stencil inside the cube
            assert reach_px <= ix < self.nx - reach_px and reach_px <= iy < self.ny - reach_px, (ix, iy, reach_px)
        self.static_u = torch.zeros((1, 1), dtype=torch.float32)
        self._gen, self._pid = None, None
        print(f'Dataset.py:  Random-geometry training dataset: {len(self.sample_centers)} centres, {len(self.cols_i)} stencil columns, '
              f'hops strides 1-{self.max_stride}, reach {self.edge_reach} Mm (central {self.central_reach} Mm), rotation {self.augment_rotation}')

    @staticmethod
    def stencil_margin(xdim, ydim, max_stride):
        """Pixels the stencil can reach from its centre: max offset x max hop."""
        return int(max(xdim, ydim) * max_stride)

    def __len__(self):
        return int(len(self.sample_centers) * self.epoch_size_fraction)

    def _rng(self):                         # one generator per process, seeded from the per-worker numpy state set by the DataLoader
        if self._gen is None or self._pid != os.getpid():
            self._gen, self._pid = np.random.default_rng(np.random.randint(0, 2 ** 31 - 1)), os.getpid()
        return self._gen

    def __getitem__(self, index):
        rng = self._rng()
        ix, iy = self.sample_centers[index]

        # vertical grid, heights above tau500 = 1 -> FITS frame; linear interpolation weights on the native levels
        h, _, _ = random_two_zone_grid(self.h_bottom, self.h_top, rng)
        zl = h + ZTAU51_BIFROST
        nl = len(zl)
        fi = np.interp(zl, self.z_native, np.arange(self.nz))
        k0 = np.clip(np.floor(fi).astype(np.int64), 0, self.nz - 2)
        wz = fi - k0

        # hops [px] and rotation
        hx, hy = (float(v) for v in rng.integers(1, self.max_stride + 1, size=2))
        rot = int(rng.integers(4)) if self.augment_rotation else 0
        if rot in (1, 3):
            hx, hy = hy, hx
        ci, cj, c0 = self.cols_i, self.cols_j, self.centre_col
        nc = len(ci)

        # gather: 2 native levels per node, node n = c * nl + l
        px, py = ix + ci * hx, iy + cj * hy
        x0, y0 = np.floor(px + 1e-9).astype(np.int64), np.floor(py + 1e-9).astype(np.int64)
        cx, cy, wc = x0[:, None], y0[:, None], np.ones((nc, 1))
        kk = np.stack([k0, k0 + 1], -1)                                                      # (nl, 2)
        wk = np.stack([1.0 - wz, wz], -1)
        flat = kk[None, :, :, None] * (self.ny * self.nx) + (cy * self.nx + cx)[:, None, None, :]   # (nc, nl, 2, ncorner)
        w = wk[None, :, :, None] * wc[:, None, None, :]
        flat = torch.from_numpy(flat.reshape(nc * nl, -1))
        w = torch.from_numpy(w.reshape(nc * nl, -1)).float().unsqueeze(-1)
        node_features = (self.features[flat] * w).sum(1)
        node_targets = (self.targets[flat] * w).sum(1)
        f = torch.clamp(10.0 ** node_targets - self.log_offset, min=0.0)                    # renormalize the interpolated fractions
        node_targets = torch.log10(f / f.sum(1, keepdim=True) + self.log_offset)

        if rot:                                # 90 deg: (vx, vy) -> (-vy, vx); 180: (-vx, -vy); 270: (vy, -vx)
            for i, j in self.rot_pairs:
                vx, vy = node_features[:, i].clone(), node_features[:, j].clone()
                if rot == 1:
                    node_features[:, i], node_features[:, j] = -vy, vx
                elif rot == 2:
                    node_features[:, i], node_features[:, j] = -vx, -vy
                else:
                    node_features[:, i], node_features[:, j] = vy, -vx

        # geometry [Mm] and edges (physical rule)
        pos_x = np.repeat(ci * hx * self.pixel, nl)
        pos_y = np.repeat(cj * hy * self.pixel, nl)
        pos_z = np.tile(zl, nc)
        lev = np.tile(np.arange(nl), nc)
        central_idx = np.arange(c0 * nl, (c0 + 1) * nl)
        is_central = np.zeros(nc * nl, dtype=bool)
        is_central[central_idx] = True
        edge_index = torch.from_numpy(physical_edges(pos_z, is_central, lev, zl, central_idx, self.edge_reach, self.central_reach))
        pos = torch.tensor(np.stack([pos_x, pos_y, pos_z], axis=1), dtype=torch.float32)
        edge_vectors = pos[edge_index[0]] - pos[edge_index[1]]
        edge_attr = torch.stack([edge_vectors[:, 2], edge_vectors[:, :2].norm(dim=1), edge_vectors.norm(dim=1)], dim=1)

        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=node_targets)
        graph_data.central_mask = torch.from_numpy(is_central)
        graph_data.u = self.static_u
        return graph_data
