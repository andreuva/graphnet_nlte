# DATASET CLASS DEFINITION

import torch
import numpy as np
from torch_geometric.data import Data

class EfficientDataset(torch.utils.data.Dataset):
    def __init__(self, list_X: list, list_Y: list, xdim: int, ydim: int, z_levels, pixel_Mm: float,
                 split: str = 'train', train_ratio: float = 0.75, epoch_size_fraction: float = 0.2,
                 max_stride: int = 1,
                 random_stride: bool = False,
                 split_stride: int = None,
                 edge_rule: str = 'index', radius_neighbors: float = 6.01, stencil_radius: float = 4.01, edge_reach_Mm: float = 0.5,
                 augment_rotation: bool = False,
                 rot_pairs=((0, 1),),
                 seed: int = 777):
        """
        One graph per central column: the (2 xdim + 1) x (2 ydim + 1) stencil of columns around it, hop = `stride` pixels
        (drawn in 1..max_stride per sample when random_stride), all nz levels of every column, star topology (every edge
        touches the central column). list_X / list_Y: lists of (nz, ny, nx, ncomp) arrays on the training grid;
        z_levels: heights of the nz levels [Mm]; pixel_Mm: horizontal pixel size [Mm]. Node features are the
        concatenated list_X columns; edge features (dz, d_h, |dr|) in Mm (dz = source minus target).

        edge_rule 'index' (ladder rungs 0-4): a node is linked to a central node when their distance in index units
        (level, pixel, pixel) is <= radius_neighbors * (xs + ys) / 3, so the vertical reach grows with the stride.
        edge_rule 'physical' (rung 5a): the column lies within stencil_radius stencil units of the centre and the
        height difference is <= edge_reach_Mm (adjacent levels of the central column are always linked), so the
        receptive field in km is the same for every stride and grid.
        """
        super(EfficientDataset, self).__init__()
        print(f'Dataset.py: Initializing {split} dataset...')
        self.epoch_size_fraction = epoch_size_fraction
        self.edge_rule = edge_rule
        self.radius = radius_neighbors
        self.stencil_radius = stencil_radius
        self.edge_reach = edge_reach_Mm
        self.xdim = xdim
        self.ydim = ydim
        self.max_stride = max_stride
        self.random_stride = random_stride
        self.augment_rotation = augment_rotation  # rung 4 (D-7): one random rotation about z per sample (training only)
        self.rot_pairs = [tuple(p) for p in rot_pairs]   # (x, y) feature-column pairs rotated as vectors, e.g. (vx, vy)

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

        valid_ix = np.arange(xdim * self.max_stride, self.nx - xdim * self.max_stride)
        valid_iy = np.arange(ydim * self.max_stride, self.ny - ydim * self.max_stride)
        all_indices = [(x, y) for x in valid_ix for y in valid_iy]

        # Shuffle
        np.random.seed(seed)
        np.random.shuffle(all_indices)

        # split into train and test. The region boundaries depend on the stride; split_stride (default:
        # max_stride) lets inference at another stride reuse the region of the training run.
        self.split_stride = split_stride or self.max_stride
        ss = self.split_stride
        x_threshold = int((self.nx - (xdim * ss)) * train_ratio) + (xdim * ss)
        y_threshold = int((self.ny - (ydim * ss)) * train_ratio) + (ydim * ss)

        if split == 'test':
            self.x0, self.y0 = x_threshold, y_threshold
            self.x1, self.y1 = self.nx - xdim * ss, self.ny - ydim * ss
            self.sample_centers = [(x, y) for x, y in all_indices
                                   if self.x0 <= x < self.x1 and self.y0 <= y < self.y1]
        elif split == 'train':
            self.x0, self.y0 = xdim * ss, ydim * ss
            self.x1, self.y1 = x_threshold, y_threshold
            self.sample_centers = [(x, y) for x, y in all_indices if x < x_threshold or y < y_threshold]
        elif split == 'full':
            self.x0, self.y0 = xdim * ss, ydim * ss
            self.x1, self.y1 = self.nx - xdim * ss, self.ny - ydim * ss
            self.sample_centers = [(x, y) for x, y in all_indices
                                   if self.x0 <= x < self.x1 and self.y0 <= y < self.y1]
        else:
            raise ValueError("split must be 'train', 'test' or 'full'")

        # PRE-ALLOCATIONS to save CPU cycles
        self.static_u = torch.zeros((1, 1), dtype=torch.float32)
        self._precompute_graph_templates()

        print(f'Dataset.py:  {split.capitalize()} dataset created. Total samples: {len(self.sample_centers)}')
        print(f'Dataset.py:  Features shape: {self.features.shape}, Targets shape: {self.targets.shape}')
        print(f'Dataset.py:  Region: x ∈ [{self.x0}, {self.x1}), y ∈ [{self.y0}, {self.y1})')
        print(f'Dataset.py:  Split ratio: {len(self.sample_centers) / len(all_indices) * 100:.2f}% of all valid samples')
        print(f'Dataset.py:  Using max stride: {self.max_stride}, Random stride: {self.random_stride}, Split stride: {self.split_stride}, '
              f'rotation augmentation: {self.augment_rotation}')
        print(f'Dataset.py:  Edge rule: {self.edge_rule} ' + (f'(radius {self.radius} index units)' if self.edge_rule == 'index' else
              f'(stencil radius {self.stencil_radius}, vertical reach {self.edge_reach} Mm)') + f', edges per graph: {self.templates[(self.max_stride, self.max_stride)]["edge_index"].shape[1]}')

    def _precompute_graph_templates(self):
        """
        Calculates all possible edge connections and flat index offsets ONCE.
        This removes O(N^2) cdist calculations from the __getitem__ loop.
        """
        self.templates = {}
        k_range = np.arange(self.nz)

        for xs in range(1, self.max_stride + 1):
            for ys in range(1, self.max_stride + 1):
                x_offs = np.arange(-self.xdim, self.xdim + 1) * xs
                y_offs = np.arange(-self.ydim, self.ydim + 1) * ys

                kv, yv, xv = np.meshgrid(k_range, y_offs, x_offs, indexing='ij')

                # Precompute the relative flatten offset for 1D slicing
                relative_flat = kv.ravel() * self.ny * self.nx + yv.ravel() * self.nx + xv.ravel()

                # Mock indices to calculate fixed graph topology
                node_pos_mock = torch.tensor(np.stack([kv.ravel(), yv.ravel(), xv.ravel()], axis=1), dtype=torch.float32)

                central_mask = (yv.ravel() == 0) & (xv.ravel() == 0)

                if self.edge_rule == 'index':
                    # edges within radius_neighbors * (xs + ys) / 3 in index units (level, pixel, pixel), star topology
                    r_eff = self.radius * (xs + ys) / 3
                    dist_matrix = torch.cdist(node_pos_mock, node_pos_mock)
                    edge_indices_tuple = torch.where((dist_matrix > 0) & (dist_matrix <= r_eff))
                    edge_index = torch.stack(edge_indices_tuple, dim=0)

                    central_nodes = torch.where(torch.tensor(central_mask))[0]
                    edge_mask = torch.isin(edge_index[0], central_nodes) | torch.isin(edge_index[1], central_nodes)
                    edge_index = edge_index[:, edge_mask]
                elif self.edge_rule == 'physical':
                    # columns within stencil_radius stencil units; levels within edge_reach Mm, or adjacent in the central column
                    kk = kv.ravel()
                    disc = np.sqrt((xv.ravel() / xs) ** 2 + (yv.ravel() / ys) ** 2) <= self.stencil_radius
                    central_idx = np.where(central_mask)[0]
                    dz = np.abs(self.zz[kk][:, None] - self.zz[kk[central_idx]][None, :])                  # (N, nz)
                    adjacent = central_mask[:, None] & (np.abs(kk[:, None] - kk[central_idx][None, :]) == 1)
                    ok = disc[:, None] & ((dz <= self.edge_reach) | adjacent)
                    ok[central_idx, np.arange(len(central_idx))] = False                                   # no self loops
                    src, dst = np.nonzero(ok)                                                             # n -> c
                    e1 = np.stack([src, central_idx[dst]])
                    nc = ~central_mask[src]                                                               # c -> n, n not central
                    e2 = np.stack([central_idx[dst[nc]], src[nc]])
                    edge_index = torch.tensor(np.concatenate([e1, e2], axis=1), dtype=torch.long)
                else:
                    raise ValueError(f"unknown edge_rule '{self.edge_rule}'")

                self.templates[(xs, ys)] = {
                    'relative_flat': torch.tensor(relative_flat, dtype=torch.long),
                    'edge_index': edge_index,
                    'central_mask': torch.tensor(central_mask, dtype=torch.bool),
                    'relative_pos': node_pos_mock
                }

    def __len__(self):
        return int(len(self.sample_centers)*self.epoch_size_fraction)

    def __getitem__(self, index):
        ix, iy = self.sample_centers[index]

        # A stride of 1 is the original contiguous block.
        if self.random_stride:
            x_stride = np.random.randint(1, self.max_stride + 1)
            y_stride = np.random.randint(1, self.max_stride + 1)
        else:
            x_stride = self.max_stride
            y_stride = self.max_stride

        # Rotation of the whole volume about z by rot * 90 deg (rung 4, D-7): the rotated stencil of strides (xs, ys)
        # is the (ys, xs) stencil of the original volume for 90/270 deg; (vx, vy) rotate below, nothing else changes
        rot = np.random.randint(4) if self.augment_rotation else 0
        if rot in (1, 3):
            x_stride, y_stride = y_stride, x_stride

        # Fetch the precomputed topology template
        template = self.templates[(x_stride, y_stride)]

        # Shift the precomputed flattened indices to the target absolute center
        center_flat = iy * self.nx + ix
        flat_indices = template['relative_flat'] + center_flat

        # Direct tensor indexing (avoids data copies and from_numpy conversions)
        node_features = self.features[flat_indices]
        node_targets = self.targets[flat_indices]

        if rot:                                # 90 deg: (vx, vy) -> (-vy, vx); 180: (-vx, -vy); 270: (vy, -vx)
            for i, j in self.rot_pairs:
                vx, vy = node_features[:, i].clone(), node_features[:, j].clone()
                if rot == 1:
                    node_features[:, i], node_features[:, j] = -vy, vx
                elif rot == 2:
                    node_features[:, i], node_features[:, j] = -vx, -vy
                else:
                    node_features[:, i], node_features[:, j] = vy, -vx

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
