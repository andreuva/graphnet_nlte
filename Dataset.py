# DATASET CLASS DEFINITION

import torch
import numpy as np
from torch_geometric.data import Data
from vertical_grids import interp_weights, random_grid


def star_graph(z, col_x, col_y, c0, edge_reach, central_reach):
    """
    Edges of one graph, their features and the central mask. Nodes are level-major, node = level * ncol + column, with
    z (nl,) the heights of the levels [Mm] and col_x, col_y (ncol,) the horizontal offsets of the columns from the
    central column c0 [Mm].
    Star topology, every edge touches the central column: a node of another column and a central node are linked (in
    both directions) when their height difference is <= edge_reach; two central nodes when it is <= central_reach or
    their levels are adjacent; no self loops. Edge features: dz (source minus target), horizontal distance, |dr| [Mm].
    """
    nl, ncol = len(z), len(col_x)
    dz = np.abs(z[:, None] - z[None, :])                       # [level of a node, level of a central node]
    l, m = np.nonzero(dz <= edge_reach + 1e-9)                 # 1e-9: a tie at the reach does not depend on rounding
    adjacent = np.abs(np.arange(nl)[:, None] - np.arange(nl)[None, :]) == 1
    lc, mc = np.nonzero(((dz <= central_reach + 1e-9) | adjacent) & ~np.eye(nl, dtype=bool))
    others = np.delete(np.arange(ncol), c0)
    src = (l[None, :] * ncol + others[:, None]).ravel()        # node (l, column) ...
    dst = np.tile(m * ncol + c0, len(others))                  # ... -> central node (m, c0)
    edge_index = np.concatenate([np.stack([src, dst]), np.stack([dst, src]),
                                 np.stack([lc * ncol + c0, mc * ncol + c0])], axis=1)

    pos = np.stack([np.tile(col_x, nl), np.tile(col_y, nl), np.repeat(z, ncol)], axis=1)
    d = pos[edge_index[0]] - pos[edge_index[1]]
    d_h = np.hypot(d[:, 0], d[:, 1])
    edge_attr = np.stack([d[:, 2], d_h, np.hypot(d_h, d[:, 2])], axis=1)
    central_mask = np.arange(nl * ncol) % ncol == c0
    return torch.from_numpy(edge_index), torch.tensor(edge_attr, dtype=torch.float32), torch.from_numpy(central_mask)


class EfficientDataset(torch.utils.data.Dataset):
    def __init__(self, list_X: list, list_Y: list, z_native, pixel_Mm: float, xdim: int, ydim: int,
                 z_levels=None, random_geometry: bool = False,
                 split: str = 'train', train_ratio: float = 0.7,
                 max_stride: int = 1, split_stride: int = None,
                 stencil_radius: float = 4.01, edge_reach_Mm: float = 0.5, central_reach_Mm: float = 2.0,
                 log_offset: float = 1e-12):
        """
        One graph per central column of a cube given on its native levels. list_X: normalized inputs, list_Y:
        log10(fractions + log_offset) (or None), lists of (nz, ny, nx, ncomp) arrays; z_native: heights of the nz native
        levels [Mm above tau500 = 1]; pixel_Mm: horizontal pixel size [Mm].
        Nodes: the columns of the (2 xdim + 1) x (2 ydim + 1) stencil that lie within stencil_radius stencil units of
        the centre, at the levels z_levels, with hops of max_stride pixels; every node is interpolated linearly in height
        between the two native levels around it. With random_geometry (training) every sample draws its own levels
        (vertical_grids.random_grid), hops of 1..max_stride pixels per axis and a rotation about the vertical axis.
        Edges and edge features: star_graph.
        Split: the centres whose stencil fits in the cube; 'test' = the corner beyond train_ratio of the domain, 'train'
        = the rest, 'full' = all. The corner is the one of the training stride split_stride (default max_stride).
        """
        super(EfficientDataset, self).__init__()
        print(f'Dataset.py: Initializing {split} dataset...')
        self.max_stride = max_stride
        self.random_geometry = random_geometry
        self.pixel = pixel_Mm
        self.edge_reach = edge_reach_Mm
        self.central_reach = central_reach_Mm
        self.log_offset = log_offset

        # Convert everything directly to PyTorch tensors to avoid runtime conversions
        self.features = torch.from_numpy(np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_X], axis=1, dtype=np.float32))
        self.targets = None if list_Y is None else \
            torch.from_numpy(np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_Y], axis=1, dtype=np.float32))

        self.nz, self.ny, self.nx = list_X[0].shape[:-1]
        self.z_native = np.asarray(z_native, dtype=np.float64)
        self.z_levels = None if z_levels is None else np.asarray(z_levels, dtype=np.float64)
        assert len(self.z_native) == self.nz, (len(self.z_native), self.nz)

        # Columns of the stencil within stencil_radius (in stencil units) and the central one
        jj, ii = np.meshgrid(np.arange(-ydim, ydim + 1), np.arange(-xdim, xdim + 1), indexing='ij')
        disc = ii ** 2 + jj ** 2 <= stencil_radius ** 2
        self.cols_i, self.cols_j = ii[disc], jj[disc]
        self.c0 = int(np.flatnonzero((self.cols_i == 0) & (self.cols_j == 0))[0])
        assert not random_geometry or xdim == ydim, 'the rotation about the vertical axis needs a square stencil'

        valid_ix = np.arange(xdim * self.max_stride, self.nx - xdim * self.max_stride)
        valid_iy = np.arange(ydim * self.max_stride, self.ny - ydim * self.max_stride)
        all_indices = [(x, y) for x in valid_ix for y in valid_iy]

        # split into train and test; the boundary is the one of the training stride split_stride, so that inference at
        # any stride evaluates the same held-out corner
        split_stride = split_stride or self.max_stride
        x_threshold = int((self.nx - (xdim * split_stride)) * train_ratio) + (xdim * split_stride)
        y_threshold = int((self.ny - (ydim * split_stride)) * train_ratio) + (ydim * split_stride)

        if split == 'test':
            self.x0, self.y0 = x_threshold, y_threshold
            self.x1, self.y1 = self.nx - xdim * self.max_stride, self.ny - ydim * self.max_stride
            self.sample_centers = [(x, y) for x, y in all_indices if x >= x_threshold and y >= y_threshold]
        elif split == 'train':
            self.x0, self.y0 = xdim * self.max_stride, ydim * self.max_stride
            self.x1, self.y1 = x_threshold, y_threshold
            self.sample_centers = [(x, y) for x, y in all_indices if x < x_threshold or y < y_threshold]
        elif split == 'full':
            self.x0, self.y0 = xdim * self.max_stride, ydim * self.max_stride
            self.x1, self.y1 = self.nx - xdim * self.max_stride, self.ny - ydim * self.max_stride
            self.sample_centers = [(x, y) for x, y in all_indices]
        else:
            raise ValueError("split must be 'train', 'test' or 'full'")

        # PRE-ALLOCATIONS to save CPU cycles
        self.static_u = torch.zeros((1, 1), dtype=torch.float32)
        self._precompute_graph_templates()

        print(f'Dataset.py:  {split.capitalize()} dataset created. Total samples: {len(self.sample_centers)}')
        print(f'Dataset.py:  Features shape: {tuple(self.features.shape)} ({self.nz} native levels)')
        print(f'Dataset.py:  Region: x ∈ [{self.x0}, {self.x1}), y ∈ [{self.y0}, {self.y1})')
        print(f'Dataset.py:  Split ratio: {len(self.sample_centers) / len(all_indices) * 100:.2f}% of all valid samples')
        if self.random_geometry:
            print(f'Dataset.py:  Random geometry: levels, hops of 1-{self.max_stride} px and rotation drawn per sample')
        else:
            print(f'Dataset.py:  Stride: {self.max_stride} (split stride {split_stride}), {len(self.z_levels)} levels, '
                  f'{len(self.cols_i)} columns, {self.template["edge_index"].shape[1]} edges per graph')

    def _precompute_graph_templates(self):
        """
        With a fixed geometry the graph is the same for every centre: its template (flat index offsets, interpolation
        weights, edges) is calculated ONCE. With random_geometry __getitem__ builds the template of every sample.
        """
        self.template = None if self.random_geometry else self._graph_template(self.z_levels, self.max_stride, self.max_stride)

    def _graph_template(self, z_levels, x_stride, y_stride):
        """
        Everything of a graph that does not depend on its centre, for the levels z_levels [Mm] and hops of x_stride,
        y_stride pixels: the flat index offset of the native level below every node and the weight of the level above,
        the edges, the edge features and the central mask.
        """
        k0, w = interp_weights(self.z_native, z_levels)
        col = self.cols_j * y_stride * self.nx + self.cols_i * x_stride                 # (y, x) offsets of the columns
        relative_flat = (k0[:, None] * self.ny * self.nx + col[None, :]).ravel()        # node = level * ncol + column
        edge_index, edge_attr, central_mask = star_graph(z_levels, self.cols_i * x_stride * self.pixel,
                                                         self.cols_j * y_stride * self.pixel, self.c0,
                                                         self.edge_reach, self.central_reach)
        return {'relative_flat': torch.from_numpy(relative_flat),
                'w': torch.tensor(np.repeat(w, len(col))[:, None], dtype=torch.float32),
                'edge_index': edge_index, 'edge_attr': edge_attr, 'central_mask': central_mask}

    def __len__(self):
        return len(self.sample_centers)

    def __getitem__(self, index):
        ix, iy = self.sample_centers[index]

        rot = 0
        if self.random_geometry:
            # The geometry of this sample: its levels, hops of 1..max_stride pixels per axis and a rotation by rot x 90
            # degrees about the vertical axis. The edge features do not change under the rotation and the hops are
            # drawn independently per axis, so only (vx, vy) have to turn.
            z_levels = random_grid(self.z_native[0], self.z_native[-1])
            x_stride, y_stride = np.random.randint(1, self.max_stride + 1, size=2)
            rot = np.random.randint(4)
            template = self._graph_template(z_levels, x_stride, y_stride)
        else:
            # Fetch the precomputed topology template
            template = self.template

        # Shift the precomputed flattened indices to the target absolute center
        center_flat = iy * self.nx + ix
        flat_indices = template['relative_flat'] + center_flat

        # Direct tensor indexing of the native levels below and above every node, interpolated linearly in height
        w = template['w']
        node_features = (1 - w) * self.features[flat_indices] + w * self.features[flat_indices + self.ny * self.nx]
        if rot:                        # (vx, vy) are the inputs 0 and 1, scaled without offset
            c, s = ((1, 0), (0, 1), (-1, 0), (0, -1))[rot]
            vx, vy = node_features[:, 0].clone(), node_features[:, 1].clone()
            node_features[:, 0], node_features[:, 1] = c * vx - s * vy, s * vx + c * vy

        graph_data = Data(x=node_features, edge_index=template['edge_index'], edge_attr=template['edge_attr'])

        if self.targets is not None:
            # log10(fractions + log_offset), renormalized so that the interpolated fractions sum to one
            node_targets = (1 - w) * self.targets[flat_indices] + w * self.targets[flat_indices + self.ny * self.nx]
            fractions = torch.clamp(10.0 ** node_targets - self.log_offset, min=0.0)
            graph_data.y = torch.log10(fractions / fractions.sum(dim=1, keepdim=True) + self.log_offset)

        graph_data.central_mask = template['central_mask']
        graph_data.u = self.static_u

        return graph_data