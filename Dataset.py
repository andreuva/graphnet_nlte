# DATASET CLASS DEFINITION

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph

class EfficientDataset(torch.utils.data.Dataset):
    def __init__(self, list_X: list, list_Y: list, radius_neighbors, xdim, ydim, pos_file=None, seed=777, train_ratio=0.75, split='train', device='cpu', 
                 logspace_fraction=0.4, nz_linear=30, nz_log=25):
        super(EfficientDataset, self).__init__()
        self.device = device
        self.radius = radius_neighbors
        self.xdim = xdim
        self.ydim = ydim

        # Store data as numpy arrays to keep them on CPU until needed
        self.features = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_X], axis=1)
        self.targets = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_Y], axis=1)

        self.nz, self.ny, self.nx = list_X[0].shape[:-1]

        # Load physical coordinates from file
        grid_pos_data = np.load(pos_file)
        self.xx = grid_pos_data['x']
        self.yy = grid_pos_data['y']
        zz_orig = grid_pos_data['z']
        nz_original = len(zz_orig)

        # Create the same interpolated z-grid as in the training script
        new_z_indices = np.concatenate([
            np.linspace(0, nz_original * logspace_fraction, nz_linear, endpoint=False),
            np.logspace(np.log10(nz_original * logspace_fraction), np.log10(nz_original - 1), nz_log)
        ])
        new_z_indices = np.clip(new_z_indices, 0, nz_original - 1)
        self.zz = np.interp(new_z_indices, np.arange(nz_original), zz_orig)
        
        # Create the full grid of physical coordinates
        zgrid, ygrid, xgrid = np.meshgrid(self.zz, self.yy, self.xx, indexing='ij')
        self.grid_pos = torch.tensor(np.stack([zgrid.ravel(), ygrid.ravel(), xgrid.ravel()], axis=1), dtype=torch.float)
        print(f'Physical Z coordinates range from {self.zz.min()} to {self.zz.max()} in {len(self.zz)} steps.')

        valid_ix = np.arange(xdim, self.nx - xdim)
        valid_iy = np.arange(ydim, self.ny - ydim)
        all_indices = [(x, y) for x in valid_ix for y in valid_iy]

        # Shuffle
        np.random.seed(seed)
        np.random.shuffle(all_indices)

        # split into train and test
        x_threshold = int((self.nx - xdim) * train_ratio) + xdim
        y_threshold = int((self.ny - ydim) * train_ratio) + ydim

        if split == 'test':
            self.sample_centers = [(x, y) for x, y in all_indices if x >= x_threshold and y >= y_threshold]
        elif split == 'train':
            self.sample_centers = [(x, y) for x, y in all_indices if x < x_threshold or y < y_threshold]
        else:
            raise ValueError("split must be 'train' or 'test'")

        print(f'{split.capitalize()} dataset created. Total samples: {len(self.sample_centers)}')
        print(f'Features shape: {self.features.shape}, Targets shape: {self.targets.shape}')
        if split == 'test':
            print(f'Test region: x ∈ [{x_threshold}, {self.nx - xdim}), y ∈ [{y_threshold}, {self.ny - ydim})')
        print(f'Split ratio: {len(self.sample_centers) / len(all_indices) * 100:.2f}% of all valid samples')

    def __len__(self):
        return len(self.sample_centers)//10

    def grid_to_graph_manual(self, grid_points, values=None, targets=None, r=1.5, xpos=None, ypos=None):
        if values is None:
            values = grid_points
        if targets is None:
            targets = grid_points

        dist_matrix = torch.cdist(grid_points, grid_points)
        edge_indices_tuple = torch.where((dist_matrix > 0) & (dist_matrix <= r))
        edge_index = torch.stack(edge_indices_tuple, dim=0)
        
        graph_data = Data(x=values, pos=grid_points, y=targets, edge_index=edge_index)

        if xpos is not None and ypos is not None:
            # Use a small tolerance for floating point comparison
            central_nodes = torch.where(
                (torch.isclose(grid_points[:, 2], torch.tensor(xpos, dtype=torch.float))) &
                (torch.isclose(grid_points[:, 1], torch.tensor(ypos, dtype=torch.float)))
            )[0]

            if len(central_nodes) > 0:
                edge_mask = torch.from_numpy(
                    np.isin(graph_data.edge_index[0, :], central_nodes) |
                    np.isin(graph_data.edge_index[1, :], central_nodes)
                )
                graph_data.edge_index = graph_data.edge_index[:, edge_mask]

        return graph_data

    def __getitem__(self, index):
        ix, iy = self.sample_centers[index]

        # Define index ranges for the sub-volume
        y_range = np.arange(iy - self.ydim, iy + self.ydim + 1)
        x_range = np.arange(ix - self.xdim, ix + self.xdim + 1)
        k_range = np.arange(self.nz)

        # Create a grid of indices for the sub-volume
        kv, yv, xv = np.meshgrid(k_range, y_range, x_range, indexing='ij')

        # Calculate flat indices to slice the full data arrays
        flat_indices = (kv.ravel() * self.ny * self.nx + yv.ravel() * self.nx + xv.ravel())

        # 1. Get the physical positions for ONLY this sub-grid
        sub_physical_pos = self.grid_pos[flat_indices]

        # 2. Get the original features and targets for this sub-grid
        node_features_original = torch.tensor(self.features[flat_indices], dtype=torch.float)
        node_targets = torch.tensor(self.targets[flat_indices], dtype=torch.float)

        # 3. Extract, normalize, and append the z-coordinate to features
        z_coords = sub_physical_pos[:, 0]
        node_features = torch.cat([node_features_original, z_coords.unsqueeze(1)], dim=1)

        # 4. Use physical positions for graph construction. Pass physical center coordinates for filtering.
        phys_x_center = self.xx[ix]
        phys_y_center = self.yy[iy]
        graph_data = self.grid_to_graph_manual(
            sub_physical_pos, node_features, node_targets, 
            r=self.radius, xpos=phys_x_center, ypos=phys_y_center
        )

        # 5. Calculate edge attributes as physical distance (vector norm)
        if graph_data.edge_attr is None:
            row, col = graph_data.edge_index
            # graph_data.pos now contains physical coordinates
            edge_vectors = graph_data.pos[row] - graph_data.pos[col]
            # The edge attribute is the physical distance
            graph_data.edge_attr = edge_vectors.norm(dim=1).unsqueeze(1)

        # 6. Add a placeholder for the global attributes 'u'
        graph_data.u = torch.zeros((1, 1), dtype=torch.float)
        return graph_data