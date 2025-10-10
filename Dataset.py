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
        self.features_base = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_X], axis=1)
        self.targets = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_Y], axis=1)

        self.nz, self.ny, self.nx = list_X[0].shape[:-1]

        grid_pos = np.load(pos_file)
        self.xx, self.yy, original_zz = grid_pos['x'], grid_pos['y'], grid_pos['z']
        nz_original = len(original_zz)

        new_z_indices = np.concatenate([
            np.linspace(0, nz_original*logspace_fraction, nz_linear, endpoint=False),
            np.logspace(np.log10(nz_original*logspace_fraction), np.log10(nz_original-1), nz_log)
        ])
        new_z_indices = np.clip(new_z_indices, 0, nz_original - 1)
        self.zz = np.interp(new_z_indices, np.arange(nz_original), original_zz)

        # Create the full grid of physical coordinates
        zgrid, ygrid, xgrid = np.meshgrid(self.zz, self.yy, self.xx, indexing='ij')
        self.grid_pos = torch.tensor(np.stack([zgrid.ravel(), ygrid.ravel(), xgrid.ravel()], axis=1), dtype=torch.float)

        z_feature = zgrid.ravel()[:, np.newaxis]
        self.features = np.concatenate([self.features_base, z_feature], axis=1)

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

    def grid_to_graph_manual(self, grid_points, values=None, targets=None, r=0.25, xpos_idx=None, ypos_idx=None):
        if values is None:
            values = grid_points
        if targets is None:
            targets = grid_points

        dist_matrix = torch.cdist(grid_points, grid_points)
        edge_indices_tuple = torch.where((dist_matrix > 0) & (dist_matrix <= r))
        edge_index = torch.stack(edge_indices_tuple, dim=0)

        graph_data = Data(x=values, pos=grid_points, y=targets, edge_index=edge_index)

        if xpos_idx is not None and ypos_idx is not None:
            # Get physical coordinates of the central column from the provided indices
            central_x_coord = self.xx[xpos_idx]
            central_y_coord = self.yy[ypos_idx]
            
            # Find nodes that belong to the central column by comparing physical coordinates
            central_nodes = torch.where(
                (torch.isclose(grid_points[:, 2], torch.tensor(central_x_coord, dtype=torch.float))) & 
                (torch.isclose(grid_points[:, 1], torch.tensor(central_y_coord, dtype=torch.float)))
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

        y_range = np.arange(iy - self.ydim, iy + self.ydim + 1)
        x_range = np.arange(ix - self.xdim, ix + self.xdim + 1)
        k_range = np.arange(self.nz)

        # Create a grid of indices for the sub-volume
        kv, yv, xv = np.meshgrid(k_range, y_range, x_range, indexing='ij')

        # Calculate flat indices to slice the original full-sized arrays
        flat_indices = (kv.ravel() * self.ny * self.nx + yv.ravel() * self.nx + xv.ravel())

        # --- MODIFIED: Use physical coordinates for node positions ---
        # Slice the pre-computed physical grid to get positions for the subgraph
        node_pos = self.grid_pos[flat_indices]

        # Get features (now including z-pos) and targets for this subgraph
        node_features = torch.tensor(self.features[flat_indices], dtype=torch.float)
        node_targets = torch.tensor(self.targets[flat_indices], dtype=torch.float)

        # --- MODIFIED: Build graph with physical positions and distances ---
        graph_data = self.grid_to_graph_manual(node_pos, node_features, node_targets, r=self.radius, xpos_idx=ix, ypos_idx=iy)

        # Calculate edge attributes based on physical distance
        if graph_data.edge_attr is None:
            row, col = graph_data.edge_index
            # edge_vectors are now differences in physical coordinates
            edge_vectors = graph_data.pos[row] - graph_data.pos[col]
            # edge_attr is the Euclidean distance in physical units
            graph_data.edge_attr = edge_vectors.norm(dim=1).unsqueeze(1)

        graph_data.u = torch.zeros((1, 1), dtype=torch.float)
        return graph_data